## Binance WSS → LLM Payload Summary

### Overview
- **Goal**: Provide the LLM with compact, recent market context per symbol.
- **Source**: Binance WebSocket streams (depth and trades) aggregated into rolling summaries.
- **Output to LLM**: `DATA_WINDOW` — array of per‑symbol summary objects.

### WebSocket inputs
- **Endpoints**:
  - Spot: `wss://stream.binance.com:9443/stream?streams=...`
  - USDⓈ‑M Futures: `wss://fstream.binance.com/stream?streams=...`
- **Subscribed per symbol**:
  - `depth@100ms` (full incremental) → top‑of‑book: `best_bid`, `best_ask`
  - Or partial books: `depth5@100ms`, `depth10@100ms`, or `depth20@100ms`
  - `aggTrade` and/or `trade` → last trade: `price`, `quantity`, `is_buyer_maker`
  - `kline_1s` is subscribed but **not** used for the LLM payload
  - Futures mode (optional): `fundingRate`, `openInterest`, `forceOrder` (liquidations)

### Derived fields sent to the LLM (per symbol)
From the rolling window of recent snapshots:
- **mid** (from `best_bid`/`best_ask`): `{ mean, min, max, slope }`
- **spread_bps** (from `best_bid`/`best_ask` and mid): `{ mean, std }`
- **ob_imbalance** (heuristic using top‑of‑book info): `{ mean, trend }`
- **volume** (from trade `quantity`): `{ sum, spike_score }`
- **count**: number of snapshots in the window
  
Additional microstructure features (averaged over the window unless noted):
- **orderflow_pressure**: normalized pressure in [−1, 1] from depth add/remove deltas
- **depth_skew**: best‑level size skew in [−1, 1] (`bid_size` vs `ask_size`)
- **cancel_intensity**: ratio of cancels to total depth changes [0, 1]
- **trade_aggression**: taker buy share in [0, 1] from `aggTrade`/`trade`
- **oi_delta**: Δ open interest per second (futures only)
- **liquidation_burst**: spike score of liquidation volume (futures only)

Notes:
- Snapshots are timestamped internally; the LLM receives only the aggregated stats above plus `symbol`.
- Intermediate metrics like `vol_1s` and `delta_1s` are used to build the summaries but are not sent directly.

### How each metric is calculated
The summary is produced by `FeatureEngine.summarize_window(symbol, window_s)` over the most recent `window_s` seconds of snapshots.

- **count**
  - Number of snapshot rows within the window for the symbol.

- **mid**
  - Per snapshot: `mid_t = (best_ask_t + best_bid_t) / 2`
  - `mean`: arithmetic mean of `mid_t` across the window
  - `min` / `max`: extrema of `mid_t` across the window
  - `slope`: price units per second, computed as `(mid_last - mid_first) / dt_seconds`, where `dt_seconds = (ts_last - ts_first) / 1000`

- **spread_bps**
  - Per snapshot: `spread_bps_t = ((best_ask_t - best_bid_t) / mid_t) * 1e4`
  - `mean`: arithmetic mean of `spread_bps_t`
  - `std`: sample standard deviation of `spread_bps_t` (uses N−1 denominator; undefined or <2 samples → omitted)
  - Units: basis points (bps). Example value `0.000896...` means a very tight spread (sub‑bps).

- **ob_imbalance**
  - With only top‑of‑book available (no depth volumes provided to the summary), a heuristic is used:
    - If `best_ask == best_bid`: `ob_imbalance_t = 0.0`
    - Else: `ob_imbalance_t = 1.0` (placeholder indicating bid‑side dominance under tight spreads)
  - `mean`: mean of `ob_imbalance_t` across the window
  - `trend`: slope over time of `ob_imbalance_t` (same method as `mid.slope`), typically ~0.0 when the series is constant
  - Note: If full depth volumes were supplied, imbalance would be `(sum(bid_volumes) - sum(ask_volumes)) / (sum(bid_volumes) + sum(ask_volumes))` in [−1, 1]. Current path uses the top‑of‑book heuristic only.

- **volume**
  - Per event: trade quantity from `aggTrade` (`last_qty`) is accumulated into the rolling window.
  - `sum`: total of trade quantities within the window
  - `spike_score`: `(max(volumes) − mean(volumes)) / std(volumes)`;
    - If fewer than 2 samples or `std == 0`, the score is `0.0`.

- **orderflow_pressure**
  - From depth deltas per update: 
    - `pos = bid_add + ask_remove`, `neg = ask_add + bid_remove`
    - `pressure_t = (pos − neg) / (pos + neg)` (guarded when denominator ≤ 0)
  - Reported as the mean of `pressure_t` across the window.

- **depth_skew**
  - Uses best level sizes when available: 
    - `skew_t = (size_best_bid − size_best_ask) / (size_best_bid + size_best_ask)` (guarded when denominator ≤ 0)
  - Reported as the mean of `skew_t` across the window.

- **cancel_intensity**
  - Per depth update: `cancel_ratio_t = cancelled_orders / total_changes` (guarded when total_changes == 0)
  - Reported as the mean of `cancel_ratio_t` across the window.

- **trade_aggression**
  - From `aggTrade`/`trade`: classify taker side using `is_buyer_maker`.
  - Accumulate taker buy and taker sell volumes; `aggr = buy_vol / (buy_vol + sell_vol)` (guarded when denominator == 0).

- **oi_delta** (futures only)
  - Requires `openInterest` stream. For successive OI ticks: 
    - `delta_per_s = (oi_t − oi_{t−1}) / Δt_seconds`
  - Reported as the mean of `delta_per_s` across the window.

- **liquidation_burst** (futures only)
  - Requires `forceOrder` stream. For each liquidation event, compute notional (price×qty) if possible (fallback: qty).
  - `spike_score` over liquidation series: `(max − mean) / std` (0.0 when <2 samples or `std == 0`).

### When fields may be null
- **oi_delta**: null if futures mode is off, `openInterest` is disabled, or fewer than two OI updates arrived within the window.
- **liquidation_burst**: null if futures mode is off, `forceOrder` is disabled, or no liquidation events occurred in the window.

### Example alignment with observed data
Given an entry like:

```
count: 1130
mid: { mean: 111587.44, min: 111522.505, max: 111634.205, slope: 0.32 }
spread_bps: { mean: 0.000896, std: 0.0000002576 }
ob_imbalance: { mean: 1.0, trend: 0.0 }
volume: { sum: 7.67241, spike_score: 18.7475 }
```

- `count` reflects the number of snapshots in the time window.
- `mid` values are computed from top‑of‑book; `slope` is price change per second across the window.
- `spread_bps` stats measure tightness of the spread in bps.
- `ob_imbalance` being `1.0` with `trend 0.0` indicates the top‑of‑book heuristic path (no depth volume), constant over the window.
- `volume.sum` totals trade quantities; `spike_score` signals a relative outlier spike vs recent volume variability.

### Variables included in the LLM request
- `symbols`: list of tracked symbols
- `window_seconds`: rolling window size used for the summaries
- `DATA_WINDOW`: array of per‑symbol summary objects described above

### Explicitly not included in the LLM payload
- Full order book depth levels beyond top‑of‑book
- Detailed kline OHLCV values
- Raw per‑trade event streams


