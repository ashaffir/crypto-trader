# Strategy Specification — Binance Spot Signal Bot

This document describes the live momentum and mean-reversion strategies implemented in the bot: inputs, computed features, signal logic, control parameters, and evaluation. It uses simple equations and a flow schematic for clarity.

## 1) Data Inputs

- Binance WebSocket streams (for symbol `S`, e.g., `BTCUSDT`):
  - aggTrade: last trade price and size
  - depth@100ms: best bid/ask updates and top-of-book changes
  - kline_1s: 1-second candle (currently lightly used)

## 2) Feature Computations

Let variables: `b_t` = best bid, `a_t` = best ask, `p_t` = last trade price, `q_t` = last trade size (when present).

Mid price:

```
m_t = (a_t + b_t) / 2
```

Spread in basis points:

```
spread_bps_t = ((a_t - b_t) / m_t) * 10000
```

1-second traded volume (cumulative, rolling window):

```
vol_1s_t = sum_{i in (t-1s, t]} q_i
```

1-second mid deviation (rolling sum):

```
delta_1s_t = sum_{i in (t-1s, t]} (p_i - m_i)
```

Order book imbalance (heuristic in [-1, 1]). With only top-of-book available, a simplified imbalance is used to reflect side dominance:

```
ob_imbalance_t ∈ [-1, 1]   (>0: bid-dominant, <0: ask-dominant)
```

Notes:
- The implementation accumulates rolling metrics and updates them on each message. When full book data is thin, imbalance uses a safe heuristic to preserve directionality under tight spreads.

## 3) Signal Logic (Momentum Rule)

Intuition: When the order book is strongly skewed and spreads are tight (cheap to cross), follow the skew direction briefly.

Parameters (from UI/runtime config):
- Imbalance threshold: `I_thr` in [0, 1]
- Max spread (bps): `S_max` > 0
- Horizon (seconds): `H`
- Signal TTL (seconds): `TTL`

Trigger condition at time `t`:

```
|ob_imbalance_t| >= I_thr  AND  spread_bps_t <= S_max
```

Direction:

```
side_t =
  long  if ob_imbalance_t > 0
  short if ob_imbalance_t < 0
```

Expected return target in bps (constant per direction):

```
expected_bps_t = +8   (if side = long)
expected_bps_t = -8   (if side = short)
```

Confidence (scaled from imbalance):

```
confidence_t = min(0.99, 0.5 + 0.5 * min(1, |ob_imbalance_t|))
```

Signal payload fields:
- `symbol`, `side`, `expected_bps`, `confidence`, `horizon_s = H`, `ttl_s = TTL`, `rule_id = "mom_v1"`, `ts_ms` (timestamp), `signal_id` (UUID)

Momentum is the primary directional rule.

### Mean-Reversion Rule (mr_v2)

Intuition: When recent trades deviate from mid enough to suggest a short-term overshoot, and spreads are acceptable, fade the move back toward mid. Optionally avoid strong one-sided order-book conditions.

Parameters (from UI/runtime config):
- MR min revert (bps): `MR_min` > 0 — minimum 1s deviation vs mid, in bps
- MR expected bps: `MR_exp` > 0 — directional return target in bps
- MR conf norm (bps): `MR_norm` > 0 — scales confidence with deviation
- MR max |imbalance|: `MR_imb_max` in [0, 1] — guard to skip when order book is too one-sided (set to 1.0 to disable)
- Max spread (bps): `S_max` > 0 — reused from momentum
- Horizon `H`, TTL `TTL`: shared with momentum

Trigger at time `t` (with mid `m_t` and 1s deviation `delta_1s_t`):

```
dev_bps_t = |delta_1s_t| / m_t * 10000
dev_bps_t >= MR_min AND spread_bps_t <= S_max AND |ob_imbalance_t| <= MR_imb_max
```

Direction and targets:

```
side_t = short if delta_1s_t > 0 else long
expected_bps_t = -MR_exp if side_t = short else +MR_exp
```

Confidence scales with deviation magnitude and penalizes strong imbalance (capped):

```
norm = min(1, dev_bps_t / MR_norm)
imb_penalty = 1 - 0.3 * min(1, |ob_imbalance_t|)
confidence_t = clamp(0.5, 0.95, 0.55 + 0.35 * norm * imb_penalty)
```

## 4) Flow Schematic

```
[Binance WS Streams: aggTrade, depth@100ms, kline@1s]
            |
            v
[Collector] --parses--> {kind, symbol, ts_ms, price/qty, bid/ask}
            |
            v
[Feature Engine] --updates--> m_t, spread_bps_t, vol_1s_t, delta_1s_t, ob_imbalance_t
            |
            v
[Rule Check]  |ob_imbalance_t| >= I_thr  AND  spread_bps_t <= S_max ?
       |                          \
       |yes                        \-- no --> [Parquet Logbook: market_snapshot]
       v
[Emit Signal] side, expected_bps, confidence, H, TTL
       |
       v
[Parquet Logbook: signal_emitted]
       |
       v
[Evaluator (after H seconds)]
       |
       v
[Parquet Logbook: signal_outcome]

UI (controls and views) ----reads----> market_snapshot / signal_emitted / signal_outcome
```

## 5) Controls and Their Effects

- Momentum enabled: turn momentum rule on/off.
- Mean-reversion enabled: turn mean-reversion on/off.
- Imbalance threshold `I_thr`: higher → fewer but stronger signals; lower → more signals.
- Max spread (bps) `S_max`: lower → only in tight markets; higher → more signals but potentially worse fills.
- MR min revert (bps) `MR_min`: higher → only larger overshoots; lower → more frequent fades.
- MR expected bps `MR_exp`: target distance for mean reversion; larger increases profit target but may reduce hit rate.
- MR conf norm (bps) `MR_norm`: how quickly confidence rises with deviation magnitude.
- MR max |imbalance| `MR_imb_max`: lower to avoid fading when the book is heavily one-sided.
- Horizon `H`: evaluation window; outcomes computed H seconds after emission.
- Signal TTL `TTL`: how long a signal stays fresh for execution.
- Bot Running: starts/stops the live pipeline.

## 6) Practical Tuning

- To increase signal frequency: lower `I_thr` (e.g., 0.45) and raise `S_max` (e.g., 2.0–3.0).
- To be more selective: raise `I_thr` and/or lower `S_max`.
- Use `H` to match the time you expect the move to resolve; `TTL` sets execution freshness.

## 7) Example Walkthrough

Assume at time `t`: `a_t = 100.02`, `b_t = 100.00`. Then `m_t = 100.01`, `spread_bps_t ≈ 2.0`.

Order book indicates bid dominance, `ob_imbalance_t = 0.7`.

With `I_thr = 0.6` and `S_max = 2.5`: condition holds → emit a long signal with `expected_bps = +8`, `confidence ≈ 0.85`, `horizon_s = H`, `ttl_s = TTL`.

## 8) Storage and Monitoring

- Signals: `data/logbook/signal_emitted/symbol=S/date=YYYY-MM-DD/*.parquet`
- Snapshots: `data/logbook/market_snapshot/...`
- Outcomes: `data/logbook/signal_outcome/...`
- UI shows recent signals/outcomes and live status; parameters hot-reload via runtime config.


