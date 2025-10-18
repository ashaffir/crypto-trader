# Strategy Specification — Binance Spot Signal Bot

This document describes the live momentum strategy implemented in the bot: inputs, computed features, signal logic, control parameters, and evaluation. It uses simple equations and a flow schematic for clarity.

## 1) Data Inputs

- Binance WebSocket streams (for symbol `S`, e.g., `BTCUSDT`):
  - aggTrade: last trade price and size
  - depth@100ms: best bid/ask updates and top-of-book changes
  - kline_1s: 1-second candle (currently lightly used)

## 2) Feature Computations

Let \( b_t \) = best bid, \( a_t \) = best ask, \( p_t \) = last trade price, \( q_t \) = last trade size (when present). Mid price:

\[ m_t = \frac{a_t + b_t}{2} \]

Spread in basis points:

\[ \text{spread\_bps}_t = \frac{a_t - b_t}{m_t} \times 10{,}000 \]

1-second traded volume (cumulative, rolling window):

\[ \text{vol\_1s}_t = \sum_{i\in (t-1s,\ t]} q_i \]

1-second mid deviation (rolling sum):

\[ \text{delta\_1s}_t = \sum_{i\in (t-1s,\ t]} (p_i - m_i) \]

Order book imbalance (heuristic in [-1, 1]). With only top-of-book available, a simplified imbalance is used to reflect side dominance:

\[ \text{ob\_imbalance}_t \in [-1, 1] \quad (>0: \text{bid-dominant}, <0: \text{ask-dominant}) \]

Notes:
- The implementation accumulates rolling metrics and updates them on each message. When full book data is thin, imbalance uses a safe heuristic to preserve directionality under tight spreads.

## 3) Signal Logic (Momentum Rule)

Intuition: When the order book is strongly skewed and spreads are tight (cheap to cross), follow the skew direction briefly.

Parameters (from UI/runtime config):
- Imbalance threshold: \( I_{thr} \in [0,1] \)
- Max spread (bps): \( S_{max} > 0 \)
- Horizon (seconds): \( H \)
- Signal TTL (seconds): \( \text{TTL} \)

Trigger condition at time \( t \):

\[ |\text{ob\_imbalance}_t| \ge I_{thr} \quad \land \quad \text{spread\_bps}_t \le S_{max} \]

Direction:

\[ \text{side}_t = \begin{cases}
\text{long}, & \text{if } \text{ob\_imbalance}_t > 0 \\
\text{short}, & \text{if } \text{ob\_imbalance}_t < 0
\end{cases} \]

Expected return target in bps (constant per direction):

\[ \text{expected\_bps}_t = \begin{cases}
 +8, & \text{if side = long} \\
 -8, & \text{if side = short}
\end{cases} \]

Confidence (scaled from imbalance):

\[ \text{confidence}_t = \min\left(0.99,\ 0.5 + 0.5\cdot\min\left(1, |\text{ob\_imbalance}_t|\right)\right) \]

Signal payload fields:
- `symbol`, `side`, `expected_bps`, `confidence`, `horizon_s = H`, `ttl_s = TTL`, `rule_id = "mom_v1"`, `ts_ms` (timestamp), `signal_id` (UUID)

Mean-reversion rule exists as a placeholder switch; momentum is the primary live rule.

## 4) Flow Schematic

```
      [Binance WS Streams]
      (aggTrade, depth@100ms, kline_1s)
                |
                v
        [Collector]  -- parses messages -->  {kind, symbol, ts_ms, price/qty, bid/ask}
                |
                v
         [Feature Engine] -- updates -->  m_t, spread_bps_t, vol_1s_t, delta_1s_t, ob_imbalance_t
                |
                v
         [Signal Engine]
     if |imbalance|>=I_thr and spread_bps<=S_max
                |
                v
           emit SIGNAL {side, expected_bps, confidence, H, TTL}
                |
                v
        [Parquet Logbook]  (signal_emitted, market_snapshot)
                |
                v
         [Evaluator]  (after H seconds computes outcomes)
                |
                v
        [Parquet Logbook]  (signal_outcome)

         [UI]
     - Controls (I_thr, S_max, H, TTL, rule toggles, RUN)
     - Recent Signals and Outcomes
```

## 5) Controls and Their Effects

- Momentum enabled: turn momentum rule on/off.
- Mean-reversion enabled: toggle placeholder rule on/off.
- Imbalance threshold \( I_{thr} \): higher → fewer but stronger signals; lower → more signals.
- Max spread (bps) \( S_{max} \): lower → only in tight markets; higher → more signals but potentially worse fills.
- Horizon \( H \): evaluation window; outcomes computed H seconds after emission.
- Signal TTL \( \text{TTL} \): how long a signal stays fresh for execution.
- Bot Running: starts/stops the live pipeline.

## 6) Practical Tuning

- To increase signal frequency: lower \( I_{thr} \) (e.g., 0.45) and raise \( S_{max} \) (e.g., 2.0–3.0).
- To be more selective: raise \( I_{thr} \) and/or lower \( S_{max} \).
- Use \( H \) to match the time you expect the move to resolve; \( \text{TTL} \) sets execution freshness.

## 7) Example Walkthrough

Assume at time \( t \): \( a_t=100.02 \), \( b_t=100.00 \). Then \( m_t=100.01 \), \( \text{spread\_bps}_t \approx 2.0 \).

Order book indicates bid dominance, \( \text{ob\_imbalance}_t = 0.7 \).

With \( I_{thr}=0.6 \) and \( S_{max}=2.5 \): condition holds → emit a long signal with `expected_bps=+8`, `confidence≈0.85`, `horizon_s=H`, `ttl_s=TTL`.

## 8) Storage and Monitoring

- Signals: `data/logbook/signal_emitted/symbol=S/date=YYYY-MM-DD/*.parquet`
- Snapshots: `data/logbook/market_snapshot/...`
- Outcomes: `data/logbook/signal_outcome/...`
- UI shows recent signals/outcomes and live status; parameters hot-reload via runtime config.


