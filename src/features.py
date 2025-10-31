from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Any


@dataclass
class RollingSeries:
    window_ms: int
    values: Deque[tuple[int, float]] = field(default_factory=collections.deque)

    def add(self, ts_ms: int, value: float) -> None:
        self.values.append((ts_ms, value))
        self.compact(ts_ms)

    def compact(self, ts_ms: int) -> None:
        cutoff = ts_ms - self.window_ms
        while self.values and self.values[0][0] < cutoff:
            self.values.popleft()

    def sum(self) -> float:
        return sum(v for _, v in self.values)

    def mean(self) -> Optional[float]:
        if not self.values:
            return None
        return self.sum() / len(self.values)


@dataclass
class SymbolFeatures:
    vol_1s: RollingSeries
    delta_1s: RollingSeries
    ma_values: Dict[int, RollingSeries]
    last_mid: Optional[float] = None
    last_best_bid: Optional[float] = None
    last_best_ask: Optional[float] = None
    last_spread_bps: Optional[float] = None
    # Recent raw snapshots window for LLM summaries
    recent_snapshots: Deque[Dict[str, Any]] = field(default_factory=collections.deque)
    # Event-derived rolling features
    depth_events_1s: RollingSeries | None = None
    cancel_ratio_rs: RollingSeries | None = None
    orderflow_pressure_rs: RollingSeries | None = None
    depth_skew_rs: RollingSeries | None = None
    taker_buy_vol_rs: RollingSeries | None = None
    taker_sell_vol_rs: RollingSeries | None = None
    taker_buy_notional_rs: RollingSeries | None = None
    taker_sell_notional_rs: RollingSeries | None = None
    basis_rs: RollingSeries | None = None
    oi_delta_rs: RollingSeries | None = None
    liq_vol_rs: RollingSeries | None = None
    # For OI delta computation
    _last_oi: Optional[float] = None
    _last_oi_ts: Optional[int] = None


class FeatureEngine:
    def __init__(
        self,
        symbols: List[str],
        vol_window_s: int = 1,
        delta_window_s: int = 1,
        ma_windows: List[int] | None = None,
        snapshot_window_s: int = 60,
    ):
        if ma_windows is None:
            ma_windows = [7, 15, 30]
        self.snapshot_window_ms = int(snapshot_window_s) * 1000
        self.state: Dict[str, SymbolFeatures] = {}
        for s in symbols:
            st = SymbolFeatures(
                vol_1s=RollingSeries(window_ms=vol_window_s * 1000),
                delta_1s=RollingSeries(window_ms=delta_window_s * 1000),
                ma_values={w: RollingSeries(window_ms=w * 1000) for w in ma_windows},
            )
            # Use the snapshot window for derived event features so summarize_window can slice down further
            st.depth_events_1s = RollingSeries(window_ms=self.snapshot_window_ms)
            st.cancel_ratio_rs = RollingSeries(window_ms=self.snapshot_window_ms)
            st.orderflow_pressure_rs = RollingSeries(window_ms=self.snapshot_window_ms)
            st.depth_skew_rs = RollingSeries(window_ms=self.snapshot_window_ms)
            st.taker_buy_vol_rs = RollingSeries(window_ms=self.snapshot_window_ms)
            st.taker_sell_vol_rs = RollingSeries(window_ms=self.snapshot_window_ms)
            st.taker_buy_notional_rs = RollingSeries(window_ms=self.snapshot_window_ms)
            st.taker_sell_notional_rs = RollingSeries(window_ms=self.snapshot_window_ms)
            st.basis_rs = RollingSeries(window_ms=self.snapshot_window_ms)
            st.oi_delta_rs = RollingSeries(window_ms=self.snapshot_window_ms)
            st.liq_vol_rs = RollingSeries(window_ms=self.snapshot_window_ms)
            self.state[s] = st

    @staticmethod
    def compute_imbalance(
        best_bid: Optional[float],
        best_ask: Optional[float],
        bids: Dict[float, float] | None = None,
        asks: Dict[float, float] | None = None,
    ) -> Optional[float]:
        # Simple heuristic when full books not available: assume bid-side dominance
        # so that momentum rule can trigger under tight spreads.
        if bids is None or asks is None:
            if best_bid is None or best_ask is None:
                return None
            if best_ask == best_bid:
                return 0.0
            # Without depth we can't compute proper imbalance; return a strong
            # placeholder signal in the direction of the top-of-book (positive).
            return 1.0
        # Depth-aware version (sum first level volumes)
        bid_vol = sum(bids.values()) if bids else 0.0
        ask_vol = sum(asks.values()) if asks else 0.0
        denom = bid_vol + ask_vol
        if denom == 0:
            return None
        return (bid_vol - ask_vol) / denom

    def on_message(self, msg: Dict) -> Optional[Dict]:
        kind = msg.get("kind")
        symbol = msg.get("symbol")
        ts_ms = msg.get("ts_ms")
        st = self.state.get(symbol)
        if st is None or ts_ms is None:
            return None

        best_bid = None
        best_ask = None
        last_px = None
        last_qty = None
        if kind == "depth":
            best_bid = msg.get("best_bid")
            best_ask = msg.get("best_ask")
            st.last_best_bid = best_bid
            st.last_best_ask = best_ask
            # Depth-derived metrics
            num_changes = int(msg.get("num_changes") or 0)
            num_cancels = int(msg.get("num_cancels") or 0)
            bid_add = float(msg.get("bid_add") or 0.0)
            bid_remove = float(msg.get("bid_remove") or 0.0)
            ask_add = float(msg.get("ask_add") or 0.0)
            ask_remove = float(msg.get("ask_remove") or 0.0)
            size_best_bid = msg.get("size_best_bid")
            size_best_ask = msg.get("size_best_ask")

            # update rate: count one event
            if st.depth_events_1s is not None:
                st.depth_events_1s.add(ts_ms, 1.0)
            # cancel intensity per event
            cancel_ratio = (num_cancels / num_changes) if num_changes > 0 else 0.0
            if st.cancel_ratio_rs is not None:
                st.cancel_ratio_rs.add(ts_ms, float(cancel_ratio))
            # orderflow pressure per event
            # additions on bid and removals on ask push up; additions on ask and removals on bid push down
            pos = bid_add + ask_remove
            neg = ask_add + bid_remove
            denom = pos + neg
            of_pressure = (pos - neg) / denom if denom > 0 else 0.0
            if st.orderflow_pressure_rs is not None:
                st.orderflow_pressure_rs.add(ts_ms, float(of_pressure))
            # depth skew using best level sizes if present
            skew = None
            try:
                if size_best_bid is not None and size_best_ask is not None:
                    total = float(size_best_bid) + float(size_best_ask)
                    skew = ((float(size_best_bid) - float(size_best_ask)) / total) if total > 0 else 0.0
            except Exception:
                skew = None
            if skew is not None and st.depth_skew_rs is not None:
                st.depth_skew_rs.add(ts_ms, float(skew))
        elif kind == "aggTrade":
            last_px = msg.get("price")
            last_qty = msg.get("qty")
            # taker side: if buyer is maker -> taker is seller
            taker_is_buy = not bool(msg.get("is_buyer_maker", False))
            if last_qty is not None:
                if taker_is_buy and st.taker_buy_vol_rs is not None:
                    st.taker_buy_vol_rs.add(ts_ms, float(last_qty))
                elif not taker_is_buy and st.taker_sell_vol_rs is not None:
                    st.taker_sell_vol_rs.add(ts_ms, float(last_qty))
            # Notional flow and basis
            try:
                px_for_calc = last_px if last_px is not None else st.last_mid
                if px_for_calc is not None and last_qty is not None:
                    notional = float(px_for_calc) * float(last_qty)
                    if taker_is_buy and st.taker_buy_notional_rs is not None:
                        st.taker_buy_notional_rs.add(ts_ms, notional)
                    elif not taker_is_buy and st.taker_sell_notional_rs is not None:
                        st.taker_sell_notional_rs.add(ts_ms, notional)
                # Basis = trade price - mid
                if last_px is not None and st.last_mid is not None and st.basis_rs is not None:
                    st.basis_rs.add(ts_ms, float(last_px) - float(st.last_mid))
            except Exception:
                pass
        elif kind == "trade":
            last_px = msg.get("price")
            last_qty = msg.get("qty")
            taker_is_buy = not bool(msg.get("is_buyer_maker", False))
            if last_qty is not None:
                if taker_is_buy and st.taker_buy_vol_rs is not None:
                    st.taker_buy_vol_rs.add(ts_ms, float(last_qty))
                elif not taker_is_buy and st.taker_sell_vol_rs is not None:
                    st.taker_sell_vol_rs.add(ts_ms, float(last_qty))
            # Notional flow and basis
            try:
                px_for_calc = last_px if last_px is not None else st.last_mid
                if px_for_calc is not None and last_qty is not None:
                    notional = float(px_for_calc) * float(last_qty)
                    if taker_is_buy and st.taker_buy_notional_rs is not None:
                        st.taker_buy_notional_rs.add(ts_ms, notional)
                    elif not taker_is_buy and st.taker_sell_notional_rs is not None:
                        st.taker_sell_notional_rs.add(ts_ms, notional)
                if last_px is not None and st.last_mid is not None and st.basis_rs is not None:
                    st.basis_rs.add(ts_ms, float(last_px) - float(st.last_mid))
            except Exception:
                pass
        elif kind == "openInterest":
            oi = msg.get("open_interest")
            if isinstance(oi, (int, float)):
                prev = st._last_oi
                prev_ts = st._last_oi_ts
                if prev is not None and prev_ts is not None and ts_ms is not None:
                    dt_s = max(0.0, (ts_ms - prev_ts) / 1000.0)
                    if dt_s > 0:
                        delta_per_s = (float(oi) - float(prev)) / dt_s
                        if st.oi_delta_rs is not None:
                            st.oi_delta_rs.add(ts_ms, delta_per_s)
                st._last_oi = float(oi)
                st._last_oi_ts = ts_ms
        elif kind == "forceOrder":
            price = msg.get("price")
            qty = msg.get("qty")
            val = None
            try:
                if price is not None and qty is not None:
                    val = float(price) * float(qty)
                elif qty is not None:
                    val = float(qty)
            except Exception:
                val = None
            if val is not None and st.liq_vol_rs is not None:
                st.liq_vol_rs.add(ts_ms, float(val))
        elif kind == "kline":
            pass

        mid = None
        if best_bid is not None and best_ask is not None:
            mid = 0.5 * (best_bid + best_ask)
            st.last_mid = mid
        else:
            mid = st.last_mid

        spread_bps = None
        # For trades, fallback to last known best bid/ask to compute spread
        bb = best_bid if best_bid is not None else st.last_best_bid
        aa = best_ask if best_ask is not None else st.last_best_ask
        if bb is not None and aa is not None and mid:
            spread_bps = ((aa - bb) / mid) * 1e4
            st.last_spread_bps = spread_bps
        else:
            spread_bps = st.last_spread_bps

        if last_qty is not None:
            st.vol_1s.add(ts_ms, float(last_qty))

        if last_px is not None and st.last_mid is not None:
            delta = float(last_px) - st.last_mid
            st.delta_1s.add(ts_ms, delta)

        if mid is not None:
            for w, rs in st.ma_values.items():
                rs.add(ts_ms, float(mid))

        ob_imbalance = self.compute_imbalance(bb, aa)
        vol_1s = st.vol_1s.sum()
        delta_1s = st.delta_1s.sum()

        snapshot = {
            "ts_ms": ts_ms,
            "symbol": symbol,
            "bid": best_bid,
            "ask": best_ask,
            "mid": mid,
            "last_px": last_px,
            "last_qty": last_qty,
            "ob_imbalance": ob_imbalance,
            "spread_bps": spread_bps,
            "vol_1s": vol_1s,
            "delta_1s": delta_1s,
            # Event-level derived metrics (may be None for non-applicable events)
            "orderflow_pressure_event": of_pressure if 'of_pressure' in locals() else None,
            "cancel_ratio_event": cancel_ratio if 'cancel_ratio' in locals() else None,
            "depth_skew_event": skew if 'skew' in locals() else None,
            "taker_buy_event": float(last_qty) if kind in ("aggTrade", "trade") and 'taker_is_buy' in locals() and taker_is_buy and last_qty is not None else None,
            "taker_sell_event": float(last_qty) if kind in ("aggTrade", "trade") and 'taker_is_buy' in locals() and (not taker_is_buy) and last_qty is not None else None,
            "taker_buy_notional_event": (float(last_px) * float(last_qty)) if kind in ("aggTrade", "trade") and 'taker_is_buy' in locals() and taker_is_buy and last_px is not None and last_qty is not None else None,
            "taker_sell_notional_event": (float(last_px) * float(last_qty)) if kind in ("aggTrade", "trade") and 'taker_is_buy' in locals() and (not taker_is_buy) and last_px is not None and last_qty is not None else None,
            "basis_event": (float(last_px) - float(st.last_mid)) if kind in ("aggTrade", "trade") and last_px is not None and st.last_mid is not None else None,
            "oi_delta_event": (st.oi_delta_rs.values[-1][1] if st.oi_delta_rs and st.oi_delta_rs.values else None),
            "liq_volume_event": (st.liq_vol_rs.values[-1][1] if st.liq_vol_rs and st.liq_vol_rs.values else None),
        }
        # Store in recent window
        st.recent_snapshots.append(snapshot)
        # Compact recent window by cutoff
        cutoff = ts_ms - self.snapshot_window_ms if ts_ms is not None else None
        if cutoff is not None:
            while (
                st.recent_snapshots
                and (st.recent_snapshots[0].get("ts_ms") or 0) < cutoff
            ):
                st.recent_snapshots.popleft()
        return snapshot

    # -------- Public APIs for LLM summaries --------
    def get_recent_window(
        self, symbol: str, window_s: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        st = self.state.get(symbol)
        if st is None:
            return []
        if not st.recent_snapshots:
            return []
        if window_s is None:
            return list(st.recent_snapshots)
        last_ts = st.recent_snapshots[-1].get("ts_ms") or 0
        cutoff = last_ts - int(window_s) * 1000
        return [row for row in st.recent_snapshots if (row.get("ts_ms") or 0) >= cutoff]

    def summarize_window(
        self, symbol: str, window_s: Optional[int] = None
    ) -> Dict[str, Any]:
        rows = self.get_recent_window(symbol, window_s)
        if not rows:
            return {"symbol": symbol, "count": 0}

        # Extract series
        ts_list = [r.get("ts_ms") for r in rows if r.get("ts_ms") is not None]
        mids = [r.get("mid") for r in rows if r.get("mid") is not None]
        spreads = [r.get("spread_bps") for r in rows if r.get("spread_bps") is not None]
        imbs = [
            r.get("ob_imbalance") for r in rows if r.get("ob_imbalance") is not None
        ]
        vols = [r.get("last_qty") for r in rows if r.get("last_qty") is not None]
        of_pressures = [
            r.get("orderflow_pressure_event")
            for r in rows
            if r.get("orderflow_pressure_event") is not None
        ]
        depth_skews = [
            r.get("depth_skew_event")
            for r in rows
            if r.get("depth_skew_event") is not None
        ]
        cancel_ratios = [
            r.get("cancel_ratio_event")
            for r in rows
            if r.get("cancel_ratio_event") is not None
        ]
        taker_buys = [
            r.get("taker_buy_event") for r in rows if r.get("taker_buy_event") is not None
        ]
        taker_sells = [
            r.get("taker_sell_event") for r in rows if r.get("taker_sell_event") is not None
        ]
        oi_deltas = [
            r.get("oi_delta_event") for r in rows if r.get("oi_delta_event") is not None
        ]
        liq_vols = [
            r.get("liq_volume_event")
            for r in rows
            if r.get("liq_volume_event") is not None
        ]
        buy_notional = [
            r.get("taker_buy_notional_event")
            for r in rows
            if r.get("taker_buy_notional_event") is not None
        ]
        sell_notional = [
            r.get("taker_sell_notional_event")
            for r in rows
            if r.get("taker_sell_notional_event") is not None
        ]
        basis_vals = [
            r.get("basis_event") for r in rows if r.get("basis_event") is not None
        ]
        basis_ts = [
            r.get("ts_ms") for r in rows if r.get("basis_event") is not None and r.get("ts_ms") is not None
        ]

        def _mean(vals: List[float]) -> Optional[float]:
            return sum(vals) / len(vals) if vals else None

        def _std(vals: List[float]) -> Optional[float]:
            if not vals or len(vals) < 2:
                return None
            m = sum(vals) / len(vals)
            var = sum((v - m) * (v - m) for v in vals) / (len(vals) - 1)
            return var**0.5

        def _slope_over_time(values: List[float], ts_ms: List[int]) -> Optional[float]:
            if not values or not ts_ms or len(values) < 2 or len(ts_ms) < 2:
                return None
            # Align lengths conservatively using the last N where both exist
            n = min(len(values), len(ts_ms))
            y = values[-n:]
            t = ts_ms[-n:]
            t0 = t[0]
            t1 = t[-1]
            dt_s = (t1 - t0) / 1000.0 if t1 is not None and t0 is not None else 0.0
            if dt_s <= 0:
                return 0.0
            return (y[-1] - y[0]) / dt_s

        def _spike_score(vals: List[float]) -> Optional[float]:
            if not vals:
                return None
            m = _mean(vals)
            s = _std(vals)
            if m is None or s is None or s == 0:
                return 0.0
            return (max(vals) - m) / s

        mid_stats = {
            "mean": _mean(mids),
            "min": min(mids) if mids else None,
            "max": max(mids) if mids else None,
            "slope": _slope_over_time(mids, ts_list),
        }
        spread_stats = {
            "mean": _mean(spreads),
            "std": _std(spreads),
        }
        imb_stats = {
            "mean": _mean(imbs),
            "trend": _slope_over_time(imbs, ts_list),
        }
        vol_stats = {
            "sum": sum(vols) if vols else 0.0,
            "spike_score": _spike_score(vols),
        }

        # Additional features
        trade_buy = sum(taker_buys) if taker_buys else 0.0
        trade_sell = sum(taker_sells) if taker_sells else 0.0
        trade_den = trade_buy + trade_sell
        trade_aggr = (trade_buy / trade_den) if trade_den > 0 else None

        # ---- Synthetic OI delta estimate ----
        # 1) Net taker notional flow (dimensionless in [-1,1])
        net_notional = (sum(buy_notional) if buy_notional else 0.0) - (
            sum(sell_notional) if sell_notional else 0.0
        )
        tot_notional = (sum(buy_notional) if buy_notional else 0.0) + (
            sum(sell_notional) if sell_notional else 0.0
        )
        norm_net_notional = (net_notional / tot_notional) if tot_notional > 0 else 0.0

        # 2) Depth persistence proxy (high when cancels are low)
        mean_cancel = _mean(cancel_ratios) if cancel_ratios else None
        persistence = max(0.0, 1.0 - float(mean_cancel)) if mean_cancel is not None else 0.0

        # 3) Short-term basis change (slope of trade-mid difference per second, normalized by mid mean)
        basis_slope = _slope_over_time(basis_vals, basis_ts) if basis_vals and basis_ts else None
        mid_mean = mid_stats.get("mean")
        basis_rel = (float(basis_slope) / float(mid_mean)) if (basis_slope is not None and mid_mean) else 0.0

        # Combine components (heuristic weights)
        alpha, beta, gamma = 0.6, 0.3, 0.1
        oi_delta_est = (alpha * norm_net_notional) + (beta * (float(_mean(of_pressures)) if of_pressures else 0.0) * persistence) + (gamma * basis_rel)
        # Clamp to [-1, 1]
        if oi_delta_est > 1.0:
            oi_delta_est = 1.0
        elif oi_delta_est < -1.0:
            oi_delta_est = -1.0

        # liquidation_burst strictly from forceOrder events: explicit 0.0 when none
        liq_burst = 0.0 if not liq_vols else _spike_score(liq_vols)

        return {
            "symbol": symbol,
            "count": len(rows),
            "mid": mid_stats,
            "spread_bps": spread_stats,
            "ob_imbalance": imb_stats,
            "volume": vol_stats,
            # New compact features for LLM
            "orderflow_pressure": _mean(of_pressures) if of_pressures else None,
            "depth_skew": _mean(depth_skews) if depth_skews else None,
            "cancel_intensity": _mean(cancel_ratios) if cancel_ratios else None,
            "trade_aggression": trade_aggr,
            "oi_delta": oi_delta_est,
            "liquidation_burst": liq_burst,
        }
