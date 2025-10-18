from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional


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


class FeatureEngine:
    def __init__(
        self,
        symbols: List[str],
        vol_window_s: int = 1,
        delta_window_s: int = 1,
        ma_windows: List[int] | None = None,
    ):
        if ma_windows is None:
            ma_windows = [7, 15, 30]
        self.state: Dict[str, SymbolFeatures] = {
            s: SymbolFeatures(
                vol_1s=RollingSeries(window_ms=vol_window_s * 1000),
                delta_1s=RollingSeries(window_ms=delta_window_s * 1000),
                ma_values={w: RollingSeries(window_ms=w * 1000) for w in ma_windows},
            )
            for s in symbols
        }

    @staticmethod
    def compute_imbalance(
        best_bid: Optional[float],
        best_ask: Optional[float],
        bids: Dict[float, float] | None = None,
        asks: Dict[float, float] | None = None,
    ) -> Optional[float]:
        # Simple micro imbalance using top only when books not available
        if bids is None or asks is None:
            if best_bid is None or best_ask is None:
                return None
            mid = 0.5 * (best_bid + best_ask)
            return (
                (best_bid - (mid - (best_ask - mid))) / (best_ask - best_bid)
                if best_ask != best_bid
                else 0.0
            )
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
        elif kind == "aggTrade":
            last_px = msg.get("price")
            last_qty = msg.get("qty")
        elif kind == "kline":
            pass

        mid = None
        if best_bid is not None and best_ask is not None:
            mid = 0.5 * (best_bid + best_ask)
            st.last_mid = mid
        else:
            mid = st.last_mid

        spread_bps = None
        if best_bid is not None and best_ask is not None and mid:
            spread_bps = ((best_ask - best_bid) / mid) * 1e4

        if last_qty is not None:
            st.vol_1s.add(ts_ms, float(last_qty))

        if last_px is not None and st.last_mid is not None:
            delta = float(last_px) - st.last_mid
            st.delta_1s.add(ts_ms, delta)

        if mid is not None:
            for w, rs in st.ma_values.items():
                rs.add(ts_ms, float(mid))

        ob_imbalance = self.compute_imbalance(best_bid, best_ask)
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
        }
        return snapshot
