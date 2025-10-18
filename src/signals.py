from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from loguru import logger


@dataclass
class Signal:
    symbol: str
    side: str  # "long" or "short"
    expected_bps: float
    confidence: float
    horizon_s: int
    ttl_s: int
    rationale: str
    rule_id: str


class SignalEngine:
    def __init__(self, thresholds: Dict, horizons: Dict, rules: Dict):
        self.thr = thresholds
        self.hz = horizons
        self.rules = rules

    def on_features(self, snap: Dict) -> Optional[Dict]:
        if not snap or not snap.get("symbol"):
            return None
        symbol = snap["symbol"]
        ob = snap.get("ob_imbalance")
        spread_bps = snap.get("spread_bps")
        delta_1s = snap.get("delta_1s")

        # Momentum rule
        if self.rules.get("momentum_enabled", True):
            if ob is not None and spread_bps is not None:
                if abs(ob) >= self.thr.get(
                    "imbalance", 0.6
                ) and spread_bps <= self.thr.get("max_spread_bps", 1.5):
                    side = "long" if ob > 0 else "short"
                    expected_bps = 8.0 if side == "long" else -8.0
                    conf = min(0.99, 0.5 + 0.5 * min(1.0, abs(ob)))
                    return {
                        "symbol": symbol,
                        "side": side,
                        "expected_bps": expected_bps,
                        "confidence": conf,
                        "horizon_s": int(self.hz.get("scalp", 30)),
                        "ttl_s": int(self.hz.get("ttl_s", 10)),
                        "rationale": "imbalance+low_spread",
                        "rule_id": "mom_v1",
                    }

        # Mean reversion rule (very simple placeholder)
        if self.rules.get("mean_reversion_enabled", True):
            if (
                delta_1s is not None
                and spread_bps is not None
                and spread_bps <= self.thr.get("max_spread_bps", 1.5)
            ):
                if abs(delta_1s) > 0 and abs(delta_1s) > (snap.get("mid") or 1) * 1e-5:
                    side = "short" if delta_1s > 0 else "long"
                    expected_bps = -5.0 if delta_1s > 0 else 5.0
                    conf = 0.55
                    return {
                        "symbol": symbol,
                        "side": side,
                        "expected_bps": expected_bps,
                        "confidence": conf,
                        "horizon_s": int(self.hz.get("scalp", 30)),
                        "ttl_s": int(self.hz.get("ttl_s", 10)),
                        "rationale": "delta_revert",
                        "rule_id": "mr_v1",
                    }

        return None
