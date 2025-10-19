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

        # Mean-reversion rule (mr_v2):
        # Trigger when recent trade-mid deviation suggests an overshoot and spreads are acceptable.
        # Optional guard: avoid strong order-book imbalance if configured.
        if self.rules.get("mean_reversion_enabled", True):
            mid = snap.get("mid") or 0.0
            mr_min_revert_bps = float(self.thr.get("mr_min_revert_bps", 2.0))
            mr_max_imbalance = float(
                self.thr.get("mr_max_imbalance", 1.0)
            )  # default: no filter
            mr_target_bps = float(self.thr.get("mr_expected_bps", 6.0))
            mr_conf_norm_bps = float(self.thr.get("mr_conf_norm_bps", 5.0))

            if (
                delta_1s is not None
                and spread_bps is not None
                and spread_bps <= self.thr.get("max_spread_bps", 1.5)
                and mid > 0
            ):
                # Deviation threshold in bps relative to mid
                dev_bps = (abs(delta_1s) / mid) * 1e4
                if dev_bps >= mr_min_revert_bps and (
                    ob is None or abs(ob) <= mr_max_imbalance
                ):
                    side = "short" if delta_1s > 0 else "long"
                    expected_bps = -mr_target_bps if side == "short" else mr_target_bps

                    # Confidence scales with deviation magnitude and penalizes imbalance
                    norm = min(1.0, dev_bps / max(1e-9, mr_conf_norm_bps))
                    imb_penalty = (
                        1.0 - min(1.0, (abs(ob) if ob is not None else 0.0)) * 0.3
                    )
                    conf = max(0.5, min(0.95, 0.55 + 0.35 * norm * imb_penalty))

                    return {
                        "symbol": symbol,
                        "side": side,
                        "expected_bps": expected_bps,
                        "confidence": conf,
                        "horizon_s": int(self.hz.get("scalp", 30)),
                        "ttl_s": int(self.hz.get("ttl_s", 10)),
                        "rationale": "delta_revert_low_imbalance",
                        "rule_id": "mr_v2",
                    }

        return None
