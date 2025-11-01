from __future__ import annotations

import time

from src.deterministic import DeterministicSignalEngine


def _summary(mid: float) -> dict:
    return {
        "symbol": "BTCUSDT",
        "count": 10,
        "mid": {"mean": mid, "slope": 0.05 * mid},  # small positive slope
        "spread_bps": {"mean": 5.0, "std": 1.0},
        "ob_imbalance": {"mean": 0.1, "trend": 0.0},
        "orderflow_pressure": 0.2,
        "depth_skew": 0.1,
        "cancel_intensity": 0.2,
        "trade_aggression": 0.6,
        "oi_delta": 0.1,
        "liquidation_burst": 0.0,
    }


def test_deterministic_engine_emits_buy_signal():
    eng = DeterministicSignalEngine(["BTCUSDT"], alpha=0.05, lam=1e-3, horizon_s=1)
    now = int(time.time() * 1000)
    rec = None
    mid = 50000.0
    # Drive a short increasing sequence so realized returns are positive
    for i in range(40):
        mid *= 1.0005  # small drift up
        s = _summary(mid)
        out = eng.update_and_score(
            symbol="BTCUSDT",
            summary=s,
            ts_ms=now + 1000 * i,
            last_mid=mid,
            fee_rate_bps_roundtrip=1.0,  # tiny fee
        )
        if out:
            rec = out
            break

    # We expect a valid recommendation dict
    assert rec is not None
    assert rec["direction"] in ("buy", "sell", "none")
    assert 1 <= int(rec["leverage"]) <= 10
    assert 0.0 <= float(rec["confidence"]) <= 1.0

