from src.features import FeatureEngine
from src.signals import SignalEngine


def test_feature_engine_basic_snapshot():
    fe = FeatureEngine(symbols=["BTCUSDT"])
    # Depth update then trade
    snap1 = fe.on_message(
        {
            "kind": "depth",
            "symbol": "BTCUSDT",
            "ts_ms": 1000,
            "best_bid": 100.0,
            "best_ask": 100.1,
        }
    )
    assert snap1["mid"] == 100.05
    assert snap1["spread_bps"] > 0

    snap2 = fe.on_message(
        {
            "kind": "aggTrade",
            "symbol": "BTCUSDT",
            "ts_ms": 1100,
            "price": 100.06,
            "qty": 1.0,
        }
    )
    assert snap2["last_px"] == 100.06
    assert snap2["vol_1s"] >= 1.0


def test_signal_engine_momentum_rule():
    fe = FeatureEngine(symbols=["BTCUSDT"])
    se = SignalEngine(
        thresholds={"imbalance": 0.6, "max_spread_bps": 5.0},
        horizons={"scalp": 30, "ttl_s": 10},
        rules={"momentum_enabled": True, "mean_reversion_enabled": False},
    )
    snap = fe.on_message(
        {
            "kind": "depth",
            "symbol": "BTCUSDT",
            "ts_ms": 1000,
            "best_bid": 100.0,
            "best_ask": 100.01,
        }
    )
    sig = se.on_features(snap)
    assert sig is not None
    assert sig["side"] in ("long", "short")
