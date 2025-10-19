import pytest
from src.features import FeatureEngine
from src.signals import SignalEngine
from src.runtime_config import RuntimeConfigManager


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


def test_runtime_apply_overrides_to_signal_engine():
    fe = FeatureEngine(symbols=["BTCUSDT"])
    se = SignalEngine(
        thresholds={"imbalance": 0.6, "max_spread_bps": 1.5},
        horizons={"scalp": 30, "ttl_s": 10},
        rules={"momentum_enabled": True, "mean_reversion_enabled": True},
    )
    # Apply runtime override to disable momentum and adjust horizons
    overrides = {
        "rules": {"momentum_enabled": False},
        "horizons": {"scalp": 45},
        "signal_thresholds": {"imbalance": 0.9},
    }
    RuntimeConfigManager.apply_to_engines(overrides, signal_engine=se)
    assert se.rules.get("momentum_enabled") is False
    assert se.hz.get("scalp") == 45
    assert se.thr.get("imbalance") == 0.9


def test_signal_engine_mean_reversion_rule_triggers_and_direction():
    fe = FeatureEngine(symbols=["BTCUSDT"])
    se = SignalEngine(
        thresholds={
            "imbalance": 0.6,
            "max_spread_bps": 5.0,
            # mean-reversion specific thresholds
            "mr_min_revert_bps": 1.0,
            "mr_expected_bps": 6.0,
            "mr_conf_norm_bps": 3.0,
            # allow signal even under heuristic ob=1.0 when no depth volumes
            "mr_max_imbalance": 1.0,
        },
        horizons={"scalp": 30, "ttl_s": 10},
        rules={"momentum_enabled": False, "mean_reversion_enabled": True},
    )

    # Provide depth to set mid and small spread
    snap_depth = fe.on_message(
        {
            "kind": "depth",
            "symbol": "BTCUSDT",
            "ts_ms": 1000,
            "best_bid": 100.0,
            "best_ask": 100.02,
        }
    )
    assert snap_depth["mid"] == pytest.approx(100.01, abs=1e-6)

    # Trade prints above mid to create positive delta -> expect short signal (revert)
    snap_trade = fe.on_message(
        {
            "kind": "aggTrade",
            "symbol": "BTCUSDT",
            "ts_ms": 1100,
            "price": 100.05,  # 4 bps above mid
            "qty": 0.5,
        }
    )
    sig = se.on_features(snap_trade)
    assert sig is not None
    assert sig["rule_id"].startswith("mr_")
    assert sig["side"] == "short"
    assert sig["expected_bps"] < 0


def test_signal_engine_mean_reversion_respects_imbalance_guard():
    fe = FeatureEngine(symbols=["BTCUSDT"])
    se = SignalEngine(
        thresholds={
            "imbalance": 0.6,
            "max_spread_bps": 5.0,
            "mr_min_revert_bps": 1.0,
            "mr_expected_bps": 6.0,
            "mr_conf_norm_bps": 3.0,
            "mr_max_imbalance": 0.1,  # very strict, should block when ob is high
        },
        horizons={"scalp": 30, "ttl_s": 10},
        rules={"momentum_enabled": False, "mean_reversion_enabled": True},
    )

    # Depth (tight spread)
    snap_depth = fe.on_message(
        {
            "kind": "depth",
            "symbol": "BTCUSDT",
            "ts_ms": 1000,
            "best_bid": 100.0,
            "best_ask": 100.02,
        }
    )

    # Because compute_imbalance returns 1.0 without depth volumes, ob will be ~1.0
    # Create a deviation that is large enough, but imbalance guard should prevent signal
    snap_trade = fe.on_message(
        {
            "kind": "aggTrade",
            "symbol": "BTCUSDT",
            "ts_ms": 1100,
            "price": 100.05,
            "qty": 0.2,
        }
    )
    sig = se.on_features(snap_trade)
    assert sig is None
