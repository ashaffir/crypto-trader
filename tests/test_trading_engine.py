from src.positions import PositionStore
from src.trading import TradingEngine, TraderSettings
from src.utils.fees import get_fee_rate


def test_open_slot_and_confidence(tmp_path):
    store = PositionStore(db_path=str(tmp_path / "pos.sqlite"))
    eng = TradingEngine(
        store, TraderSettings(concurrent_positions=1, confidence_threshold=0.8)
    )
    ts = 1_700_000_000_000
    # Below threshold: should not open
    assert (
        eng.maybe_open_from_recommendation(
            symbol="BTCUSDT",
            direction="buy",
            leverage=5,
            confidence=0.5,
            ts_ms=ts,
            price_info={"mid": 100.0, "last_px": 100.0},
            llm_model="x",
        )
        is None
    )
    # Meets threshold: opens
    pid = eng.maybe_open_from_recommendation(
        symbol="BTCUSDT",
        direction="buy",
        leverage=5,
        confidence=0.9,
        ts_ms=ts,
        price_info={"mid": 100.0, "last_px": 100.0},
        llm_model="x",
    )
    assert isinstance(pid, int)
    assert store.count_open() == 1
    # Slot full: another open should fail
    assert (
        eng.maybe_open_from_recommendation(
            symbol="ETHUSDT",
            direction="buy",
            leverage=3,
            confidence=0.95,
            ts_ms=ts + 1,
            price_info={"mid": 10.0, "last_px": 10.0},
            llm_model="x",
        )
        is None
    )


def test_inverse_close(tmp_path):
    store = PositionStore(db_path=str(tmp_path / "pos.sqlite"))
    eng = TradingEngine(
        store, TraderSettings(concurrent_positions=2, confidence_threshold=0.8)
    )
    ts = 1_700_000_000_000
    # Open long
    pid = eng.maybe_open_from_recommendation(
        symbol="BTCUSDT",
        direction="buy",
        leverage=2,
        confidence=0.9,
        ts_ms=ts,
        price_info={"mid": 100.0, "last_px": 100.0},
        llm_model="x",
    )
    assert pid is not None and store.count_open() == 1
    # Inverse rec should close regardless of confidence
    closed = eng.maybe_close_on_inverse_or_tp_sl(
        symbol="BTCUSDT",
        recommendation_direction="sell",
        confidence=0.1,  # below threshold, should still close
        ts_ms=ts + 1000,
        price_info={"mid": 99.0, "last_px": 99.0},
    )
    assert isinstance(closed, int)


def test_inverse_close_without_confidence(tmp_path):
    store = PositionStore(db_path=str(tmp_path / "pos.sqlite"))
    eng = TradingEngine(
        store, TraderSettings(concurrent_positions=2, confidence_threshold=0.95)
    )
    ts = 1_700_000_000_000
    # Open short
    pid = eng.maybe_open_from_recommendation(
        symbol="ETHUSDT",
        direction="sell",
        leverage=3,
        confidence=0.99,
        ts_ms=ts,
        price_info={"mid": 200.0, "last_px": 200.0},
        llm_model="x",
    )
    assert pid is not None and store.count_open() == 1
    # Opposite rec with confidence None should still close
    closed = eng.maybe_close_on_inverse_or_tp_sl(
        symbol="ETHUSDT",
        recommendation_direction="buy",
        confidence=None,
        ts_ms=ts + 1000,
        price_info={"mid": 201.0, "last_px": 201.0},
    )
    assert isinstance(closed, int)


def test_tp_sl(tmp_path):
    store = PositionStore(db_path=str(tmp_path / "pos.sqlite"))
    eng = TradingEngine(
        store,
        TraderSettings(
            concurrent_positions=1,
            confidence_threshold=0.0,
            tp_percent=1.0,
            sl_percent=0.5,
            tp_disabled=False,
        ),
    )
    ts = 1_700_000_000_000
    _ = eng.maybe_open_from_recommendation(
        symbol="BTCUSDT",
        direction="buy",
        leverage=1,
        confidence=1.0,
        ts_ms=ts,
        price_info={"mid": 100.0, "last_px": 100.0},
        llm_model="x",
    )
    # Move up 1% -> TP closes
    closed = eng.maybe_close_on_inverse_or_tp_sl(
        symbol="BTCUSDT",
        recommendation_direction=None,
        confidence=None,
        ts_ms=ts + 1000,
        price_info={"mid": 101.0, "last_px": 101.0},
    )
    assert isinstance(closed, int)


def test_pnl_includes_fees_when_enabled(tmp_path):
    store = PositionStore(db_path=str(tmp_path / "pos.sqlite"))
    eng = TradingEngine(
        store,
        TraderSettings(
            concurrent_positions=1,
            confidence_threshold=0.0,
            default_position_size_usd=1000.0,
            default_leverage=1,
            tp_percent=1.0,
            tp_disabled=False,
            # fees enabled: spot taker VIP0 with BNB discount off
            fees_enabled=True,
            fee_market="spot",
            fee_vip_tier=0,
            fee_liquidity="taker",
            fee_bnb_discount=False,
        ),
    )
    ts = 1_700_000_000_000
    entry_px = 100.0
    # open
    _ = eng.maybe_open_from_recommendation(
        symbol="BTCUSDT",
        direction="buy",
        leverage=1,
        confidence=1.0,
        ts_ms=ts,
        price_info={"mid": entry_px, "last_px": entry_px},
        llm_model="x",
    )
    pos = store.get_latest_open_for_symbol("BTCUSDT")
    assert pos is not None
    qty = float(pos.get("qty"))
    # close on +1% to trigger TP
    exit_px = entry_px * 1.01
    closed = eng.maybe_close_on_inverse_or_tp_sl(
        symbol="BTCUSDT",
        recommendation_direction=None,
        confidence=None,
        ts_ms=ts + 1000,
        price_info={"mid": exit_px, "last_px": exit_px},
    )
    assert isinstance(closed, int)
    rows = store.all_positions()
    assert len(rows) == 1
    gross = (exit_px - entry_px) * qty * 1
    # spot taker vip0 no bnb => 0.1% each leg
    rate = get_fee_rate(
        market="spot", vip_tier=0, liquidity="taker", bnb_discount=False
    )
    expected_fees = (entry_px * qty) * rate + (exit_px * qty) * rate
    expected_net = gross - expected_fees
    assert abs(float(rows[0]["pnl"]) - expected_net) < 1e-6


def test_default_settings_keep_previous_no_fee_behavior(tmp_path):
    store = PositionStore(db_path=str(tmp_path / "pos.sqlite"))
    eng = TradingEngine(
        store,
        TraderSettings(
            concurrent_positions=1,
            confidence_threshold=0.0,
            default_position_size_usd=1000.0,
            default_leverage=2,
            tp_percent=1.0,
            tp_disabled=False,
            # fees disabled by default
        ),
    )
    ts = 1_700_000_000_000
    entry_px = 50.0
    _ = eng.maybe_open_from_recommendation(
        symbol="ETHUSDT",
        direction="buy",
        leverage=2,
        confidence=1.0,
        ts_ms=ts,
        price_info={"mid": entry_px, "last_px": entry_px},
        llm_model="x",
    )
    pos = store.get_latest_open_for_symbol("ETHUSDT")
    qty = float(pos.get("qty"))
    exit_px = entry_px * 1.02  # +2%
    closed = eng.maybe_close_on_inverse_or_tp_sl(
        symbol="ETHUSDT",
        recommendation_direction=None,
        confidence=None,
        ts_ms=ts + 1000,
        price_info={"mid": exit_px, "last_px": exit_px},
    )
    assert isinstance(closed, int)
    rows = store.all_positions()
    gross = (exit_px - entry_px) * qty * 2
    assert abs(float(rows[0]["pnl"]) - gross) < 1e-6


def test_auto_expire_stale(tmp_path):
    store = PositionStore(db_path=str(tmp_path / "pos.sqlite"))
    eng = TradingEngine(
        store,
        TraderSettings(
            concurrent_positions=1,
            confidence_threshold=0.0,
            auto_expire_minutes=1,
        ),
    )
    ts = 1_700_000_000_000
    pid = eng.maybe_open_from_recommendation(
        symbol="ETHUSDT",
        direction="buy",
        leverage=1,
        confidence=1.0,
        ts_ms=ts,
        price_info={"mid": 100.0, "last_px": 100.0},
        llm_model="x",
    )
    assert isinstance(pid, int)
    # Advance > 1 minute, keep price flat; should close as Stale
    closed = eng.maybe_close_on_inverse_or_tp_sl(
        symbol="ETHUSDT",
        recommendation_direction=None,
        confidence=None,
        ts_ms=ts + 61_000,
        price_info={"mid": 100.0, "last_px": 100.0},
    )
    assert isinstance(closed, int)
    rows = store.all_positions()
    assert rows and rows[0].get("close_reason") == "Stale"
