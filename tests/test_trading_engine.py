from src.positions import PositionStore
from src.trading import TradingEngine, TraderSettings


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
    # Inverse rec with enough confidence -> close
    closed = eng.maybe_close_on_inverse_or_tp_sl(
        symbol="BTCUSDT",
        recommendation_direction="sell",
        confidence=0.9,
        ts_ms=ts + 1000,
        price_info={"mid": 99.0, "last_px": 99.0},
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
