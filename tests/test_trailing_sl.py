from src.positions import PositionStore
from src.trading import TradingEngine, TraderSettings


def test_trailing_sl(tmp_path):
    store = PositionStore(db_path=str(tmp_path / "pos.sqlite"))
    eng = TradingEngine(
        store,
        TraderSettings(
            concurrent_positions=1,
            long_confidence_threshold=0.0,
            short_confidence_threshold=0.0,
            sl_percent=0.5,
            trailing_sl_enabled=True,
            tp_disabled=True,
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
        llm_model="m",
    )
    # Price rises to 101 (best_favorable updated), then drops to 100.3 -> -0.69% from best -> SL
    _ = eng.maybe_close_on_inverse_or_tp_sl(
        symbol="BTCUSDT",
        recommendation_direction=None,
        confidence=None,
        ts_ms=ts + 1000,
        price_info={"mid": 101.0, "last_px": 101.0},
    )
    closed = eng.maybe_close_on_inverse_or_tp_sl(
        symbol="BTCUSDT",
        recommendation_direction=None,
        confidence=None,
        ts_ms=ts + 2000,
        price_info={"mid": 100.3, "last_px": 100.3},
    )
    assert isinstance(closed, int)
