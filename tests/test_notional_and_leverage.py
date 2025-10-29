from src.positions import PositionStore
from src.trading import TradingEngine, TraderSettings


def test_notional_stored_and_pnl_uses_leverage(tmp_path):
    store = PositionStore(db_path=str(tmp_path / "pos.sqlite"))
    eng = TradingEngine(
        store,
        TraderSettings(
            concurrent_positions=1,
            long_confidence_threshold=0.0,
            short_confidence_threshold=0.0,
        ),
    )

    ts = 1_700_000_000_000
    entry_px = 100.0
    leverage = 3
    # default_position_size_usd -> qty = usd / entry_px
    eng.update_settings(
        TraderSettings(
            concurrent_positions=1,
            long_confidence_threshold=0.0,
            short_confidence_threshold=0.0,
            default_position_size_usd=1000.0,
            default_leverage=leverage,
            tp_percent=0.1,  # enable small TP to trigger close on +1%
            tp_disabled=False,
        )
    )

    pid = eng.maybe_open_from_recommendation(
        symbol="TESTUSDT",
        direction="buy",
        leverage=leverage,
        confidence=1.0,
        ts_ms=ts,
        price_info={"mid": entry_px},
        llm_model="x",
    )
    assert isinstance(pid, int)

    # Fetch the open position and verify notional
    pos = store.get_latest_open_for_symbol("TESTUSDT")
    assert pos is not None
    qty = float(pos.get("qty"))
    assert qty > 0
    assert abs(float(pos.get("notional")) - qty * entry_px * leverage) < 1e-6

    # Close with +1 price move -> pnl = (101-100)*qty*leverage
    exit_px = 101.0
    closed = eng.maybe_close_on_inverse_or_tp_sl(
        symbol="TESTUSDT",
        recommendation_direction=None,
        confidence=None,
        ts_ms=ts + 1000,
        price_info={"mid": exit_px},
    )
    assert isinstance(closed, int)

    rows = store.all_positions()
    assert len(rows) == 1
    expected_pnl = (exit_px - entry_px) * qty * leverage
    assert abs(float(rows[0]["pnl"]) - expected_pnl) < 1e-6
