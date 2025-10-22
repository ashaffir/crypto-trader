from src.positions import PositionStore


def test_open_close_roundtrip(tmp_path):
    store = PositionStore(db_path=str(tmp_path / "pos.sqlite"))
    pid = store.open_position(
        symbol="ETHUSDT",
        direction="buy",
        leverage=3,
        opened_ts_ms=1,
        qty=0.1,
        entry_px=2000.0,
        confidence=0.9,
    )
    assert isinstance(pid, int)
    assert store.count_open() == 1
    pos = store.get_latest_open_for_symbol("ETHUSDT")
    assert pos and pos["symbol"] == "ETHUSDT"
    store.close_position(
        pid, closed_ts_ms=2, exit_px=2020.0, pnl=None, close_reason="TP"
    )
    assert store.count_open() == 0
    # Verify close_reason persisted
    all_rows = store.all_positions()
    assert len(all_rows) == 1
    assert all_rows[0].get("close_reason") == "TP"
