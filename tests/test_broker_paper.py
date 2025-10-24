from src.positions import PositionStore
import tempfile
from src.broker import PaperBroker


def test_paper_broker_open_and_close():
    tmpdb = tempfile.NamedTemporaryFile(suffix=".sqlite")
    store = PositionStore(db_path=tmpdb.name)
    broker = PaperBroker(store)
    pid = broker.open_position(
        symbol="BTCUSDT",
        direction="long",
        leverage=10,
        qty=0.001,
        entry_px=50000.0,
        ts_ms=1234567890,
        meta={"confidence": 0.9, "llm_model": "test"},
    )
    assert isinstance(pid, int)
    pos = store.get_latest_open_for_symbol("BTCUSDT")
    assert pos is not None
    assert int(pos["id"]) == int(pid)
    assert pos["direction"] == "long"
    ok = broker.close_position(
        position_id=int(pid),
        symbol="BTCUSDT",
        exit_px=50500.0,
        ts_ms=1234567999,
        pnl=None,
        reason="Test",
    )
    assert ok is True
