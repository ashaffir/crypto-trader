import os
import shutil
import tempfile
from src.control import Control
from src.runtime_config import RuntimeConfigManager


def test_control_roundtrip():
    tmp = tempfile.mkdtemp()
    try:
        c = Control(base_dir=tmp)
        assert c.get_desired_state() == "stopped"
        c.set_desired_state("running")
        assert c.get_desired_state() == "running"
        c.write_status({"status": "running", "queue_size": 0})
        st = c.read_status()
        assert st.get("status") == "running"
        assert "heartbeat_ts" in st
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_runtime_config_write_and_load():
    tmp = tempfile.mkdtemp()
    try:
        base = os.path.join(tmp, "control")
        os.environ["CONTROL_DIR"] = base
        rcm = RuntimeConfigManager(base)

        data = {
            "rules": {"momentum_enabled": False},
            "signal_thresholds": {"imbalance": 0.8, "max_spread_bps": 2.0},
            "horizons": {"scalp": 45, "ttl_s": 12},
        }
        assert rcm.write(data)
        changed, cfg = rcm.load_if_changed()
        assert changed is True
        assert cfg["rules"]["momentum_enabled"] is False
        # No change on second load
        changed2, _ = rcm.load_if_changed()
        assert changed2 is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
