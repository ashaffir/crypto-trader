import os
import json
import tempfile

from ui.lib.settings_state import (
    load_positions_settings,
    save_positions_settings,
)


def _runtime_config_path(base_dir: str) -> str:
    return os.path.join(base_dir, "runtime_config.json")


def test_load_positions_settings_defaults_when_missing():
    with tempfile.TemporaryDirectory() as tmp:
        out = load_positions_settings(base_dir=tmp)
        assert isinstance(out, dict)
        assert out.get("total_pnl_latest_n") == 100


def test_save_and_load_positions_settings_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        ok = save_positions_settings({"total_pnl_latest_n": 250}, base_dir=tmp)
        assert ok is True

        out = load_positions_settings(base_dir=tmp)
        assert out.get("total_pnl_latest_n") == 250

        # Ensure persisted structure in runtime_config.json
        path = _runtime_config_path(tmp)
        assert os.path.exists(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)
        positions = data.get("positions")
        assert isinstance(positions, dict)
        assert positions.get("total_pnl_latest_n") == 250


def test_save_positions_settings_clamps_min_value():
    with tempfile.TemporaryDirectory() as tmp:
        ok = save_positions_settings({"total_pnl_latest_n": 0}, base_dir=tmp)
        assert ok is True
        out = load_positions_settings(base_dir=tmp)
        assert out.get("total_pnl_latest_n") == 1
