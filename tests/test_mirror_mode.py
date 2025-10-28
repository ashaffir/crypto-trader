import json
from ui.lib.settings_state import load_mirror_mode, save_mirror_mode
import os


def test_mirror_mode_persist_roundtrip(tmp_path, monkeypatch):
    # Point CONTROL_DIR to temp folder
    monkeypatch.setenv("CONTROL_DIR", str(tmp_path))

    # Default should be False when file missing
    assert load_mirror_mode(base_dir=str(tmp_path)) is False

    # Enable and persist
    assert save_mirror_mode(True, base_dir=str(tmp_path)) is True
    assert load_mirror_mode(base_dir=str(tmp_path)) is True

    # Ensure flag is stored under llm.mirror_mode in runtime_config.json
    runtime_file = os.path.join(str(tmp_path), "runtime_config.json")
    with open(runtime_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data.get("llm"), dict)
    assert data["llm"].get("mirror_mode") is True

    # Disable and persist
    assert save_mirror_mode(False, base_dir=str(tmp_path)) is True
    assert load_mirror_mode(base_dir=str(tmp_path)) is False
