import json
import os
import shutil

from ui.lib.retention_state import load_retention_settings, save_retention_settings


def _mk_tmp(tmp_path):
    base = tmp_path / "control"
    base.mkdir(parents=True, exist_ok=True)
    return str(base)


def test_load_defaults_when_missing(tmp_path):
    base = _mk_tmp(tmp_path)
    settings = load_retention_settings(base)
    assert settings["mode"] == "days"
    assert settings["max_days"] == 7
    assert "size_cap" in settings


def test_save_and_load_roundtrip(tmp_path):
    base = _mk_tmp(tmp_path)
    ok = save_retention_settings(
        {
            "mode": "size",
            "size_cap": "123MB",
            "dry_run_default": False,
        },
        base,
    )
    assert ok
    s = load_retention_settings(base)
    assert s["mode"] == "size"
    assert s["size_cap"] == "123MB"
    assert s["dry_run_default"] is False


def test_preserve_existing_runtime_keys(tmp_path):
    base = _mk_tmp(tmp_path)
    # Pre-populate runtime_config.json with other keys
    pre = {
        "rules": {"momentum_enabled": True},
        "retention": {"max_days": 10},
    }
    path = os.path.join(base, "runtime_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pre, f)
    # Save retention settings; other keys should remain
    save_retention_settings({"max_days": 5}, base)
    with open(path, "r", encoding="utf-8") as f:
        after = json.load(f)
    assert after["rules"]["momentum_enabled"] is True
    assert after["retention"]["max_days"] == 5
