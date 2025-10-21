import json
import os
from typing import Optional

from .common import CONTROL_DIR


_DEFAULTS: dict[str, object] = {
    "mode": "days",  # "days" | "size"
    "max_days": 7,
    "size_cap": "50GB",
    "dry_run_default": True,
}


def _runtime_file(base_dir: Optional[str] = None) -> str:
    b = base_dir or CONTROL_DIR
    return os.path.join(b, "runtime_config.json")


def _safe_read_json(path: str) -> dict:
    try:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
            return {}
    except Exception:
        return {}


def _safe_write_json(path: str, data: dict) -> bool:
    tmp = f"{path}.tmp"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, path)
        return True
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        return False


def load_retention_settings(base_dir: Optional[str] = None) -> dict:
    """Return persisted retention settings merged with defaults.

    Data lives under key "retention" inside runtime_config.json. Unknown keys are
    ignored. Defaults are returned when file or key is missing.
    """
    path = _runtime_file(base_dir)
    cfg = _safe_read_json(path)
    raw = cfg.get("retention") if isinstance(cfg, dict) else None
    out: dict[str, object] = dict(_DEFAULTS)
    if isinstance(raw, dict):
        for k, v in raw.items():
            if k in out:
                out[k] = v
    return out


def save_retention_settings(settings: dict, base_dir: Optional[str] = None) -> bool:
    """Persist provided retention settings under the "retention" key.

    Only recognized keys are stored; others are ignored. Existing runtime
    config keys (rules, thresholds, etc.) are preserved.
    """
    path = _runtime_file(base_dir)
    cfg = _safe_read_json(path)
    keep: dict[str, object] = {}
    for k in _DEFAULTS.keys():
        if k in settings:
            keep[k] = settings[k]
    merged = dict(cfg or {})
    merged["retention"] = dict(load_retention_settings(base_dir))
    merged["retention"].update(keep)
    return _safe_write_json(path, merged)


__all__ = [
    "load_retention_settings",
    "save_retention_settings",
]
