import json
import os

from .common import CONTROL_DIR


_DEFAULTS: dict[str, object] = {
    "logical_max_files": 10,
    "quality_max_files": None,  # None means "all"
    "chart_points": 200,
}


def _runtime_file(base_dir: str | None = None) -> str:
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


def load_backtesting_settings(base_dir: str | None = None) -> dict:
    path = _runtime_file(base_dir)
    cfg = _safe_read_json(path)
    raw = cfg.get("backtesting") if isinstance(cfg, dict) else None
    out: dict[str, object] = dict(_DEFAULTS)
    if isinstance(raw, dict):
        for k, v in raw.items():
            if k in out:
                out[k] = v
    return out


def save_backtesting_settings(settings: dict, base_dir: str | None = None) -> bool:
    path = _runtime_file(base_dir)
    cfg = _safe_read_json(path)
    keep: dict[str, object] = {}
    for k in _DEFAULTS.keys():
        if k in settings:
            keep[k] = settings[k]
    merged = dict(cfg or {})
    merged["backtesting"] = dict(load_backtesting_settings(base_dir))
    merged["backtesting"].update(keep)
    return _safe_write_json(path, merged)


__all__ = [
    "load_backtesting_settings",
    "save_backtesting_settings",
]
