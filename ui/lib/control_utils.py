import json
import os
from typing import Dict

from .common import CONTROL_DIR
from src.runtime_config import RuntimeConfigManager


def read_status() -> dict:
    path = os.path.join(CONTROL_DIR, "bot_status.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def read_desired() -> str:
    path = os.path.join(CONTROL_DIR, "bot_control.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
            desired = str(data.get("desired") or "stopped").lower()
            return "running" if desired == "running" else "stopped"
    except Exception:
        return "stopped"


def set_desired_state(running: bool) -> bool:
    os.makedirs(CONTROL_DIR, exist_ok=True)
    path = os.path.join(CONTROL_DIR, "bot_control.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"desired": "running" if running else "stopped"}, f)
            try:
                f.flush()
                os.fsync(f.fileno())
            except Exception:
                # Best-effort flush; on some hosts fsync on mounted volumes may fail
                pass
    except Exception:
        return False
    # Consider write successful even if immediate re-read fails due to fs caching
    return True


RCM = RuntimeConfigManager(CONTROL_DIR)


def get_effective_status(heartbeat_fresh_ms: int = 5000) -> str:
    """Return 'running' only when status explicitly says running AND heartbeat is fresh.

    Falls back to 'stopped' if no status, missing ts, or stale heartbeat.
    """
    st = read_status()
    try:
        hb_ts = (
            int(st.get("heartbeat_ts")) if st.get("heartbeat_ts") is not None else None
        )
    except Exception:
        hb_ts = None
    now_ms = int(__import__("time").time() * 1000)
    fresh = bool(hb_ts is not None and (now_ms - hb_ts) < int(heartbeat_fresh_ms))
    if st.get("status") == "running" and fresh:
        return "running"
    return "stopped"


__all__ = [
    "read_status",
    "read_desired",
    "set_desired_state",
    "RCM",
    "get_effective_status",
]
