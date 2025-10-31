from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple


def _now_ms() -> int:
    return int(time.time() * 1000)


def ms_to_human(ms: Optional[int]) -> str:
    if ms is None or ms < 0:
        return "n/a"
    seconds = ms / 1000.0
    if seconds < 1:
        return f"{int(ms)} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes = int(seconds // 60)
    rem = int(seconds % 60)
    return f"{minutes}m {rem}s"


def compute_heartbeat_fresh(status: Dict[str, Any], threshold_ms: int = 5000, now_ms: Optional[int] = None) -> Tuple[bool, Optional[int]]:
    try:
        hb_ts = int(status.get("heartbeat_ts")) if status.get("heartbeat_ts") is not None else None
    except Exception:
        hb_ts = None
    if hb_ts is None:
        return (False, None)
    now = _now_ms() if now_ms is None else int(now_ms)
    age = max(0, now - hb_ts)
    fresh = age < int(threshold_ms)
    return (fresh, age)


def summarize_streams(status: Dict[str, Any], now_ms: Optional[int] = None, fresh_threshold_ms: int = 5000) -> Dict[str, Any]:
    streams = status.get("streams") if isinstance(status, dict) else None
    if not isinstance(streams, dict):
        streams = {}
    event_counts = status.get("event_counts") if isinstance(status, dict) else None
    if not isinstance(event_counts, dict):
        event_counts = {}
    last_events = status.get("last_events") if isinstance(status, dict) else None
    if not isinstance(last_events, dict):
        last_events = {}

    now = _now_ms() if now_ms is None else int(now_ms)

    per_stream = {}
    enabled_keys = [k for k, v in streams.items() if bool(v)]
    for key in sorted(set(list(streams.keys()) + list(event_counts.keys()) + list(last_events.keys()))):
        enabled = bool(streams.get(key))
        count = int(event_counts.get(key) or 0)
        ts = last_events.get(key)
        try:
            ts = int(ts) if ts is not None else None
        except Exception:
            ts = None
        age_ms = (now - ts) if ts is not None else None
        recent = bool(age_ms is not None and age_ms < int(fresh_threshold_ms))
        per_stream[key] = {
            "enabled": enabled,
            "event_count": count,
            "last_event_age_ms": age_ms,
            "recent": recent,
        }

    enabled_count = len(enabled_keys)
    recent_enabled = sum(1 for k, v in per_stream.items() if v["enabled"] and v["recent"])
    any_enabled = enabled_count > 0
    overall = "ok" if (not any_enabled or recent_enabled == enabled_count) else ("degraded" if recent_enabled > 0 else "down")

    return {
        "enabled_count": enabled_count,
        "any_enabled": any_enabled,
        "recent_enabled": recent_enabled,
        "overall": overall,
        "per_stream": per_stream,
    }


def summarize_status(status: Dict[str, Any], now_ms: Optional[int] = None) -> Dict[str, Any]:
    now = _now_ms() if now_ms is None else int(now_ms)
    running_flag = str(status.get("status") or "").lower() == "running"
    fresh, age_ms = compute_heartbeat_fresh(status, threshold_ms=5000, now_ms=now)
    market = str(status.get("market") or "").lower() or None
    symbols = status.get("symbols") if isinstance(status.get("symbols"), list) else []
    queue_size = None
    try:
        queue_size = int(status.get("queue_size")) if status.get("queue_size") is not None else None
    except Exception:
        queue_size = None

    streams_summary = summarize_streams(status, now_ms=now)

    # Simple queue classification
    queue_health = None
    if queue_size is not None:
        if queue_size < 100:
            queue_health = "low"
        elif queue_size < 1000:
            queue_health = "moderate"
        else:
            queue_health = "high"

    return {
        "is_running": running_flag and fresh,
        "heartbeat_fresh": fresh,
        "heartbeat_age_ms": age_ms,
        "market": market,
        "symbols": [str(s).upper() for s in symbols],
        "symbols_count": len(symbols),
        "queue_size": queue_size,
        "queue_health": queue_health,
        "streams": streams_summary,
    }


__all__ = [
    "ms_to_human",
    "compute_heartbeat_fresh",
    "summarize_streams",
    "summarize_status",
]


