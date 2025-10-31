import time

from ui.lib.monitor_utils import (
    ms_to_human,
    compute_heartbeat_fresh,
    summarize_streams,
    summarize_status,
)


def test_ms_to_human_basic():
    assert ms_to_human(None) == "n/a"
    assert ms_to_human(-1) == "n/a"
    assert ms_to_human(250).endswith(" ms")
    assert ms_to_human(1500).endswith(" s")
    assert "m" in ms_to_human(61_000)


def test_heartbeat_fresh_and_age():
    now = int(time.time() * 1000)
    status = {"heartbeat_ts": now - 4000}
    fresh, age = compute_heartbeat_fresh(status, threshold_ms=5000, now_ms=now)
    assert fresh is True
    assert 3000 <= age <= 5000

    status = {"heartbeat_ts": now - 7000}
    fresh, age = compute_heartbeat_fresh(status, threshold_ms=5000, now_ms=now)
    assert fresh is False
    assert age >= 7000


def test_summarize_streams_no_enabled_ok():
    now = int(time.time() * 1000)
    status = {
        "streams": {"trade": False, "bookTicker": False},
        "event_counts": {},
        "last_events": {},
    }
    s = summarize_streams(status, now_ms=now, fresh_threshold_ms=5000)
    assert s["any_enabled"] is False
    assert s["overall"] == "ok"


def test_summarize_streams_all_recent_ok():
    now = int(time.time() * 1000)
    status = {
        "streams": {"trade": True, "bookTicker": True},
        "event_counts": {"trade": 100, "bookTicker": 50},
        "last_events": {"trade": now - 1000, "bookTicker": now - 100},
    }
    s = summarize_streams(status, now_ms=now, fresh_threshold_ms=5000)
    assert s["overall"] == "ok"
    assert s["recent_enabled"] == 2


def test_summarize_streams_some_recent_degraded():
    now = int(time.time() * 1000)
    status = {
        "streams": {"trade": True, "bookTicker": True},
        "event_counts": {"trade": 100, "bookTicker": 50},
        "last_events": {"trade": now - 1000, "bookTicker": now - 10_000},
    }
    s = summarize_streams(status, now_ms=now, fresh_threshold_ms=5000)
    assert s["overall"] == "degraded"
    assert s["recent_enabled"] == 1


def test_summarize_streams_enabled_no_recent_down():
    now = int(time.time() * 1000)
    status = {
        "streams": {"trade": True},
        "event_counts": {"trade": 0},
        "last_events": {"trade": now - 60_000},
    }
    s = summarize_streams(status, now_ms=now, fresh_threshold_ms=5000)
    assert s["overall"] == "down"
    assert s["recent_enabled"] == 0


def test_summarize_status_top_level_fields():
    now = int(time.time() * 1000)
    raw = {
        "status": "running",
        "heartbeat_ts": now - 1000,
        "market": "spot",
        "symbols": ["btcusdt", "ethusdt"],
        "queue_size": 42,
        "streams": {"trade": True},
        "event_counts": {"trade": 10},
        "last_events": {"trade": now - 1000},
    }
    s = summarize_status(raw, now_ms=now)
    assert s["is_running"] is True
    assert s["heartbeat_fresh"] is True
    assert s["symbols_count"] == 2
    assert s["queue_size"] == 42
    assert s["streams"]["overall"] == "ok"


