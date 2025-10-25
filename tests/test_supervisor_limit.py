import time

from src.supervisor import should_stop_by_time_limit


def test_should_stop_no_limit():
    now = int(time.time() * 1000)
    assert should_stop_by_time_limit(now, 0, now + 10_000) is False
    assert should_stop_by_time_limit(now, -5, now + 10_000) is False
    assert should_stop_by_time_limit(None, 10, now) is False
    assert should_stop_by_time_limit(now, 10, None) is False


def test_should_stop_not_reached():
    start = int(time.time() * 1000)
    now = start + 59_000  # less than 1 minute
    assert should_stop_by_time_limit(start, 1, now) is False


def test_should_stop_reached_and_after():
    start = int(time.time() * 1000)
    reached = start + 60_000
    after = start + 61_000
    assert should_stop_by_time_limit(start, 1, reached) is True
    assert should_stop_by_time_limit(start, 1, after) is True

