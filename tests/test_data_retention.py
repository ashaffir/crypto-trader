import os
import tempfile
from datetime import datetime, timezone, timedelta

import pyarrow as pa
import pyarrow.parquet as pq

from src.data_retention import (
    iter_date_partitions,
    prune_by_days,
    prune_to_size_cap,
    parse_size_cap,
)


def _write_partition(base: str, table: str, symbol: str, date: str, rows: int = 1):
    ddir = os.path.join(base, table, f"symbol={symbol}", f"date={date}")
    os.makedirs(ddir, exist_ok=True)
    tbl = pa.Table.from_pydict({"x": list(range(rows))})
    pq.write_table(tbl, os.path.join(ddir, "part-1.parquet"))


def test_iter_and_prune_by_days():
    with tempfile.TemporaryDirectory() as tmp:
        old_date = (datetime.now(timezone.utc) - timedelta(days=10)).strftime(
            "%Y-%m-%d"
        )
        new_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        _write_partition(tmp, "market_snapshot", "BTCUSDT", old_date)
        _write_partition(tmp, "market_snapshot", "BTCUSDT", new_date)
        parts = list(iter_date_partitions(tmp))
        assert len(parts) == 2
        removed = prune_by_days(tmp, max_days=5, dry_run=False)
        assert len(removed) == 1
        parts = list(iter_date_partitions(tmp))
        assert len(parts) == 1


def test_prune_to_size_cap():
    with tempfile.TemporaryDirectory() as tmp:
        # Create three dates with increasing sizes
        base_date = datetime(2025, 10, 1, tzinfo=timezone.utc)
        for i in range(3):
            d = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
            _write_partition(
                tmp, "trade_recommendation", "BTCUSDT", d, rows=(i + 1) * 1000
            )
        # Compute total actual size and set a cap that forces pruning
        total = 0
        for root, _dirs, files in os.walk(tmp):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
        cap = max(1, total // 2)
        removed = prune_to_size_cap(tmp, size_cap_bytes=cap, dry_run=False)
        assert len(removed) >= 1


def test_parse_size_cap():
    assert parse_size_cap("1024") == 1024
    assert parse_size_cap("1KB") == 1024
    assert parse_size_cap("1 MB") == 1024 * 1024
    assert parse_size_cap("0.5GB") == int(0.5 * 1024 * 1024 * 1024)
