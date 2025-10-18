import os
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.parquet_inspector import (
    list_tables,
    list_symbols,
    list_dates,
    glob_parquet_files,
)


def _write_partition(base: str, table: str, symbol: str, date: str) -> None:
    ddir = os.path.join(base, table, f"symbol={symbol}", f"date={date}")
    os.makedirs(ddir, exist_ok=True)
    tbl = pa.Table.from_pydict({"a": [1, 2], "b": ["x", "y"]})
    pq.write_table(tbl, os.path.join(ddir, "part-1.parquet"))


def test_listing_helpers_work():
    with tempfile.TemporaryDirectory() as tmp:
        _write_partition(tmp, "market_snapshot", "BTCUSDT", "2025-10-18")
        _write_partition(tmp, "signal_emitted", "BTCUSDT", "2025-10-18")
        assert list_tables(tmp) == ["market_snapshot", "signal_emitted"]
        assert list_symbols(tmp, "market_snapshot") == ["BTCUSDT"]
        assert list_dates(tmp, "market_snapshot", "BTCUSDT") == ["2025-10-18"]
        files = glob_parquet_files(tmp, "market_snapshot", "BTCUSDT", None)
        assert len(files) == 1
