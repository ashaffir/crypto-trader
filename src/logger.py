from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, List

import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class PartitionInfo:
    symbol: str
    date: str  # YYYY-MM-DD


class ParquetLogbook:
    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    @staticmethod
    def _today_utc_str(ts_ms: int | None) -> str:
        if ts_ms is None:
            dt = datetime.now(timezone.utc)
        else:
            dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d")

    def _dataset_path(self, table: str, symbol: str, date_str: str) -> str:
        return os.path.join(
            self.base_dir, table, f"symbol={symbol}", f"date={date_str}"
        )

    def _write_rows(
        self, table: str, rows: List[Dict[str, Any]], partition: PartitionInfo
    ) -> None:
        if not rows:
            return
        ds_path = self._dataset_path(table, partition.symbol, partition.date)
        os.makedirs(ds_path, exist_ok=True)
        # Create a filename with timestamp to avoid collisions
        filename = os.path.join(
            ds_path, f"part-{int(datetime.now().timestamp() * 1e6)}.parquet"
        )
        # Avoid dictionary encoding to reduce schema drift across parts
        tbl = pa.Table.from_pylist(rows)
        pq.write_table(tbl, filename, use_dictionary=False)

    # Public APIs
    def append_market_snapshot(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        symbol = rows[0]["symbol"]
        ts_ms = rows[0].get("ts_ms")
        date_str = self._today_utc_str(ts_ms)
        self._write_rows(
            "market_snapshot", rows, PartitionInfo(symbol=symbol, date=date_str)
        )

    def append_signal_emitted(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        symbol = rows[0]["symbol"]
        ts_ms = rows[0].get("ts_ms")
        date_str = self._today_utc_str(ts_ms)
        self._write_rows(
            "signal_emitted", rows, PartitionInfo(symbol=symbol, date=date_str)
        )

    def append_signal_outcome(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        # Outcome rows must include symbol for partitioning consistency
        symbol = rows[0]["symbol"]
        ts_ms = rows[0].get("resolved_ts_ms") or rows[0].get("ts_ms")
        date_str = self._today_utc_str(ts_ms)
        self._write_rows(
            "signal_outcome", rows, PartitionInfo(symbol=symbol, date=date_str)
        )

    def append_trade_recommendation(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        symbol = rows[0]["symbol"]
        ts_ms = rows[0].get("ts_ms")
        date_str = self._today_utc_str(ts_ms)
        self._write_rows(
            "trade_recommendation", rows, PartitionInfo(symbol=symbol, date=date_str)
        )


__all__ = ["ParquetLogbook"]
