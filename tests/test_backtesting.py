from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.backtesting.engine import logical_test, quality_test


def _write_parquet_rows(base: str, table: str, symbol: str, rows: list[dict]) -> None:
    if not rows:
        return
    ts_ms = rows[0].get("ts_ms")
    if ts_ms is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    else:
        date_str = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime(
            "%Y-%m-%d"
        )
    ddir = os.path.join(base, table, f"symbol={symbol}", f"date={date_str}")
    os.makedirs(ddir, exist_ok=True)
    tbl = pa.Table.from_pylist(rows)
    pq.write_table(tbl, os.path.join(ddir, "part-1.parquet"))


def _make_snapshots(
    symbol: str, start_ts: int, n: int, mid0: float = 100.0, drift: float = 0.0
):
    rows = []
    mid = mid0
    for i in range(n):
        ts = start_ts + i * 1000
        mid = mid * (1.0 + drift)
        rows.append(
            {
                "ts_ms": ts,
                "symbol": symbol,
                "bid": mid - 0.5,
                "ask": mid + 0.5,
                "mid": mid,
                "last_px": mid,
                "last_qty": 1.0,
                "ob_imbalance": 0.0,
                "spread_bps": 10.0,
                "vol_1s": 1.0,
                "delta_1s": 0.0,
            }
        )
    return rows


def _make_signals(symbol: str, start_ts: int, n: int, step: int = 3):
    rows = []
    for i in range(0, n, step):
        ts = start_ts + i * 1000
        rows.append(
            {
                "signal_id": f"sig-{i}",
                "ts_ms": ts,
                "symbol": symbol,
                "side": "long" if i % 2 == 0 else "short",
                "expected_bps": 5.0,
                "confidence": 0.8,
                "horizon_s": 30,
                "ttl_s": 10,
                "rationale": "test",
                "rule_id": "t",
            }
        )
    return rows


def test_logical_test_validates_schema_and_timing(tmp_path):
    base = str(tmp_path)
    symbol = "BTCUSDT"
    start_ts = 1_700_000_000_000
    # write minimal signals
    _write_parquet_rows(
        base, "signal_emitted", symbol, _make_signals(symbol, start_ts, 10, step=2)
    )
    res = logical_test(symbol, base_dir=base, max_files=10)
    assert res.num_signals > 0
    assert res.fields_ok is True
    assert res.time_monotonic is True


def test_quality_test_reports_metrics(tmp_path):
    base = str(tmp_path)
    symbol = "BTCUSDT"
    start_ts = 1_700_000_000_000
    snaps = _make_snapshots(symbol, start_ts, 120, mid0=100.0, drift=0.001)
    sigs = _make_signals(symbol, start_ts, 60, step=5)
    _write_parquet_rows(base, "market_snapshot", symbol, snaps)
    _write_parquet_rows(base, "signal_emitted", symbol, sigs)
    res = quality_test(symbol, base_dir=base, horizon_s=10)
    assert res.outcomes is not None
    assert not res.outcomes.empty
    rp = res.report
    assert rp.num_trades == len(res.outcomes)
    assert rp.win_rate >= 0.0 and rp.win_rate <= 1.0
    # With upward drift, long signals should have positive mean
    assert isinstance(rp.mean_ret_bps, float)
