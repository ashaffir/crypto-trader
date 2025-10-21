"""Tests for DATA_WINDOW construction from market snapshots."""

import os
import tempfile
from datetime import datetime, timezone

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.utils.data_window import construct_data_window, _read_recent_snapshots


@pytest.fixture
def temp_logbook():
    """Create a temporary logbook directory with test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def _create_test_snapshots(
    logbook_dir: str,
    symbol: str,
    date_str: str,
    snapshots: list[dict],
) -> None:
    """Helper to create parquet files with test snapshots."""
    snapshot_path = os.path.join(
        logbook_dir, "market_snapshot", f"symbol={symbol}", f"date={date_str}"
    )
    os.makedirs(snapshot_path, exist_ok=True)

    # Write snapshots to a single parquet file
    table = pa.Table.from_pylist(snapshots)
    file_path = os.path.join(snapshot_path, "part-test.parquet")
    pq.write_table(table, file_path, use_dictionary=False)


def test_read_recent_snapshots_empty(temp_logbook):
    """Test reading snapshots when no data exists."""
    df = _read_recent_snapshots(temp_logbook, "BTCUSDT", 60)
    assert df.empty


def test_read_recent_snapshots_with_data(temp_logbook):
    """Test reading snapshots with data."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Create test snapshots spanning 10 seconds
    snapshots = []
    for i in range(10):
        snapshots.append(
            {
                "ts_ms": now_ms - ((10 - i) * 1000),
                "symbol": "BTCUSDT",
                "bid": 50000.0 + i,
                "ask": 50001.0 + i,
                "mid": 50000.5 + i,
                "last_px": 50000.5 + i,
                "last_qty": 1.5,
                "ob_imbalance": 0.1,
                "spread_bps": 2.0,
                "vol_1s": 10.0,
                "delta_1s": 0.5,
            }
        )

    _create_test_snapshots(temp_logbook, "BTCUSDT", date_str, snapshots)

    # Read with 15 second window (should get all 10)
    df = _read_recent_snapshots(temp_logbook, "BTCUSDT", 15, now_ms)
    assert len(df) == 10
    assert df["symbol"].iloc[0] == "BTCUSDT"


def test_read_recent_snapshots_window_filter(temp_logbook):
    """Test that window filtering works correctly."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Create snapshots: 5 old (outside window) + 5 recent (inside window)
    snapshots = []
    for i in range(10):
        snapshots.append(
            {
                "ts_ms": now_ms - ((10 - i) * 1000),
                "symbol": "ETHUSDT",
                "bid": 2500.0,
                "ask": 2501.0,
                "mid": 2500.5,
                "last_px": 2500.5,
                "last_qty": 2.0,
                "ob_imbalance": 0.2,
                "spread_bps": 2.0,
                "vol_1s": 5.0,
                "delta_1s": 0.3,
            }
        )

    _create_test_snapshots(temp_logbook, "ETHUSDT", date_str, snapshots)

    # Read with 5 second window (should get only last 5-6 snapshots)
    df = _read_recent_snapshots(temp_logbook, "ETHUSDT", 5, now_ms)
    assert len(df) <= 6  # May include boundary


def test_construct_data_window_empty(temp_logbook):
    """Test constructing DATA_WINDOW with no data."""
    result = construct_data_window(temp_logbook, ["BTCUSDT"], 60)

    assert "timestamp" in result
    assert "window_seconds" in result
    assert result["window_seconds"] == 60
    assert "assets" in result
    assert len(result["assets"]) == 0  # No data available


def test_construct_data_window_single_symbol(temp_logbook):
    """Test constructing DATA_WINDOW for a single symbol."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Create 5 snapshots
    snapshots = []
    for i in range(5):
        snapshots.append(
            {
                "ts_ms": now_ms - ((5 - i) * 1000),
                "symbol": "BTCUSDT",
                "bid": 67240.0 + i,
                "ask": 67242.0 + i,
                "mid": 67241.0 + i,
                "last_px": 67241.0 + i,
                "last_qty": 1.0 + (i * 0.1),
                "ob_imbalance": 0.68 + (i * 0.01),
                "spread_bps": 1.8,
                "vol_1s": 10.0 + i,
                "delta_1s": 0.5,
            }
        )

    _create_test_snapshots(temp_logbook, "BTCUSDT", date_str, snapshots)

    result = construct_data_window(temp_logbook, ["BTCUSDT"], 10, now_ms)

    assert result["window_seconds"] == 10
    assert len(result["assets"]) == 1

    asset = result["assets"][0]
    assert asset["symbol"] == "BTCUSDT"
    assert len(asset["recent_prices"]) == 5
    assert len(asset["recent_volumes"]) == 5
    assert len(asset["recent_bid_ask_spreads_bps"]) == 5
    assert len(asset["recent_imbalance"]) == 5

    # Check that prices are increasing
    assert asset["recent_prices"][0] < asset["recent_prices"][-1]

    # Check aggregates
    assert "price_change_bps" in asset
    assert "volume_total" in asset
    assert asset["volume_total"] > 0


def test_construct_data_window_multiple_symbols(temp_logbook):
    """Test constructing DATA_WINDOW for multiple symbols."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Create snapshots for BTC
    btc_snapshots = []
    for i in range(3):
        btc_snapshots.append(
            {
                "ts_ms": now_ms - ((3 - i) * 1000),
                "symbol": "BTCUSDT",
                "bid": 67000.0,
                "ask": 67002.0,
                "mid": 67001.0,
                "last_px": 67001.0,
                "last_qty": 1.0,
                "ob_imbalance": 0.5,
                "spread_bps": 2.0,
                "vol_1s": 5.0,
                "delta_1s": 0.1,
            }
        )

    # Create snapshots for ETH
    eth_snapshots = []
    for i in range(3):
        eth_snapshots.append(
            {
                "ts_ms": now_ms - ((3 - i) * 1000),
                "symbol": "ETHUSDT",
                "bid": 2550.0,
                "ask": 2552.0,
                "mid": 2551.0,
                "last_px": 2551.0,
                "last_qty": 2.0,
                "ob_imbalance": -0.3,
                "spread_bps": 2.5,
                "vol_1s": 10.0,
                "delta_1s": -0.2,
            }
        )

    _create_test_snapshots(temp_logbook, "BTCUSDT", date_str, btc_snapshots)
    _create_test_snapshots(temp_logbook, "ETHUSDT", date_str, eth_snapshots)

    result = construct_data_window(temp_logbook, ["BTCUSDT", "ETHUSDT"], 10, now_ms)

    assert len(result["assets"]) == 2

    symbols = {asset["symbol"] for asset in result["assets"]}
    assert "BTCUSDT" in symbols
    assert "ETHUSDT" in symbols


def test_construct_data_window_price_change(temp_logbook):
    """Test that price_change_bps is calculated correctly."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Create snapshots with 1% price increase
    snapshots = []
    for i in range(2):
        price = 50000.0 if i == 0 else 50500.0  # 1% increase
        snapshots.append(
            {
                "ts_ms": now_ms - ((2 - i) * 1000),
                "symbol": "BTCUSDT",
                "bid": price - 0.5,
                "ask": price + 0.5,
                "mid": price,
                "last_px": price,
                "last_qty": 1.0,
                "ob_imbalance": 0.0,
                "spread_bps": 2.0,
                "vol_1s": 5.0,
                "delta_1s": 0.0,
            }
        )

    _create_test_snapshots(temp_logbook, "BTCUSDT", date_str, snapshots)

    result = construct_data_window(temp_logbook, ["BTCUSDT"], 10, now_ms)

    asset = result["assets"][0]
    # 1% = 100 bps
    assert abs(asset["price_change_bps"] - 100.0) < 1.0


def test_construct_data_window_sampling(temp_logbook):
    """Test that large datasets are sampled correctly."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Create 200 snapshots (should be sampled to 100)
    snapshots = []
    for i in range(200):
        snapshots.append(
            {
                "ts_ms": now_ms - ((200 - i) * 1000),
                "symbol": "BTCUSDT",
                "bid": 50000.0,
                "ask": 50001.0,
                "mid": 50000.5,
                "last_px": 50000.5,
                "last_qty": 1.0,
                "ob_imbalance": 0.0,
                "spread_bps": 2.0,
                "vol_1s": 5.0,
                "delta_1s": 0.0,
            }
        )

    _create_test_snapshots(temp_logbook, "BTCUSDT", date_str, snapshots)

    result = construct_data_window(temp_logbook, ["BTCUSDT"], 300, now_ms)

    asset = result["assets"][0]
    # Should be sampled to max 100 points
    assert len(asset["recent_prices"]) <= 100


def test_construct_data_window_handles_missing_fields(temp_logbook):
    """Test that missing/NaN fields are handled gracefully."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Create snapshots with some missing fields
    snapshots = [
        {
            "ts_ms": now_ms - 2000,
            "symbol": "BTCUSDT",
            "bid": 50000.0,
            "ask": 50001.0,
            "mid": 50000.5,
            "last_px": None,  # Missing
            "last_qty": 1.0,
            "ob_imbalance": None,  # Missing
            "spread_bps": 2.0,
            "vol_1s": 5.0,
            "delta_1s": 0.0,
        },
        {
            "ts_ms": now_ms - 1000,
            "symbol": "BTCUSDT",
            "bid": 50100.0,
            "ask": 50101.0,
            "mid": 50100.5,
            "last_px": 50100.5,
            "last_qty": 1.5,
            "ob_imbalance": 0.3,
            "spread_bps": 2.0,
            "vol_1s": 6.0,
            "delta_1s": 0.1,
        },
    ]

    _create_test_snapshots(temp_logbook, "BTCUSDT", date_str, snapshots)

    # Should not raise an error
    result = construct_data_window(temp_logbook, ["BTCUSDT"], 10, now_ms)

    assert len(result["assets"]) == 1
    asset = result["assets"][0]

    # Should have data for both rows (using mid when last_px is missing)
    assert len(asset["recent_prices"]) == 2

    # Should only have imbalance for the second row
    assert len(asset["recent_imbalance"]) == 1
