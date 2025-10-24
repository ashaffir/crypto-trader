"""Utility to construct DATA_WINDOW JSON from market snapshot parquet files."""

from __future__ import annotations

import glob
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow.parquet as pq
from loguru import logger


def _read_recent_snapshots(
    base_dir: str, symbol: str, window_seconds: int, end_ts_ms: Optional[int] = None
) -> pd.DataFrame:
    """
    Read market snapshots for a symbol within the last N seconds.

    Args:
        base_dir: Base logbook directory
        symbol: Trading symbol (e.g., BTCUSDT)
        window_seconds: Number of seconds to look back
        end_ts_ms: End timestamp in ms (default: now)

    Returns:
        DataFrame with columns: ts_ms, symbol, bid, ask, mid, last_px, last_qty,
                                ob_imbalance, spread_bps, vol_1s, delta_1s
    """
    if end_ts_ms is None:
        end_ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    start_ts_ms = end_ts_ms - (window_seconds * 1000)

    # Get today's date for partitioning
    date_str = datetime.fromtimestamp(end_ts_ms / 1000.0, tz=timezone.utc).strftime(
        "%Y-%m-%d"
    )

    # Path to market_snapshot for this symbol and date
    snapshot_path = os.path.join(
        base_dir, "market_snapshot", f"symbol={symbol}", f"date={date_str}"
    )

    if not os.path.exists(snapshot_path):
        return pd.DataFrame()

    # Find all parquet files regardless of naming pattern
    all_files = glob.glob(os.path.join(snapshot_path, "*.parquet"))

    if not all_files:
        return pd.DataFrame()

    # Try to use filename timestamps if present; otherwise include all files
    # Filename format sometimes: part-{microsecond_timestamp}.parquet
    start_file_ts = (start_ts_ms - 2000) * 1000  # 2s buffer before
    end_file_ts = (end_ts_ms + 2000) * 1000  # 2s buffer after

    relevant_files = []
    for file_path in all_files:
        basename = os.path.basename(file_path)
        ts_candidate = basename.replace("part-", "").replace(".parquet", "")
        try:
            file_ts = int(ts_candidate)
            if start_file_ts <= file_ts <= end_file_ts:
                relevant_files.append(file_path)
        except Exception:
            # If no parseable timestamp, include and let record-level ts_ms filter apply
            relevant_files.append(file_path)

    logger.debug(
        f"Filtered {len(relevant_files)} of {len(all_files)} files for {symbol}"
    )

    if not relevant_files:
        return pd.DataFrame()

    # Sort by timestamp (most recent last)
    relevant_files.sort()

    # Read filtered files and filter by actual timestamp
    dfs = []
    for file_path in relevant_files:
        try:
            # Use ParquetFile to avoid dataset schema merging issues
            pf = pq.ParquetFile(file_path)
            df = pf.read().to_pandas()
            if not df.empty and "ts_ms" in df.columns:
                # Filter to exact window
                mask = (df["ts_ms"] >= start_ts_ms) & (df["ts_ms"] <= end_ts_ms + 100)
                filtered = df[mask]
                if not filtered.empty:
                    dfs.append(filtered)
        except Exception:
            # Skip corrupted files
            continue

    if not dfs:
        return pd.DataFrame()

    # Combine and sort (filter out empty DataFrames to avoid FutureWarning)
    non_empty_dfs = [df for df in dfs if not df.empty]
    if not non_empty_dfs:
        return pd.DataFrame()

    combined = pd.concat(non_empty_dfs, ignore_index=True)
    combined = combined.sort_values("ts_ms").reset_index(drop=True)

    return combined


def _process_symbol(
    base_dir: str,
    symbol: str,
    window_seconds: int,
    end_ts_ms: int,
    max_samples: int,
) -> Optional[Dict[str, Any]]:
    """Process a single symbol and return asset data."""
    logger.debug(f"Reading snapshots for {symbol}")
    df = _read_recent_snapshots(base_dir, symbol, window_seconds, end_ts_ms)

    if df.empty:
        logger.warning(f"No data available for {symbol} in last {window_seconds}s")
        return None

    logger.debug(f"Found {len(df)} snapshots for {symbol}")

    # Sample evenly across the window if we have too many points
    if len(df) > max_samples:
        indices = [int(i * len(df) / max_samples) for i in range(max_samples)]
        df_sampled = df.iloc[indices]
    else:
        df_sampled = df

    # Build arrays
    recent_prices = []
    recent_volumes = []
    recent_spreads = []
    recent_imbalance = []

    for _, row in df_sampled.iterrows():
        # Price: use mid, last_px, or fallback to bid/ask average
        price = row.get("mid")
        if pd.isna(price) or price is None:
            price = row.get("last_px")
        if pd.isna(price) or price is None:
            bid = row.get("bid")
            ask = row.get("ask")
            if not pd.isna(bid) and not pd.isna(ask):
                price = (bid + ask) / 2.0

        if price is not None and not pd.isna(price):
            recent_prices.append(round(float(price), 2))

        # Volume (from vol_1s)
        vol = row.get("vol_1s")
        if vol is not None and not pd.isna(vol):
            recent_volumes.append(round(float(vol), 4))

        # Spread in bps
        spread = row.get("spread_bps")
        if spread is not None and not pd.isna(spread):
            recent_spreads.append(round(float(spread), 2))

        # Imbalance
        imb = row.get("ob_imbalance")
        if imb is not None and not pd.isna(imb):
            recent_imbalance.append(round(float(imb), 4))

    # Calculate aggregates
    price_change_bps = 0.0
    if len(recent_prices) >= 2:
        first_price = recent_prices[0]
        last_price = recent_prices[-1]
        if first_price > 0:
            price_change_bps = ((last_price - first_price) / first_price) * 1e4

    volume_total = sum(recent_volumes)

    asset_data = {
        "symbol": symbol,
        "recent_prices": recent_prices,
        "recent_volumes": recent_volumes,
        "recent_bid_ask_spreads_bps": recent_spreads,
        "recent_imbalance": recent_imbalance,
        "price_change_bps": round(price_change_bps, 2),
        "volume_total": round(volume_total, 2),
    }

    logger.debug(
        f"Added {symbol}: {len(recent_prices)} prices, {price_change_bps:.2f} bps change"
    )
    return asset_data


def construct_data_window(
    base_dir: str,
    symbols: List[str],
    window_seconds: int,
    end_ts_ms: Optional[int] = None,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Construct DATA_WINDOW JSON for LLM analysis.

    Args:
        base_dir: Base logbook directory (e.g., /path/to/data/logbook)
        symbols: List of trading symbols
        window_seconds: Number of seconds of data to include
        end_ts_ms: End timestamp in ms (default: now)

    Returns:
        JSON dict with structure:
        {
            "timestamp": "2025-10-20T12:34:56Z",
            "window_seconds": 60,
            "assets": [
                {
                    "symbol": "BTCUSDT",
                    "recent_prices": [...],
                    "recent_volumes": [...],
                    "recent_bid_ask_spreads_bps": [...],
                    "recent_imbalance": [...],
                    "price_change_bps": 5.3,
                    "volume_total": 125.6
                },
                ...
            ]
        }
    """
    if end_ts_ms is None:
        end_ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    # Auto-scale max_samples based on window size if not provided
    if max_samples is None:
        if window_seconds <= 10:
            max_samples = 30
        elif window_seconds <= 30:
            max_samples = 50
        elif window_seconds <= 60:
            max_samples = 75
        else:
            max_samples = 100

    timestamp = datetime.fromtimestamp(end_ts_ms / 1000.0, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    logger.info(
        f"Constructing DATA_WINDOW: symbols={symbols}, window={window_seconds}s, max_samples={max_samples}"
    )

    # Process symbols in parallel for speed
    assets = []
    with ThreadPoolExecutor(max_workers=min(len(symbols), 3)) as executor:
        futures = [
            executor.submit(
                _process_symbol,
                base_dir,
                symbol,
                window_seconds,
                end_ts_ms,
                max_samples,
            )
            for symbol in symbols
        ]
        for future in futures:
            result = future.result()
            if result is not None:
                assets.append(result)

    result = {
        "timestamp": timestamp,
        "window_seconds": window_seconds,
        "assets": assets,
    }

    logger.info(
        f"DATA_WINDOW constructed: {len(assets)} assets, total size ~{len(str(result))} chars"
    )

    return result


__all__ = ["construct_data_window"]
