import os
import glob
from typing import List, Optional

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc

from .common import LOGBOOK_DIR


def tail_parquet_table(table: str, symbol: str, tail_files: int = 20) -> pd.DataFrame:
    base = os.path.join(LOGBOOK_DIR, table, f"symbol={symbol}")
    files = sorted(glob.glob(os.path.join(base, "date=*", "*.parquet")))
    if not files:
        return pd.DataFrame()
    subset = files[-tail_files:]
    # First, try to read as a single logical table (fast path)
    try:
        tbl = pq.read_table(subset)
        return _table_cast_dictionary_to_string(tbl).to_pandas()
    except Exception:
        # Fallback: read file-by-file, and if a file has mixed row-group schemas,
        # read row groups individually. Cast any dictionary columns to strings to
        # avoid merge/type conflicts across parts.
        frames: list[pd.DataFrame] = []
        for f in subset:
            try:
                t = pq.read_table(f)
                frames.append(_table_cast_dictionary_to_string(t).to_pandas())
                continue
            except Exception:
                pass
            # Per-row-group fallback
            try:
                pf = pq.ParquetFile(f)
            except Exception:
                continue
            for i in range(getattr(pf, "num_row_groups", 0)):
                try:
                    rg = pf.read_row_group(i)
                    frames.append(_table_cast_dictionary_to_string(rg).to_pandas())
                except Exception:
                    continue
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True, sort=False)


def _table_cast_dictionary_to_string(tbl: "pa.Table") -> "pa.Table":
    try:
        fields = tbl.schema
        new_cols = []
        changed = False
        for idx, field in enumerate(fields):
            col = tbl.column(idx)
            if pa.types.is_dictionary(field.type):
                try:
                    col = pc.cast(col, pa.string())
                    changed = True
                except Exception:
                    pass
            new_cols.append(col)
        if changed:
            return pa.table(new_cols, names=fields.names)
        return tbl
    except Exception:
        return tbl


def read_latest_file(table: str, symbol: str) -> pd.DataFrame:
    base = os.path.join(LOGBOOK_DIR, table, f"symbol={symbol}")
    pattern = os.path.join(base, "date=*", "*.parquet")
    files = glob.glob(pattern)
    if not files:
        return pd.DataFrame()
    latest = max(files, key=os.path.getmtime)
    try:
        tbl = pq.read_table(latest)
        return _table_cast_dictionary_to_string(tbl).to_pandas()
    except Exception:
        try:
            pf = pq.ParquetFile(latest)
        except Exception:
            return pd.DataFrame()
        frames: list[pd.DataFrame] = []
        for i in range(getattr(pf, "num_row_groups", 0)):
            try:
                rg = pf.read_row_group(i)
                frames.append(_table_cast_dictionary_to_string(rg).to_pandas())
            except Exception:
                continue
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True, sort=False)


__all__ = [
    "tail_parquet_table",
    "read_latest_file",
]


def list_symbols_with_data(table: str) -> list[str]:
    """Return symbols that have at least one parquet file under table/."""
    base = os.path.join(LOGBOOK_DIR, table)
    out: list[str] = []
    try:
        for name in os.listdir(base):
            p = os.path.join(base, name)
            if name.startswith("symbol=") and os.path.isdir(p):
                sym = name.split("=", 1)[1]
                # Ensure there is at least one file present
                if glob.glob(os.path.join(p, "date=*", "*.parquet")):
                    out.append(sym)
    except Exception:
        return []
    return sorted(out)


__all__.append("list_symbols_with_data")


# ---- Price helpers for UI ----


def _row_to_price(row: pd.Series) -> Optional[float]:
    """Extract a representative price from a market snapshot row.

    Preference order:
    - mid
    - last_px
    - (best_bid + best_ask) / 2 if both present
    """
    try:
        price = row.get("mid")
        if price is None or pd.isna(price):
            price = row.get("last_px")
        if (price is None or pd.isna(price)) and (
            "best_bid" in row and "best_ask" in row
        ):
            bid = row.get("best_bid")
            ask = row.get("best_ask")
            if (
                bid is not None
                and ask is not None
                and not pd.isna(bid)
                and not pd.isna(ask)
            ):
                price = (float(bid) + float(ask)) / 2.0
        if price is None or pd.isna(price):
            return None
        return float(price)
    except Exception:
        return None


def latest_price(symbol: str) -> Optional[float]:
    """Return the most recent price for a symbol from market snapshots.

    Uses the latest parquet part; falls back to a small tail window if needed.
    """
    df = read_latest_file("market_snapshot", symbol)
    if df.empty:
        df = tail_parquet_table("market_snapshot", symbol, tail_files=5)
    if df.empty:
        return None
    try:
        if "ts_ms" in df.columns:
            df = df.sort_values("ts_ms")
        row = df.tail(1).iloc[0]
        return _row_to_price(row)
    except Exception:
        return None


def price_at_ts(
    symbol: str, ts_ms: int, search_tail_files: int = 50
) -> Optional[float]:
    """Return the snapshot price at or nearest before ts_ms for the symbol.

    Reads recent market snapshots (limited by search_tail_files) and finds the
    last row with ts_ms <= target; if none, uses the earliest after.
    """
    if ts_ms is None:
        return None
    df = tail_parquet_table("market_snapshot", symbol, tail_files=search_tail_files)
    if df.empty:
        # As a last resort, try the latest file only
        df = read_latest_file("market_snapshot", symbol)
    if df.empty:
        return None
    try:
        if "ts_ms" not in df.columns:
            # Without timestamps, use last available snapshot
            row = df.tail(1).iloc[0]
            return _row_to_price(row)
        sdf = df.dropna(subset=["ts_ms"]).sort_values("ts_ms")
        before = sdf[sdf["ts_ms"] <= int(ts_ms)]
        if not before.empty:
            row = before.tail(1).iloc[0]
            return _row_to_price(row)
        # No snapshot before ts_ms, take the first after
        after = sdf[sdf["ts_ms"] > int(ts_ms)]
        if not after.empty:
            row = after.head(1).iloc[0]
            return _row_to_price(row)
        # Fallback to last row
        row = sdf.tail(1).iloc[0]
        return _row_to_price(row)
    except Exception:
        return None


__all__ += ["latest_price", "price_at_ts"]
