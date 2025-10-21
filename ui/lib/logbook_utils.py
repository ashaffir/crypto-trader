import os
import glob
from typing import List

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


__all__ = ["tail_parquet_table", "read_latest_file"]


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
