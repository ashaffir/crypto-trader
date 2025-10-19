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
    try:
        tbl = pq.read_table(subset)
        return tbl.to_pandas()
    except Exception:
        frames: list[pd.DataFrame] = []
        for f in subset:
            try:
                frames.append(pq.read_table(f).to_pandas())
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
