from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc

from src.utils.logbook_utils import resolve_logbook_dir


@dataclass
class DataSlice:
    table: str
    symbol: str
    files: List[str]


def _symbol_dir(base_dir: str, table: str, symbol: str) -> str:
    return os.path.join(base_dir, table, f"symbol={symbol}")


def list_dates(base_dir: str, table: str, symbol: str) -> List[str]:
    out: List[str] = []
    sdir = _symbol_dir(base_dir, table, symbol)
    if not os.path.isdir(sdir):
        return out
    for name in os.listdir(sdir):
        if name.startswith("date=") and os.path.isdir(os.path.join(sdir, name)):
            out.append(name.split("=", 1)[1])
    return sorted(out)


def glob_parquet_files(
    base_dir: str, table: str, symbol: str, date: str | None
) -> List[str]:
    base = _symbol_dir(base_dir, table, symbol)
    if date:
        pattern = os.path.join(base, f"date={date}", "*.parquet")
    else:
        pattern = os.path.join(base, "date=*", "*.parquet")
    return sorted(glob.glob(pattern))


def load_snapshots(
    symbol: str,
    base_dir: Optional[str] = None,
    *,
    dates: Optional[Iterable[str]] = None,
    max_files: Optional[int] = None,
) -> pd.DataFrame:
    base = base_dir or resolve_logbook_dir()
    files: List[str] = []
    if dates is None:
        files = glob_parquet_files(base, "market_snapshot", symbol, None)
    else:
        for d in dates:
            files.extend(glob_parquet_files(base, "market_snapshot", symbol, d))
        files = sorted(files)
    if max_files is not None and max_files > 0:
        files = files[-max_files:]
    if not files:
        return pd.DataFrame()
    frames: List[pd.DataFrame] = []
    for f in files:
        df_part = _read_file_to_df(f)
        if df_part is not None and not df_part.empty:
            frames.append(df_part)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    if "ts_ms" in df.columns:
        df = df.sort_values("ts_ms").reset_index(drop=True)
    return df


def load_signals(
    symbol: str,
    base_dir: Optional[str] = None,
    *,
    dates: Optional[Iterable[str]] = None,
    max_files: Optional[int] = None,
) -> pd.DataFrame:
    base = base_dir or resolve_logbook_dir()
    files: List[str] = []
    if dates is None:
        files = glob_parquet_files(base, "signal_emitted", symbol, None)
    else:
        for d in dates:
            files.extend(glob_parquet_files(base, "signal_emitted", symbol, d))
        files = sorted(files)
    if max_files is not None and max_files > 0:
        files = files[-max_files:]
    if not files:
        return pd.DataFrame()
    frames: List[pd.DataFrame] = []
    for f in files:
        df_part = _read_file_to_df(f)
        if df_part is not None and not df_part.empty:
            frames.append(df_part)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    if "ts_ms" in df.columns:
        df = df.sort_values("ts_ms").reset_index(drop=True)
    return df


__all__ = [
    "DataSlice",
    "list_dates",
    "glob_parquet_files",
    "load_snapshots",
    "load_signals",
]


# ---------- Internals ----------
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


def _read_file_to_df(path: str) -> Optional[pd.DataFrame]:
    try:
        # Use row-group based reading to avoid partition inference conflicts
        pf = pq.ParquetFile(path)
    except Exception:
        return None
    frames: List[pd.DataFrame] = []
    for i in range(getattr(pf, "num_row_groups", 0)):
        try:
            rg = pf.read_row_group(i)
            rg = _table_cast_dictionary_to_string(rg)
            frames.append(rg.to_pandas())
        except Exception:
            continue
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True, sort=False)
