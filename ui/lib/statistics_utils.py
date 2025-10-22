import os
import sys
from typing import Optional, Tuple

import pandas as pd

# Ensure project root on sys.path for `src` imports when running Streamlit from ui/
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.positions import PositionStore, DEFAULT_DB_PATH


def load_positions_dataframe(db_path: Optional[str] = None) -> pd.DataFrame:
    """Load all positions into a pandas DataFrame, sorted by open/close time.

    Returns empty DataFrame if no rows.
    """
    store = PositionStore(db_path or DEFAULT_DB_PATH)
    rows = store.all_positions()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Sort for deterministic processing
    if "closed_ts_ms" in df.columns:
        df = df.sort_values(["closed_ts_ms", "id"], na_position="last")
    else:
        df = df.sort_values(["id"])
    return df.reset_index(drop=True)


def compute_pnl_series(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-trade and cumulative PnL series for closed positions.

    Parameters:
    - df: DataFrame containing positions with at least columns: id, symbol, pnl, opened_ts_ms, closed_ts_ms.

    Returns:
    - trades_df: per-trade PnL with timestamp index in seconds and columns ["pnl", "symbol", "id"]. Only closed trades kept.
    - cum_df: cumulative PnL over time with columns ["cum_pnl"], indexed by timestamp seconds.
    """
    if df.empty:
        return pd.DataFrame(columns=["pnl", "symbol", "id"]).set_index(
            pd.Index([], name="ts")
        ), pd.DataFrame(columns=["cum_pnl"]).set_index(pd.Index([], name="ts"))

    work = df.copy()
    # Keep only closed positions with defined pnl and closed_ts_ms
    work = work[(~work["closed_ts_ms"].isna())]
    if work.empty:
        return pd.DataFrame(columns=["pnl", "symbol", "id"]).set_index(
            pd.Index([], name="ts")
        ), pd.DataFrame(columns=["cum_pnl"]).set_index(pd.Index([], name="ts"))

    # Coerce
    for c in ("pnl", "closed_ts_ms"):
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=["closed_ts_ms"]).copy()
    if work.empty:
        return pd.DataFrame(columns=["pnl", "symbol", "id"]).set_index(
            pd.Index([], name="ts")
        ), pd.DataFrame(columns=["cum_pnl"]).set_index(pd.Index([], name="ts"))

    # Map timestamp (seconds) for x-axis stability
    work["ts"] = (work["closed_ts_ms"].astype("Int64") // 1000).astype("Int64")
    trades = work[["ts", "id", "symbol", "pnl"]].copy()
    trades = trades.dropna(subset=["ts"])  # ensure ts exists
    # Fill missing pnl as 0 for visualization; we still show bars at close time
    trades["pnl"] = pd.to_numeric(trades["pnl"], errors="coerce").fillna(0.0)
    trades["ts"] = trades["ts"].astype("int64")

    # Per-trade series indexed by ts
    trades_df = trades.set_index("ts").sort_index()
    trades_df.index.name = "ts"

    # Aggregate by ts and cumulative
    agg = trades_df.groupby(level=0)["pnl"].sum().to_frame()
    agg.index.name = "ts"
    agg = agg.sort_index()
    cum = agg.cumsum()
    cum.columns = ["cum_pnl"]

    return trades_df, cum


__all__ = [
    "load_positions_dataframe",
    "compute_pnl_series",
]
