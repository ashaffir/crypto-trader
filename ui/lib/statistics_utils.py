import os
import sys
from typing import Optional, Tuple

import pandas as pd
import numpy as np

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


def compute_pnl_series_by_model(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-trade and per-model cumulative PnL series.

    Returns:
    - trades_df: index ts (seconds), columns ["pnl", "symbol", "id", "llm_model"]
    - cum_df: index ts (seconds), columns ["llm_model", "cum_pnl"] (tidy format)
    """
    if df.empty:
        empty_idx = pd.Index([], name="ts")
        return (
            pd.DataFrame(columns=["pnl", "symbol", "id", "llm_model"]).set_index(
                empty_idx
            ),
            pd.DataFrame(columns=["llm_model", "cum_pnl"]).set_index(empty_idx),
        )

    work = df.copy()
    work = work[(~work["closed_ts_ms"].isna())]
    if work.empty:
        empty_idx = pd.Index([], name="ts")
        return (
            pd.DataFrame(columns=["pnl", "symbol", "id", "llm_model"]).set_index(
                empty_idx
            ),
            pd.DataFrame(columns=["llm_model", "cum_pnl"]).set_index(empty_idx),
        )

    # Coerce numeric
    for c in ("pnl", "closed_ts_ms"):
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=["closed_ts_ms"]).copy()
    if work.empty:
        empty_idx = pd.Index([], name="ts")
        return (
            pd.DataFrame(columns=["pnl", "symbol", "id", "llm_model"]).set_index(
                empty_idx
            ),
            pd.DataFrame(columns=["llm_model", "cum_pnl"]).set_index(empty_idx),
        )

    # Timestamp in seconds
    work["ts"] = (work["closed_ts_ms"].astype("Int64") // 1000).astype("Int64")

    # Normalize model label; include unknowns as "Unknown" to keep them in groups
    if "llm_model" not in work.columns:
        work["llm_model"] = "Unknown"
    else:
        try:
            work["llm_model"] = (
                work["llm_model"].astype("string").fillna("Unknown").str.strip()
            )
            work.loc[work["llm_model"].isin(["", "nan", "None"]), "llm_model"] = (
                "Unknown"
            )
        except Exception:
            work["llm_model"] = "Unknown"

    trades = work[["ts", "id", "symbol", "pnl", "llm_model"]].copy()
    trades = trades.dropna(subset=["ts"])  # ensure ts exists
    trades["pnl"] = pd.to_numeric(trades["pnl"], errors="coerce").fillna(0.0)
    trades["ts"] = trades["ts"].astype("int64")

    trades_df = trades.set_index("ts").sort_index()
    trades_df.index.name = "ts"

    # Aggregate and cumulative per model
    agg = (
        trades_df.reset_index()
        .groupby(["ts", "llm_model"], as_index=False)["pnl"]
        .sum()
        .sort_values(["llm_model", "ts"])
    )
    agg["cum_pnl"] = agg.groupby("llm_model")["pnl"].cumsum()
    cum_df = agg[["ts", "llm_model", "cum_pnl"]].copy().set_index("ts").sort_index()
    cum_df.index.name = "ts"

    return trades_df, cum_df


__all__.append("compute_pnl_series_by_model")


def compute_close_reason_distribution_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy DataFrame with columns [llm_model, close_reason, count]."""
    if df.empty:
        return pd.DataFrame(columns=["llm_model", "close_reason", "count"])
    work = df.copy()
    if "llm_model" not in work.columns:
        work["llm_model"] = "Unknown"
    else:
        try:
            work["llm_model"] = (
                work["llm_model"].astype("string").fillna("Unknown").str.strip()
            )
            work.loc[work["llm_model"].isin(["", "nan", "None"]), "llm_model"] = (
                "Unknown"
            )
        except Exception:
            work["llm_model"] = "Unknown"
    reason = work.get("close_reason")
    if reason is None:
        work["close_reason"] = "Unknown"
    else:
        work["close_reason"] = (
            work["close_reason"].astype("string").fillna("Unknown").str.strip()
        )
        work.loc[work["close_reason"].isin(["", "nan", "None"]), "close_reason"] = (
            "Unknown"
        )
    grp = (
        work.groupby(["llm_model", "close_reason"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["llm_model", "count"], ascending=[True, False])
    )
    return grp


def compute_window_pnl_correlation(
    df: pd.DataFrame, window_seconds: Optional[int]
) -> pd.DataFrame:
    """Return tidy frame mapping each closed trade to (ts, llm_model, pnl, window_seconds).

    Note: if per-trade llm_window_s exists in the positions data, it is used; otherwise
    the provided window_seconds value is attached as a fallback.
    """
    if df.empty:
        return pd.DataFrame(columns=["ts", "llm_model", "pnl", "window_seconds"])
    work = df.copy()
    work = work[(~work["closed_ts_ms"].isna())]
    if work.empty:
        return pd.DataFrame(columns=["ts", "llm_model", "pnl", "window_seconds"])
    for c in ("pnl", "closed_ts_ms"):
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=["closed_ts_ms"]).copy()
    work["ts"] = (work["closed_ts_ms"].astype("Int64") // 1000).astype("Int64")
    if "llm_model" not in work.columns:
        work["llm_model"] = "Unknown"
    else:
        try:
            work["llm_model"] = (
                work["llm_model"].astype("string").fillna("Unknown").str.strip()
            )
            work.loc[work["llm_model"].isin(["", "nan", "None"]), "llm_model"] = (
                "Unknown"
            )
        except Exception:
            work["llm_model"] = "Unknown"
    # Prefer stored per-trade window; else fallback provided
    has_per_trade = "llm_window_s" in work.columns
    out = work[
        ["ts", "llm_model", "pnl"] + (["llm_window_s"] if has_per_trade else [])
    ].copy()
    out["ts"] = out["ts"].astype("int64")
    out["pnl"] = pd.to_numeric(out["pnl"], errors="coerce").fillna(0.0)
    if has_per_trade:
        try:
            out["window_seconds"] = pd.to_numeric(
                out["llm_window_s"], errors="coerce"
            ).astype("Int64")
        except Exception:
            out["window_seconds"] = None
        out = out.drop(columns=["llm_window_s"], errors="ignore")
        # Fill missing with fallback
        if window_seconds is not None:
            out["window_seconds"] = out["window_seconds"].fillna(int(window_seconds))
    else:
        out["window_seconds"] = (
            int(window_seconds) if window_seconds is not None else None
        )
    return out


__all__ += [
    "compute_close_reason_distribution_by_model",
    "compute_window_pnl_correlation",
]


def summarize_window_pnl_correlation(corr_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-model correlation metrics between window_seconds and pnl.

    Returns DataFrame with columns [llm_model, n, pearson_r, slope, intercept].
    Only models with at least 2 points are included.
    """
    if corr_df is None or corr_df.empty:
        return pd.DataFrame(
            columns=["llm_model", "n", "pearson_r", "slope", "intercept"]
        )
    df = corr_df.copy()
    # Clean
    try:
        df = df.dropna(subset=["window_seconds"]).copy()
        df["window_seconds"] = pd.to_numeric(df["window_seconds"], errors="coerce")
        df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
        df = df.dropna(subset=["window_seconds", "pnl"])  # both numeric
    except Exception:
        return pd.DataFrame(
            columns=["llm_model", "n", "pearson_r", "slope", "intercept"]
        )

    rows = []
    for mdl, g in df.groupby("llm_model"):
        x = g["window_seconds"].to_numpy(dtype=float)
        y = g["pnl"].to_numpy(dtype=float)
        if len(x) < 2:
            continue
        # Pearson r
        try:
            r = float(np.corrcoef(x, y)[0, 1])
        except Exception:
            r = float("nan")
        # OLS slope/intercept
        try:
            slope, intercept = np.polyfit(x, y, 1)
            slope = float(slope)
            intercept = float(intercept)
        except Exception:
            slope, intercept = float("nan"), float("nan")
        rows.append(
            {
                "llm_model": mdl,
                "n": int(len(x)),
                "pearson_r": r,
                "slope": slope,
                "intercept": intercept,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["llm_model", "n", "pearson_r", "slope", "intercept"]
        )
    out = pd.DataFrame(rows).sort_values(["llm_model"]).reset_index(drop=True)
    return out


__all__.append("summarize_window_pnl_correlation")
