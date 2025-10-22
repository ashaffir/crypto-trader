import streamlit as st
import sys as _sys
import os as _os
import pandas as pd

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from ui.lib.common import render_status_badge
from ui.lib.settings_state import load_sidebar_settings

try:
    from streamlit_autorefresh import st_autorefresh as _st_autorefresh  # type: ignore
except Exception:  # pragma: no cover
    _st_autorefresh = None
from src.positions import PositionStore, DEFAULT_DB_PATH


st.set_page_config(page_title="Positions", layout="wide")
render_status_badge(st)
st.subheader("Positions")

# Lightweight auto-refresh using sidebar setting (non-disruptive)
try:
    _sidebar = load_sidebar_settings()
    _rs = max(1, int(_sidebar.get("refresh_seconds", 2)))
    if _st_autorefresh is not None:
        _st_autorefresh(interval=_rs * 1000, key="positions_auto_refresh")
        st.caption(f"Auto-refresh every {_rs}s")
    else:
        st.caption("Auto-refresh unsupported in this Streamlit version")
except Exception:
    st.caption("Auto-refresh unavailable")


def _read_all() -> pd.DataFrame:
    store = PositionStore(DEFAULT_DB_PATH)
    rows = store.all_positions()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Friendly ordering
    preferred = [
        "id",
        "symbol",
        "direction",
        "leverage",
        "opened_ts_ms",
        "entry_px",
        "confidence",
        "llm_model",
        "best_favorable_px",
        "closed_ts_ms",
        "exit_px",
        "pnl",
        "close_reason",
    ]
    order = [c for c in preferred if c in df.columns] + [
        c for c in df.columns if c not in preferred
    ]
    return df[order]


df = _read_all()
if df.empty:
    st.info("No positions yet.")
else:
    # Filters
    symbols = sorted(df["symbol"].dropna().unique().tolist()) if "symbol" in df else []
    col_f1, col_f2, col_f3 = st.columns([1, 1, 2])
    with col_f1:
        sel_syms = st.multiselect(
            "Symbols",
            options=symbols,
            default=symbols,
        )
    with col_f2:
        status_opt = st.selectbox("Status", options=["All", "Open", "Closed"], index=0)

    # Derived columns
    df = df.copy()
    df["status"] = df["closed_ts_ms"].apply(
        lambda x: "Open" if pd.isna(x) or x is None else "Closed"
    )

    # Human-readable timestamps
    def _ts(x):
        try:
            if pd.isna(x) or x is None:
                return None
            return (
                pd.to_datetime(int(x), unit="ms", utc=True)
                .tz_convert(None)
                .strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            )
        except Exception:
            return None

    if "opened_ts_ms" in df:
        df["opened_at"] = df["opened_ts_ms"].apply(_ts)
    if "closed_ts_ms" in df:
        df["closed_at"] = df["closed_ts_ms"].apply(_ts)

    def _pnl_pct(row):
        pnl = row.get("pnl")
        qty = row.get("qty")
        ent = row.get("entry_px")
        try:
            if (
                pnl is None
                or qty is None
                or ent is None
                or float(qty) == 0
                or float(ent) == 0
            ):
                return None
            return float(pnl) / (float(qty) * float(ent)) * 100.0
        except Exception:
            return None

    df["pnl_pct"] = df.apply(_pnl_pct, axis=1)

    # Duration (seconds)
    def _dur_min(row):
        t0 = row.get("opened_ts_ms")
        t1 = row.get("closed_ts_ms")
        if t0 is None or pd.isna(t0):
            return None
        if t1 is None or pd.isna(t1):
            return None
        try:
            diff_ms = int(t1) - int(t0)
            secs = max(0, diff_ms // 1000)
            return int(secs)
        except Exception:
            return None

    df["duration_s"] = df.apply(_dur_min, axis=1)

    # Apply filters
    if sel_syms:
        df = df[df["symbol"].isin(sel_syms)]
    if status_opt != "All":
        df = df[df["status"] == status_opt]

    # Pagination
    page_size = st.number_input(
        "Rows per page", min_value=10, max_value=1000, value=50, step=10
    )
    total = len(df)
    max_page = max(1, (total - 1) // int(page_size) + 1)
    page = st.number_input("Page", min_value=1, max_value=max_page, value=1, step=1)
    start = (int(page) - 1) * int(page_size)
    end = start + int(page_size)
    page_df = df.iloc[start:end]

    # Drop internal timestamp columns from view
    for _c in ("opened_ts_ms", "closed_ts_ms"):
        if _c in page_df.columns:
            page_df = page_df.drop(columns=[_c])
    # Display
    st.dataframe(page_df, use_container_width=True, hide_index=True)

    # Totals footer (sum of pnl)
    try:
        total_pnl = pd.to_numeric(df["pnl"], errors="coerce").sum()
    except Exception:
        total_pnl = None
    if total_pnl is not None:
        st.caption(f"Total PnL (all rows): {total_pnl:.2f}")

    # Download full dataset
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV", data=csv, file_name="positions.csv", mime="text/csv"
    )
