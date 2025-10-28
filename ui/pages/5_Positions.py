import streamlit as st
import sys as _sys
import os as _os
import pandas as pd

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from ui.lib.common import render_status_badge
from ui.lib.settings_state import (
    load_sidebar_settings,
    load_positions_settings,
    save_positions_settings,
)

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
        "notional",
        "confidence",
        "llm_model",
        "best_favorable_px",
        "closed_ts_ms",
        "exit_px",
        "pnl",
        "close_reason",
        # New optional metadata columns
        "venue",
        "exec_mode",
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
    venues = sorted(df["venue"].dropna().unique().tolist()) if "venue" in df else []
    modes = (
        sorted(df["exec_mode"].dropna().unique().tolist()) if "exec_mode" in df else []
    )
    col_f1, col_f2, col_f3, col_f4 = st.columns([1, 1, 1, 2])
    with col_f1:
        sel_syms = st.multiselect(
            "Symbols",
            options=symbols,
            default=symbols,
        )
    with col_f2:
        sel_venue = st.selectbox(
            "Venue",
            options=(["All"] + venues if venues else ["All"]),
            index=0,
        )
    with col_f3:
        sel_mode = st.selectbox(
            "Mode",
            options=(["All"] + modes if modes else ["All"]),
            index=0,
        )
    with col_f4:
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
        notional = row.get("notional")
        qty = row.get("qty")
        ent = row.get("entry_px")
        try:
            if pnl is None or (
                (notional is None or float(notional) == 0.0)
                and (qty is None or ent is None or float(qty) == 0 or float(ent) == 0)
            ):
                return None
            denom = None
            try:
                if notional is not None and float(notional) != 0.0:
                    denom = float(notional)
                else:
                    denom = float(qty) * float(ent)
            except Exception:
                return None
            return float(pnl) / denom * 100.0
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
    if sel_venue != "All" and "venue" in df.columns:
        df = df[df["venue"] == sel_venue]
    if sel_mode != "All" and "exec_mode" in df.columns:
        df = df[df["exec_mode"] == sel_mode]
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

    # Latest-N Total PnL control (persisted)
    try:
        max_rows = max(1, len(df))
        # Load persisted default; clamp to [1, max_rows]
        try:
            _persisted = load_positions_settings()
            _persisted_n = int(_persisted.get("total_pnl_latest_n", 100))
        except Exception:
            _persisted = {}
            _persisted_n = 100
        default_n = int(max(1, min(_persisted_n, max_rows)))
        n_latest = st.number_input(
            "Total PnL window (latest N rows)",
            min_value=1,
            max_value=max_rows,
            value=int(default_n),
            step=1,
            key="positions_total_pnl_latest_n",
            help="Sum PnL over the most recent N rows by table order (newest first).",
        )

        # Persist if changed
        try:
            if int(n_latest) != int(_persisted_n):
                save_positions_settings({"total_pnl_latest_n": int(n_latest)})
        except Exception:
            pass

        # Use visible table ordering: id DESC (matches SELECT ORDER BY id DESC)
        if "id" in df.columns:
            latest_df = df.sort_values("id", ascending=False).head(int(n_latest))
        else:
            # Fallback to timestamp if id missing
            sort_series = None
            if "closed_ts_ms" in df.columns and "opened_ts_ms" in df.columns:
                sort_series = (
                    df["closed_ts_ms"].fillna(df["opened_ts_ms"]).astype("Int64")
                )
            elif "opened_ts_ms" in df.columns:
                sort_series = df["opened_ts_ms"].astype("Int64")
            latest_df = (
                df.assign(_sort=sort_series)
                .sort_values("_sort", ascending=False)
                .head(int(n_latest))
                if sort_series is not None
                else df.head(int(n_latest))
            )

        try:
            latest_total_pnl = pd.to_numeric(latest_df["pnl"], errors="coerce").sum()
            st.caption(
                f"Total PnL (latest {int(n_latest)} rows): {latest_total_pnl:.2f}"
            )
        except Exception:
            pass
    except Exception:
        pass

    # Download full dataset
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV", data=csv, file_name="positions.csv", mime="text/csv"
    )
