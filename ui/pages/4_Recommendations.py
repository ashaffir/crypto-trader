import os as _os
import sys as _sys
import pandas as pd
import json
import streamlit as st

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from ui.lib.common import (
    LOGBOOK_DIR,
    PAGE_HEADER_TITLE,
    render_status_badge,
    CONTROL_DIR,
)
from ui.lib.settings_state import (
    load_tracked_symbols,
    get_active_llm_config,
)
from ui.lib.logbook_utils import (
    tail_parquet_table,
    list_symbols_with_data,
    latest_price,
    price_at_ts,
)


st.set_page_config(page_title="Recommendations", layout="wide")
render_status_badge(st)

# Build page-scoped symbol selector based on available data
_default_symbol = "BTCUSDT"

# Build options: start with tracked symbols (preserve order),
# then append any symbols that have data but aren't tracked yet
try:
    _tracked = load_tracked_symbols()
except Exception:
    _tracked = []
try:
    _with_data = list_symbols_with_data("trade_recommendation")
except Exception:
    _with_data = []

_available = list(_tracked) if _tracked else []
for s in _with_data:
    if s not in _available:
        _available.append(s)

st.subheader("Recommendations")
if not _available:
    st.caption(f"LOGBOOK_DIR: {LOGBOOK_DIR}")
    st.info("No recommendations found in logbook.")
    st.stop()

symbol_index = _available.index(_default_symbol) if _default_symbol in _available else 0
symbol = st.selectbox(
    "Symbol", _available, index=symbol_index, key="recs_symbol_select"
)
st.caption(f"LOGBOOK_DIR: {LOGBOOK_DIR}")

df = tail_parquet_table("trade_recommendation", symbol, tail_files=50)
if df.empty:
    st.info("No recommendations yet.")
else:
    try:
        df = df.sort_values("ts_ms")
        df["ts"] = (
            pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
            .dt.tz_localize(None)
            .dt.strftime("%Y-%m-%d %H:%M:%S.%f")
            .str[:-3]
        )
    except Exception:
        df["ts"] = ""

    # Backfill asset if missing (older rows)
    if "asset" not in df.columns and "symbol" in df.columns:
        df["asset"] = df["symbol"]

    # Reset index for pagination calculations only (do not display as column)
    df = df.reset_index(drop=True)

    # Model filter (only when column exists with real values)
    if "llm_model" in df.columns:
        model_values = sorted(
            [str(m) for m in pd.Series(df["llm_model"]).dropna().unique() if str(m)]
        )
        if model_values:
            selected_model = st.selectbox(
                "LLM model",
                options=["All"] + model_values,
                index=0,
                key="recs_model_filter",
            )
            if selected_model != "All":
                df = df[df["llm_model"] == selected_model]

    # Pagination controls
    total_rows = len(df)
    left, right = st.columns([1, 3])
    with left:
        page_size = st.number_input(
            "Rows per page", 10, 200, 50, 10, key="recs_page_size"
        )
    total_pages = max(1, (total_rows + page_size - 1) // page_size)
    with right:
        page_num = st.number_input(
            "Page",
            min_value=1,
            max_value=int(total_pages),
            value=int(total_pages),
            step=1,
            key="recs_page_num",
        )

    start = (int(page_num) - 1) * int(page_size)
    end = min(start + int(page_size), total_rows)
    page_df = df.iloc[start:end].copy()

    # Compute prices
    # Current price per asset (usually single symbol on this page)
    assets = list(pd.Series(page_df.get("asset", [])).dropna().unique())
    current_price_map = {}
    for a in assets:
        try:
            cp = latest_price(str(a))
        except Exception:
            cp = None
        current_price_map[str(a)] = cp

    # Preload recent market snapshots once and align prices to signal timestamps
    snaps = tail_parquet_table("market_snapshot", symbol, tail_files=800)
    signal_prices_series = None
    if not snaps.empty and "ts_ms" in snaps.columns:
        try:
            snaps = snaps.sort_values("ts_ms")

            def _snap_price_row(r: pd.Series) -> float | None:
                p = r.get("mid")
                if p is None or pd.isna(p):
                    p = r.get("last_px")
                if (p is None or pd.isna(p)) and ("best_bid" in r and "best_ask" in r):
                    b = r.get("best_bid")
                    a = r.get("best_ask")
                    if (
                        b is not None
                        and a is not None
                        and not pd.isna(b)
                        and not pd.isna(a)
                    ):
                        p = (float(b) + float(a)) / 2.0
                try:
                    return float(p) if p is not None and not pd.isna(p) else None
                except Exception:
                    return None

            snaps["_snap_price"] = snaps.apply(_snap_price_row, axis=1)
            # Merge-asof to get price at or immediately before signal ts
            left = page_df[["ts_ms"]].copy().astype({"ts_ms": int}).sort_values("ts_ms")
            right = snaps[["ts_ms", "_snap_price"]].dropna().astype({"ts_ms": int})
            merged = pd.merge_asof(left, right, on="ts_ms", direction="backward")
            # Map back to original order
            signal_prices_series = merged.set_index(left.index)["_snap_price"]
        except Exception:
            signal_prices_series = None

    if signal_prices_series is not None:
        page_df["Signal price"] = signal_prices_series
    else:
        # Fallback to per-row lookup (slower)
        page_df["Signal price"] = page_df.apply(
            lambda r: price_at_ts(str(r.get("asset")), int(r.get("ts_ms", 0))),
            axis=1,
        )

    page_df["Current price"] = page_df.get("asset").map(current_price_map)

    # Compute numeric delta and a formatted display with percent
    try:
        page_df["_DeltaNum"] = page_df["Current price"].astype(float) - page_df[
            "Signal price"
        ].astype(float)
    except Exception:
        page_df["_DeltaNum"] = float("nan")

    def _fmt_delta(row: pd.Series) -> str:
        try:
            d = float(row.get("_DeltaNum"))
            sp = float(row.get("Signal price"))
            if pd.isna(d) or pd.isna(sp) or sp == 0:
                return ""
            d2 = round(d, 2)
            pct = round(d / sp * 100.0, 2)
            return f"{d2:.2f} ({pct:.2f}%)"
        except Exception:
            return ""

    page_df["Delta"] = page_df.apply(_fmt_delta, axis=1)

    # Confidence (if present in data); format to two decimals
    if "confidence" in page_df.columns:
        try:
            page_df["Confidence"] = page_df["confidence"].astype(float).round(2)
        except Exception:
            page_df["Confidence"] = page_df["confidence"]
    else:
        page_df["Confidence"] = ""

    # Per-row LLM model display
    if "llm_model" in page_df.columns:
        page_df["LLM model"] = page_df["llm_model"].astype(str)
    else:
        page_df["LLM model"] = ""

    # Human-friendly direction and leverage
    if "direction" in page_df.columns:
        page_df["Direction"] = page_df["direction"].astype(str).str.capitalize()
    else:
        page_df["Direction"] = ""
    if "leverage" in page_df.columns:
        try:
            page_df["Leverage"] = page_df["leverage"].astype(int)
        except Exception:
            page_df["Leverage"] = page_df["leverage"]
    else:
        page_df["Leverage"] = ""

    # Final column selection and formatting
    view_cols = [
        "ts",
        "Direction",
        "Leverage",
        "Signal price",
        "Current price",
        "Delta",
        "Confidence",
        "LLM model",
    ]
    view = page_df[[c for c in view_cols if c in page_df.columns]].copy()
    # Round numeric columns for display
    for c in ("Signal price", "Current price", "Delta"):
        if c in view.columns:
            try:
                view[c] = view[c].astype(float).round(4)
            except Exception:
                pass

    # Delta styling based on numeric delta sign
    sign_map = {}
    try:
        sign_map = {
            idx: (
                "pos"
                if (not pd.isna(v) and float(v) > 0)
                else ("neg" if (not pd.isna(v) and float(v) < 0) else "zero")
            )
            for idx, v in page_df["_DeltaNum"].items()
        }
    except Exception:
        sign_map = {}

    def _delta_style_col(s: pd.Series):
        out = []
        for idx in s.index:
            tag = sign_map.get(idx)
            if tag == "pos":
                out.append("color: #137333; font-weight: 600;")
            elif tag == "neg":
                out.append("color: #c5221f; font-weight: 600;")
            else:
                out.append("")
        return out

    # Apply Confidence formatting (two decimals) and Delta color styling
    styler = view.style
    if "Confidence" in view.columns:
        try:
            styler = styler.format({"Confidence": "{:.2f}"})
        except Exception:
            pass
    if "Delta" in view.columns:
        styler = styler.apply(_delta_style_col, subset=["Delta"])
    styled = styler
    st.dataframe(styled, use_container_width=True)
