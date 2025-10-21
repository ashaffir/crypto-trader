import os as _os
import sys as _sys
import pandas as pd
import streamlit as st

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from ui.lib.common import (
    LOGBOOK_DIR,
    PAGE_HEADER_TITLE,
    render_status_badge,
)
from ui.lib.settings_state import load_sidebar_settings, load_tracked_symbols
from ui.lib.logbook_utils import tail_parquet_table, list_symbols_with_data


st.set_page_config(page_title="Trades", layout="wide")
st.title(PAGE_HEADER_TITLE)
render_status_badge(st)

# Build page-scoped symbol selector based on available data
_persisted = load_sidebar_settings()
_default_symbol = _persisted.get("symbol", "BTCUSDT")

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

st.subheader("Trades")
if not _available:
    st.caption(f"LOGBOOK_DIR: {LOGBOOK_DIR}")
    st.info("No trade data found in logbook.")
    st.stop()

symbol_index = _available.index(_default_symbol) if _default_symbol in _available else 0
symbol = st.selectbox(
    "Symbol", _available, index=symbol_index, key="trades_symbol_select"
)
st.caption(f"LOGBOOK_DIR: {LOGBOOK_DIR}")

df = tail_parquet_table("trade_recommendation", symbol, tail_files=50)
if df.empty:
    st.info("No trade recommendations yet.")
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
    cols = [
        "ts",
        "asset",
        "direction",
        "leverage",
        "source",
    ]
    # Backfill asset if missing (older rows)
    if "asset" not in df.columns and "symbol" in df.columns:
        df["asset"] = df["symbol"]
    view = df[[c for c in cols if c in df.columns]].tail(200)
    st.dataframe(view, use_container_width=True)
