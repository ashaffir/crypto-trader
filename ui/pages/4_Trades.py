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
from ui.lib.settings_state import load_sidebar_settings
from ui.lib.logbook_utils import tail_parquet_table


st.set_page_config(page_title="Trades", layout="wide")
st.title(PAGE_HEADER_TITLE)
render_status_badge(st)
# Use persisted sidebar settings but do not render the sidebar controls here
_persisted = load_sidebar_settings()
symbol = _persisted.get("symbol", "BTCUSDT")
st.subheader("Trades")
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
