import os
import json
import sys as _sys
import os as _os
import pandas as pd
import streamlit as st

# Ensure project root (parent of `ui/`) is on sys.path so `ui.*` and `src.*` are importable
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from ui.lib.common import (
    LOGBOOK_DIR,
    CONTROL_DIR,
    render_common_sidebar,
    PAGE_HEADER_TITLE,
)
from ui.lib.logbook_utils import tail_parquet_table
from ui.lib.control_utils import (
    read_status,
    read_desired,
    set_desired_state,
    get_effective_status,
)


st.set_page_config(page_title="Home", layout="wide")

st.title(PAGE_HEADER_TITLE)
symbol, refresh, show_price_panel = render_common_sidebar(st)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Recent Signals")
    sig_df = tail_parquet_table("signal_emitted", symbol)
    if not sig_df.empty:
        # Add human-readable timestamp string (no timezone suffix)
        try:
            sig_df["ts"] = (
                pd.to_datetime(sig_df["ts_ms"], unit="ms", utc=True)
                .dt.tz_localize(None)
                .dt.strftime("%Y-%m-%d %H:%M:%S.%f")
                .str[:-3]
            )
        except Exception:
            sig_df["ts"] = ""
        sig_df = sig_df.sort_values("ts_ms", ascending=False).head(50)
        st.dataframe(
            sig_df[["ts", "symbol", "side", "expected_bps", "confidence", "rule_id"]]
        )
    else:
        st.info("No signals yet")

with col2:
    st.subheader("Recent Performance (outcomes)")
    out_df = tail_parquet_table("signal_outcome", symbol)
    if not out_df.empty:
        # Add human-readable resolved timestamp string (no timezone suffix)
        try:
            out_df["resolved_ts"] = (
                pd.to_datetime(out_df["resolved_ts_ms"], unit="ms", utc=True)
                .dt.tz_localize(None)
                .dt.strftime("%Y-%m-%d %H:%M:%S.%f")
                .str[:-3]
            )
        except Exception:
            out_df["resolved_ts"] = ""
        out_df = out_df.sort_values("resolved_ts_ms", ascending=False).head(100)
        st.metric("Hit-rate (last 100)", f"{(out_df['hit'].mean()*100):.1f}%")
        st.metric("Mean return (bps)", f"{out_df['ret_bps'].mean():.2f}")
        st.dataframe(
            out_df[
                [
                    "signal_id",
                    "resolved_ts",
                    "ret_bps",
                    "hit",
                    "max_adverse_bps",
                    "max_favorable_bps",
                ]
            ]
        )
    else:
        st.info("No outcomes yet")

# Live price/heartbeat panel
st.divider()
st.subheader("Live Status")
status = read_status()
if status:
    effective_status = get_effective_status()

    qsz = status.get("queue_size")
    if qsz is not None:
        cols = st.columns(2)
        cols[0].metric("Status", effective_status)
        cols[1].metric("Backlog (msgs)", int(qsz))
    else:
        cols = st.columns(1)
        cols[0].metric("Status", effective_status)

    # Switch reflects actual; desired mismatch banner suppressed
else:
    st.info("No heartbeat yet. Start the bot to see status.")

# Show control dir for clarity only
st.caption(f"CONTROL_DIR: {CONTROL_DIR}")

st.caption("Auto-refreshing…")
# Use non-blocking auto-refresh to avoid keeping the script in RUNNING state
try:
    if hasattr(st, "autorefresh"):
        st.autorefresh(interval=int(refresh) * 1000, key="home_autorefresh")
    else:
        # Fallback: lightweight JS reload after the interval
        st.markdown(
            f"<script>setTimeout(function(){{window.location.reload();}}, {int(refresh) * 1000});</script>",
            unsafe_allow_html=True,
        )
except Exception:
    # As a last resort, do nothing (no auto-refresh) rather than blocking the UI
    pass
