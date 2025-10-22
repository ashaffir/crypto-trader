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

st.write("")

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
