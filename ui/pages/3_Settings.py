import streamlit as st
import sys as _sys
import os as _os

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from ui.lib.common import PAGE_HEADER_TITLE
from ui.lib.settings_state import (
    load_backtesting_settings,
    save_backtesting_settings,
)


st.set_page_config(page_title="Settings", layout="wide")
st.title(PAGE_HEADER_TITLE)
st.subheader("Settings")

cfg = load_backtesting_settings()

st.markdown("**Backtesting**")
c1, c2, c3 = st.columns(3)
with c1:
    logical_max = st.number_input(
        "Logical: max files",
        min_value=1,
        max_value=200,
        value=int(cfg.get("logical_max_files", 10)),
        step=1,
        help="Number of latest files to read for logical test",
    )
with c2:
    quality_max = st.number_input(
        "Quality: max files (0=all)",
        min_value=0,
        max_value=5000,
        value=int(cfg.get("quality_max_files", 0) or 0),
        step=10,
        help="0 means use all available files",
    )
with c3:
    chart_points = st.number_input(
        "Chart points",
        min_value=50,
        max_value=5000,
        value=int(cfg.get("chart_points", 200)),
        step=50,
        help="Max points to plot in charts",
    )


def _save() -> None:
    ok = save_backtesting_settings(
        {
            "logical_max_files": int(logical_max),
            "quality_max_files": int(quality_max) if int(quality_max) > 0 else None,
            "chart_points": int(chart_points),
        }
    )
    if ok:
        st.success("Saved")
    else:
        st.error("Failed to save settings")


st.button("Save", on_click=_save)
