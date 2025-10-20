import streamlit as st
import sys as _sys
import os as _os

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from ui.lib.common import PAGE_HEADER_TITLE, render_common_sidebar
from ui.lib.settings_state import (
    load_tracked_symbols,
    save_tracked_symbols,
    load_llm_settings,
    save_llm_settings,
)
from ui.lib.logbook_utils import read_latest_file
from src.utils.llm_client import LLMClient, LLMConfig
import asyncio


st.set_page_config(page_title="Settings", layout="wide")
st.title(PAGE_HEADER_TITLE)
with st.sidebar:
    _symbol, _refresh, _show = render_common_sidebar(st)
st.subheader("Settings")
st.markdown("**Tracked Symbols**")
symbols = load_tracked_symbols()
sym_text = st.text_input(
    "Symbols (comma separated)",
    value=", ".join(symbols) if symbols else "BTCUSDT",
    help="Enter spot symbols like BTCUSDT, ETHUSDT",
)


def _save_symbols() -> None:
    parts = [s.strip().upper() for s in sym_text.split(",") if s.strip()]
    if not parts:
        st.error("Provide at least one symbol")
        return
    ok = save_tracked_symbols(parts)
    if ok:
        st.success("Symbols saved")
    else:
        st.error("Failed to save symbols")


st.button("Save Symbols", on_click=_save_symbols)

st.divider()
