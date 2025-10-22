import sys as _sys
import os as _os
import streamlit as st

# Ensure project root (parent of `ui/`) is on sys.path so `ui.*` and `src.*` are importable
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from ui.lib.control_utils import (
    read_desired,
    set_desired_state,
    get_effective_status,
)


st.set_page_config(page_title="Home", layout="wide")

# Minimal controls panel with only the Bot Running toggle
st.subheader("Controls")

desired_current = read_desired() == "running"
if st.session_state.get("desired_running") != desired_current:
    st.session_state["desired_running"] = bool(desired_current)


def _apply_desired_change() -> None:
    val = bool(st.session_state.get("desired_running", desired_current))
    ok = set_desired_state(val)
    if ok:
        st.toast("Desired state updated")
    else:
        from ui.lib.common import CONTROL_DIR as _CTRL

        st.error(f"Failed to update control at {_CTRL}")


st.toggle(
    "Bot Running",
    help="Start/Stop the bot",
    key="desired_running",
    on_change=_apply_desired_change,
)

# Subtle hint if desired and effective diverge (e.g., waiting for heartbeat)
try:
    desired_label = "running" if desired_current else "stopped"
    effective = get_effective_status()
    if desired_label != effective:
        st.caption(
            f"Desired: {desired_label} • Actual: {effective} — waiting for heartbeat…"
        )
except Exception:
    pass

st.write("")
