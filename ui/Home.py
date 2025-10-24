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
from ui.lib.settings_state import load_supervisor_settings, save_supervisor_settings


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

# --- Max runtime control ---
try:
    sup = load_supervisor_settings()
    current_minutes = int(sup.get("max_run_minutes", 0))
except Exception:
    current_minutes = 0

col_left, col_right = st.columns([1, 3])
with col_left:
    new_minutes = st.number_input(
        "Max run time [minutes]",
        min_value=0,
        max_value=24 * 60,
        value=current_minutes,
        help="Set to 0 for unlimited. When non-zero, the bot will stop automatically once the time limit is reached.",
        step=1,
        key="max_run_minutes_input",
    )

if int(new_minutes) != int(current_minutes):
    ok = save_supervisor_settings({"max_run_minutes": int(new_minutes)})
    if ok:
        st.toast("Max runtime saved")
    else:
        st.error("Failed to save max runtime setting")

# Spacer
st.write("")
