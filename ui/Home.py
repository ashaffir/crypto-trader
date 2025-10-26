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
from ui.lib.settings_state import (
    load_consensus_settings,
    save_consensus_settings,
    load_llm_configs,
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

# --- Consensus Mode ---
try:
    cons = load_consensus_settings()
    all_llms = [cfg.get("name") for cfg in load_llm_configs()] or []
    all_llms = [x for x in all_llms if isinstance(x, str) and x]

    col_c1, col_c2 = st.columns([1, 2])
    with col_c1:
        new_enabled = st.toggle(
            "Consensus mode",
            value=bool(cons.get("enabled", False)),
            help="When enabled, trade openings require unanimous agreement across selected LLMs.",
            key="consensus_enabled",
        )

    # Persist enabled toggle immediately (keep current members)
    if new_enabled != bool(cons.get("enabled", False)):
        ok = save_consensus_settings(
            {
                "enabled": bool(new_enabled),
                "members": list(cons.get("members", [])),
            }
        )
        if ok:
            st.toast("Consensus mode updated")
        else:
            st.error("Failed to update consensus mode")

    with col_c2:
        with st.form("consensus_members_form", clear_on_submit=False):
            st.caption("Select LLMs required for consensus:")
            current = set(cons.get("members", []))
            chosen: dict[str, bool] = {}
            for name in all_llms:
                chosen[name] = st.checkbox(
                    name,
                    value=(name in current),
                    key=f"consensus_member_{name}",
                )
            submitted = st.form_submit_button("Select")
            if submitted:
                new_members = [n for n, v in chosen.items() if v]
                ok = save_consensus_settings(
                    {"enabled": bool(new_enabled), "members": new_members}
                )
                if ok:
                    st.toast("Consensus members saved")
                else:
                    st.error("Failed to save consensus members")
except Exception as e:
    st.warning(f"Consensus controls unavailable: {e}")

# Spacer
st.write("")
