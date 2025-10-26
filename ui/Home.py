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

# --- Account Balances ---
try:
    from ui.lib.common import render_status_badge

    render_status_badge(st)

    try:
        from streamlit_autorefresh import st_autorefresh as _st_autorefresh  # type: ignore
    except Exception:  # pragma: no cover
        _st_autorefresh = None

    # Auto-refresh every N seconds based on config.ui.auto_refresh_seconds (fallback 10s)
    try:
        from src.config import load_app_config

        _cfg = load_app_config()
        _rs = int(getattr(_cfg.ui, "auto_refresh_seconds", 10))
    except Exception:
        _rs = 10

    if _st_autorefresh is not None and _rs > 0:
        _st_autorefresh(interval=_rs * 1000, key="home_balance_auto_refresh")
        st.caption(f"Auto-refresh every {_rs}s")

    st.subheader("Account Balances")
    st.caption("Shows Binance balances using keys saved in Settings (execution mode).")

    import asyncio
    from src.utils.binance_account import fetch_balances_from_runtime_config

    async def _load():
        return await fetch_balances_from_runtime_config()

    data = asyncio.run(_load())

    if not isinstance(data, dict) or not data.get("ok"):
        err = data.get("error") if isinstance(data, dict) else "unknown"
        if err == "missing_credentials":
            st.info("Add API key/secret in Settings → Execution to view live balances.")
        else:
            st.warning(f"Unable to load balances: {err}")
    else:
        venue = str(data.get("venue") or "spot").lower()
        network = str(data.get("network") or "mainnet").lower()
        bals = data.get("balances", [])
        # Show top non-zero first
        non_zero = [
            b
            for b in bals
            if float(b.get("total", 0)) > 0
            or float(b.get("available", 0)) > 0
            or float(b.get("locked", 0)) > 0
        ]
        zeros = [b for b in bals if b not in non_zero]
        ordered = non_zero + zeros

        st.caption(f"Venue: {venue.upper()} · Network: {network.upper()}")
        import pandas as _pd

        if ordered:
            df = _pd.DataFrame(ordered)
            # Column order
            cols = [
                c for c in ["asset", "available", "locked", "total"] if c in df.columns
            ]
            df = df[cols]
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No assets found.")
except Exception as e:  # pragma: no cover
    st.warning(f"Balances widget unavailable: {e}")

# Spacer
st.write("")

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
