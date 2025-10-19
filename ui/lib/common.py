import os as _os
import sys as _sys

# Ensure project root on sys.path for `src` imports when running Streamlit from ui/
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

LOGBOOK_DIR = _os.getenv("LOGBOOK_DIR", "data/logbook")
CONTROL_DIR = _os.getenv("CONTROL_DIR", _os.path.join("data", "control"))

__all__ = ["LOGBOOK_DIR", "CONTROL_DIR"]
PAGE_HEADER_TITLE = "Binance Spot Signal Bot — Live Monitor"


def render_common_sidebar(st):
    from .control_utils import (
        read_status,
        read_desired,
        set_desired_state,
        get_effective_status,
    )

    st.subheader("Controls")
    current_status = read_status()
    effective = get_effective_status()
    hb_ts = current_status.get("heartbeat_ts")
    alive = False
    try:
        alive = bool(
            hb_ts is not None
            and (int(__import__("time").time() * 1000) - int(hb_ts)) < 3000
        )
    except Exception:
        alive = False

    # Keep desired state synchronized across pages: always reconcile from file
    desired_current = read_desired() == "running"
    if st.session_state.get("desired_running") != desired_current:
        st.session_state["desired_running"] = bool(desired_current)
    # Persist "show_price_panel" across hard reloads via query params (price=1|0)
    if "show_price_panel" not in st.session_state:
        try:
            qp = getattr(st, "query_params", {}) or {}
            raw = None
            try:
                raw = qp.get(
                    "price"
                )  # may be list or str depending on Streamlit version
            except Exception:
                raw = None
            if isinstance(raw, (list, tuple)):
                raw = raw[0] if raw else None
            default_show = str(raw).lower() in ("1", "true", "yes")
        except Exception:
            default_show = False
        st.session_state["show_price_panel"] = default_show

    def _apply_desired_change() -> None:
        val = bool(st.session_state.get("desired_running", desired_current))
        ok = set_desired_state(val)
        if ok:
            st.toast("Desired state updated")
            # No explicit rerun in callback; Streamlit re-runs automatically.
        else:
            st.error(f"Failed to update control at {CONTROL_DIR}")

    st.toggle(
        "Bot Running",
        value=bool(desired_current),
        help="Start/Stop the bot",
        key="desired_running",
        on_change=_apply_desired_change,
    )

    # Subtle hint if desired and effective diverge (e.g., waiting for heartbeat)
    try:
        desired_label = "running" if desired_current else "stopped"
        if desired_label != effective:
            st.caption(
                f"Desired: {desired_label} • Actual: {effective} — waiting for heartbeat…"
            )
    except Exception:
        pass
    # Initialize shared sidebar state for symbol/refresh so it's consistent across pages
    if "sidebar_symbol" not in st.session_state:
        st.session_state["sidebar_symbol"] = "BTCUSDT"
    if "sidebar_refresh" not in st.session_state:
        st.session_state["sidebar_refresh"] = 2

    symbol = st.selectbox(
        "Symbol",
        ["BTCUSDT"],  # extend later
        key="sidebar_symbol",
    )
    refresh = st.number_input(
        "Auto-refresh (s)", min_value=1, max_value=30, step=1, key="sidebar_refresh"
    )

    # Hide live price chart control and always return False for panel visibility
    return symbol, int(refresh), False
