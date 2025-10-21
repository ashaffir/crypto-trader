import os as _os
import sys as _sys

# Ensure project root on sys.path for `src` imports when running Streamlit from ui/
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

LOGBOOK_DIR = _os.getenv("LOGBOOK_DIR", "data/logbook")
CONTROL_DIR = _os.getenv("CONTROL_DIR", _os.path.join("data", "control"))

__all__ = ["LOGBOOK_DIR", "CONTROL_DIR"]
PAGE_HEADER_TITLE = "Crypto Bot"


def render_common_sidebar(st):
    from .control_utils import (
        read_status,
        read_desired,
        set_desired_state,
        get_effective_status,
    )
    from ui.lib.settings_state import (
        load_tracked_symbols,
        load_sidebar_settings,
        save_sidebar_settings,
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
    # Initialize shared sidebar state for symbol/refresh using persisted values
    try:
        symbols = load_tracked_symbols()
        if not symbols:
            symbols = ["BTCUSDT"]
    except Exception:
        symbols = ["BTCUSDT"]

    try:
        persisted = load_sidebar_settings()
    except Exception:
        persisted = {"symbol": symbols[0], "refresh_seconds": 2}

    default_symbol = (
        persisted.get("symbol") if persisted.get("symbol") in symbols else symbols[0]
    )
    default_refresh = int(persisted.get("refresh_seconds", 2) or 2)

    if (
        "sidebar_symbol" not in st.session_state
        or st.session_state["sidebar_symbol"] not in symbols
    ):
        st.session_state["sidebar_symbol"] = default_symbol
    if "sidebar_refresh" not in st.session_state:
        st.session_state["sidebar_refresh"] = default_refresh

    def _persist():
        try:
            save_sidebar_settings(
                {
                    "symbol": st.session_state.get("sidebar_symbol"),
                    "refresh_seconds": int(st.session_state.get("sidebar_refresh", 2)),
                }
            )
        except Exception:
            pass

    symbol = st.selectbox(
        "Symbol",
        symbols,
        key="sidebar_symbol",
        on_change=_persist,
        index=(
            symbols.index(st.session_state["sidebar_symbol"])
            if st.session_state.get("sidebar_symbol") in symbols
            else 0
        ),
    )
    refresh = st.number_input(
        "Auto-refresh (s)",
        min_value=1,
        max_value=30,
        step=1,
        key="sidebar_refresh",
        on_change=_persist,
    )

    # Hide live price chart control and always return False for panel visibility
    return symbol, int(refresh), False


def render_status_badge(st) -> None:
    """Render a small right-aligned status badge (Running/Stopped).

    Intended for use on non-Home pages. Reads the effective bot status and shows
    a colored pill: green when running, red when stopped.
    """
    from .control_utils import get_effective_status

    status = get_effective_status()
    running = str(status).lower() == "running"

    color = "#16a34a" if running else "#dc2626"  # green-600 / red-600
    bg = "rgba(22,163,74,0.12)" if running else "rgba(220,38,38,0.12)"
    text = "RUNNING" if running else "STOPPED"

    st.markdown(
        f"""
<div style="display:flex; justify-content:flex-end; margin-top:-0.5rem;">
  <span style="display:inline-flex; align-items:center; gap:8px; padding:4px 10px; border-radius:999px; background:{bg}; color:{color}; font-weight:600; font-size:12px; letter-spacing:0.02em;">
    <span style="width:8px; height:8px; border-radius:999px; background:{color}; box-shadow:0 0 0 2px rgba(0,0,0,0.05);"></span>
    {text}
  </span>
</div>
        """,
        unsafe_allow_html=True,
    )
