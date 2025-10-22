import os as _os
import sys as _sys

# Ensure project root on sys.path for `src` imports when running Streamlit from ui/
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)


def _resolve_path(env_var_name: str, default_rel: str) -> str:
    """Resolve a data path that works in both Docker and local runs.

    Preference order:
      1) Environment variable if set and exists
      2) Project-root absolute default (ui/..../default_rel)
      3) Plain relative default string
    """
    env_val = _os.getenv(env_var_name)
    if env_val and _os.path.isdir(env_val):
        return env_val

    # Try absolute path anchored at project root (parent of `ui/`)
    abs_default = _os.path.abspath(_os.path.join(_ROOT, default_rel))
    if _os.path.isdir(abs_default):
        return abs_default

    # Fallback to plain relative for last resort
    return default_rel


LOGBOOK_DIR = _resolve_path("LOGBOOK_DIR", _os.path.join("data", "logbook"))
CONTROL_DIR = _resolve_path("CONTROL_DIR", _os.path.join("data", "control"))

__all__ = ["LOGBOOK_DIR", "CONTROL_DIR"]
PAGE_HEADER_TITLE = "Crypto Bot"


def render_common_sidebar(st):
    """Deprecated shim: sidebar controls removed. Returns neutral defaults.

    This function remains temporarily to avoid import errors if any page still
    references it. It returns a placeholder tuple and renders nothing.
    """
    return "BTCUSDT", 2, False


def render_status_badge(st) -> None:
    """Render a small right-aligned status badge (Running/Stopped).

    Intended for use on non-Home pages. Reads the effective bot status and shows
    a colored pill: green when running, red when stopped.
    """
    pass


#     from .control_utils import get_effective_status

#     status = get_effective_status()
#     running = str(status).lower() == "running"

#     color = "#16a34a" if running else "#dc2626"  # green-600 / red-600
#     bg = "rgba(22,163,74,0.12)" if running else "rgba(220,38,38,0.12)"
#     text = "RUNNING" if running else "STOPPED"

#     st.markdown(
#         f"""
# <div style="display:flex; justify-content:flex-end; margin-top:-0.5rem;">
#   <span style="display:inline-flex; align-items:center; gap:8px; padding:4px 10px; border-radius:999px; background:{bg}; color:{color}; font-weight:600; font-size:12px; letter-spacing:0.02em;">
#     <span style="width:8px; height:8px; border-radius:999px; background:{color}; box-shadow:0 0 0 2px rgba(0,0,0,0.05);"></span>
#     {text}
#   </span>
# </div>
#         """,
#         unsafe_allow_html=True,
#     )
