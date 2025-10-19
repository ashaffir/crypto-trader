"""Top-level package for the Streamlit UI.

Having this file prevents accidental shadowing by third-party packages named
`ui` and makes absolute imports like `from ui.lib.common import ...` reliable
in all environments (local and inside containers).
"""

# Re-export a few commonly used helpers for convenience
try:
    from .lib.common import LOGBOOK_DIR, CONTROL_DIR, PAGE_HEADER_TITLE  # noqa: F401
except Exception:
    # Keep package importable even if optional deps are missing at import time
    pass
