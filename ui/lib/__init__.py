"""Internal utilities for the Streamlit UI.

This subpackage groups helpers used across UI pages and the entrypoint.
Explicit `__init__` ensures Python treats `ui.lib` as a regular package,
avoiding collisions with any third-party `ui` namespace packages.
"""

from .common import LOGBOOK_DIR, CONTROL_DIR, PAGE_HEADER_TITLE  # noqa: F401
from .logbook_utils import tail_parquet_table, read_latest_file  # noqa: F401
from .control_utils import (  # noqa: F401
    read_status,
    read_desired,
    set_desired_state,
    get_effective_status,
    RCM,
)

__all__ = [
    "LOGBOOK_DIR",
    "CONTROL_DIR",
    "PAGE_HEADER_TITLE",
    "tail_parquet_table",
    "read_latest_file",
    "read_status",
    "read_desired",
    "set_desired_state",
    "get_effective_status",
    "RCM",
]
