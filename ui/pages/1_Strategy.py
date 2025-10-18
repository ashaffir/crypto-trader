import streamlit as st
import sys as _sys
import os as _os

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from ui.lib.common import CONTROL_DIR, render_common_sidebar, PAGE_HEADER_TITLE
from ui.lib.control_utils import RCM


st.set_page_config(page_title="Strategy Controls", layout="wide")
st.title(PAGE_HEADER_TITLE)
with st.sidebar:
    _symbol, _refresh, _show = render_common_sidebar(st)
st.subheader("Strategy Controls")

st.caption(f"CONTROL_DIR: {CONTROL_DIR}")

current_overrides = RCM.read() or {}
cur_thr = current_overrides.get("signal_thresholds") or {}
cur_hz = current_overrides.get("horizons") or {}
cur_rules = current_overrides.get("rules") or {}

col1, col2 = st.columns(2)
with col1:
    st.subheader("Rules")
    mom_enabled = st.checkbox(
        "Momentum enabled", value=bool(cur_rules.get("momentum_enabled", True))
    )
    mr_enabled = st.checkbox(
        "Mean-reversion enabled",
        value=bool(cur_rules.get("mean_reversion_enabled", True)),
    )

with col2:
    st.subheader("Thresholds & Horizons")
    imb = st.slider(
        "Imbalance threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(cur_thr.get("imbalance", 0.6)),
        step=0.05,
    )
    spd = st.slider(
        "Max spread (bps)",
        min_value=0.1,
        max_value=10.0,
        value=float(cur_thr.get("max_spread_bps", 1.5)),
        step=0.1,
    )
    scalp = st.number_input(
        "Horizon (s)",
        min_value=5,
        max_value=300,
        value=int(cur_hz.get("scalp", 30)),
        step=5,
    )
    ttl = st.number_input(
        "Signal TTL (s)",
        min_value=1,
        max_value=120,
        value=int(cur_hz.get("ttl_s", 10)),
        step=1,
    )


def _persist() -> None:
    new_cfg = {
        "rules": {
            "momentum_enabled": bool(mom_enabled),
            "mean_reversion_enabled": bool(mr_enabled),
        },
        "signal_thresholds": {
            "imbalance": float(imb),
            "max_spread_bps": float(spd),
        },
        "horizons": {"scalp": int(scalp), "ttl_s": int(ttl)},
    }
    if RCM.write(new_cfg):
        st.success("Updated. Bot will hot-reload within ~1s.")
    else:
        st.error("Failed to write runtime_config.json")


st.button("Apply strategy", on_click=_persist)
