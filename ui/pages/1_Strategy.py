import streamlit as st
import sys as _sys
import os as _os

# Ensure project root (parent of `ui/`) is on sys.path so absolute imports work
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
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

# Load current overrides for defaults
current_overrides = RCM.read() or {}
cur_thr = current_overrides.get("signal_thresholds") or {}
cur_hz = current_overrides.get("horizons") or {}
cur_rules = current_overrides.get("rules") or {}

# Initialize parameters with current defaults so they always exist for persistence
imb = float(cur_thr.get("imbalance", 0.6))
spd = float(cur_thr.get("max_spread_bps", 1.5))
scalp = int(cur_hz.get("scalp", 30))
ttl = int(cur_hz.get("ttl_s", 10))
mr_min_revert_bps = float(cur_thr.get("mr_min_revert_bps", 2.0))
mr_expected_bps = float(cur_thr.get("mr_expected_bps", 6.0))
mr_conf_norm_bps = float(cur_thr.get("mr_conf_norm_bps", 5.0))
mr_max_imbalance = float(cur_thr.get("mr_max_imbalance", 1.0))

col_left, col_right = st.columns([1, 2])
with col_left:
    st.subheader("Rules")
    mom_enabled = st.checkbox(
        "Momentum enabled",
        value=bool(cur_rules.get("momentum_enabled", True)),
        key="rule_momentum_enabled",
    )
    mr_enabled = st.checkbox(
        "Mean-reversion enabled",
        value=bool(cur_rules.get("mean_reversion_enabled", True)),
        key="rule_mr_enabled",
    )

    # Select which rule to configure on the right side
    rule_display_names = {
        "momentum": "Momentum",
        "mean_reversion": "Mean-Reversion",
    }
    default_rule_key = st.session_state.get("selected_rule_key", "momentum")
    selected_rule = st.radio(
        "Configure rule",
        options=list(rule_display_names.keys()),
        format_func=lambda k: rule_display_names[k],
        index=list(rule_display_names.keys()).index(default_rule_key),
        key="selected_rule_key",
        help="Choose a rule to edit its parameters",
    )

with col_right:
    if selected_rule == "momentum":
        st.subheader("Momentum Settings")
        imb = st.slider(
            "Imbalance threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(imb),
            step=0.05,
            key="mom_imbalance_thr",
        )
        spd = st.slider(
            "Max spread (bps)",
            min_value=0.1,
            max_value=10.0,
            value=float(spd),
            step=0.1,
            key="mom_max_spread_bps",
        )
        st.divider()
        scalp = st.number_input(
            "Horizon (s)",
            min_value=5,
            max_value=300,
            value=int(scalp),
            step=5,
            key="mom_horizon_s",
        )
        ttl = st.number_input(
            "Signal TTL (s)",
            min_value=1,
            max_value=120,
            value=int(ttl),
            step=1,
            key="mom_ttl_s",
        )
    else:
        st.subheader("Mean-Reversion Settings")
        mr_cols = st.columns(2)
        with mr_cols[0]:
            mr_min_revert_bps = st.number_input(
                "MR min revert (bps)",
                min_value=0.1,
                max_value=50.0,
                value=float(mr_min_revert_bps),
                step=0.1,
                key="mr_min_revert_bps",
            )
            mr_expected_bps = st.number_input(
                "MR expected bps",
                min_value=0.1,
                max_value=50.0,
                value=float(mr_expected_bps),
                step=0.1,
                key="mr_expected_bps",
            )
        with mr_cols[1]:
            mr_conf_norm_bps = st.number_input(
                "MR conf norm (bps)",
                min_value=0.1,
                max_value=50.0,
                value=float(mr_conf_norm_bps),
                step=0.1,
                key="mr_conf_norm_bps",
            )
            mr_max_imbalance = st.slider(
                "MR max |imbalance|",
                min_value=0.0,
                max_value=1.0,
                value=float(mr_max_imbalance),
                step=0.05,
                key="mr_max_imbalance",
            )
        st.divider()
        # Show shared thresholds/horizons here as they affect MR as well
        spd = st.slider(
            "Max spread (bps)",
            min_value=0.1,
            max_value=10.0,
            value=float(spd),
            step=0.1,
            key="mr_max_spread_bps",
            help="Shared across rules",
        )
        scalp = st.number_input(
            "Horizon (s)",
            min_value=5,
            max_value=300,
            value=int(scalp),
            step=5,
            key="mr_horizon_s",
        )
        ttl = st.number_input(
            "Signal TTL (s)",
            min_value=1,
            max_value=120,
            value=int(ttl),
            step=1,
            key="mr_ttl_s",
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
            # mean-reversion
            "mr_min_revert_bps": float(mr_min_revert_bps),
            "mr_expected_bps": float(mr_expected_bps),
            "mr_conf_norm_bps": float(mr_conf_norm_bps),
            "mr_max_imbalance": float(mr_max_imbalance),
        },
        "horizons": {"scalp": int(scalp), "ttl_s": int(ttl)},
    }
    if RCM.write(new_cfg):
        st.success("Updated. Bot will hot-reload within ~1s.")
    else:
        st.error("Failed to write runtime_config.json")


st.button("Apply strategy", on_click=_persist)
