import streamlit as st
import sys as _sys
import os as _os

# Ensure project root (parent of `ui/`) is on sys.path so absolute imports work
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from ui.lib.common import (
    CONTROL_DIR,
    render_common_sidebar,
    PAGE_HEADER_TITLE,
    LOGBOOK_DIR,
)
from ui.lib.control_utils import RCM
from ui.lib.settings_state import load_backtesting_settings
import pandas as pd
import importlib.util as _importlib_util


def _load_backtesting_engine():
    """Load logical_test and quality_test with robust fallbacks.

    - First try normal package import (src.backtesting.engine).
    - If unavailable (e.g., when running Streamlit with different CWD),
      import directly from the file path.
    """
    try:
        from src.backtesting.engine import logical_test, quality_test  # type: ignore

        return logical_test, quality_test
    except Exception as _e:
        engine_path = _os.path.abspath(
            _os.path.join(
                _os.path.dirname(__file__),
                "..",
                "..",
                "src",
                "backtesting",
                "engine.py",
            )
        )
        if not _os.path.exists(engine_path):
            raise RuntimeError(
                "Backtesting engine not found. Ensure the UI container mounts ./src or install the package."
            ) from _e
        spec = _importlib_util.spec_from_file_location("_bt_engine", engine_path)
        if spec and spec.loader:
            mod = _importlib_util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            return getattr(mod, "logical_test"), getattr(mod, "quality_test")
        raise


def _load_backtesting_loader():
    try:
        from src.backtesting.loader import load_signals  # type: ignore

        return load_signals
    except Exception as _e:
        loader_path = _os.path.abspath(
            _os.path.join(
                _os.path.dirname(__file__),
                "..",
                "..",
                "src",
                "backtesting",
                "loader.py",
            )
        )
        if not _os.path.exists(loader_path):
            raise RuntimeError(
                "Backtesting loader not found. Ensure the UI container mounts ./src or install the package."
            ) from _e
        spec = _importlib_util.spec_from_file_location("_bt_loader", loader_path)
        if spec and spec.loader:
            mod = _importlib_util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            return getattr(mod, "load_signals")
        raise


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

# ---------------- Backtesting ----------------
st.divider()
st.subheader("Backtesting")

bt_cols = st.columns(2)

with bt_cols[0]:
    st.caption("Logical Test — validates signal schema/timing on a small slice")
    if st.button("Run Logical Test", key="btn_logical_test"):
        with st.spinner("Running logical test..."):
            try:
                _logical_test, _quality_test = _load_backtesting_engine()
                bt_cfg = load_backtesting_settings()
                max_files = int(bt_cfg.get("logical_max_files", 10) or 10)
                res = _logical_test(_symbol, base_dir=LOGBOOK_DIR, max_files=max_files)
                st.session_state["_bt_logical_res"] = {
                    "num_signals": res.num_signals,
                    "fields_ok": res.fields_ok,
                    "time_monotonic": res.time_monotonic,
                    "sample": (
                        res.sample.copy()
                        if isinstance(res.sample, pd.DataFrame)
                        else pd.DataFrame()
                    ),
                    "t0": getattr(res, "timeframe_start_ms", None),
                    "t1": getattr(res, "timeframe_end_ms", None),
                }
            except Exception as e:
                st.session_state["_bt_logical_res"] = {"error": str(e)}

with bt_cols[1]:
    st.caption("Performance Test — simulates outcomes and computes metrics over data")
    if st.button("Run Performance Test", key="btn_quality_test"):
        with st.spinner("Running performance test..."):
            try:
                _logical_test, _quality_test = _load_backtesting_engine()
                bt_cfg = load_backtesting_settings()
                q_max = bt_cfg.get("quality_max_files")
                res = _quality_test(
                    _symbol,
                    base_dir=LOGBOOK_DIR,
                    horizon_s=int(scalp),
                    max_files=(
                        int(q_max) if isinstance(q_max, int) and q_max > 0 else None
                    ),
                )
                st.session_state["_bt_quality_res"] = {
                    "report": res.report,
                    "outcomes": (
                        res.outcomes.copy()
                        if isinstance(res.outcomes, pd.DataFrame)
                        else pd.DataFrame()
                    ),
                }
            except Exception as e:
                st.session_state["_bt_quality_res"] = {"error": str(e)}


# Results display (inline, expandable). Can be switched to modals if desired.
log_res = st.session_state.get("_bt_logical_res")
if log_res:
    with st.expander("Logical Test Results", expanded=True):
        if "error" in log_res:
            st.error(f"Logical test failed: {log_res['error']}")
        else:
            cols = st.columns(4)
            cols[0].metric("Signals", int(log_res.get("num_signals", 0)))
            cols[1].metric("Fields OK", "Yes" if log_res.get("fields_ok") else "No")
            cols[2].metric(
                "Time Monotonic", "Yes" if log_res.get("time_monotonic") else "No"
            )
            t0 = log_res.get("t0")
            t1 = log_res.get("t1")
            if isinstance(t0, int) and isinstance(t1, int):
                try:
                    t0s = pd.to_datetime(t0, unit="ms", utc=True).tz_localize(None)
                    t1s = pd.to_datetime(t1, unit="ms", utc=True).tz_localize(None)
                    cols[3].metric("Timeframe", f"{t0s} → {t1s}")
                except Exception:
                    cols[3].metric("Timeframe", "-")
            else:
                cols[3].metric("Timeframe", "-")
            sample = log_res.get("sample")
            if not isinstance(sample, pd.DataFrame):
                sample = pd.DataFrame()
            if not sample.empty:
                if "ts_ms" in sample.columns:
                    try:
                        sample["ts"] = (
                            pd.to_datetime(sample["ts_ms"], unit="ms", utc=True)
                            .dt.tz_localize(None)
                            .dt.strftime("%Y-%m-%d %H:%M:%S.%f")
                            .str[:-3]
                        )
                    except Exception:
                        pass
                st.caption("Sample (up to 20 rows)")
                # Hide raw technical columns in logical test view
                try:
                    _drop = [c for c in ["ts_ms", "signal_id"] if c in sample.columns]
                    sample_to_show = sample.drop(columns=_drop) if _drop else sample
                except Exception:
                    sample_to_show = sample
                st.dataframe(sample_to_show)

                # Enhanced signals chart
                try:
                    import altair as alt

                    # Prefer full slice for chart (bounded by settings)
                    bt_cfg = load_backtesting_settings()
                    chart_cap = int(bt_cfg.get("chart_points", 200) or 200)
                    chart_df = None
                    try:
                        _load_sig = _load_backtesting_loader()
                        _max_files = int(bt_cfg.get("logical_max_files", 10) or 10)
                        chart_df = _load_sig(
                            _symbol, base_dir=LOGBOOK_DIR, max_files=_max_files
                        )
                    except Exception:
                        pass
                    if (
                        chart_df is None
                        or not isinstance(chart_df, pd.DataFrame)
                        or chart_df.empty
                    ):
                        chart_df = sample.copy()

                    if "ts" not in chart_df.columns and "ts_ms" in chart_df.columns:
                        chart_df["ts"] = pd.to_datetime(
                            chart_df["ts_ms"], unit="ms", utc=True
                        ).dt.tz_localize(None)

                    # Keep only the latest chart_cap points
                    try:
                        chart_df = chart_df.sort_values("ts").tail(chart_cap)
                    except Exception:
                        pass

                    # Filters
                    fcols = st.columns(3)
                    with fcols[0]:
                        _sides = (
                            sorted(
                                [
                                    s
                                    for s in chart_df.get("side", [])
                                    .dropna()
                                    .unique()
                                    .tolist()
                                ]
                            )
                            if "side" in chart_df
                            else []
                        )
                        sel_sides = st.multiselect(
                            "Side", _sides, default=_sides, key="bt_side_filter"
                        )
                    with fcols[1]:
                        _rules = (
                            sorted(
                                [
                                    r
                                    for r in chart_df.get("rule_id", [])
                                    .dropna()
                                    .unique()
                                    .tolist()
                                ]
                            )
                            if "rule_id" in chart_df
                            else []
                        )
                        sel_rules = st.multiselect(
                            "Rule", _rules, default=_rules, key="bt_rule_filter"
                        )
                    with fcols[2]:
                        min_conf = st.slider(
                            "Min confidence", 0.0, 1.0, 0.0, 0.01, key="bt_min_conf"
                        )

                    def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
                        out = df
                        if sel_sides:
                            out = (
                                out[out["side"].isin(sel_sides)]
                                if "side" in out.columns
                                else out
                            )
                        if sel_rules:
                            out = (
                                out[out["rule_id"].isin(sel_rules)]
                                if "rule_id" in out.columns
                                else out
                            )
                        if "confidence" in out.columns:
                            out = out[out["confidence"] >= float(min_conf)]
                        return out

                    chart_df = _apply_filters(chart_df)

                    st.caption("Signals over time")
                    scatter = (
                        alt.Chart(chart_df)
                        .mark_circle(opacity=0.6)
                        .encode(
                            x=alt.X("ts:T", title="Time"),
                            y=alt.Y("expected_bps:Q", title="Expected bps"),
                            size=alt.Size(
                                "confidence:Q",
                                title="Confidence",
                                scale=alt.Scale(range=[40, 200]),
                            ),
                            color=alt.Color("side:N", title="Side"),
                            shape=alt.Shape("rule_id:N", title="Rule"),
                            tooltip=[
                                "symbol",
                                "side",
                                "expected_bps",
                                "confidence",
                                "rule_id",
                                alt.Tooltip("ts:T", title="time"),
                            ],
                        )
                        .properties(height=260)
                    )

                    # Density of signals (per second)
                    try:
                        density = (
                            alt.Chart(chart_df)
                            .mark_area(opacity=0.25)
                            .encode(
                                x=alt.X(
                                    "yearmonthdatehoursminutesseconds(ts):T",
                                    title="Time",
                                ),
                                y=alt.Y("count()", title="Signals"),
                                tooltip=[alt.Tooltip("count()", title="signals")],
                            )
                            .properties(height=100)
                        )
                        chart = alt.vconcat(scatter.interactive(), density, spacing=8)
                    except Exception:
                        chart = scatter.interactive()

                    st.altair_chart(chart, use_container_width=True)
                except Exception:
                    pass
            else:
                st.info("No sample to display.")

qual_res = st.session_state.get("_bt_quality_res")
if qual_res:
    with st.expander("Performance Test Results", expanded=True):
        if "error" in qual_res:
            st.error(f"Performance test failed: {qual_res['error']}")
        else:
            rpt = qual_res.get("report")
            if rpt:
                mcols = st.columns(6)
                mcols[0].metric("Trades", int(getattr(rpt, "num_trades", 0)))
                mcols[1].metric("Win Rate", f"{getattr(rpt, 'win_rate', 0.0)*100:.1f}%")
                mcols[2].metric(
                    "Mean (bps)", f"{getattr(rpt, 'mean_ret_bps', 0.0):.2f}"
                )
                mcols[3].metric("Sharpe", f"{getattr(rpt, 'sharpe', 0.0):.2f}")
                mcols[4].metric("PnL (bps)", f"{getattr(rpt, 'pnl_bps', 0.0):.1f}")
                mcols[5].metric("Score", f"{getattr(rpt, 'score', 0.0):.2f}")
            out = qual_res.get("outcomes")
            if not isinstance(out, pd.DataFrame):
                out = pd.DataFrame()
            if not out.empty:
                try:
                    out = out.sort_values("resolved_ts_ms", ascending=False)
                except Exception:
                    pass
                st.caption("Outcomes (latest 200)")
                st.dataframe(out.head(200))
            else:
                st.info("No outcomes to display.")
