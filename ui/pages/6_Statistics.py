import os as _os
import sys as _sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timezone

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from ui.lib.common import render_status_badge
from ui.lib.statistics_utils import (
    load_positions_dataframe,
    compute_pnl_series,
    compute_pnl_series_by_model,
    compute_close_reason_distribution_by_model,
    compute_window_pnl_correlation,
    summarize_window_pnl_correlation,
    compute_confidence_vs_pnl,
)
from ui.lib.settings_state import (
    load_tracked_symbols,
    load_sidebar_settings,
    load_statistics_settings,
    save_statistics_settings,
    load_window_seconds,
)

try:
    from streamlit_autorefresh import st_autorefresh as _st_autorefresh  # type: ignore
except Exception:  # pragma: no cover
    _st_autorefresh = None


st.set_page_config(page_title="Statistics", layout="wide")
render_status_badge(st)
st.subheader("Statistics · PnL Over Time")

# Live update controls
try:
    _sidebar = load_sidebar_settings()
    _default_rs = max(1, int(_sidebar.get("refresh_seconds", 2)))
except Exception:
    _default_rs = 2

col_live, col_int = st.columns([1, 2])
with col_live:
    live_update = st.checkbox("Live update", value=True)
with col_int:
    refresh_seconds = st.slider(
        "Refresh interval (s)",
        min_value=1,
        max_value=10,
        value=int(_default_rs),
        step=1,
    )

if live_update and _st_autorefresh is not None:
    _st_autorefresh(interval=int(refresh_seconds) * 1000, key="stats_auto_refresh")
    st.caption(f"Auto-refresh every {int(refresh_seconds)}s")
elif live_update:
    st.caption("Auto-refresh unsupported in this Streamlit version")
else:
    st.caption("Live updates off")

# Controls row
all_tracked = load_tracked_symbols()
col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
with col1:
    selected_symbols = st.multiselect(
        "Symbols",
        options=all_tracked or ["BTCUSDT"],
        default=all_tracked or ["BTCUSDT"],
    )
with col2:
    show_cum = st.checkbox("Show Cumulative", value=True)
with col3:
    show_trades = st.checkbox("Show Per-Trade Bars", value=True)
with col4:
    bar_mode = st.selectbox("Bar Mode", options=["stack", "group", "overlay"], index=0)

# Additional grouping control (remember last)
_stats = load_statistics_settings()
group_by = st.selectbox(
    "Group By",
    options=["LLM Model", "Symbol", "None"],
    index=["LLM Model", "Symbol", "None"].index(
        str(_stats.get("pnl_group_by", "LLM Model"))
    ),
)
try:
    save_statistics_settings({"pnl_group_by": group_by})
except Exception:
    pass

# Load and filter data
df = load_positions_dataframe()
if df.empty:
    st.info("No positions yet.")
    st.stop()

if selected_symbols:
    df = df[df["symbol"].isin([s.upper() for s in selected_symbols])]

if group_by == "LLM Model":
    trades_df, cum_df = compute_pnl_series_by_model(df)
else:
    trades_df, cum_df = compute_pnl_series(df)

if trades_df.empty and cum_df.empty:
    st.info("No closed positions with PnL to plot yet.")
    st.stop()

# Build figure
fig = go.Figure()

# Convert index seconds to datetime for display


def _to_datetime_index(idx: pd.Index) -> pd.Index:
    try:
        return pd.to_datetime(idx.astype("int64"), unit="s")
    except Exception:
        return pd.to_datetime([], unit="s")


# Per-trade bars: group by selected dimension
if show_trades and not trades_df.empty:
    bars_df = trades_df.copy()
    bars_df["dt"] = _to_datetime_index(bars_df.index)
    if group_by == "LLM Model" and "llm_model" in bars_df.columns:
        for mdl, g in bars_df.groupby("llm_model"):
            fig.add_trace(
                go.Bar(
                    x=g["dt"],
                    y=g["pnl"],
                    name=f"Trade PnL · {mdl}",
                    opacity=0.75,
                )
            )
    elif (
        group_by == "Symbol"
        and "symbol" in bars_df.columns
        and len(selected_symbols) > 1
    ):
        for sym, g in bars_df.groupby("symbol"):
            fig.add_trace(
                go.Bar(
                    x=g["dt"],
                    y=g["pnl"],
                    name=f"Trade PnL · {sym}",
                    opacity=0.7,
                )
            )
    else:
        fig.add_trace(
            go.Bar(
                x=bars_df["dt"],
                y=bars_df["pnl"],
                name="Trade PnL",
                opacity=0.8,
            )
        )

# Cumulative line(s)
if show_cum and not cum_df.empty:
    if group_by == "LLM Model" and "llm_model" in cum_df.columns:
        for mdl, g in cum_df.reset_index().groupby("llm_model"):
            g = g.sort_values("ts")
            dt = _to_datetime_index(pd.Index(g["ts"].astype("int64")))
            fig.add_trace(
                go.Scatter(
                    x=dt,
                    y=g["cum_pnl"],
                    mode="lines",
                    name=f"Cumulative · {mdl}",
                    yaxis="y2" if show_trades else "y",
                )
            )
    else:
        c = cum_df.copy()
        c["dt"] = _to_datetime_index(c.index)
        fig.add_trace(
            go.Scatter(
                x=c["dt"],
                y=c["cum_pnl"],
                mode="lines",
                name="Cumulative PnL",
                line=dict(color="#2563eb", width=2),
                yaxis="y2" if show_trades else "y",
            )
        )

# Layout with range slider + selectors and uirevision (preserve zoom on rerun)
fig.update_layout(
    template="plotly_white",
    height=520,
    barmode=bar_mode,
    uirevision="stats_pnl_chart_v1",
    xaxis=dict(
        title="Time",
        showgrid=True,
        rangeslider=dict(visible=True),
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="7d", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(step="all"),
                ]
            ),
        ),
    ),
    yaxis=dict(title="Trade PnL"),
)

if show_trades and show_cum:
    fig.update_layout(
        yaxis2=dict(
            title="Cumulative PnL",
            overlaying="y",
            side="right",
            showgrid=False,
        )
    )

# Auto-annotations: biggest win/loss by absolute PnL
annos = []
try:
    if not trades_df.empty:
        abs_order = trades_df.copy()
        abs_order["abs"] = abs_order["pnl"].abs()
        abs_order = abs_order.sort_values("abs", ascending=False)
        for i, (_, row) in enumerate(abs_order.head(2).iterrows()):
            ts = int(row.name)
            p = float(row["pnl"]) if row.get("pnl") is not None else 0.0
            dt = pd.to_datetime(ts, unit="s")
            label = "Biggest Win" if p >= 0 else "Biggest Loss"
            annos.append(
                dict(
                    x=dt,
                    y=p,
                    xref="x",
                    yref="y",
                    text=f"{label}: {p:.2f}",
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-40 if p >= 0 else 40,
                    bgcolor="rgba(0,0,0,0.05)",
                )
            )
except Exception:
    pass

if annos:
    fig.update_layout(annotations=annos)

config = {
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtonsToAdd": [
        "drawline",
        "drawopenpath",
        "eraseshape",
    ],
}

st.plotly_chart(fig, use_container_width=True, config=config)

# ---- Additional charts ----
st.subheader("Close Reasons by LLM Model")
try:
    dist_df = compute_close_reason_distribution_by_model(df)
    if dist_df.empty:
        st.caption("No close reason data available yet.")
    else:
        import plotly.express as px  # local import to avoid global dependency at top

        fig2 = px.bar(
            dist_df,
            x="llm_model",
            y="count",
            color="close_reason",
            barmode="stack",
            title=None,
        )
        fig2.update_layout(height=380, template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True, config={"displaylogo": False})
except Exception:
    st.caption("Close reason chart unavailable.")

st.subheader("Window Size vs PnL by LLM Model")
try:
    current_window = load_window_seconds()
    corr_df = compute_window_pnl_correlation(df, current_window)
    if corr_df.empty:
        st.caption("No closed trades to compute correlation.")
    else:
        import plotly.express as px

        # Sanitize dtypes for robust plotting and correlation
        plot_df = corr_df.copy()
        try:
            plot_df["window_seconds"] = pd.to_numeric(
                plot_df["window_seconds"], errors="coerce"
            )
            plot_df["pnl"] = pd.to_numeric(plot_df["pnl"], errors="coerce")
            plot_df = plot_df.dropna(subset=["window_seconds", "pnl"]).copy()
        except Exception:
            pass

        # Use OLS trendline only if statsmodels is available
        try:
            import statsmodels.api as _sm  # type: ignore  # noqa: F401

            _trend = "ols"
        except Exception:
            _trend = None

        col_a, col_b = st.columns([3, 2])
        with col_a:
            fig3 = px.scatter(
                plot_df,
                x="window_seconds",
                y="pnl",
                color="llm_model",
                trendline=_trend,
                title=None,
            )
            fig3.update_layout(
                height=420,
                template="plotly_white",
                xaxis_title="Window (s)",
                yaxis_title="Trade PnL",
            )
            st.plotly_chart(
                fig3, use_container_width=True, config={"displaylogo": False}
            )
        with col_b:
            summ = summarize_window_pnl_correlation(plot_df)
            if summ.empty:
                st.caption("Not enough data for summary.")
            else:
                # Display compact table with rounded values
                df_show = summ.copy()
                try:
                    df_show["pearson_r"] = df_show["pearson_r"].round(3)
                    df_show["slope"] = df_show["slope"].round(5)
                    df_show["intercept"] = df_show["intercept"].round(2)
                except Exception:
                    pass
                st.dataframe(
                    df_show,
                    use_container_width=True,
                    hide_index=True,
                )
except Exception:
    st.caption("Correlation chart unavailable.")

# Confidence vs PnL scatter with model toggle
st.subheader("Confidence vs PnL")
try:
    conf_df = compute_confidence_vs_pnl(df)
    if conf_df.empty:
        st.caption("No confidence-PnL data available.")
    else:
        import plotly.express as px

        # Collect model options
        models = (
            conf_df["llm_model"].dropna().astype(str).unique().tolist()
            if "llm_model" in conf_df.columns
            else []
        )
        models = sorted(models)
        sel = st.selectbox(
            "LLM Model",
            options=["All"] + models,
            index=0,
            key="conf_pnl_model_toggle",
        )
        plot_df = conf_df if sel == "All" else conf_df[conf_df["llm_model"] == sel]

        # Use OLS trendline only if statsmodels is available
        try:
            import statsmodels.api as _sm  # type: ignore  # noqa: F401

            _trend = "ols"
        except Exception:
            _trend = None

        if sel == "All":
            fig4 = px.scatter(
                plot_df,
                x="confidence",
                y="pnl",
                color="llm_model",
                trendline=_trend,
                title=None,
            )
        else:
            fig4 = px.scatter(
                plot_df,
                x="confidence",
                y="pnl",
                trendline=_trend,
                title=None,
            )

        fig4.update_layout(
            height=420,
            template="plotly_white",
            xaxis_title="Confidence (0..1)",
            yaxis_title="Trade PnL",
        )
        st.plotly_chart(fig4, use_container_width=True, config={"displaylogo": False})
except Exception:
    st.caption("Confidence chart unavailable.")

# Optional user annotation controls
with st.expander("Add custom annotation"):
    try:
        # Pick a trade ID to annotate (if available)
        trade_ids = (
            trades_df["id"].dropna().astype(int).unique().tolist()
            if not trades_df.empty
            else []
        )
        selected_trade = None
        if trade_ids:
            selected_trade = st.selectbox("Trade ID", options=trade_ids, index=0)
            text = st.text_input("Annotation text", value="Note")
            if st.button("Annotate") and selected_trade is not None:
                row = trades_df[trades_df["id"] == selected_trade].head(1)
                if not row.empty:
                    ts = int(row.index[0])
                    p = float(row["pnl"].iloc[0] or 0.0)
                    dt = pd.to_datetime(ts, unit="s")
                    fig.add_annotation(
                        x=dt, y=p, text=text, showarrow=True, arrowhead=1
                    )
                    st.plotly_chart(fig, use_container_width=True, config=config)
        else:
            st.caption("No trades to annotate yet.")
    except Exception:
        st.caption("Annotation controls unavailable.")

# Footer: last updated time
_now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
st.caption(f"Last updated: {_now}")
