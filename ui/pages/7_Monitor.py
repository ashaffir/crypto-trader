import os as _os
import sys as _sys

import streamlit as st

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from ui.lib.common import render_status_badge
from ui.lib.control_utils import read_status, RCM
from ui.lib.monitor_utils import summarize_status, ms_to_human


st.set_page_config(page_title="Monitor", layout="wide")
render_status_badge(st)
st.subheader("Monitor · Operational Status")

try:
    from streamlit_autorefresh import st_autorefresh as _st_autorefresh  # type: ignore
except Exception:  # pragma: no cover
    _st_autorefresh = None

# Lightweight auto-refresh honoring UI config if present
try:
    overrides = RCM.read() or {}
    rs = int((overrides.get("ui") or {}).get("auto_refresh_seconds", 5))
except Exception:
    rs = 5
if _st_autorefresh is not None and rs > 0:
    _st_autorefresh(interval=rs * 1000, key="monitor_auto_refresh")
    st.caption(f"Auto-refresh every {rs}s")


def _badge(text: str, ok: bool) -> str:
    color = "#16a34a" if ok else "#dc2626"
    bg = "rgba(22,163,74,0.12)" if ok else "rgba(220,38,38,0.12)"
    return (
        f"<span style='display:inline-flex;align-items:center;gap:6px;padding:4px 10px;"
        f"border-radius:999px;background:{bg};color:{color};font-weight:600;font-size:12px;'>"
        f"<span style='width:8px;height:8px;border-radius:999px;background:{color};'></span>{text}</span>"
    )


# Read latest status and build summary
raw = read_status() or {}
summary = summarize_status(raw)

# Read runtime config for authoritative market/mode (aligns with status badge)
try:
    _overrides = RCM.read() or {}
except Exception:
    _overrides = {}
runtime_market = str((_overrides.get("market") or "")).lower() or None

# Top row cards
col1, col2, col3, col4 = st.columns([1.3, 1, 1, 1])
with col1:
    st.markdown("**Bot**")
    st.markdown(
        _badge(
            "RUNNING" if summary.get("is_running") else "STOPPED",
            bool(summary.get("is_running")),
        ),
        unsafe_allow_html=True,
    )
with col2:
    st.markdown("**Heartbeat**")
    hb_age = summary.get("heartbeat_age_ms")
    st.write(ms_to_human(hb_age))
with col3:
    st.markdown("**Queue Size**")
    qs = summary.get("queue_size")
    st.write("n/a" if qs is None else str(qs))
with col4:
    st.markdown("**Market**")
    st.write((runtime_market or "spot").upper())

# Symbols and Streams section
col_a, col_b = st.columns([1.2, 1.8])
with col_a:
    st.markdown("### Symbols")
    syms = summary.get("symbols") or []
    st.write(
        f"{len(syms)} tracked"
        + (f": {', '.join(syms[:10])}{' …' if len(syms) > 10 else ''}" if syms else "")
    )

with col_b:
    st.markdown("### Streams")
    st.caption("Health is based on recent events vs. enabled streams.")
    streams = summary.get("streams") or {}
    per = streams.get("per_stream") or {}
    if not per:
        st.info("No stream info available.")
    else:
        import pandas as pd
        # Enrich recency using event_count deltas across refreshes
        state_key = "_monitor_prev_counts"
        prev_counts = st.session_state.get(state_key, {})
        rows = []
        enabled_count = 0
        recent_enabled = 0
        computed_rows = {}
        for name, info in sorted(per.items()):
            enabled = bool(info.get("enabled"))
            count = int(info.get("event_count") or 0)
            # Delta-based recent if no timestamp
            delta_recent = False
            try:
                prev = int(prev_counts.get(name)) if prev_counts.get(name) is not None else None
            except Exception:
                prev = None
            # If there are no timestamps, consider recent when count increases
            age = info.get("last_event_age_ms")
            if age is None and prev is not None and count > prev:
                delta_recent = True
            recent = bool(info.get("recent")) or delta_recent
            row = {
                "Stream": name,
                "Enabled": enabled,
                "Recent": recent,
                "Event Count": count,
            }
            computed_rows[name] = row
        # Save counts for next refresh
        st.session_state[state_key] = {k: int(v.get("event_count") or 0) for k, v in per.items()}

        # Consolidate depth variants into a single "depth" row
        depth_aliases = ["depth", "depth_100ms", "depth5_100ms", "depth10_100ms", "depth20_100ms"]
        if any(k in computed_rows for k in depth_aliases):
            # Enabled if any alias is enabled
            depth_enabled = any(computed_rows.get(k, {}).get("Enabled", False) for k in depth_aliases)
            # Prefer normalized depth for recent/count
            base = computed_rows.get("depth") or {}
            depth_recent = bool(base.get("Recent")) if base else False
            depth_count = int(base.get("Event Count") or 0) if base else 0
            rows.append({
                "Stream": "depth",
                "Enabled": depth_enabled,
                "Recent": depth_recent,
                "Event Count": depth_count,
            })
            # Remove aliases from computed rows to avoid duplicates
            for k in depth_aliases:
                computed_rows.pop(k, None)

        # Append remaining non-depth rows
        for name, row in sorted(computed_rows.items()):
            rows.append(row)

        # Determine overall from enriched recency
        for r in rows:
            if r.get("Enabled"):
                enabled_count += 1
                if r.get("Recent"):
                    recent_enabled += 1
        if enabled_count == 0:
            overall = "ok"
        elif recent_enabled == enabled_count:
            overall = "ok"
        elif recent_enabled > 0:
            overall = "degraded"
        else:
            overall = "down"

        badge = _badge(
            {
                "ok": "HEALTHY",
                "degraded": "DEGRADED",
                "down": "NO ACTIVITY",
            }.get(overall, "UNKNOWN"),
            overall == "ok",
        )
        st.markdown(badge, unsafe_allow_html=True)

        # Render table
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

# Raw status expander for debugging
with st.expander("Raw status (debug)"):
    import json as _json

    st.code(_json.dumps(raw, indent=2), language="json")


