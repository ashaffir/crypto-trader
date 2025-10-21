import os
import sys as _sys
import os as _os
import pandas as pd
import streamlit as st
from datetime import datetime, timezone, timedelta

# Ensure project root (parent of `ui/`) is on sys.path so `ui.*` and `src.*` are importable
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from ui.lib.common import (
    LOGBOOK_DIR,
    CONTROL_DIR,
    PAGE_HEADER_TITLE,
    render_status_badge,
)
from ui.lib.retention_utils import (
    iter_date_partitions,
    prune_by_days,
    prune_to_size_cap,
    parse_size_cap,
    humanize_bytes,
    dir_size_bytes,
)
from ui.lib.logbook_utils import read_latest_file

try:
    from ui.lib.retention_state import load_retention_settings, save_retention_settings
except Exception:
    # Fallback inline persistence if module is unavailable
    import json as _json

    def _runtime_file(base_dir: str | None = None) -> str:
        b = base_dir or CONTROL_DIR
        return os.path.join(b, "runtime_config.json")

    _DEFAULTS = {
        "mode": "days",
        "max_days": 7,
        "size_cap": "50GB",
        "dry_run_default": True,
    }

    def load_retention_settings(base_dir: str | None = None) -> dict:
        path = _runtime_file(base_dir)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = _json.load(f)
        except Exception:
            data = {}
        raw = data.get("retention") if isinstance(data, dict) else None
        out = dict(_DEFAULTS)
        if isinstance(raw, dict):
            for k in out.keys():
                if k in raw:
                    out[k] = raw[k]
        return out

    def save_retention_settings(settings: dict, base_dir: str | None = None) -> bool:
        path = _runtime_file(base_dir)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            current = {}
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    current = _json.load(f) or {}
        except Exception:
            current = {}
        keep = {
            k: settings[k]
            for k in ("mode", "max_days", "size_cap", "dry_run_default")
            if k in settings
        }
        merged_ret = load_retention_settings(base_dir)
        merged_ret.update(keep)
        current["retention"] = merged_ret
        tmp = f"{path}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                _json.dump(current, f)
            os.replace(tmp, path)
            return True
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            return False


st.set_page_config(page_title="Data Tools", layout="wide")
st.title(PAGE_HEADER_TITLE)
render_status_badge(st)
st.subheader("Data Tools")
st.caption(f"LOGBOOK_DIR: {LOGBOOK_DIR}")

ins_tab, ret_tab = st.tabs(["Inspect", "Retention"])

with ins_tab:
    # Inspect latest file for selected table/symbol
    tables = (
        sorted(
            [
                d
                for d in os.listdir(LOGBOOK_DIR)
                if os.path.isdir(os.path.join(LOGBOOK_DIR, d))
            ]
        )
        if os.path.isdir(LOGBOOK_DIR)
        else []
    )
    if not tables:
        st.info("No tables found.")
    else:
        t = st.selectbox("Table", tables, index=0, key="data_tools_table")
        syms = []
        tdir = os.path.join(LOGBOOK_DIR, t)
        for name in os.listdir(tdir):
            if name.startswith("symbol=") and os.path.isdir(os.path.join(tdir, name)):
                syms.append(name.split("=", 1)[1])
        syms = sorted(syms)
        if not syms:
            st.info("No symbols found for selected table.")
        else:
            s = st.selectbox("Symbol", syms, index=0, key=f"data_tools_symbol_{t}")
            df = read_latest_file(t, s)
            if df.empty:
                st.info("Empty dataset or failed to read latest parquet")
            else:
                if "ts_ms" in df.columns:
                    try:
                        df["ts"] = (
                            pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
                            .dt.tz_localize(None)
                            .dt.strftime("%Y-%m-%d %H:%M:%S.%f")
                            .str[:-3]
                        )
                    except Exception:
                        df["ts"] = ""
                    df = df.sort_values("ts_ms")
                if len(df) > 5000:
                    st.caption(f"Showing last 5000 of {len(df)} rows")
                    df = df.tail(5000)
                cols = ["ts"] + [c for c in df.columns if c not in ("ts", "ts_ms")]
                st.dataframe(df[cols])

with ret_tab:
    # Load persisted UI defaults
    persisted = load_retention_settings()
    total_size = dir_size_bytes(LOGBOOK_DIR) if os.path.isdir(LOGBOOK_DIR) else 0
    parts = (
        list(iter_date_partitions(LOGBOOK_DIR)) if os.path.isdir(LOGBOOK_DIR) else []
    )
    cols = st.columns(3)
    cols[0].metric("Total size", humanize_bytes(total_size))
    cols[1].metric("Partitions", f"{len(parts)}")
    cols[2].metric("Tables", f"{len({p[0] for p in parts})}")

    st.markdown("Retention mode")
    mode_map = {
        "minutes": "Keep last N minutes",
        "days": "Keep last N days",
        "size": "Cap total size",
    }
    mode_default = mode_map.get(str(persisted.get("mode")), "Keep last N days")
    mode = st.radio(
        "Mode",
        ["Keep last N minutes", "Keep last N days", "Cap total size"],
        horizontal=True,
        index=["Keep last N minutes", "Keep last N days", "Cap total size"].index(
            mode_default
        ),
    )
    dry = st.checkbox(
        "Dry-run (preview only)", value=bool(persisted.get("dry_run_default", True))
    )

    preview = []
    if mode == "Keep last N minutes":
        mins = st.number_input(
            "Max minutes to keep",
            min_value=1,
            max_value=60 * 24 * 30,
            value=int(persisted.get("max_minutes", 60)),
        )
        # Approx preview by days granularity: show partitions older than cutoff minute
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=int(mins))
        for t, s, d, pdir in parts:
            try:
                dt = datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except Exception:
                continue
            if dt < cutoff:
                preview.append((t, s, d, pdir))
        if st.button("Apply", key="retention_apply_minutes"):
            try:
                from src.data_retention import prune_by_minutes

                removed = prune_by_minutes(LOGBOOK_DIR, int(mins), dry)
            except Exception:
                removed = []
            if dry:
                st.info(f"Would remove {len(preview)} partitions (dry-run)")
            else:
                st.success(f"Removed {len(removed)} partitions")
            save_retention_settings(
                {
                    "mode": "minutes",
                    "max_minutes": int(mins),
                    "dry_run_default": bool(dry),
                }
            )
    elif mode == "Keep last N days":
        max_days = st.number_input(
            "Max days to keep",
            min_value=1,
            max_value=365,
            value=int(persisted.get("max_days", 7)),
        )
        cutoff = datetime.now(timezone.utc) - timedelta(days=int(max_days))
        for t, s, d, pdir in parts:
            try:
                dt = datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except Exception:
                continue
            if dt < cutoff:
                preview.append((t, s, d, pdir))
        if st.button("Apply", key="retention_apply_days"):
            removed = prune_by_days(LOGBOOK_DIR, int(max_days), dry)
            if dry:
                st.info(f"Would remove {len(preview)} partitions (dry-run)")
            else:
                st.success(f"Removed {len(removed)} partitions")
            # Persist settings
            save_retention_settings(
                {
                    "mode": "days",
                    "max_days": int(max_days),
                    "dry_run_default": bool(dry),
                }
            )
    else:
        cap_str = st.text_input(
            "Size cap (e.g., 50GB, 500MB)", value=str(persisted.get("size_cap", "50GB"))
        )
        if cap_str:
            try:
                cap = parse_size_cap(cap_str)
            except Exception:
                st.error("Invalid size cap")
                cap = None
        else:
            cap = None
        if cap is not None:
            part_list = sorted(parts, key=lambda x: x[2])
            sizes = {}
            total = 0
            for _t, _s, _d, ddir in part_list:
                sz = dir_size_bytes(ddir)
                sizes[ddir] = sz
                total += sz
            to_remove = []
            for entry in part_list:
                if total <= cap:
                    break
                _t, _s, _d, ddir = entry
                total -= sizes[ddir]
                to_remove.append(entry)
            preview = to_remove
        if st.button("Apply", key="retention_apply_size"):
            if cap is None:
                st.error("Provide a valid size cap")
            else:
                removed = prune_to_size_cap(LOGBOOK_DIR, cap, dry)
                if dry:
                    st.info(f"Would remove {len(preview)} partitions (dry-run)")
                else:
                    st.success(f"Removed {len(removed)} partitions")
                # Persist settings
                save_retention_settings(
                    {
                        "mode": "size",
                        "size_cap": str(cap_str),
                        "dry_run_default": bool(dry),
                    }
                )

    if preview:
        st.markdown("Preview of partitions to remove")
        prev_df = pd.DataFrame(preview, columns=["table", "symbol", "date", "path"])
        st.dataframe(prev_df)
