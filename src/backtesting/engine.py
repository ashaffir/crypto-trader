from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd

from src.backtesting.loader import load_signals, load_snapshots
from src.backtesting.metrics import PerformanceReport, aggregate_outcomes


@dataclass
class LogicalTestResult:
    num_signals: int
    fields_ok: bool
    time_monotonic: bool
    sample: pd.DataFrame
    timeframe_start_ms: Optional[int] = None
    timeframe_end_ms: Optional[int] = None


def _validate_signal_schema(df: pd.DataFrame) -> bool:
    required = {"ts_ms", "symbol", "side", "expected_bps", "confidence", "rule_id"}
    return required.issubset(set(df.columns))


def _validate_signal_timing(df: pd.DataFrame) -> bool:
    if df.empty or "ts_ms" not in df.columns:
        return True
    ts = df["ts_ms"].to_numpy()
    return bool((ts[1:] >= ts[:-1]).all())


def logical_test(
    symbol: str,
    *,
    base_dir: Optional[str] = None,
    dates: Optional[Iterable[str]] = None,
    max_files: int = 10,
) -> LogicalTestResult:
    sig = load_signals(symbol, base_dir=base_dir, dates=dates, max_files=max_files)
    if not sig.empty and "ts_ms" in sig.columns:
        sig = sig.sort_values("ts_ms").reset_index(drop=True)
    fields_ok = _validate_signal_schema(sig)
    monotonic = _validate_signal_timing(sig)
    sample = sig.head(20).copy() if not sig.empty else pd.DataFrame()
    timeframe_start = (
        int(sig["ts_ms"].min()) if (not sig.empty and "ts_ms" in sig.columns) else None
    )
    timeframe_end = (
        int(sig["ts_ms"].max()) if (not sig.empty and "ts_ms" in sig.columns) else None
    )
    return LogicalTestResult(
        num_signals=int(len(sig)),
        fields_ok=fields_ok,
        time_monotonic=monotonic,
        sample=sample,
        timeframe_start_ms=timeframe_start,
        timeframe_end_ms=timeframe_end,
    )


@dataclass
class QualityTestResult:
    outcomes: pd.DataFrame
    report: PerformanceReport


def _simulate_outcomes_from_mid(
    signals: pd.DataFrame,
    snapshots: pd.DataFrame,
    horizon_ms: int,
) -> pd.DataFrame:
    if signals.empty or snapshots.empty:
        return pd.DataFrame(
            columns=[
                "signal_id",
                "symbol",
                "resolved_ts_ms",
                "ret_bps",
                "hit",
                "max_adverse_bps",
                "max_favorable_bps",
            ]
        )
    # Ensure sorted
    signals = signals.sort_values("ts_ms").reset_index(drop=True)
    snapshots = snapshots.sort_values("ts_ms").reset_index(drop=True)

    # Build pointer traversal to avoid O(N*M)
    times = snapshots["ts_ms"].to_numpy()
    mids = snapshots["mid"].to_numpy()
    out_rows: List[Dict] = []
    j = 0
    for _, s in signals.iterrows():
        ts = int(s.get("ts_ms"))
        side = s.get("side")
        symbol = s.get("symbol")
        # advance j until time >= ts
        while j < len(times) and (times[j] is None or times[j] < ts):
            j += 1
        if j >= len(times):
            break
        start_mid = mids[j]
        if start_mid is None or not pd.notna(start_mid):
            continue
        max_fav = None
        max_adv = None
        end_mid = None
        k = j
        while k < len(times):
            t = times[k]
            m = mids[k]
            if t is None or m is None or not pd.notna(m):
                k += 1
                continue
            if end_mid is None:
                end_mid = m
            dt = int(t) - ts
            ret = (float(m) - float(start_mid)) / float(start_mid) * 1e4
            if side == "short":
                ret = -ret
            max_fav = ret if (max_fav is None or ret > max_fav) else max_fav
            max_adv = ret if (max_adv is None or ret < max_adv) else max_adv
            end_mid = m
            if dt >= horizon_ms:
                break
            k += 1
        if end_mid is None:
            continue
        ret_bps = (float(end_mid) - float(start_mid)) / float(start_mid) * 1e4
        if side == "short":
            ret_bps = -ret_bps
        out_rows.append(
            {
                "signal_id": s.get("signal_id"),
                "symbol": symbol,
                "resolved_ts_ms": ts + horizon_ms,
                "ret_bps": float(ret_bps),
                "hit": int(ret_bps > 0),
                "max_adverse_bps": float(max_adv or 0.0),
                "max_favorable_bps": float(max_fav or 0.0),
            }
        )
    return pd.DataFrame(out_rows)


def quality_test(
    symbol: str,
    *,
    base_dir: Optional[str] = None,
    dates: Optional[Iterable[str]] = None,
    horizon_s: int = 30,
    max_files: Optional[int] = None,
) -> QualityTestResult:
    snaps = load_snapshots(symbol, base_dir=base_dir, dates=dates, max_files=max_files)
    sigs = load_signals(symbol, base_dir=base_dir, dates=dates, max_files=max_files)
    outcomes = _simulate_outcomes_from_mid(sigs, snaps, horizon_s * 1000)
    report = aggregate_outcomes(outcomes)
    return QualityTestResult(outcomes=outcomes, report=report)


__all__ = [
    "LogicalTestResult",
    "QualityTestResult",
    "logical_test",
    "quality_test",
]
