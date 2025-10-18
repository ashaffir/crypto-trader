from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import List, Dict, Optional

import pyarrow.parquet as pq


@dataclass
class Outcome:
    signal_id: str
    symbol: str
    resolved_ts_ms: int
    ret_bps: float
    hit: int
    max_adverse_bps: float
    max_favorable_bps: float


class Evaluator:
    def __init__(self, logbook_dir: str, horizon_s: int = 30) -> None:
        self.dir = logbook_dir
        self.horizon_ms = horizon_s * 1000

    def _read_latest_snapshots(
        self, symbol: str, date: Optional[str] = None
    ) -> Optional[pq.Table]:
        base = os.path.join(self.dir, "market_snapshot", f"symbol={symbol}")
        if date:
            base = os.path.join(base, f"date={date}")
            files = sorted(glob.glob(os.path.join(base, "*.parquet")))
        else:
            files = sorted(glob.glob(os.path.join(base, "date=*", "*.parquet")))
        if not files:
            return None
        return pq.read_table(files[-20:])  # tail a subset

    def _read_latest_signals(
        self, symbol: str, date: Optional[str] = None
    ) -> Optional[pq.Table]:
        base = os.path.join(self.dir, "signal_emitted", f"symbol={symbol}")
        if date:
            base = os.path.join(base, f"date={date}")
            files = sorted(glob.glob(os.path.join(base, "*.parquet")))
        else:
            files = sorted(glob.glob(os.path.join(base, "date=*", "*.parquet")))
        if not files:
            return None
        return pq.read_table(files[-20:])

    def _compute_outcomes(
        self, signals_tbl: pq.Table, snaps_tbl: pq.Table
    ) -> List[Dict]:
        if signals_tbl is None or snaps_tbl is None:
            return []
        sig = signals_tbl.to_pydict()
        snap = snaps_tbl.to_pydict()
        # Build time series
        times = snap.get("ts_ms", [])
        mids = snap.get("mid", [])
        series = list(zip(times, mids))
        outcomes: List[Dict] = []
        for i in range(len(sig.get("signal_id", []))):
            s_id = sig["signal_id"][i]
            symbol = sig["symbol"][i]
            ts = sig["ts_ms"][i]
            side = sig["side"][i]
            # find future window
            start_mid = None
            max_fav = None
            max_adv = None
            end_mid = None
            for t, m in series:
                if t is None or m is None:
                    continue
                if t < ts:
                    continue
                if start_mid is None:
                    start_mid = m
                dt = t - ts
                ret = (m - start_mid) / start_mid * 1e4
                if side == "short":
                    ret = -ret
                max_fav = ret if (max_fav is None or ret > max_fav) else max_fav
                max_adv = ret if (max_adv is None or ret < max_adv) else max_adv
                end_mid = m
                if dt >= self.horizon_ms:
                    break
            if start_mid is None or end_mid is None:
                continue
            ret_bps = (end_mid - start_mid) / start_mid * 1e4
            if side == "short":
                ret_bps = -ret_bps
            outcomes.append(
                {
                    "signal_id": s_id,
                    "symbol": symbol,
                    "resolved_ts_ms": ts + self.horizon_ms,
                    "ret_bps": float(ret_bps),
                    "hit": int(ret_bps > 0),
                    "max_adverse_bps": float(max_adv or 0.0),
                    "max_favorable_bps": float(max_fav or 0.0),
                }
            )
        return outcomes

    async def run_periodic(self, interval_seconds: int = 5) -> None:
        # Lightweight periodic evaluator (reads tail and computes)
        import asyncio
        from src.logger import ParquetLogbook

        lb = ParquetLogbook(self.dir)
        while True:
            try:
                # For now, assume at least BTCUSDT
                for symbol in ["BTCUSDT"]:
                    sig_tbl = self._read_latest_signals(symbol)
                    snap_tbl = self._read_latest_snapshots(symbol)
                    if sig_tbl is None or snap_tbl is None:
                        continue
                    outcomes = self._compute_outcomes(sig_tbl, snap_tbl)
                    if outcomes:
                        lb.append_signal_outcome(outcomes)
            except Exception:
                # Best-effort evaluator
                pass
            await asyncio.sleep(interval_seconds)
