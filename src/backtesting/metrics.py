from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


@dataclass
class PerformanceReport:
    num_trades: int
    win_rate: float
    mean_ret_bps: float
    sharpe: float
    pnl_bps: float
    score: float


def compute_trade_metrics(returns_bps: Iterable[float]) -> PerformanceReport:
    arr = np.array(list(returns_bps), dtype=float)
    n = int(arr.size)
    if n == 0:
        return PerformanceReport(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    win_rate = float(np.mean(arr > 0.0))
    mean_ret = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    sharpe = float(mean_ret / std) if std > 1e-12 else 0.0
    pnl = float(np.sum(arr))

    # Composite score balances profitability, win-rate and risk-adjusted return
    score = (
        0.5 * np.tanh(pnl / 1000.0)
        + 0.3 * (win_rate - 0.5) * 2
        + 0.2 * np.tanh(sharpe / 3.0)
    )
    return PerformanceReport(n, win_rate, mean_ret, sharpe, pnl, float(score))


def aggregate_outcomes(df: pd.DataFrame) -> PerformanceReport:
    if df is None or df.empty or "ret_bps" not in df.columns:
        return PerformanceReport(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return compute_trade_metrics(df["ret_bps"].to_numpy())


__all__ = ["PerformanceReport", "compute_trade_metrics", "aggregate_outcomes"]
