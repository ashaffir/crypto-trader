import pandas as pd

from ui.lib.statistics_utils import compute_pnl_series


def test_compute_pnl_series_empty():
    df = pd.DataFrame()
    trades, cum = compute_pnl_series(df)
    assert trades.empty
    assert cum.empty


def test_compute_pnl_series_basic():
    data = [
        {"id": 1, "symbol": "BTCUSDT", "pnl": 10.0, "closed_ts_ms": 1000},
        {"id": 2, "symbol": "BTCUSDT", "pnl": -5.0, "closed_ts_ms": 2000},
        {"id": 3, "symbol": "ETHUSDT", "pnl": 2.5, "closed_ts_ms": 2000},
    ]
    df = pd.DataFrame(data)
    trades, cum = compute_pnl_series(df)
    # trades index is seconds
    assert (trades.index.values == pd.Index([1, 2, 2], dtype="int64").values).all()
    # cumulative should aggregate by ts: at t=1 -> 10, t=2 -> 10 + (-5 + 2.5) = 7.5
    assert cum.loc[1, "cum_pnl"] == 10.0
    assert abs(cum.loc[2, "cum_pnl"] - 7.5) < 1e-9
