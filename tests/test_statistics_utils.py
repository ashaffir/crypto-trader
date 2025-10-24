import pandas as pd

from ui.lib.statistics_utils import compute_pnl_series, compute_pnl_series_by_model
from ui.lib.statistics_utils import (
    compute_window_pnl_correlation,
    summarize_window_pnl_correlation,
)


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


def test_compute_pnl_series_by_model_basic():
    data = [
        {
            "id": 1,
            "symbol": "BTCUSDT",
            "pnl": 10.0,
            "closed_ts_ms": 1000,
            "llm_model": "gpt-a",
        },
        {
            "id": 2,
            "symbol": "BTCUSDT",
            "pnl": -5.0,
            "closed_ts_ms": 2000,
            "llm_model": "gpt-a",
        },
        {
            "id": 3,
            "symbol": "ETHUSDT",
            "pnl": 2.5,
            "closed_ts_ms": 2000,
            "llm_model": "gpt-b",
        },
    ]
    df = pd.DataFrame(data)
    trades, cum = compute_pnl_series_by_model(df)
    # trades index seconds and contains model column
    assert (trades.index.values == pd.Index([1, 2, 2], dtype="int64").values).all()
    assert "llm_model" in trades.columns
    # cumulative per model
    gpta = cum.reset_index().query('llm_model == "gpt-a"').sort_values("ts")
    gptb = cum.reset_index().query('llm_model == "gpt-b"').sort_values("ts")
    assert list(gpta["ts"]) == [1, 2]
    assert abs(float(gpta.iloc[0]["cum_pnl"]) - 10.0) < 1e-9
    assert abs(float(gpta.iloc[1]["cum_pnl"]) - 5.0) < 1e-9  # 10 + (-5)
    assert list(gptb["ts"]) == [2]
    assert abs(float(gptb.iloc[0]["cum_pnl"]) - 2.5) < 1e-9


def test_compute_pnl_series_by_model_unknown_label():
    data = [
        {
            "id": 1,
            "symbol": "BTCUSDT",
            "pnl": 3.0,
            "closed_ts_ms": 3000,
            "llm_model": None,
        },
    ]
    df = pd.DataFrame(data)
    trades, cum = compute_pnl_series_by_model(df)
    assert "llm_model" in trades.columns
    # Unknown should be present
    labs = set(trades["llm_model"].unique().tolist())
    assert "Unknown" in labs


def test_summarize_window_pnl_correlation():
    # Build a small corr_df
    import pandas as pd

    corr_df = pd.DataFrame(
        {
            "llm_model": ["m1", "m1", "m2", "m2"],
            "window_seconds": [10, 20, 10, 30],
            "pnl": [1.0, 3.0, -1.0, 2.0],
        }
    )
    summ = summarize_window_pnl_correlation(corr_df)
    assert not summ.empty
    assert set(summ["llm_model"]) == {"m1", "m2"}
    # n must be >= 2 for each
    assert (summ["n"] >= 2).all()
    # r in [-1, 1]
    assert (summ["pearson_r"].abs() <= 1.0).all()
