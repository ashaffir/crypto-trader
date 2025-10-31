import time

from src.collector import SpotCollector, FuturesCollector
from src.features import FeatureEngine


def test_spot_collector_stream_selection_trade_and_partial_depth():
    q = __import__("asyncio").Queue()
    c = SpotCollector(
        ["BTCUSDT"],
        {
            "aggTrade": True,
            "trade": True,
            "depth_100ms": False,
            "depth10_100ms": True,
            "kline_1s": False,
        },
        q,
    )
    url = c._url()
    assert "btcusdt@aggTrade" in url
    assert "btcusdt@trade" in url
    assert "btcusdt@depth10@100ms" in url


def test_futures_collector_includes_futures_streams():
    q = __import__("asyncio").Queue()
    c = FuturesCollector(
        ["BTCUSDT"],
        {
            "aggTrade": True,
            "fundingRate": True,
            "openInterest": True,
            "forceOrder": True,
        },
        q,
    )
    url = c._url()
    assert url.startswith("wss://fstream.binance.com/stream?streams=")
    assert "btcusdt@aggTrade" in url
    assert "btcusdt@fundingRate" in url
    assert "btcusdt@openInterest" in url
    assert "btcusdt@forceOrder" in url


def test_feature_engine_new_metrics_summarize_window():
    # Build engine
    eng = FeatureEngine(symbols=["BTCUSDT"], snapshot_window_s=60)
    now_ms = int(time.time() * 1000)

    # Feed a depth message with metrics
    eng.on_message(
        {
            "kind": "depth",
            "symbol": "BTCUSDT",
            "ts_ms": now_ms,
            "best_bid": 100.0,
            "best_ask": 100.1,
            "num_changes": 10,
            "num_cancels": 2,
            "bid_add": 5.0,
            "bid_remove": 1.0,
            "ask_add": 2.0,
            "ask_remove": 3.0,
            "size_best_bid": 4.0,
            "size_best_ask": 3.0,
        }
    )

    # Feed trades: one taker buy, one taker sell
    eng.on_message(
        {
            "kind": "aggTrade",
            "symbol": "BTCUSDT",
            "ts_ms": now_ms + 10,
            "price": 100.05,
            "qty": 2.0,
            "is_buyer_maker": False,  # taker is buyer
        }
    )
    eng.on_message(
        {
            "kind": "trade",
            "symbol": "BTCUSDT",
            "ts_ms": now_ms + 20,
            "price": 100.02,
            "qty": 1.0,
            "is_buyer_maker": True,  # taker is seller
        }
    )

    # Feed open interest updates
    eng.on_message(
        {
            "kind": "openInterest",
            "symbol": "BTCUSDT",
            "ts_ms": now_ms + 30,
            "open_interest": 1000.0,
        }
    )
    eng.on_message(
        {
            "kind": "openInterest",
            "symbol": "BTCUSDT",
            "ts_ms": now_ms + 1030,
            "open_interest": 1010.0,
        }
    )

    # Feed a liquidation event
    eng.on_message(
        {
            "kind": "forceOrder",
            "symbol": "BTCUSDT",
            "ts_ms": now_ms + 40,
            "price": 100.0,
            "qty": 5.0,
        }
    )

    summary = eng.summarize_window("BTCUSDT", window_s=30)
    assert summary["symbol"] == "BTCUSDT"
    # Check presence of new metrics
    assert "orderflow_pressure" in summary
    assert "depth_skew" in summary
    assert "cancel_intensity" in summary
    assert "trade_aggression" in summary
    assert "oi_delta" in summary
    assert "liquidation_burst" in summary

    # Sanity checks on values ranges
    if summary["orderflow_pressure"] is not None:
        assert -1.0 <= summary["orderflow_pressure"] <= 1.0
    if summary["depth_skew"] is not None:
        assert -1.0 <= summary["depth_skew"] <= 1.0
    if summary["cancel_intensity"] is not None:
        assert 0.0 <= summary["cancel_intensity"] <= 1.0
    if summary["trade_aggression"] is not None:
        assert 0.0 <= summary["trade_aggression"] <= 1.0

