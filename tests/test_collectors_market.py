from src.collector import SpotCollector, FuturesCollector


def test_spot_collector_url(monkeypatch):
    q = __import__("asyncio").Queue()
    c = SpotCollector(["BTCUSDT"], {"aggTrade": True}, q)
    url = c._url()
    assert url.startswith("wss://stream.binance.com:9443/stream?streams=")
    assert "btcusdt@aggTrade" in url


def test_futures_collector_url(monkeypatch):
    q = __import__("asyncio").Queue()
    c = FuturesCollector(["BTCUSDT"], {"aggTrade": True}, q)
    url = c._url()
    assert url.startswith("wss://fstream.binance.com/stream?streams=")
    assert "btcusdt@aggTrade" in url
