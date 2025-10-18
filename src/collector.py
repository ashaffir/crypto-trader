from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from loguru import logger
import websockets

BINANCE_WS_BASE = "wss://stream.binance.com:9443/stream?streams="


@dataclass
class OrderBookTop:
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None


@dataclass
class SymbolState:
    orderbook_bids: Dict[float, float] = field(default_factory=dict)
    orderbook_asks: Dict[float, float] = field(default_factory=dict)
    top: OrderBookTop = field(default_factory=OrderBookTop)
    last_px: Optional[float] = None
    last_qty: Optional[float] = None


class SpotCollector:
    def __init__(
        self, symbols: List[str], streams: Dict[str, bool], out_queue: asyncio.Queue
    ):
        self.symbols = symbols
        self.streams = streams
        self.out_queue = out_queue
        self.state: Dict[str, SymbolState] = {s: SymbolState() for s in symbols}
        self._stop = asyncio.Event()

    def build_streams(self) -> List[str]:
        names: List[str] = []
        for s in self.symbols:
            lower = s.lower()
            if self.streams.get("aggTrade", True):
                names.append(f"{lower}@aggTrade")
            if self.streams.get("depth_100ms", True):
                names.append(f"{lower}@depth@100ms")
            if self.streams.get("kline_1s", True):
                names.append(f"{lower}@kline_1s")
        return names

    def _url(self) -> str:
        return BINANCE_WS_BASE + "/".join(self.build_streams())

    async def stop(self) -> None:
        self._stop.set()

    @staticmethod
    def _safe_float(x) -> Optional[float]:
        try:
            return float(x)
        except Exception:
            return None

    def _apply_depth_update(self, symbol: str, data: Dict) -> None:
        st = self.state[symbol]
        # Bids
        for price_str, qty_str in data.get("b", []):
            price = float(price_str)
            qty = float(qty_str)
            if qty == 0.0:
                st.orderbook_bids.pop(price, None)
            else:
                st.orderbook_bids[price] = qty
        # Asks
        for price_str, qty_str in data.get("a", []):
            price = float(price_str)
            qty = float(qty_str)
            if qty == 0.0:
                st.orderbook_asks.pop(price, None)
            else:
                st.orderbook_asks[price] = qty
        # Update top of book
        st.top.best_bid = max(st.orderbook_bids.keys()) if st.orderbook_bids else None
        st.top.best_ask = min(st.orderbook_asks.keys()) if st.orderbook_asks else None

    async def run(self) -> None:
        backoff = 1
        while not self._stop.is_set():
            url = self._url()
            try:
                logger.info(f"Connecting to {url}")
                async with websockets.connect(
                    url, ping_interval=20, ping_timeout=20
                ) as ws:
                    backoff = 1
                    async for raw in ws:
                        if self._stop.is_set():
                            break
                        try:
                            msg = json.loads(raw)
                        except Exception:
                            logger.exception("Failed to parse message")
                            continue
                        stream = msg.get("stream", "")
                        data = msg.get("data", {})
                        symbol = (data.get("s") or data.get("ps") or "").upper()
                        ts_ms = data.get("T") or data.get("E")
                        if not symbol:
                            # kline is nested under k
                            k = data.get("k")
                            if k:
                                symbol = k.get("s", "").upper()
                                ts_ms = k.get("T") or k.get("t") or ts_ms
                        if not symbol:
                            continue

                        # Route by event type
                        if stream.endswith("aggTrade") or data.get("e") == "aggTrade":
                            price = self._safe_float(data.get("p"))
                            qty = self._safe_float(data.get("q"))
                            st = self.state[symbol]
                            st.last_px = price
                            st.last_qty = qty
                            await self.out_queue.put(
                                {
                                    "kind": "aggTrade",
                                    "symbol": symbol,
                                    "ts_ms": ts_ms,
                                    "price": price,
                                    "qty": qty,
                                    "is_buyer_maker": bool(data.get("m", False)),
                                }
                            )
                        elif "depth@" in stream or data.get("e") == "depthUpdate":
                            self._apply_depth_update(symbol, data)
                            top = self.state[symbol].top
                            await self.out_queue.put(
                                {
                                    "kind": "depth",
                                    "symbol": symbol,
                                    "ts_ms": ts_ms,
                                    "best_bid": top.best_bid,
                                    "best_ask": top.best_ask,
                                }
                            )
                        elif stream.endswith("kline_1s") or data.get("e") == "kline":
                            k = data.get("k", {})
                            await self.out_queue.put(
                                {
                                    "kind": "kline",
                                    "symbol": symbol,
                                    "ts_ms": ts_ms,
                                    "open": self._safe_float(k.get("o")),
                                    "high": self._safe_float(k.get("h")),
                                    "low": self._safe_float(k.get("l")),
                                    "close": self._safe_float(k.get("c")),
                                    "volume": self._safe_float(k.get("v")),
                                    "closed": bool(k.get("x", False)),
                                }
                            )
                        else:
                            # Unknown event type, ignore
                            continue
            except Exception as e:
                logger.warning(f"WebSocket error: {e}. Reconnecting in {backoff}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)
