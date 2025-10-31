from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from loguru import logger
import websockets

BINANCE_WS_BASE = "wss://stream.binance.com:9443/stream?streams="
# Binance USDⓈ-M Futures combined stream base
BINANCE_FUTURES_WS_BASE = "wss://fstream.binance.com/stream?streams="


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
            if self.streams.get("trade", False):
                names.append(f"{lower}@trade")
            if self.streams.get("depth_100ms", True):
                names.append(f"{lower}@depth@100ms")
            else:
                # Partial book depths (pick at most one)
                if self.streams.get("depth10_100ms", False):
                    names.append(f"{lower}@depth10@100ms")
                elif self.streams.get("depth5_100ms", False):
                    names.append(f"{lower}@depth5@100ms")
                elif self.streams.get("depth20_100ms", False):
                    names.append(f"{lower}@depth20@100ms")
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

    def _apply_depth_update(self, symbol: str, data: Dict) -> Dict[str, float | int | None]:
        st = self.state[symbol]
        # Track per-event deltas for feature calculations
        num_changes = 0
        num_cancels = 0
        bid_add_vol = 0.0
        bid_remove_vol = 0.0
        ask_add_vol = 0.0
        ask_remove_vol = 0.0

        # Bids
        for price_str, qty_str in data.get("b", []):
            price = float(price_str)
            new_qty = float(qty_str)
            prev_qty = st.orderbook_bids.get(price, 0.0)
            if new_qty == 0.0:
                if prev_qty > 0.0:
                    bid_remove_vol += prev_qty
                    num_cancels += 1
                st.orderbook_bids.pop(price, None)
            else:
                st.orderbook_bids[price] = new_qty
                delta = new_qty - prev_qty
                if delta > 0:
                    bid_add_vol += delta
                elif delta < 0:
                    bid_remove_vol += -delta
            num_changes += 1

        # Asks
        for price_str, qty_str in data.get("a", []):
            price = float(price_str)
            new_qty = float(qty_str)
            prev_qty = st.orderbook_asks.get(price, 0.0)
            if new_qty == 0.0:
                if prev_qty > 0.0:
                    ask_remove_vol += prev_qty
                    num_cancels += 1
                st.orderbook_asks.pop(price, None)
            else:
                st.orderbook_asks[price] = new_qty
                delta = new_qty - prev_qty
                if delta > 0:
                    ask_add_vol += delta
                elif delta < 0:
                    ask_remove_vol += -delta
            num_changes += 1

        # Update top of book and sizes
        st.top.best_bid = max(st.orderbook_bids.keys()) if st.orderbook_bids else None
        st.top.best_ask = min(st.orderbook_asks.keys()) if st.orderbook_asks else None
        best_bid_size = (
            st.orderbook_bids.get(st.top.best_bid) if st.top.best_bid is not None else None
        )
        best_ask_size = (
            st.orderbook_asks.get(st.top.best_ask) if st.top.best_ask is not None else None
        )

        return {
            "num_changes": num_changes,
            "num_cancels": num_cancels,
            "bid_add": bid_add_vol,
            "bid_remove": bid_remove_vol,
            "ask_add": ask_add_vol,
            "ask_remove": ask_remove_vol,
            "size_best_bid": best_bid_size,
            "size_best_ask": best_ask_size,
        }

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
                        elif stream.endswith("trade") or data.get("e") == "trade":
                            price = self._safe_float(data.get("p"))
                            qty = self._safe_float(data.get("q"))
                            st = self.state[symbol]
                            st.last_px = price
                            st.last_qty = qty
                            await self.out_queue.put(
                                {
                                    "kind": "trade",
                                    "symbol": symbol,
                                    "ts_ms": ts_ms,
                                    "price": price,
                                    "qty": qty,
                                    "is_buyer_maker": bool(data.get("m", False)),
                                }
                            )
                        elif "depth@" in stream or data.get("e") == "depthUpdate":
                            metrics = self._apply_depth_update(symbol, data)
                            top = self.state[symbol].top
                            await self.out_queue.put(
                                {
                                    "kind": "depth",
                                    "symbol": symbol,
                                    "ts_ms": ts_ms,
                                    "best_bid": top.best_bid,
                                    "best_ask": top.best_ask,
                                    **metrics,
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
                        elif stream.endswith("openInterest") or data.get("e") == "openInterest":
                            # Futures: open interest updates
                            oi_val = data.get("oi") or data.get("openInterest")
                            oi = self._safe_float(oi_val)
                            await self.out_queue.put(
                                {
                                    "kind": "openInterest",
                                    "symbol": symbol,
                                    "ts_ms": ts_ms,
                                    "open_interest": oi,
                                }
                            )
                        elif stream.endswith("forceOrder") or data.get("e") == "forceOrder":
                            # Futures: liquidation orders
                            o = data.get("o") or {}
                            side = (o.get("S") or o.get("s") or "").lower()
                            price = self._safe_float(o.get("p"))
                            qty = self._safe_float(o.get("q"))
                            await self.out_queue.put(
                                {
                                    "kind": "forceOrder",
                                    "symbol": symbol,
                                    "ts_ms": ts_ms,
                                    "side": side,
                                    "price": price,
                                    "qty": qty,
                                }
                            )
                        elif stream.endswith("fundingRate") or data.get("e") == "fundingRate":
                            # Futures: funding rate updates
                            rate = self._safe_float(data.get("r"))
                            await self.out_queue.put(
                                {
                                    "kind": "fundingRate",
                                    "symbol": symbol,
                                    "ts_ms": ts_ms,
                                    "rate": rate,
                                }
                            )
                        else:
                            # Unknown event type, ignore
                            continue
            except Exception as e:
                logger.warning(f"WebSocket error: {e}. Reconnecting in {backoff}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)


class FuturesCollector(SpotCollector):
    """Collector for Binance USDⓈ-M Futures market streams.

    Reuses SpotCollector behavior but connects to the futures combined stream host.
    """

    def _url(self) -> str:
        return BINANCE_FUTURES_WS_BASE + "/".join(self.build_streams())

    def build_streams(self) -> List[str]:
        # Start with spot-like streams (aggTrade/trade/depth/kline)
        names = super().build_streams()
        # Append futures-specific streams when enabled
        for s in self.symbols:
            lower = s.lower()
            if self.streams.get("fundingRate", False):
                names.append(f"{lower}@fundingRate")
            if self.streams.get("openInterest", False):
                names.append(f"{lower}@openInterest")
            if self.streams.get("forceOrder", False):
                names.append(f"{lower}@forceOrder")
        return names
