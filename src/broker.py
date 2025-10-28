from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol
import time
import hmac
import hashlib
import httpx

from loguru import logger

from .positions import PositionStore


@dataclass
class ExecutionSettings:
    mode: str = "paper"  # "paper" | "live"
    venue: str = "spot"  # "spot" | "futures"
    network: str = "testnet"  # "testnet" | "mainnet"
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    recv_window_ms: int = 60000

    @staticmethod
    def from_overrides(overrides: Optional[dict[str, Any]]) -> "ExecutionSettings":
        if not isinstance(overrides, dict):
            return ExecutionSettings()
        ex = (
            overrides.get("execution")
            if isinstance(overrides.get("execution"), dict)
            else {}
        )
        out = ExecutionSettings()
        try:
            m = str(ex.get("mode", out.mode)).lower()
            if m in ("paper", "live"):
                out.mode = m
            v = str(ex.get("venue", out.venue)).lower()
            if v in ("spot", "futures"):
                out.venue = v
            n = str(ex.get("network", out.network)).lower()
            if n in ("testnet", "mainnet"):
                out.network = n
            ak = ex.get("api_key")
            if isinstance(ak, str) and ak:
                out.api_key = ak
            sk = ex.get("api_secret")
            if isinstance(sk, str) and sk:
                out.api_secret = sk
            try:
                rw = ex.get("recv_window_ms")
                if rw is not None:
                    out.recv_window_ms = max(0, int(rw))
            except Exception:
                pass
        except Exception:
            pass
        return out


class Broker(Protocol):
    def open_position(
        self,
        *,
        symbol: str,
        direction: str,  # "buy"|"sell"|"long"|"short"
        leverage: int,
        qty: Optional[float],
        entry_px: Optional[float],
        ts_ms: int,
        meta: Optional[dict[str, Any]] = None,
    ) -> Optional[int]: ...

    def close_position(
        self,
        *,
        position_id: int,
        symbol: str,
        exit_px: Optional[float],
        ts_ms: int,
        pnl: Optional[float],
        reason: str,
    ) -> bool: ...


class PaperBroker:
    def __init__(self, store: PositionStore, *, venue: str = "spot") -> None:
        self.store = store
        self.venue = "futures" if str(venue).lower() == "futures" else "spot"

    def open_position(
        self,
        *,
        symbol: str,
        direction: str,
        leverage: int,
        qty: Optional[float],
        entry_px: Optional[float],
        ts_ms: int,
        meta: Optional[dict[str, Any]] = None,
    ) -> Optional[int]:
        pid = self.store.open_position(
            symbol=symbol,
            direction=direction,
            leverage=int(leverage),
            opened_ts_ms=int(ts_ms),
            qty=qty,
            entry_px=entry_px,
            confidence=(meta or {}).get("confidence"),
            llm_model=(meta or {}).get("llm_model"),
            llm_window_s=(meta or {}).get("llm_window_s"),
            venue=self.venue,
            exec_mode="paper",
        )
        return pid

    def close_position(
        self,
        *,
        position_id: int,
        symbol: str,
        exit_px: Optional[float],
        ts_ms: int,
        pnl: Optional[float],
        reason: str,
    ) -> bool:
        try:
            self.store.close_position(
                int(position_id),
                int(ts_ms),
                exit_px=exit_px,
                pnl=pnl,
                close_reason=reason,
            )
            return True
        except Exception:
            return False


class BinanceBrokerSkeleton:
    """Placeholder for live execution (spot/futures). Safe by default (no-op).

    This skeleton logs intents. Wiring real order placement should use Binance REST
    and user data streams. Until credentials are provided and an execution flag is set,
    it behaves like a dry-run that only mirrors to the PositionStore.
    """

    def __init__(self, store: PositionStore, settings: ExecutionSettings) -> None:
        self.store = store
        self.settings = settings
        self.enabled = settings.mode == "live" and bool(
            settings.api_key and settings.api_secret
        )
        logger.info(
            f"BinanceBrokerSkeleton initialized: venue={settings.venue}, network={settings.network}, enabled={self.enabled}"
        )

    # ---- Minimal synchronous spot MARKET order (quote notional) ----
    @staticmethod
    def _spot_base(network: str) -> str:
        return (
            "https://api.binance.com"
            if str(network).lower() == "mainnet"
            else "https://testnet.binance.vision"
        )

    @staticmethod
    def _sign(query: str, secret: str) -> str:
        return hmac.new(
            secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def _place_spot_market_order_sync(
        self,
        *,
        symbol: str,
        side: str,
        quote_qty: float,
    ) -> dict | None:
        base = self._spot_base(self.settings.network)
        url = f"{base}/api/v3/order"
        with httpx.Client(timeout=15.0) as client:
            # server time
            ts = int(time.time() * 1000)
            try:
                t = client.get(f"{base}/api/v3/time")
                t.raise_for_status()
                jd = t.json() or {}
                if isinstance(jd, dict) and jd.get("serverTime"):
                    ts = int(jd["serverTime"])  # ms
            except Exception:
                pass

            params = {
                "symbol": symbol.upper(),
                "side": side.upper(),
                "type": "MARKET",
                "quoteOrderQty": f"{quote_qty}",
                "recvWindow": int(self.settings.recv_window_ms),
                "timestamp": ts,
            }
            query = "&".join([f"{k}={params[k]}" for k in params])
            sig = self._sign(query, str(self.settings.api_secret))
            headers = {
                "X-MBX-APIKEY": str(self.settings.api_key),
                "Content-Type": "application/x-www-form-urlencoded",
            }
            r = client.post(url, headers=headers, content=f"{query}&signature={sig}")
            r.raise_for_status()
            return r.json()

    # ---- Minimal USD-M futures MARKET order ----
    @staticmethod
    def _futures_base(network: str) -> str:
        return (
            "https://fapi.binance.com"
            if str(network).lower() == "mainnet"
            else "https://testnet.binancefuture.com"
        )

    def _place_futures_market_order_sync(
        self,
        *,
        symbol: str,
        side: str,
        leverage: int,
        quantity: float,
    ) -> dict | None:
        base = self._futures_base(self.settings.network)
        with httpx.Client(timeout=15.0) as client:
            # server time
            ts = int(time.time() * 1000)
            try:
                t = client.get(f"{base}/fapi/v1/time")
                t.raise_for_status()
                jd = t.json() or {}
                if isinstance(jd, dict) and jd.get("serverTime"):
                    ts = int(jd["serverTime"])  # ms
            except Exception:
                pass

            # Ensure leverage for the symbol
            try:
                p = {
                    "symbol": symbol.upper(),
                    "leverage": int(leverage),
                    "recvWindow": int(self.settings.recv_window_ms),
                    "timestamp": ts,
                }
                q = "&".join([f"{k}={p[k]}" for k in p])
                sig = self._sign(q, str(self.settings.api_secret))
                hdrs = {
                    "X-MBX-APIKEY": str(self.settings.api_key),
                    "Content-Type": "application/x-www-form-urlencoded",
                }
                r = client.post(
                    f"{base}/fapi/v1/leverage",
                    headers=hdrs,
                    content=f"{q}&signature={sig}",
                )
                r.raise_for_status()
                # refresh ts after leverage set
                ts = int(time.time() * 1000)
                try:
                    t = client.get(f"{base}/fapi/v1/time")
                    t.raise_for_status()
                    jd = t.json() or {}
                    if isinstance(jd, dict) and jd.get("serverTime"):
                        ts = int(jd["serverTime"])  # ms
                except Exception:
                    pass
            except httpx.HTTPStatusError as e:
                try:
                    body = e.response.json()
                except Exception:
                    body = {"status": e.response.status_code, "text": e.response.text}
                logger.error(
                    f"Set futures leverage failed: HTTP {e.response.status_code} {body}"
                )
                return None

            # Place order
            params = {
                "symbol": symbol.upper(),
                "side": side.upper(),
                "type": "MARKET",
                "quantity": f"{quantity}",
                "recvWindow": int(self.settings.recv_window_ms),
                "timestamp": ts,
            }
            query = "&".join([f"{k}={params[k]}" for k in params])
            sig = self._sign(query, str(self.settings.api_secret))
            headers = {
                "X-MBX-APIKEY": str(self.settings.api_key),
                "Content-Type": "application/x-www-form-urlencoded",
            }
            r = client.post(
                f"{base}/fapi/v1/order",
                headers=headers,
                content=f"{query}&signature={sig}",
            )
            r.raise_for_status()
            return r.json()

    def open_position(
        self,
        *,
        symbol: str,
        direction: str,
        leverage: int,
        qty: Optional[float],
        entry_px: Optional[float],
        ts_ms: int,
        meta: Optional[dict[str, Any]] = None,
    ) -> Optional[int]:
        if self.enabled:
            side = "BUY" if str(direction).lower() in ("buy", "long") else "SELL"
            if self.settings.venue == "spot":
                # Use quote amount if possible
                quote_amt: Optional[float] = None
                try:
                    if qty is not None and entry_px is not None and float(entry_px) > 0:
                        # For spot (no leverage), notional ~= qty * price
                        quote_amt = float(qty) * float(entry_px)
                except Exception:
                    quote_amt = None
                if quote_amt is None:
                    logger.warning(
                        "Cannot compute quote amount; skipping live order and not opening position."
                    )
                    return None
                try:
                    _ = self._place_spot_market_order_sync(
                        symbol=symbol, side=side, quote_qty=float(quote_amt)
                    )
                    logger.info(
                        f"[LIVE spot/{self.settings.network}] Order accepted for {symbol} {side} notional≈{quote_amt}"
                    )
                except httpx.HTTPStatusError as e:
                    try:
                        body = e.response.json()
                    except Exception:
                        body = {
                            "status": e.response.status_code,
                            "text": e.response.text,
                        }
                    logger.error(
                        f"Live order failed: HTTP {e.response.status_code} {body}"
                    )
                    return None
                except Exception as e:
                    logger.error(f"Live order failed: {e}")
                    return None
            else:
                # USD-M futures: derive notional from leverage and qty/price
                notional: Optional[float] = None
                price_for_calc: Optional[float] = None
                try:
                    if entry_px is not None and float(entry_px) > 0:
                        price_for_calc = float(entry_px)
                except Exception:
                    price_for_calc = None
                # If no entry price, fetch mark price
                if price_for_calc is None:
                    try:
                        base = self._futures_base(self.settings.network)
                        with httpx.Client(timeout=10.0) as client:
                            r = client.get(
                                f"{base}/fapi/v1/ticker/price?symbol={symbol.upper()}"
                            )
                            r.raise_for_status()
                            data = r.json() or {}
                            p = (
                                float(data.get("price"))
                                if isinstance(data, dict)
                                else None
                            )
                            if p and p > 0:
                                price_for_calc = p
                    except Exception:
                        price_for_calc = None
                try:
                    if qty is not None and price_for_calc is not None:
                        notional = (
                            float(qty) * float(price_for_calc) * float(leverage or 1)
                        )
                except Exception:
                    notional = None
                if notional is None or price_for_calc is None:
                    logger.warning(
                        "Cannot compute futures notional/price; skipping live order."
                    )
                    return None
                # Quantity in base asset for futures
                fut_qty = max(0.0, notional / float(price_for_calc))
                try:
                    _ = self._place_futures_market_order_sync(
                        symbol=symbol,
                        side=side,
                        leverage=int(leverage or 1),
                        quantity=fut_qty,
                    )
                    logger.info(
                        f"[LIVE futures/{self.settings.network}] Order accepted for {symbol} {side} qty≈{fut_qty} lev={leverage} notional≈{notional}"
                    )
                except httpx.HTTPStatusError as e:
                    try:
                        body = e.response.json()
                    except Exception:
                        body = {
                            "status": e.response.status_code,
                            "text": e.response.text,
                        }
                    logger.error(
                        f"Live futures order failed: HTTP {e.response.status_code} {body}"
                    )
                    return None
                except Exception as e:
                    logger.error(f"Live futures order failed: {e}")
                    return None
        return self.store.open_position(
            symbol=symbol,
            direction=direction,
            leverage=int(leverage),
            opened_ts_ms=int(ts_ms),
            qty=qty,
            entry_px=entry_px,
            confidence=(meta or {}).get("confidence"),
            llm_model=(meta or {}).get("llm_model"),
            llm_window_s=(meta or {}).get("llm_window_s"),
            venue=str(self.settings.venue).lower(),
            exec_mode=(
                "live" if self.enabled and self.settings.venue == "spot" else "paper"
            ),
        )

    def close_position(
        self,
        *,
        position_id: int,
        symbol: str,
        exit_px: Optional[float],
        ts_ms: int,
        pnl: Optional[float],
        reason: str,
    ) -> bool:
        if self.enabled:
            logger.info(
                f"[LIVE {self.settings.venue}/{self.settings.network}] Close {symbol} pos_id={position_id} reason={reason}"
            )
            # TODO: send close order; on success, mirror locally
        try:
            self.store.close_position(
                int(position_id),
                int(ts_ms),
                exit_px=exit_px,
                pnl=pnl,
                close_reason=reason,
            )
            return True
        except Exception:
            return False


__all__ = [
    "ExecutionSettings",
    "Broker",
    "PaperBroker",
    "BinanceBrokerSkeleton",
]
