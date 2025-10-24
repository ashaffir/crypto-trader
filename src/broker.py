from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

from loguru import logger

from .positions import PositionStore


@dataclass
class ExecutionSettings:
    mode: str = "paper"  # "paper" | "live"
    venue: str = "spot"  # "spot" | "futures"
    network: str = "testnet"  # "testnet" | "mainnet"
    api_key: Optional[str] = None
    api_secret: Optional[str] = None

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
    def __init__(self, store: PositionStore) -> None:
        self.store = store

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
            logger.info(
                f"[LIVE {self.settings.venue}/{self.settings.network}] Place MARKET {direction} {symbol} qty={qty} lev={leverage}"
            )
            # TODO: send real order; on success, mirror locally
        return self.store.open_position(
            symbol=symbol,
            direction=direction,
            leverage=int(leverage),
            opened_ts_ms=int(ts_ms),
            qty=qty,
            entry_px=entry_px,
            confidence=(meta or {}).get("confidence"),
            llm_model=(meta or {}).get("llm_model"),
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
