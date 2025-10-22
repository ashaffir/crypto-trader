from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from loguru import logger

from .positions import PositionStore


@dataclass
class TraderSettings:
    concurrent_positions: int = 1
    confidence_threshold: float = 0.8
    default_position_size_usd: float = 0.0
    default_leverage: Optional[int] = None
    tp_percent: float = 0.0  # e.g., 1.0 -> +1%
    sl_percent: float = 0.0  # e.g., 0.5 -> -0.5%
    trailing_sl_enabled: bool = False
    tp_disabled: bool = False
    auto_expire_minutes: Optional[int] = None


def load_trader_settings(overrides: Optional[Dict[str, Any]]) -> TraderSettings:
    tr = (overrides or {}).get("trader") if isinstance(overrides, dict) else None
    if not isinstance(tr, dict):
        return TraderSettings()
    out = TraderSettings()
    try:
        if "concurrent_positions" in tr:
            out.concurrent_positions = max(0, int(tr.get("concurrent_positions", 1)))
        if "confidence_threshold" in tr:
            out.confidence_threshold = float(tr.get("confidence_threshold", 0.8))
        if "default_position_size_usd" in tr:
            out.default_position_size_usd = float(
                tr.get("default_position_size_usd", 0.0)
            )
        if "default_leverage" in tr and tr.get("default_leverage") not in (None, ""):
            out.default_leverage = int(tr.get("default_leverage"))
        if "tp_percent" in tr:
            out.tp_percent = float(tr.get("tp_percent", 0.0))
        if "sl_percent" in tr:
            out.sl_percent = float(tr.get("sl_percent", 0.0))
        if "trailing_sl_enabled" in tr:
            out.trailing_sl_enabled = bool(tr.get("trailing_sl_enabled", False))
        if "tp_disabled" in tr:
            out.tp_disabled = bool(tr.get("tp_disabled", False))
        if "auto_expire_minutes" in tr and tr.get("auto_expire_minutes") not in (
            None,
            "",
        ):
            out.auto_expire_minutes = max(0, int(tr.get("auto_expire_minutes")))
    except Exception as e:
        logger.warning(f"Invalid trader overrides; using defaults: {e}")
    return out


class TradingEngine:
    def __init__(self, store: PositionStore, settings: TraderSettings) -> None:
        self.store = store
        self.settings = settings

    def update_settings(self, settings: TraderSettings) -> None:
        self.settings = settings

    # price_source: dict from FeatureEngine snapshot to evaluate TP/SL
    def maybe_open_from_recommendation(
        self,
        *,
        symbol: str,
        direction: str,
        leverage: int,
        confidence: Optional[float],
        ts_ms: int,
        price_info: Dict[str, Any] | None,
        llm_model: Optional[str],
    ) -> Optional[int]:
        # Slot check
        if self.store.count_open() >= self.settings.concurrent_positions:
            return None
        # Confidence check
        if confidence is not None and confidence < self.settings.confidence_threshold:
            return None
        # Leverage choice
        lev = (
            self.settings.default_leverage
            if self.settings.default_leverage
            else leverage
        )

        entry_px = None
        if isinstance(price_info, dict):
            # Favor last trade if present, else mid
            last_px = price_info.get("last_px")
            mid = price_info.get("mid")
            entry_px = last_px if last_px is not None else mid

        # Record open; qty kept optional; store model via confidence field? we add later in schema if needed
        # Compute paper qty from default_position_size_usd and entry price if available
        qty = None
        try:
            if self.settings.default_position_size_usd and entry_px and entry_px > 0:
                qty = float(self.settings.default_position_size_usd) / float(entry_px)
        except Exception:
            qty = None

        pid = self.store.open_position(
            symbol=symbol,
            direction=direction,
            leverage=int(lev or 1),
            opened_ts_ms=int(ts_ms),
            qty=qty,
            entry_px=entry_px,
            confidence=confidence,
            llm_model=llm_model,
        )
        return pid

    def maybe_close_on_inverse_or_tp_sl(
        self,
        *,
        symbol: str,
        recommendation_direction: Optional[str],
        confidence: Optional[float],
        ts_ms: int,
        price_info: Dict[str, Any] | None,
    ) -> Optional[int]:
        """Close existing position if inverse rec above threshold or TP/SL triggers.

        Returns closed position id or None.
        """
        pos = self.store.get_latest_open_for_symbol(symbol)
        if not pos:
            return None

        # TP/SL evaluation
        exit_px = None
        if isinstance(price_info, dict):
            last_px = price_info.get("last_px")
            mid = price_info.get("mid")
            exit_px = last_px if last_px is not None else mid

        # Auto-expire stale position
        try:
            expire_min = self.settings.auto_expire_minutes
            if expire_min and expire_min > 0:
                opened = (
                    int(pos.get("opened_ts_ms"))
                    if pos.get("opened_ts_ms") is not None
                    else None
                )
                if opened is not None:
                    max_age_ms = int(expire_min) * 60_000
                    if int(ts_ms) - opened >= max_age_ms:
                        pnl = None
                        try:
                            if (
                                pos.get("qty") is not None
                                and pos.get("entry_px") is not None
                                and exit_px is not None
                            ):
                                qty = float(pos.get("qty"))
                                entry = float(pos.get("entry_px"))
                                if str(pos.get("direction")) == "long":
                                    pnl = (float(exit_px) - entry) * qty
                                else:
                                    pnl = (entry - float(exit_px)) * qty
                        except Exception:
                            pnl = None
                        self.store.close_position(
                            pos["id"],
                            ts_ms,
                            exit_px=exit_px,
                            pnl=pnl,
                            close_reason="Stale",
                        )
                        return int(pos["id"])  # closed by Stale
        except Exception:
            # Do not block other closures if expiry computation fails
            pass

        if exit_px is not None and pos.get("entry_px") is not None:
            change_pct = 0.0
            try:
                direction = str(pos.get("direction"))
                entry_px = float(pos.get("entry_px"))
                change = (float(exit_px) - entry_px) / entry_px * 100.0
                change_pct = change if direction == "long" else -change
            except Exception:
                change_pct = 0.0

            # Maintain trailing best-favorable price if trailing is enabled
            try:
                if self.settings.trailing_sl_enabled:
                    self.store.update_best_favorable(int(pos["id"]), float(exit_px))
            except Exception:
                pass

            # TP
            if (
                not self.settings.tp_disabled
                and self.settings.tp_percent > 0
                and change_pct >= self.settings.tp_percent
            ):
                pnl = None
                try:
                    if pos.get("qty") is not None and pos.get("entry_px") is not None:
                        qty = float(pos.get("qty"))
                        entry = float(pos.get("entry_px"))
                        if str(pos.get("direction")) == "long":
                            pnl = (float(exit_px) - entry) * qty
                        else:
                            pnl = (entry - float(exit_px)) * qty
                except Exception:
                    pnl = None
                self.store.close_position(
                    pos["id"], ts_ms, exit_px=exit_px, pnl=pnl, close_reason="TP"
                )
                return int(pos["id"])  # closed by TP

            # SL (no trailing implementation stateful here; trailing requires tracking max_favorable)
            sl_trigger = False
            if self.settings.sl_percent > 0:
                if not self.settings.trailing_sl_enabled:
                    sl_trigger = change_pct <= -self.settings.sl_percent
                else:
                    try:
                        # Compute drawdown from best favorable
                        best = pos.get("best_favorable_px")
                        direction = str(pos.get("direction"))
                        if best is not None:
                            if direction == "long":
                                drawdown_pct = (
                                    (float(exit_px) - float(best)) / float(best) * 100.0
                                )
                            else:
                                drawdown_pct = (
                                    (float(best) - float(exit_px)) / float(best) * 100.0
                                )
                            sl_trigger = drawdown_pct <= -self.settings.sl_percent
                    except Exception:
                        sl_trigger = False
            if sl_trigger:
                pnl = None
                try:
                    if pos.get("qty") is not None and pos.get("entry_px") is not None:
                        qty = float(pos.get("qty"))
                        entry = float(pos.get("entry_px"))
                        if str(pos.get("direction")) == "long":
                            pnl = (float(exit_px) - entry) * qty
                        else:
                            pnl = (entry - float(exit_px)) * qty
                except Exception:
                    pnl = None
                self.store.close_position(
                    pos["id"], ts_ms, exit_px=exit_px, pnl=pnl, close_reason="SL"
                )
                return int(pos["id"])  # closed by SL

        # Inverse recommendation close (slots not relevant for close)
        if recommendation_direction and confidence is not None:
            inv = (
                recommendation_direction.lower() in ("buy", "long")
                and pos.get("direction") == "short"
                or (
                    recommendation_direction.lower() in ("sell", "short")
                    and pos.get("direction") == "long"
                )
            )
            if inv and confidence >= self.settings.confidence_threshold:
                pnl = None
                try:
                    if (
                        pos.get("qty") is not None
                        and pos.get("entry_px") is not None
                        and exit_px is not None
                    ):
                        qty = float(pos.get("qty"))
                        entry = float(pos.get("entry_px"))
                        if str(pos.get("direction")) == "long":
                            pnl = (float(exit_px) - entry) * qty
                        else:
                            pnl = (entry - float(exit_px)) * qty
                except Exception:
                    pnl = None
                self.store.close_position(
                    pos["id"], ts_ms, exit_px=exit_px, pnl=pnl, close_reason="Inverse"
                )
                return int(pos["id"])

        return None


__all__ = [
    "TraderSettings",
    "load_trader_settings",
    "TradingEngine",
]
