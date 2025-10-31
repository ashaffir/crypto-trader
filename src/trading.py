from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from loguru import logger

from .positions import PositionStore
from .broker import Broker, PaperBroker
from .utils.fees import estimate_trade_fees_usd


@dataclass
class TraderSettings:
    concurrent_positions: int = 1
    # Per-direction confidence thresholds (required; defaults provided)
    long_confidence_threshold: float = 0.8
    short_confidence_threshold: float = 0.8
    # Optional filter to require a stronger reversal signal than the one used to open
    confidence_diff_filter_enabled: bool = False
    confidence_diff_delta: float = 0.2
    default_position_size_usd: float = 0.0
    default_leverage: Optional[int] = None
    max_leverage: int = 0  # 0 means unlimited
    tp_percent: float = 0.0  # e.g., 1.0 -> +1%
    sl_percent: float = 0.0  # e.g., 0.5 -> -0.5%
    trailing_sl_enabled: bool = False
    tp_disabled: bool = False
    auto_expire_minutes: Optional[int] = None
    # Disable closing positions purely on inverse LLM recommendation
    inverse_close_disabled: bool = False
    # --- Fee settings ---
    fees_enabled: bool = False
    fee_market: str = "spot"  # "spot" | "futures"
    fee_vip_tier: int = 0
    fee_liquidity: str = "taker"  # default assume market orders
    fee_bnb_discount: bool = False  # only spot


def load_trader_settings(overrides: Optional[Dict[str, Any]]) -> TraderSettings:
    tr = (overrides or {}).get("trader") if isinstance(overrides, dict) else None
    if not isinstance(tr, dict):
        return TraderSettings()
    out = TraderSettings()
    try:
        if "concurrent_positions" in tr:
            out.concurrent_positions = max(0, int(tr.get("concurrent_positions", 1)))
        # Per-direction thresholds (required; default to 0.8)
        try:
            if tr.get("long_confidence_threshold") not in (None, ""):
                out.long_confidence_threshold = float(
                    tr.get("long_confidence_threshold")
                )
        except Exception:
            pass
        try:
            if tr.get("short_confidence_threshold") not in (None, ""):
                out.short_confidence_threshold = float(
                    tr.get("short_confidence_threshold")
                )
        except Exception:
            pass
        # Optional confidence differential filter
        if "confidence_diff_filter_enabled" in tr:
            out.confidence_diff_filter_enabled = bool(
                tr.get("confidence_diff_filter_enabled", False)
            )
        if "confidence_diff_delta" in tr:
            try:
                out.confidence_diff_delta = max(
                    0.0, float(tr.get("confidence_diff_delta", 0.2))
                )
            except Exception:
                pass
        if "default_position_size_usd" in tr:
            out.default_position_size_usd = float(
                tr.get("default_position_size_usd", 0.0)
            )
        if "default_leverage" in tr and tr.get("default_leverage") not in (None, ""):
            out.default_leverage = int(tr.get("default_leverage"))
        if "max_leverage" in tr and tr.get("max_leverage") not in (None, ""):
            try:
                ml = int(tr.get("max_leverage"))
                out.max_leverage = max(0, ml)
            except Exception:
                pass
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
        # Inverse-close disable switch (supports two keys for convenience)
        if "inverse_close_disabled" in tr:
            out.inverse_close_disabled = bool(tr.get("inverse_close_disabled", False))
        elif "disable_inverse" in tr:
            out.inverse_close_disabled = bool(tr.get("disable_inverse", False))
        # Fees
        fees = tr.get("fees") if isinstance(tr.get("fees"), dict) else {}
        if fees:
            out.fees_enabled = bool(fees.get("enabled", False))
            if fees.get("market") in ("spot", "futures"):
                out.fee_market = str(fees.get("market"))
            if fees.get("vip_tier") not in (None, ""):
                try:
                    out.fee_vip_tier = int(fees.get("vip_tier"))
                except Exception:
                    pass
            if str(fees.get("liquidity", "taker")).lower() in ("maker", "taker"):
                out.fee_liquidity = str(fees.get("liquidity")).lower()
            out.fee_bnb_discount = bool(fees.get("bnb_discount", False))
    except Exception as e:
        logger.warning(f"Invalid trader overrides; using defaults: {e}")
    return out


class TradingEngine:
    def __init__(
        self,
        store: PositionStore,
        settings: TraderSettings,
        broker: Broker | None = None,
    ) -> None:
        self.store = store
        self.settings = settings
        # Default to paper broker if none provided
        self.broker: Broker = broker or PaperBroker(store)

    def update_settings(self, settings: TraderSettings) -> None:
        self.settings = settings

    def _confidence_threshold_for(self, direction: str) -> float:
        d = str(direction).lower()
        if d in ("buy", "long"):
            return float(self.settings.long_confidence_threshold)
        return float(self.settings.short_confidence_threshold)

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
        llm_window_s: Optional[int] = None,
    ) -> Optional[int]:
        # Slot check
        if self.store.count_open() >= self.settings.concurrent_positions:
            return None
        # Confidence check (per-direction if configured)
        try:
            threshold = self._confidence_threshold_for(direction)
        except Exception:
            threshold = 0.8
        if confidence is not None and float(confidence) < float(threshold):
            return None
        # Leverage choice
        lev = (
            self.settings.default_leverage
            if self.settings.default_leverage
            else leverage
        )
        # Enforce maximum leverage if configured
        try:
            if (
                isinstance(self.settings.max_leverage, int)
                and self.settings.max_leverage > 0
            ):
                lev = min(int(lev or 1), int(self.settings.max_leverage))
        except Exception:
            pass

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

        pid = self.broker.open_position(
            symbol=symbol,
            direction=direction,
            leverage=int(lev or 1),
            qty=qty,
            entry_px=entry_px,
            ts_ms=int(ts_ms),
            meta={
                "confidence": confidence,
                "llm_model": llm_model,
                "llm_window_s": llm_window_s,
            },
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
        """Evaluate all open positions for this symbol and close as needed.

        Applies: auto-expire, trailing best-favorable maintenance, TP/SL, and inverse.
        Returns the last closed position id (if any), else None.
        """
        opens = self.store.get_open_positions(symbol)
        if not opens:
            return None

        # Determine a single exit price snapshot for this evaluation tick
        exit_px: Optional[float] = None
        if isinstance(price_info, dict):
            last_px = price_info.get("last_px")
            mid = price_info.get("mid")
            exit_px = last_px if last_px is not None else mid

        last_closed: Optional[int] = None

        # Auto-expire window (minutes) if configured
        expire_min = None
        try:
            expire_min = self.settings.auto_expire_minutes
        except Exception:
            expire_min = None

        for pos in list(opens):
            # 1) Auto-expire stale position
            try:
                if expire_min and expire_min > 0:
                    opened = (
                        int(pos.get("opened_ts_ms"))
                        if pos.get("opened_ts_ms") is not None
                        else None
                    )
                    if opened is not None:
                        max_age_ms = int(expire_min) * 60_000
                        if int(ts_ms) - opened >= max_age_ms:
                            self.broker.close_position(
                                position_id=int(pos["id"]),
                                symbol=symbol,
                                exit_px=exit_px,
                                ts_ms=int(ts_ms),
                                pnl=None,
                                reason="Stale",
                            )
                            last_closed = int(pos["id"])  # continue to evaluate others
                            continue
            except Exception:
                # Do not block other closures if expiry computation fails
                pass

            # 2) Maintain trailing best-favorable if enabled and we have a price
            try:
                if exit_px is not None and self.settings.trailing_sl_enabled:
                    self.store.update_best_favorable(int(pos["id"]), float(exit_px))
            except Exception:
                pass

            # 3) TP/SL evaluation (requires entry and a price)
            if exit_px is not None and pos.get("entry_px") is not None:
                change_pct = 0.0
                try:
                    direction = str(pos.get("direction"))
                    entry_px = float(pos.get("entry_px"))
                    change = (float(exit_px) - entry_px) / entry_px * 100.0
                    change_pct = change if direction == "long" else -change
                except Exception:
                    change_pct = 0.0

                # TP
                if (
                    not self.settings.tp_disabled
                    and self.settings.tp_percent > 0
                    and change_pct >= self.settings.tp_percent
                ):
                    self._close_with_fees(pos, ts_ms, float(exit_px), close_reason="TP")
                    last_closed = int(pos["id"])  # continue to evaluate others
                    continue

                # SL (supports trailing via best_favorable)
                sl_trigger = False
                if self.settings.sl_percent > 0:
                    if not self.settings.trailing_sl_enabled:
                        sl_trigger = change_pct <= -self.settings.sl_percent
                    else:
                        try:
                            best = pos.get("best_favorable_px")
                            direction = str(pos.get("direction"))
                            if best is not None:
                                if direction == "long":
                                    drawdown_pct = (
                                        (float(exit_px) - float(best))
                                        / float(best)
                                        * 100.0
                                    )
                                else:
                                    drawdown_pct = (
                                        (float(best) - float(exit_px))
                                        / float(best)
                                        * 100.0
                                    )
                                sl_trigger = drawdown_pct <= -self.settings.sl_percent
                        except Exception:
                            sl_trigger = False
                if sl_trigger:
                    self._close_with_fees(pos, ts_ms, float(exit_px), close_reason="SL")
                    last_closed = int(pos["id"])  # continue to evaluate others
                    continue

            # 4) Inverse recommendation close
            # New default behavior: require opposite direction AND confidence >= threshold.
            # Optionally, if confidence differential filter is enabled, also require
            # (new_confidence - old_confidence) > confidence_diff_delta to avoid false reversals.
            # If disabled via settings, skip inverse-based closures entirely
            if self.settings.inverse_close_disabled:
                continue
            if recommendation_direction:
                inv = (
                    recommendation_direction.lower() in ("buy", "long")
                    and pos.get("direction") == "short"
                    or (
                        recommendation_direction.lower() in ("sell", "short")
                        and pos.get("direction") == "long"
                    )
                )
                if inv:
                    # Require LLM confidence meets threshold
                    meets_threshold = False
                    try:
                        # Use per-direction threshold based on the new recommendation direction
                        threshold = self._confidence_threshold_for(
                            recommendation_direction
                        )
                        if confidence is not None and float(confidence) >= float(
                            threshold
                        ):
                            meets_threshold = True
                    except Exception:
                        meets_threshold = False

                    # Optional confidence differential filter
                    passes_diff = True
                    try:
                        if self.settings.confidence_diff_filter_enabled:
                            old_conf = (
                                float(pos.get("confidence"))
                                if pos.get("confidence") is not None
                                else None
                            )
                            new_conf = (
                                float(confidence) if confidence is not None else None
                            )
                            # If we cannot compute diff reliably, reject close to avoid false reversals
                            if old_conf is None or new_conf is None:
                                passes_diff = False
                            else:
                                passes_diff = (
                                    float(new_conf) - float(old_conf)
                                ) > float(self.settings.confidence_diff_delta)
                    except Exception:
                        passes_diff = False

                    if meets_threshold and passes_diff:
                        self._close_with_fees(
                            pos,
                            ts_ms,
                            float(exit_px) if exit_px is not None else None,
                            close_reason="Inverse",
                        )
                        last_closed = int(pos["id"])  # continue to evaluate others

        return last_closed

    def maybe_close_on_consensus_break(
        self,
        *,
        symbol: str,
        ts_ms: int,
        price_info: Dict[str, Any] | None,
    ) -> Optional[int]:
        """Close position immediately if it was opened under consensus-llm and consensus broke.

        Returns closed position id or None.
        """
        pos = self.store.get_latest_open_for_symbol(symbol)
        if not pos:
            return None
        try:
            if str(pos.get("llm_model") or "") != "consensus-llm":
                return None
        except Exception:
            return None

        exit_px = None
        if isinstance(price_info, dict):
            last_px = price_info.get("last_px")
            mid = price_info.get("mid")
            exit_px = last_px if last_px is not None else mid

        # Close with fees if configured
        self._close_with_fees(
            pos,
            ts_ms,
            float(exit_px) if exit_px is not None else None,
            close_reason="NoConsensus",
        )
        return int(pos["id"])

    def _close_with_fees(
        self,
        pos: Dict[str, Any],
        ts_ms: int,
        exit_px: Optional[float],
        *,
        close_reason: str,
    ) -> None:
        """Close a position computing net PnL after Binance fees when enabled.

        We estimate two legs of fees (open and close) based on price*qty and configured rates.
        Defaults keep previous behavior when fees are disabled or data missing.
        """
        # Default: let store compute pnl if fees disabled or insufficient data
        if not self.settings.fees_enabled or exit_px is None:
            self.broker.close_position(
                position_id=int(pos["id"]),
                symbol=str(pos.get("symbol")),
                exit_px=exit_px,
                ts_ms=int(ts_ms),
                pnl=None,
                reason=close_reason,
            )
            return

        try:
            entry_px = (
                float(pos.get("entry_px")) if pos.get("entry_px") is not None else None
            )
            qty = float(pos.get("qty")) if pos.get("qty") is not None else None
            lev = int(pos.get("leverage")) if pos.get("leverage") is not None else 1
            direction = str(pos.get("direction") or "long")
        except Exception:
            entry_px = None
            qty = None
            lev = 1
            direction = "long"

        if entry_px is None or qty is None:
            # Cannot compute pnl; fallback
            self.store.close_position(
                pos["id"], ts_ms, exit_px=exit_px, pnl=None, close_reason=close_reason
            )
            return

        # Gross PnL as before
        if direction == "long":
            gross_pnl = (float(exit_px) - float(entry_px)) * float(qty) * int(lev)
        else:
            gross_pnl = (float(entry_px) - float(exit_px)) * float(qty) * int(lev)

        # Estimate fees on open and close trades (price * qty each leg)
        open_notional = float(entry_px) * float(qty)
        close_notional = float(exit_px) * float(qty)
        try:
            market = self.settings.fee_market
            vip = self.settings.fee_vip_tier
            liq = self.settings.fee_liquidity
            bnb = self.settings.fee_bnb_discount
            open_fee = estimate_trade_fees_usd(
                market=market,
                vip_tier=vip,
                liquidity=liq,
                trade_quote_value_usd=open_notional,
                bnb_discount=bnb,
            )
            close_fee = estimate_trade_fees_usd(
                market=market,
                vip_tier=vip,
                liquidity=liq,
                trade_quote_value_usd=close_notional,
                bnb_discount=bnb,
            )
            net_pnl = gross_pnl - (open_fee + close_fee)
        except Exception:
            net_pnl = gross_pnl

        self.broker.close_position(
            position_id=int(pos["id"]),
            symbol=str(pos.get("symbol")),
            exit_px=exit_px,
            ts_ms=int(ts_ms),
            pnl=float(net_pnl),
            reason=close_reason,
        )


__all__ = [
    "TraderSettings",
    "load_trader_settings",
    "TradingEngine",
]
