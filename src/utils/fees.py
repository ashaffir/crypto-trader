from __future__ import annotations

from typing import Literal


Liquidity = Literal["maker", "taker"]
Market = Literal["spot", "futures"]


def _clamp_vip(vip_tier: int) -> int:
    try:
        v = int(vip_tier)
        if v < 0:
            return 0
        if v > 9:
            return 9
        return v
    except Exception:
        return 0


def get_spot_fee_rate(vip_tier: int, liquidity: Liquidity, bnb_discount: bool) -> float:
    """Approximate Binance Spot fee rate per VIP tier and liquidity.

    Notes:
    - VIP 0 base: 0.1% maker, 0.1% taker
    - Paying fees with BNB typically provides a 25% discount on spot: 0.075%
    - For higher VIP tiers, rates are lower; we conservatively fall back to VIP 0 if not mapped.
    Reference: https://www.binance.com/en/fee
    """
    v = _clamp_vip(vip_tier)

    # Minimal schedule; default to VIP 0 if not explicitly listed
    base_schedule = {
        0: {"maker": 0.0010, "taker": 0.0010},
        # Examples for a couple tiers; others fall back to VIP 0
        1: {"maker": 0.0009, "taker": 0.0009},
        2: {"maker": 0.0008, "taker": 0.0008},
    }
    rates = base_schedule.get(v, base_schedule[0])
    rate = float(rates[liquidity])
    if bnb_discount:
        # Apply 25% discount on spot when paying fees with BNB
        rate *= 0.75
    return rate


def get_futures_fee_rate(vip_tier: int, liquidity: Liquidity) -> float:
    """Approximate Binance USDⓈ-M Futures fee rate.

    Notes:
    - VIP 0 base: 0.02% maker, 0.04% taker
    - We conservatively fall back to VIP 0 if not mapped.
    Reference: https://www.binance.com/en/fee
    """
    v = _clamp_vip(vip_tier)
    base_schedule = {
        0: {"maker": 0.0002, "taker": 0.0004},
        1: {"maker": 0.00016, "taker": 0.00036},
        2: {"maker": 0.00014, "taker": 0.00032},
    }
    rates = base_schedule.get(v, base_schedule[0])
    return float(rates[liquidity])


def get_fee_rate(
    *,
    market: Market,
    vip_tier: int = 0,
    liquidity: Liquidity = "taker",
    bnb_discount: bool = False,
) -> float:
    """Return the fee rate (as a decimal) for the given market and parameters.

    - market: "spot" or "futures"
    - vip_tier: 0..9 (values outside are clamped)
    - liquidity: "maker" or "taker"
    - bnb_discount: only applicable to spot
    """
    m = str(market).lower()
    lq = "maker" if str(liquidity).lower() == "maker" else "taker"
    if m == "spot":
        return get_spot_fee_rate(vip_tier, lq, bnb_discount)
    return get_futures_fee_rate(vip_tier, lq)


def estimate_trade_fees_usd(
    *,
    market: Market,
    vip_tier: int,
    liquidity: Liquidity,
    trade_quote_value_usd: float,
    bnb_discount: bool = False,
) -> float:
    """Estimate the fee in quote currency (USD terms) for one trade.

    trade_quote_value_usd should be price * quantity in quote currency units (e.g., USDT).
    For leveraged futures, leverage does not change the traded notional; pass price * qty.
    """
    rate = get_fee_rate(
        market=market, vip_tier=vip_tier, liquidity=liquidity, bnb_discount=bnb_discount
    )
    try:
        return float(trade_quote_value_usd) * float(rate)
    except Exception:
        return 0.0


__all__ = [
    "Liquidity",
    "Market",
    "get_fee_rate",
    "estimate_trade_fees_usd",
]
