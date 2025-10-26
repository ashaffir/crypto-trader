from __future__ import annotations

import hashlib
import hmac
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx


BINANCE_SPOT_BASE = "https://api.binance.com"
BINANCE_FUTURES_BASE = "https://fapi.binance.com"
BINANCE_SPOT_TESTNET_BASE = "https://testnet.binance.vision"
BINANCE_FUTURES_TESTNET_BASE = "https://testnet.binancefuture.com"


def _choose_base(venue: str, network: str) -> str:
    v = str(venue).lower()
    n = str(network).lower()
    if v == "futures":
        return BINANCE_FUTURES_TESTNET_BASE if n == "testnet" else BINANCE_FUTURES_BASE
    return BINANCE_SPOT_TESTNET_BASE if n == "testnet" else BINANCE_SPOT_BASE


def _sign(query: str, secret: str) -> str:
    return hmac.new(
        secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def _headers(api_key: str) -> Dict[str, str]:
    return {
        "X-MBX-APIKEY": api_key,
        "User-Agent": "crypto-trader/1.0",
    }


async def _get(client: httpx.AsyncClient, url: str, headers: Dict[str, str]) -> Any:
    r = await client.get(url, headers=headers, timeout=10.0)
    r.raise_for_status()
    return r.json()


async def fetch_spot_account_info(
    *, api_key: str, api_secret: str, network: str = "mainnet"
) -> Dict[str, Any]:
    base = _choose_base("spot", network)
    ts = int(time.time() * 1000)
    query = f"timestamp={ts}"
    sig = _sign(query, api_secret)
    url = f"{base}/api/v3/account?{query}&signature={sig}"
    async with httpx.AsyncClient() as client:
        data = await _get(client, url, _headers(api_key))
        return data  # contains balances list with free/locked


async def fetch_futures_account_info(
    *, api_key: str, api_secret: str, network: str = "mainnet"
) -> Dict[str, Any]:
    base = _choose_base("futures", network)
    ts = int(time.time() * 1000)
    query = f"timestamp={ts}"
    sig = _sign(query, api_secret)
    url = f"{base}/fapi/v2/account?{query}&signature={sig}"
    async with httpx.AsyncClient() as client:
        data = await _get(client, url, _headers(api_key))
        return data  # assets array with walletBalance/crossUnPnl etc


def _env(key: str) -> Optional[str]:
    v = os.getenv(key)
    return v if isinstance(v, str) and v else None


async def fetch_balances_from_env(
    *, venue: str = "spot", network: str = "mainnet"
) -> Dict[str, Any]:
    """Fetch balances using environment variables.

    Expected env vars:
    - BINANCE_API_KEY
    - BINANCE_API_SECRET
    Optionally:
    - BINANCE_NETWORK: "mainnet" | "testnet" (overrides network param)
    - BINANCE_VENUE: "spot" | "futures" (overrides venue param)
    """
    api_key = _env("BINANCE_API_KEY")
    api_secret = _env("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        return {"ok": False, "error": "missing_credentials"}

    v = str(_env("BINANCE_VENUE") or venue or "spot").lower()
    n = str(_env("BINANCE_NETWORK") or network or "mainnet").lower()

    try:
        if v == "futures":
            info = await fetch_futures_account_info(
                api_key=api_key, api_secret=api_secret, network=n
            )
            # Normalize
            assets = info.get("assets", []) if isinstance(info, dict) else []
            normalized = []
            for a in assets:
                asset = a.get("asset")
                wb = float(a.get("walletBalance", 0))
                cw = float(a.get("crossWalletBalance", a.get("availableBalance", 0)))
                normalized.append({"asset": asset, "total": wb, "available": cw})
            return {
                "ok": True,
                "venue": "futures",
                "network": n,
                "balances": normalized,
            }
        else:
            info = await fetch_spot_account_info(
                api_key=api_key, api_secret=api_secret, network=n
            )
            bals = info.get("balances", []) if isinstance(info, dict) else []
            normalized = []
            for b in bals:
                asset = b.get("asset")
                free = float(b.get("free", 0))
                locked = float(b.get("locked", 0))
                normalized.append(
                    {
                        "asset": asset,
                        "available": free,
                        "locked": locked,
                        "total": free + locked,
                    }
                )
            return {"ok": True, "venue": "spot", "network": n, "balances": normalized}
    except httpx.HTTPStatusError as e:
        try:
            detail = e.response.json()
        except Exception:
            detail = {"msg": str(e)}
        return {"ok": False, "error": "http_error", "detail": detail}
    except Exception as e:
        return {"ok": False, "error": "exception", "detail": str(e)}


__all__ = [
    "fetch_balances_from_env",
    "fetch_balances_from_runtime_config",
]


async def fetch_balances_from_runtime_config() -> Dict[str, Any]:
    """Fetch balances using keys and venue/network from runtime_config.json.

    Reads CONTROL_DIR/runtime_config.json > execution:{mode,venue,network,api_key,api_secret}.
    Falls back to market (top-level) when venue missing. Defaults: venue=spot, network=testnet.
    """
    try:
        from src.runtime_config import (
            RuntimeConfigManager,
        )  # local import to avoid cycles

        rcm = RuntimeConfigManager()
        cfg = rcm.read() or {}
        ex = cfg.get("execution") if isinstance(cfg, dict) else None
        api_key = (ex or {}).get("api_key")
        api_secret = (ex or {}).get("api_secret")
        if not api_key or not api_secret:
            return {"ok": False, "error": "missing_credentials"}

        venue = str((ex or {}).get("venue") or cfg.get("market") or "spot").lower()
        network = str((ex or {}).get("network") or "testnet").lower()

        if venue == "futures":
            info = await fetch_futures_account_info(
                api_key=api_key, api_secret=api_secret, network=network
            )
            assets = info.get("assets", []) if isinstance(info, dict) else []
            normalized = []
            for a in assets:
                asset = a.get("asset")
                wb = float(a.get("walletBalance", 0))
                cw = float(a.get("crossWalletBalance", a.get("availableBalance", 0)))
                normalized.append({"asset": asset, "total": wb, "available": cw})
            return {
                "ok": True,
                "venue": "futures",
                "network": network,
                "balances": normalized,
            }
        else:
            info = await fetch_spot_account_info(
                api_key=api_key, api_secret=api_secret, network=network
            )
            bals = info.get("balances", []) if isinstance(info, dict) else []
            normalized = []
            for b in bals:
                asset = b.get("asset")
                free = float(b.get("free", 0))
                locked = float(b.get("locked", 0))
                normalized.append(
                    {
                        "asset": asset,
                        "available": free,
                        "locked": locked,
                        "total": free + locked,
                    }
                )
            return {
                "ok": True,
                "venue": "spot",
                "network": network,
                "balances": normalized,
            }
    except Exception as e:  # pragma: no cover
        return {"ok": False, "error": "exception", "detail": str(e)}
