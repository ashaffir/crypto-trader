from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any, Dict, Optional

import httpx
from loguru import logger


CONTROL_DIR = os.getenv("CONTROL_DIR", os.path.join("data", "control"))
RUNTIME_FILE = os.path.join(CONTROL_DIR, "runtime_config.json")


def _read_runtime() -> Dict[str, Any]:
    try:
        if os.path.exists(RUNTIME_FILE):
            with open(RUNTIME_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        return {}
    except Exception:
        return {}


def _load_execution() -> Dict[str, Any]:
    cfg = _read_runtime()
    ex = cfg.get("execution") if isinstance(cfg, dict) else None
    out = {
        "mode": "paper",
        "venue": "spot",
        "network": "testnet",
        "api_key": None,
        "api_secret": None,
    }
    if isinstance(ex, dict):
        for k in out.keys():
            if k in ex:
                out[k] = ex[k]
    return out


def _binance_endpoints(network: str) -> Dict[str, str]:
    if str(network).lower() == "mainnet":
        return {
            "base": "https://api.binance.com",
        }
    # default testnet
    return {
        "base": "https://testnet.binance.vision",
    }


def _timestamp_ms() -> int:
    return int(time.time() * 1000)


def _sign(query_string: str, secret: str) -> str:
    return hmac.new(
        secret.encode("utf-8"), query_string.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def _build_headers(api_key: str) -> Dict[str, str]:
    return {
        "X-MBX-APIKEY": api_key,
    }


async def place_spot_market_order(
    *,
    symbol: str,
    side: str,
    quote_qty: float,
    ex: Dict[str, Any],
) -> Dict[str, Any]:
    side_norm = side.upper()
    if side_norm not in ("BUY", "SELL"):
        raise ValueError("side must be BUY or SELL")

    endpoints = _binance_endpoints(str(ex.get("network", "testnet")))
    base = endpoints["base"]
    url = f"{base}/api/v3/order"

    async with httpx.AsyncClient(timeout=15.0) as client:
        # Sync timestamp with server to avoid 400 timestamp errors
        ts = _timestamp_ms()
        try:
            t = await client.get(f"{base}/api/v3/time")
            t.raise_for_status()
            data = t.json() or {}
            if isinstance(data, dict) and data.get("serverTime"):
                ts = int(data["serverTime"])  # use server-provided ms
        except Exception:
            # Fall back to local clock
            pass

        params = {
            "symbol": symbol.upper(),
            "side": side_norm,
            "type": "MARKET",
            "quoteOrderQty": f"{quote_qty}",
            "recvWindow": 5000,
            "timestamp": ts,
            # Optional: reduce risk of duplicates
            "newClientOrderId": base64.urlsafe_b64encode(f"tiny-{ts}".encode())
            .decode()
            .strip("="),
        }

        # Serialize as query string in canonical order (dict preserves insertion order)
        items = [f"{k}={params[k]}" for k in params]
        query = "&".join(items)
        signature = _sign(query, str(ex.get("api_secret")))
        query_signed = f"{query}&signature={signature}"

        headers = _build_headers(str(ex.get("api_key")))
        headers.setdefault("Content-Type", "application/x-www-form-urlencoded")

        resp = await client.post(url, headers=headers, content=query_signed)
        resp.raise_for_status()
        return resp.json()


def _validate_ready(ex: Dict[str, Any]) -> Optional[str]:
    if str(ex.get("venue", "spot")).lower() != "spot":
        return "This helper only supports spot for now. Set Market Mode to spot."
    if str(ex.get("mode", "paper")).lower() != "live":
        return "Execution mode is not live. Set Mode to live in Settings."
    if not ex.get("api_key") or not ex.get("api_secret"):
        return "Missing API credentials. Save API Key and Secret in Settings."
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Place a tiny Binance spot MARKET order using saved execution settings."
    )
    parser.add_argument("symbol", help="Trading pair, e.g., BTCUSDT")
    parser.add_argument(
        "side", choices=["BUY", "SELL", "buy", "sell"], help="Order side"
    )
    parser.add_argument(
        "quote_amount", type=float, help="Amount in quote currency (e.g., USDT)"
    )
    args = parser.parse_args()

    ex = _load_execution()
    err = _validate_ready(ex)
    if err:
        logger.error(err)
        raise SystemExit(2)

    logger.info(
        f"Submitting MARKET {args.side.upper()} {args.symbol} for quote {args.quote_amount} on {ex.get('network')}."
    )

    try:
        import asyncio

        res = asyncio.run(
            place_spot_market_order(
                symbol=args.symbol,
                side=args.side,
                quote_qty=float(args.quote_amount),
                ex=ex,
            )
        )
        logger.success("Order accepted")
        print(json.dumps(res, indent=2))
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP {e.response.status_code}: {e.response.text}")
        raise SystemExit(1)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to place order: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
