from __future__ import annotations

import asyncio
import json
import logging
import time
import argparse

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb

from prepare_data import add_indicators  # same feature engineering as training


@dataclass
class LiveConfig:
    argparser = argparse.ArgumentParser(description="Realtime Trading Classifier")
    argparser.add_argument(
        "--symbol",
        type=str,
        default="btcusdt",
        help="Trading symbol (lowercase, e.g. 'btcusdt').",
    )
    argparser.add_argument(
        "--interval",
        type=str,
        default="1m",
        help="Candle interval (e.g. '1m', '5m').",
    )
    argparser.add_argument(
        "--ws_url",
        type=str,
        default="wss://stream.binance.com:9443/ws",
        help="WebSocket URL for Binance.",
    )
    argparser.add_argument(
        "--max_window",
        type=int,
        default=1000,
        help="Max number of candles to keep in memory for indicators.",
    )
    argparser.add_argument(
        "--model_path",
        type=str,
        default="models/lightgbm_direction_btcusdt_1m.txt",
        help="Path to the trained LightGBM model.",
    )
    argparser.add_argument(
        "--features_path",
        type=str,
        default="models/lightgbm_direction_btcusdt_1m.features.txt",
        help="Path to the feature names file.",
    )
    argparser.add_argument(
        "--p_threshold",
        type=float,
        default=0.48,
        help="Probability threshold for opening LONG positions.",
    )
    argparser.add_argument(
        "--p_down_threshold",
        type=float,
        default=0.50,
        help="Probability threshold for opening SHORT positions.",
    )
    argparser.add_argument(
        "--max_leverage",
        type=float,
        default=3.0,
        help="Maximum leverage for positions.",
    )
    argparser.add_argument(
        "--horizon_minutes",
        type=int,
        default=15,
        help="Label horizon in minutes (for Sharpe scaling).",
    )
    argparser.add_argument(
        "--vol_percentile",
        type=float,
        default=60.0,
        help="Only trade when vol_15m is above this percentile (0 to disable).",
    )
    argparser.add_argument(
        "--min_history",
        type=int,
        default=200,
        help="Minimum number of candles before starting trading.",
    )
    argparser.add_argument(
        "--starting_equity",
        type=float,
        default=1.0,
        help="Starting equity for PnL tracking.",
    )
    argparser.add_argument(
        "--log_path",
        type=str,
        default="logs/realtime_trader.log",
        help="Path to the log file.",
    )

    args = argparser.parse_args()

    symbol: str = args.symbol  # lowercase for WS stream e.g. 'btcusdt'
    interval: str = args.interval  # e.g. '1m'
    ws_url: str = args.ws_url  # e.g. 'wss://stream.binance.com:9443/ws'
    max_window: int = (
        args.max_window
    )  # candles kept for indicators & vol percentile e.g. 1000
    model_path: Path = Path(
        args.model_path
    )  # e.g. 'models/lightgbm_direction_btcusdt_1m.txt'
    features_path: Path = Path(
        args.features_path
    )  # e.g. 'models/lightgbm_direction_btcusdt_1m.features.txt'
    p_threshold: float = args.p_threshold  # long threshold (P(up)) e.g. 0.48
    p_down_threshold: float = (
        args.p_down_threshold
    )  # short threshold (P(down)) e.g. 0.50
    max_leverage: float = args.max_leverage  # e.g. 3.0
    horizon_minutes: int = args.horizon_minutes  # e.g. 15
    vol_percentile: float = (
        args.vol_percentile
    )  # only trade if vol_15m above this pct e.g. 60.0
    min_history: int = (
        args.min_history
    )  # need enough history before trusting indicators e.g. 200
    starting_equity: float = args.starting_equity  # e.g. 1.0
    log_path: Path = Path(args.log_path)  # e.g. 'logs/realtime_trader.log'

    # symbol: str = "btcusdt"  # lowercase for WS stream
    # interval: str = "1m"
    # ws_url: str = "wss://stream.binance.com:9443/ws"
    # max_window: int = 1000  # candles kept for indicators & vol percentile

    # model_path: Path = Path("models/lightgbm_direction_btcusdt_1m.txt")
    # features_path: Path = Path("models/lightgbm_direction_btcusdt_1m.features.txt")

    # p_threshold: float = 0.48  # long threshold (P(up))
    # p_down_threshold: float = 0.50  # short threshold (P(down))

    # max_leverage: float = 3.0
    # horizon_minutes: int = 15

    # vol_percentile: float = 60.0  # match backtest: only trade if vol_15m above this pct
    # min_history: int = 200  # need enough history before trusting indicators

    # starting_equity: float = 1.0
    # log_path: Path = Path("logs/realtime_trader.log")


def setup_logger(config: LiveConfig) -> logging.Logger:
    config.log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("realtime_trader")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        fh = logging.FileHandler(config.log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    return logger


async def submit_order(
    action: str,
    leverage: float,
    price: float,
    ts: pd.Timestamp,
    equity: float,
    logger: logging.Logger,
) -> None:
    """
    Placeholder for real order execution.

    Replace this with Binance REST/Futures API calls.
    """
    logger.info(
        f"[ORDER] ts={ts} action={action} leverage={leverage:.2f} "
        f"price={price:.2f} equity={equity:.4f}"
    )


async def cancel_all_orders(logger: logging.Logger) -> None:
    """
    Placeholder for cancelling all open orders.
    """
    logger.info("[ORDER] cancel_all_orders() called")


def load_model_and_features(config: LiveConfig) -> Tuple[lgb.Booster, List[str]]:
    model = lgb.Booster(model_file=str(config.model_path))
    feature_names: List[str] = [
        line.strip()
        for line in config.features_path.read_text().splitlines()
        if line.strip()
    ]
    return model, feature_names


def proba_policy(
    proba: np.ndarray,
    p_threshold: float,
    max_leverage: float,
) -> float:
    """
    proba = [P(down), P(flat), P(up)]
    if max(P(up), P(down)) < p_threshold -> no trade
    else go long/short with max_leverage in direction of higher prob
    """
    p_down = float(proba[0])
    p_flat = float(proba[1])
    p_up = float(proba[2])

    best_p = max(p_up, p_down)
    if best_p < p_threshold:
        return 0.0

    if p_up > p_down:
        return max_leverage
    return -max_leverage


def kline_to_row(msg: dict) -> Tuple[pd.Timestamp, float, float, float, float, float]:
    k = msg["k"]
    close_time_ms = k["T"]
    ts = pd.to_datetime(close_time_ms, unit="ms", utc=True)
    o = float(k["o"])
    h = float(k["h"])
    l = float(k["l"])
    c = float(k["c"])
    v = float(k["v"])
    return ts, o, h, l, c, v


def preload_history(
    config: LiveConfig,
    window: Deque[Tuple[pd.Timestamp, float, float, float, float, float]],
    logger: logging.Logger,
    max_candles: int | None = None,
) -> float | None:
    """
    Preload recent candles from Binance REST (via ccxt) into `window`.

    Returns last close price (for initial PnL reference), or None on failure.
    """
    import ccxt  # type: ignore[import]

    if max_candles is None:
        max_candles = config.min_history

    symbol_ccxt = config.symbol.upper().replace("USDT", "/USDT")

    ex = ccxt.binance({"enableRateLimit": True})

    logger.info(
        f"Preloading last {max_candles} candles for {symbol_ccxt} {config.interval}..."
    )
    try:
        ohlcv = ex.fetch_ohlcv(
            symbol_ccxt,
            timeframe=config.interval,
            limit=max_candles,
        )
    except Exception as e:
        logger.error(f"Failed to preload history: {e}")
        return None

    window.clear()

    last_close: float | None = None
    for ts_ms, o, h, l, c, v in ohlcv:
        ts = pd.to_datetime(ts_ms, unit="ms", utc=True)
        window.append((ts, float(o), float(h), float(l), float(c), float(v)))
        last_close = float(c)

    logger.info(
        f"Preloaded {len(window)} candles, last ts={window[-1][0]} close={last_close}"
    )
    return last_close


async def run_live_once(config: LiveConfig, logger: logging.Logger) -> None:
    import websockets  # type: ignore[import]

    stream_name = f"{config.symbol}@kline_{config.interval}"
    url = f"{config.ws_url}/{stream_name}"

    model, feature_names = load_model_and_features(config)

    window: Deque[Tuple[pd.Timestamp, float, float, float, float, float]] = deque(
        maxlen=config.max_window
    )

    equity = config.starting_equity
    current_pos = 0.0  # same units as in backtest (direction * leverage)
    last_price: float | None = None
    trade_count = 0
    last_heartbeat = time.time()

    logger.info(f"Connecting to {url} ...")

    last_price = preload_history(config, window, logger)

    async with websockets.connect(url) as ws:
        logger.info("Connected. Waiting for closed klines...")
        async for message in ws:
            data = json.loads(message)

            if "k" not in data:
                continue

            k = data["k"]
            if not k.get("x"):  # only closed candle
                continue

            ts, o, h, l, c, v = kline_to_row(data)
            window.append((ts, o, h, l, c, v))

            # --- PnL update from previous candle ---
            close_price = c
            if last_price is not None:
                ret = np.log(close_price / last_price)
                step_pnl = current_pos * ret
                equity += step_pnl

            last_price = close_price

            if len(window) < config.min_history:
                logger.info(f"{ts} - warming up ({len(window)}/{config.min_history})")
                continue

            # build DF and indicators on full window
            df = pd.DataFrame(
                list(window),
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            ).set_index("timestamp")

            df_ind = add_indicators(df)
            last_row = df_ind.iloc[-1]

            if last_row.isna().any():
                logger.info(f"{ts} - indicators not ready, skipping.")
                continue

            # feature vector
            try:
                X_live = (
                    last_row[feature_names].to_numpy(dtype=np.float32).reshape(1, -1)
                )
            except KeyError as e:
                logger.error(f"{ts} - missing features for model input: {e}")
                continue

            proba = model.predict(X_live)[0]  # [P(down), P(flat), P(up)]
            p_down, p_flat, p_up = proba

            logger.info(
                f"[DEBUG_PROBA] ts={ts} RAW_P = {proba} "
                f"(down={proba[0]:.3f}, flat={proba[1]:.3f}, up={proba[2]:.3f})"
            )

            raw_pos = proba_policy(
                proba,
                config.p_threshold,
                config.p_down_threshold,
                config.max_leverage,
            )
            pos = raw_pos

            close_price = float(last_row["close"])
            ema50 = float(last_row.get("ema_50", np.nan))
            vol15 = float(last_row.get("vol_15m", np.nan))

            # --- soft trend filter (EMA50) ---
            trend_note = "neutral"
            if not np.isnan(ema50):
                if close_price > ema50 and raw_pos < 0:
                    pos *= 0.5
                    trend_note = "countertrend_short (halved)"
                elif close_price < ema50 and raw_pos > 0:
                    pos *= 0.5
                    trend_note = "countertrend_long (halved)"
                elif raw_pos != 0:
                    trend_note = "with_trend"

            # --- volatility filter (vol_15m percentile) ---
            vol_note = "no_vol_filter"
            if "vol_15m" in df_ind.columns and config.vol_percentile > 0:
                vol_series = df_ind["vol_15m"].dropna()
                if len(vol_series) > 50:
                    vol_thr = np.nanpercentile(vol_series, config.vol_percentile)
                    if not np.isnan(vol15) and vol15 < vol_thr:
                        pos = 0.0
                        vol_note = f"vol_low (< p{config.vol_percentile})"
                    else:
                        vol_note = f"vol_ok (>= p{config.vol_percentile})"

            # map final pos to action
            if pos > 0:
                action = "BUY"
                leverage = abs(pos)
            elif pos < 0:
                action = "SELL"
                leverage = abs(pos)
            else:
                action = "HOLD"
                leverage = 0.0

            prev_pos = current_pos
            current_pos = pos

            # if position changes -> "order"
            if current_pos != prev_pos:
                trade_count += 1
                await submit_order(action, leverage, close_price, ts, equity, logger)

            logger.info(
                f"{ts} | close={close_price:.2f} | "
                f"P(down)={p_down:.3f}, P(flat)={p_flat:.3f}, P(up)={p_up:.3f} | "
                f"raw_pos={raw_pos:+.2f}, final_pos={pos:+.2f} | "
                f"trend={trend_note}, vol={vol_note} | "
                f"equity={equity:.4f}, trades={trade_count} | "
                f"signal={action}, leverage={leverage:.2f}x, horizon={config.horizon_minutes}m"
            )

            # --- heartbeat every 60 seconds ---
            now = time.time()
            if now - last_heartbeat > 60:
                logger.info(
                    f"[HEARTBEAT] ts={ts} equity={equity:.4f} "
                    f"pos={current_pos:+.2f} trades={trade_count}"
                )
                last_heartbeat = now


async def run_live(config: LiveConfig) -> None:
    """
    Wrapper with auto-reconnect and error logging.
    """
    logger = setup_logger(config)
    logger.info("Starting realtime classifier loop with auto-reconnect.")

    while True:
        try:
            await run_live_once(config, logger)
        except asyncio.CancelledError:
            logger.info("Cancelled, shutting down.")
            break
        except Exception:
            logger.exception("WebSocket loop crashed; reconnecting in 5 seconds...")
            await cancel_all_orders(logger)
            await asyncio.sleep(5)


if __name__ == "__main__":
    cfg = LiveConfig()
    asyncio.run(run_live(cfg))
