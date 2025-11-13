from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Tuple
from collections import deque

import numpy as np
import pandas as pd
import lightgbm as lgb

from prepare_data import add_indicators  # reuse training-time feature engineering


@dataclass
class LiveConfig:
    symbol: str = "btcusdt"  # lowercase for WS stream
    interval: str = "1m"
    ws_url: str = "wss://stream.binance.com:9443/ws"
    max_window: int = 500  # number of candles to keep for indicators
    model_path: Path = Path("models/lightgbm_btcusdt_1m.txt")
    features_path: Path = Path("models/lightgbm_btcusdt_1m.features.txt")
    threshold: float = 0.0005
    max_leverage: float = 3.0
    horizon_minutes: int = 15


def load_model_and_features(config: LiveConfig) -> Tuple[lgb.Booster, List[str]]:
    model = lgb.Booster(model_file=str(config.model_path))
    feature_names: List[str] = [
        line.strip()
        for line in config.features_path.read_text().splitlines()
        if line.strip()
    ]
    return model, feature_names


def decide_action(
    pred: float,
    threshold: float,
    max_leverage: float,
    horizon_minutes: int,
) -> Tuple[str, float, int]:
    """
    Convert model prediction into (action, leverage, duration_minutes).

    You can later tune this mapping or make it piecewise by magnitude of pred.
    """
    if pred > threshold:
        return "BUY", max_leverage, horizon_minutes
    if pred < -threshold:
        return "SELL", max_leverage, horizon_minutes
    return "HOLD", 0.0, horizon_minutes


def kline_to_row(msg: dict) -> Tuple[pd.Timestamp, float, float, float, float, float]:
    k = msg["k"]
    close_time_ms = k["T"]
    ts = pd.to_datetime(close_time_ms, unit="ms", utc=True)
    open_ = float(k["o"])
    high = float(k["h"])
    low = float(k["l"])
    close = float(k["c"])
    volume = float(k["v"])
    return ts, open_, high, low, close, volume


async def run_live(config: LiveConfig) -> None:
    import websockets  # type: ignore[import]

    stream_name = f"{config.symbol}@kline_{config.interval}"
    url = f"{config.ws_url}/{stream_name}"

    model, feature_names = load_model_and_features(config)

    window: Deque[Tuple[pd.Timestamp, float, float, float, float, float]] = deque(
        maxlen=config.max_window
    )

    print(f"Connecting to {url} ...")
    async with websockets.connect(url) as ws:
        print("Connected. Waiting for closed klines...")
        async for message in ws:
            data = json.loads(message)

            if "k" not in data:
                continue

            k = data["k"]
            if not k.get("x"):  # only closed candle
                continue

            ts, o, h, l, c, v = kline_to_row(data)
            window.append((ts, o, h, l, c, v))

            # Build DataFrame of recent candles and compute indicators
            df = pd.DataFrame(
                list(window),
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            ).set_index("timestamp")

            df_ind = add_indicators(df)

            last_row = df_ind.iloc[-1]

            # Warmup: not enough history to compute all indicators
            if last_row.isna().any():
                print(f"{ts} - not enough history yet, skipping.")
                continue

            X_live = last_row[feature_names].to_numpy(dtype=np.float32).reshape(1, -1)
            pred = float(model.predict(X_live)[0])

            action, leverage, duration = decide_action(
                pred=pred,
                threshold=config.threshold,
                max_leverage=config.max_leverage,
                horizon_minutes=config.horizon_minutes,
            )

            print(
                f"{ts} - pred={pred:+.6f} -> action={action}, "
                f"leverage={leverage}x, duration={duration}m, price={c}"
            )


if __name__ == "__main__":
    cfg = LiveConfig()
    asyncio.run(run_live(cfg))
