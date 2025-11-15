from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


@dataclass
class FetchConfig:
    symbol: str = "BTC/USDT"
    timeframe: str = "1m"
    start: str = "2023-01-01"
    end: str = "2023-03-01"
    output_path: Path = Path("data/raw_ohlcv.parquet")


def fetch_ohlcv(config: FetchConfig) -> pd.DataFrame:
    """
    Fetch OHLCV data from Binance using ccxt.

    This function requires `pip install ccxt`.
    """
    import ccxt  # type: ignore[import]

    exchange = ccxt.binance({"enableRateLimit": True})

    start_ts = int(pd.Timestamp(config.start, tz="UTC").timestamp() * 1000)
    end_ts = int(pd.Timestamp(config.end, tz="UTC").timestamp() * 1000)

    all_rows: List[List[float]] = []
    cursor = start_ts

    while True:
        ohlcv = exchange.fetch_ohlcv(
            config.symbol,
            timeframe=config.timeframe,
            since=cursor,
            limit=1000,
        )

        if not ohlcv:
            break

        all_rows.extend(ohlcv)
        last_ts = ohlcv[-1][0]

        # Stop if we've reached or passed the end timestamp
        if last_ts >= end_ts:
            break

        # Move cursor to last timestamp + 1 ms to avoid duplicates
        cursor = last_ts + 1

    if not all_rows:
        raise RuntimeError(
            "No data fetched from Binance. Check symbol/timeframe/date range."
        )

    df = pd.DataFrame(
        all_rows,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()

    # Trim exactly to requested range
    df = df.loc[config.start : config.end]

    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a set of technical indicators and return a new DataFrame.

    Requires `pip install pandas_ta`.
    """
    import pandas_ta as ta  # type: ignore[import]

    df = df.copy()

    # Basic log-returns
    df["ret_1m"] = np.log(df["close"]).diff()
    df["ret_5m"] = np.log(df["close"]).diff(5)
    df["ret_15m"] = np.log(df["close"]).diff(15)

    # Rolling volatility
    df["vol_15m"] = df["ret_1m"].rolling(15).std()
    df["vol_60m"] = df["ret_1m"].rolling(60).std()

    # EMAs
    df["ema_10"] = ta.ema(df["close"], length=10)
    df["ema_20"] = ta.ema(df["close"], length=20)
    df["ema_50"] = ta.ema(df["close"], length=50)

    # MACD
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macd"] = macd.iloc[:, 0]
    df["macd_signal"] = macd.iloc[:, 1]
    df["macd_hist"] = macd.iloc[:, 2]

    # RSI
    df["rsi_14"] = ta.rsi(df["close"], length=14)

    # Bollinger Bands (column names differ between versions → detect)
    bb = ta.bbands(df["close"], length=20, std=2.0)

    def pick_bb(col_key: str) -> pd.Series:
        candidates = [c for c in bb.columns if col_key in c]
        if not candidates:
            raise ValueError(
                f"Could not find Bollinger column containing '{col_key}'. "
                f"Available columns: {list(bb.columns)}"
            )
        return bb[candidates[0]]

    df["bb_low"] = pick_bb("BBL")  # lower band
    df["bb_mid"] = pick_bb("BBM")  # middle band
    df["bb_high"] = pick_bb("BBU")  # upper band
    df["bb_width"] = (df["bb_high"] - df["bb_low"]) / df["bb_mid"]

    # ATR
    df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    # Volume features
    df["vol_ma_20"] = df["volume"].rolling(20).mean()
    df["vol_ma_ratio"] = df["volume"] / df["vol_ma_20"]

    return df


def create_supervised(
    df: pd.DataFrame,
    horizon: int = 15,
    thr: float = 0.0007,
) -> pd.DataFrame:
    """
    Create supervised dataset for *direction* prediction.

    - future_ret: future log-return over `horizon` minutes  (kept for analysis)
    - direction:  1 = up, 0 = flat/no-trade, -1 = down
    """
    df = df.copy()

    # Future log-return
    df["future_ret"] = np.log(df["close"].shift(-horizon) / df["close"])

    # Direction label
    df["direction"] = 0  # flat / no-trade
    df.loc[df["future_ret"] > thr, "direction"] = 1  # up
    df.loc[df["future_ret"] < -thr, "direction"] = -1  # down

    df = df.dropna()

    return df


def create_supervised_size(
    df: pd.DataFrame,
    horizon: int = 15,
) -> pd.DataFrame:
    """
    Create a supervised learning dataset.

    Label: future log-return over `horizon` minutes.
    """
    df = df.copy()

    df["future_ret"] = np.log(df["close"].shift(-horizon) / df["close"])

    # Drop rows with NaNs from indicators and labeling
    df = df.dropna()

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Binance OHLCV and prepare dataset."
    )
    parser.add_argument(
        "--symbol", type=str, default="BTC/USDT", help="Symbol, e.g. BTC/USDT"
    )
    parser.add_argument(
        "--timeframe", type=str, default="1m", help="Timeframe, e.g. 1m, 5m"
    )
    parser.add_argument(
        "--start", type=str, default="2023-01-01", help="Start date (UTC, ISO)"
    )
    parser.add_argument(
        "--end", type=str, default="2023-03-01", help="End date (UTC, ISO)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/dataset.parquet",
        help="Path to save the prepared dataset.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=15,
        help="Prediction horizon in minutes.",
    )
    parser.add_argument(
        "--thr",
        type=float,
        default=0.0007,
        help="Threshold for direction labeling.",
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = FetchConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start=args.start,
        end=args.end,
        output_path=output_path,
    )

    print(
        f"Fetching OHLCV for {config.symbol} {config.timeframe} from {config.start} to {config.end}..."
    )
    ohlcv = fetch_ohlcv(config)
    print(f"Fetched {len(ohlcv)} rows.")

    print("Adding technical indicators...")
    df_ind = add_indicators(ohlcv)

    print(f"Creating supervised dataset with horizon={args.horizon} minutes...")
    dataset = create_supervised(df_ind, horizon=args.horizon, thr=args.thr)
    print(f"Final dataset rows: {len(dataset)}")

    print(f"Saving dataset to {output_path} ...")
    dataset.to_parquet(output_path)
    print("Done.")


if __name__ == "__main__":
    main()
