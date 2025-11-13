from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import lightgbm as lgb


def load_model(model_path: Path) -> lgb.Booster:
    return lgb.Booster(model_file=str(model_path))


def load_dataset_and_features(
    data_path: Path,
    features_path: Path,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_parquet(data_path)
    if "future_ret" not in df.columns:
        raise ValueError("Dataset must contain 'future_ret' column as label.")

    feature_names: List[str] = [
        line.strip() for line in features_path.read_text().splitlines() if line.strip()
    ]

    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Features missing from dataset: {missing}")

    X = df[feature_names].to_numpy(dtype=np.float32)
    y = df["future_ret"].to_numpy(dtype=np.float32)
    return df, X, y


def simple_policy(pred: float, threshold: float, max_leverage: float) -> float:
    """
    Piecewise leverage: small edge -> small size, big edge -> bigger size.
    """
    a = threshold
    b = 2 * threshold

    if pred > b:
        return max_leverage
    if pred > a:
        return max_leverage * 0.5
    if pred < -b:
        return -max_leverage
    if pred < -a:
        return -max_leverage * 0.5
    return 0.0


def run_threshold_sweep(
    df: pd.DataFrame,
    model: lgb.Booster,
    X: np.ndarray,
    y: np.ndarray,
    thresholds: List[float],
    max_leverage: float,
    transaction_cost_bps: float,
    horizon_minutes: int,
) -> None:
    print("Threshold sweep:")
    print("thr\tSharpe\tWinRate\tTrades\tFinalEq")

    for thr in thresholds:
        n = len(df)
        preds = model.predict(X)

        positions = np.zeros(n, dtype=np.float32)
        for t in range(1, n):
            positions[t] = simple_policy(preds[t - 1], thr, max_leverage)

        cost_per_unit = transaction_cost_bps / 10000.0
        costs = np.zeros(n, dtype=np.float32)
        for t in range(1, n):
            costs[t] = cost_per_unit * abs(positions[t] - positions[t - 1])

        pnl = positions * y - costs
        equity = 1.0 + np.cumsum(pnl)

        valid_mask = ~np.isnan(pnl)
        pnl_valid = pnl[valid_mask]
        if len(pnl_valid) < 2:
            continue

        mean_pnl = pnl_valid.mean()
        std_pnl = pnl_valid.std(ddof=1)
        steps_per_day = int(round(24 * 60 / horizon_minutes))
        annual_factor = np.sqrt(steps_per_day * 365)
        sharpe = (mean_pnl / std_pnl * annual_factor) if std_pnl > 0 else np.nan

        trade_change = np.abs(np.diff(positions)) > 1e-6
        n_trades = int(np.sum(trade_change))
        trade_indices = np.where(trade_change)[0] + 1
        trade_pnls = pnl[trade_indices]
        wins = np.sum(trade_pnls > 0)
        win_rate = wins / len(trade_pnls) if len(trade_pnls) > 0 else np.nan

        print(f"{thr:.4g}\t{sharpe:.3f}\t{win_rate:.2%}\t{n_trades}\t{equity[-1]:.3f}")


def backtest(
    df: pd.DataFrame,
    model: lgb.Booster,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float,
    max_leverage: float,
    transaction_cost_bps: float,
    horizon_minutes: int,
) -> None:
    n = len(df)
    if n < 3:
        raise ValueError("Dataset too small for backtest.")

    preds = model.predict(X)

    # Position at time t is decided using prediction at t-1
    positions = np.zeros(n, dtype=np.float32)
    for t in range(1, n):
        positions[t] = simple_policy(preds[t - 1], threshold, max_leverage)

    # Simple transaction cost model: cost when position changes
    cost_per_unit = transaction_cost_bps / 10000.0
    costs = np.zeros(n, dtype=np.float32)
    for t in range(1, n):
        costs[t] = cost_per_unit * abs(positions[t] - positions[t - 1])

    # PnL per step uses future_ret as label
    pnl = positions * y - costs
    equity = 1.0 + np.cumsum(pnl)

    valid_mask = ~np.isnan(pnl)
    pnl_valid = pnl[valid_mask]

    mean_pnl = pnl_valid.mean()
    std_pnl = pnl_valid.std(ddof=1) if len(pnl_valid) > 1 else np.nan

    # Approx annualized Sharpe, using label horizon for scaling
    steps_per_day = int(round(24 * 60 / horizon_minutes))
    annual_factor = np.sqrt(steps_per_day * 365)
    sharpe = (mean_pnl / std_pnl * annual_factor) if std_pnl > 0 else np.nan

    # Rough trade stats: a "trade" when position changes
    trade_change = np.abs(np.diff(positions)) > 1e-6
    n_trades = int(np.sum(trade_change))
    trade_indices = np.where(trade_change)[0] + 1
    trade_pnls = pnl[trade_indices]
    wins = np.sum(trade_pnls > 0)
    win_rate = wins / len(trade_pnls) if len(trade_pnls) > 0 else np.nan

    print("Backtest results")
    print("================")
    print(f"Start: {df.index[0]}  End: {df.index[-1]}")
    print(f"Final equity: {equity[-1]:.4f} (from 1.0)")
    print(f"Total steps: {n}")
    print(f"Mean per-step PnL: {mean_pnl:.6f}")
    print(f"Std per-step PnL: {std_pnl:.6f}")
    print(f"Annualized Sharpe (approx): {sharpe:.3f}")
    print(f"Trades: {n_trades}")
    print(f"Win rate: {win_rate:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest LightGBM model on historical dataset."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/btcusdt_1m_2023_q1.parquet",
        help="Path to dataset parquet file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/lightgbm_btcusdt_1m.txt",
        help="Path to LightGBM model file.",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="models/lightgbm_btcusdt_1m.features.txt",
        help="Path to feature name list (one per line).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0005,
        help="Prediction threshold for opening a position (log-return).",
    )
    parser.add_argument(
        "--max_leverage",
        type=float,
        default=3.0,
        help="Absolute leverage when in a position.",
    )
    parser.add_argument(
        "--transaction_cost_bps",
        type=float,
        default=1.0,
        help="Round-trip transaction cost in basis points.",
    )
    parser.add_argument(
        "--horizon_minutes",
        type=int,
        default=15,
        help="Label horizon in minutes (used for Sharpe scaling).",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="If set, run a threshold sweep before the final backtest.",
    )

    args = parser.parse_args()

    data_path = Path(args.data)
    model_path = Path(args.model)
    features_path = Path(args.features)

    print(f"Loading dataset from {data_path} ...")
    df, X, y = load_dataset_and_features(data_path, features_path)
    print(f"Dataset length: {len(df)}")

    print(f"Loading model from {model_path} ...")
    model = load_model(model_path)

    if args.sweep:
        thresholds = [0.0002, 0.0003, 0.0005, 0.0008, 0.001, 0.0015]
        run_threshold_sweep(
            df=df,
            model=model,
            X=X,
            y=y,
            thresholds=thresholds,
            max_leverage=args.max_leverage,
            transaction_cost_bps=args.transaction_cost_bps,
            horizon_minutes=args.horizon_minutes,
        )
        print()

    backtest(
        df=df,
        model=model,
        X=X,
        y=y,
        threshold=args.threshold,
        max_leverage=args.max_leverage,
        transaction_cost_bps=args.transaction_cost_bps,
        horizon_minutes=args.horizon_minutes,
    )


if __name__ == "__main__":
    main()
