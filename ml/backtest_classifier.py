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
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_parquet(data_path)

    if "future_ret" not in df.columns or "direction" not in df.columns:
        raise ValueError("Dataset must contain 'future_ret' and 'direction' columns.")

    feature_names: List[str] = [
        line.strip() for line in features_path.read_text().splitlines() if line.strip()
    ]

    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Features missing from dataset: {missing}")

    X = df[feature_names].to_numpy(dtype=np.float32)

    # continuous return for PnL
    y_ret = df["future_ret"].to_numpy(dtype=np.float32)

    # mapped labels for accuracy stats: {-1,0,1} -> {0,1,2}
    label_map = {-1: 0, 0: 1, 1: 2}
    y_dir = df["direction"].map(label_map).to_numpy(dtype=int)

    return df, X, y_ret, y_dir


def proba_policy(
    proba: np.ndarray,
    p_threshold: float,
    max_leverage: float,
) -> float:
    """
    proba = [P(down), P(flat), P(up)] (matching label mapping 0,1,2)

    - If max(P(up), P(down)) < p_threshold -> no trade
    - Else go long/short with max_leverage in direction of higher prob
    """
    p_down = float(proba[0])
    p_flat = float(proba[1])
    p_up = float(proba[2])

    best_p = max(p_up, p_down)
    if best_p < p_threshold:
        return 0.0

    if p_up > p_down:
        return max_leverage
    else:
        return -max_leverage


def backtest_classifier(
    df: pd.DataFrame,
    model: lgb.Booster,
    X: np.ndarray,
    y_ret: np.ndarray,
    y_dir: np.ndarray,
    p_threshold: float,
    max_leverage: float,
    transaction_cost_bps: float,
    horizon_minutes: int,
    vol_percentile: float,
    tp: float,
    sl: float,
) -> None:
    n = len(df)
    if n < 3:
        raise ValueError("Dataset too small for backtest.")

    proba_preds = model.predict(X)  # shape (n, 3)

    trend_filter = True

    # --- volatility filter setup ---
    vol = df["vol_15m"].to_numpy() if "vol_15m" in df.columns else None
    vol_thr = None
    if vol is not None and vol_percentile > 0:
        vol_thr = np.nanpercentile(vol, vol_percentile)

    positions = np.zeros(n, dtype=np.float32)
    if trend_filter:
        close = df["close"].to_numpy()
        ema50 = df["ema_50"].to_numpy()

        for t in range(1, n):
            raw_pos = proba_policy(proba_preds[t - 1], p_threshold, max_leverage)

            # default: no position if we don't have EMA yet
            if np.isnan(ema50[t]) or np.isnan(close[t]):
                positions[t] = 0.0
                continue

            # trend filter:
            # - if price > EMA50 → only allow longs
            # - if price < EMA50 → only allow shorts
            pos = raw_pos

            # if countertrend, cut size instead of killing it
            if close[t] > ema50[t] and raw_pos < 0:
                pos *= 0.5  # or 0.25
            elif close[t] < ema50[t] and raw_pos > 0:
                pos *= 0.5

            # volatility filter: only trade when vol is high enough
            if vol_thr is not None and not np.isnan(vol[t]) and vol[t] < vol_thr:
                pos = 0.0

            positions[t] = pos

    else:
        for t in range(1, n):
            positions[t] = proba_policy(proba_preds[t - 1], p_threshold, max_leverage)

    cost_per_unit = transaction_cost_bps / 10000.0
    costs = np.zeros(n, dtype=np.float32)
    for t in range(1, n):
        costs[t] = cost_per_unit * abs(positions[t] - positions[t - 1])

    # --- TP/SL cap on the realized future_ret label ---
    # NOTE: this is an approximation, since we only know aggregate future_ret.
    y_eff = y_ret.copy()
    # take profit: cap positive returns
    if tp is not None and tp > 0:
        y_eff = np.minimum(y_eff, tp)
    # stop loss: cap negative returns
    if sl is not None and sl < 0:
        y_eff = np.maximum(y_eff, sl)

    pnl = positions * y_ret - costs
    equity = 1.0 + np.cumsum(pnl)

    valid_mask = ~np.isnan(pnl)
    pnl_valid = pnl[valid_mask]
    mean_pnl = pnl_valid.mean()
    std_pnl = pnl_valid.std(ddof=1) if len(pnl_valid) > 1 else float("nan")

    steps_per_day = int(round(24 * 60 / horizon_minutes))
    annual_factor = np.sqrt(steps_per_day * 365)
    sharpe = (mean_pnl / std_pnl * annual_factor) if std_pnl > 0 else float("nan")

    trade_change = np.abs(np.diff(positions)) > 1e-6
    n_trades = int(np.sum(trade_change))
    trade_indices = np.where(trade_change)[0] + 1
    trade_pnls = pnl[trade_indices]
    wins = np.sum(trade_pnls > 0)
    win_rate = wins / len(trade_pnls) if len(trade_pnls) > 0 else float("nan")

    # directional accuracy only when in a trade
    dir_pred = proba_preds.argmax(axis=1)
    trade_mask = positions != 0
    dir_true_trades = y_dir[trade_mask]
    dir_pred_trades = dir_pred[trade_mask]
    dir_acc = (
        np.mean(dir_true_trades == dir_pred_trades)
        if len(dir_true_trades) > 0
        else float("nan")
    )

    print("Classifier-based backtest")
    print("=========================")
    print(f"Start: {df.index[0]}  End: {df.index[-1]}")
    print(f"Final equity: {equity[-1]:.4f} (from 1.0)")
    print(f"Total steps: {n}")
    print(f"Mean per-step PnL: {mean_pnl:.6f}")
    print(f"Std per-step PnL: {std_pnl:.6f}")
    print(f"Annualized Sharpe (approx): {sharpe:.3f}")
    print(f"Trades: {n_trades}")
    print(f"Win rate: {win_rate:.2%}")
    print(f"Directional accuracy on traded steps: {dir_acc:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest LightGBM direction classifier on historical dataset."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/btcusdt_1m_2025_q3.parquet",
        help="Path to dataset parquet file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/lightgbm_direction_btcusdt_1m.txt",
        help="Path to LightGBM classifier model file.",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="models/lightgbm_direction_btcusdt_1m.features.txt",
        help="Path to feature name list (one per line).",
    )
    parser.add_argument(
        "--p_threshold",
        type=float,
        default=0.55,
        help="Min(max(P(up), P(down))) to open a trade.",
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
        help="Label horizon in minutes (for Sharpe scaling).",
    )

    parser.add_argument(
        "--vol_percentile",
        type=float,
        default=60.0,
        help="Only trade when vol_15m is above this percentile (0 to disable).",
    )
    parser.add_argument(
        "--tp",
        type=float,
        default=0.004,
        help="Take-profit cap on future_ret (e.g. 0.004 ~ +0.4%%, <=0 to disable).",
    )
    parser.add_argument(
        "--sl",
        type=float,
        default=-0.003,
        help="Stop-loss cap on future_ret (e.g. -0.003 ~ -0.3%%, >=0 to disable).",
    )

    args = parser.parse_args()

    data_path = Path(args.data)
    model_path = Path(args.model)
    features_path = Path(args.features)

    print(f"Loading dataset from {data_path} ...")
    df, X, y_ret, y_dir = load_dataset_and_features(data_path, features_path)
    print(f"Dataset length: {len(df)}")

    print(f"Loading model from {model_path} ...")
    model = load_model(model_path)

    backtest_classifier(
        df=df,
        model=model,
        X=X,
        y_ret=y_ret,
        y_dir=y_dir,
        p_threshold=args.p_threshold,
        max_leverage=args.max_leverage,
        transaction_cost_bps=args.transaction_cost_bps,
        horizon_minutes=args.horizon_minutes,
        vol_percentile=args.vol_percentile,
        tp=args.tp if args.tp > 0 else None,
        sl=args.sl if args.sl < 0 else None,
    )


if __name__ == "__main__":
    main()
