from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error


def time_series_split(
    X: np.ndarray,
    y: np.ndarray,
    train_size: float = 0.7,
    val_size: float = 0.15,
) -> Tuple[np.ndarray, ...]:
    """
    Chronological split: [train | val | test].
    """
    n = len(X)
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_dataset(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(path)
    if "future_ret" not in df.columns:
        raise ValueError("Dataset must contain a 'future_ret' column as target.")
    y = df["future_ret"]
    X = df.drop(columns=["future_ret"])
    return X, y


def train_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    model_output: Path,
) -> None:
    feature_names = list(X.columns)

    X_values = X.to_numpy(dtype=np.float32)
    y_values = y.to_numpy(dtype=np.float32)

    X_train, y_train, X_val, y_val, X_test, y_test = time_series_split(
        X_values, y_values
    )

    train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    val_set = lgb.Dataset(X_val, label=y_val, feature_name=feature_names)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 50,
        "verbose": -1,
    }

    print("Training LightGBM model...")
    model = lgb.train(
        params,
        train_set,
        num_boost_round=2000,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=50),
        ],
    )

    print(f"Best iteration: {model.best_iteration}")
    print("Evaluating on test set...")

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    corr = np.corrcoef(y_test, y_pred)[0, 1] if len(y_test) > 1 else np.nan

    print(f"Test RMSE: {rmse:.6f}")
    print(f"Test correlation: {corr:.4f}")

    model_output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_output))
    feature_path = model_output.with_suffix(".features.txt")
    feature_path.write_text("\n".join(feature_names))

    print(f"Model saved to {model_output}")
    print(f"Feature list saved to {feature_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train LightGBM model on prepared dataset."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/dataset.parquet",
        help="Path to the dataset parquet file.",
    )
    parser.add_argument(
        "--model_output",
        type=str,
        default="models/lightgbm_model.txt",
        help="Path to save trained LightGBM model.",
    )

    args = parser.parse_args()

    data_path = Path(args.data)
    model_output = Path(args.model_output)

    print(f"Loading dataset from {data_path} ...")
    X, y = load_dataset(data_path)
    print(f"Dataset shape: {X.shape}, target length: {len(y)}")

    train_lightgbm(X, y, model_output)


if __name__ == "__main__":
    main()
