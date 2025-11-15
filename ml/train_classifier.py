from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report


def time_series_split(
    X: np.ndarray,
    y: np.ndarray,
    train_size: float = 0.7,
    val_size: float = 0.15,
):
    n = len(X)
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_dataset(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(path)

    if "direction" not in df.columns:
        raise ValueError("Dataset must contain 'direction' column.")

    # Target = direction; keep future_ret as a feature if you want, or drop it:
    y = df["direction"]
    X = df.drop(columns=["direction", "future_ret"])

    return X, y


def train_lightgbm_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    model_output: Path,
) -> None:
    # Support old scheme (-1,0,1) and new (0,1,2)
    if y.min() < 0:
        # old: -1,0,1  -> 0,1,2
        label_map = {-1: 0, 0: 1, 1: 2}
        y_mapped = y.map(label_map).astype(int)
    else:
        # already 0,1,2
        y_mapped = y.astype(int)

    feature_names: List[str] = list(X.columns)
    X_values = X.to_numpy(dtype=np.float32)
    y_values = y_mapped.to_numpy()

    n = len(X_values)
    train_size = 0.7
    val_size = 0.15
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)

    X_train, y_train = X_values[:train_end], y_values[:train_end]
    X_val, y_val = X_values[train_end:val_end], y_values[train_end:val_end]
    X_test, y_test = X_values[val_end:], y_values[val_end:]

    # ----- class weights via sample weights -----
    # classes: 0=down, 1=flat, 2=up
    unique, counts = np.unique(y_values, return_counts=True)
    n_classes = len(unique)
    total = len(y_values)

    # simple inverse-frequency weighting (no extra boost)
    base_weights = {c: total / (n_classes * cnt) for c, cnt in zip(unique, counts)}
    class_weight = {
        0: base_weights.get(0, 1.0),
        1: base_weights.get(1, 1.0),
        2: base_weights.get(2, 1.0),
    }

    sample_weights = np.array(
        [class_weight[int(c)] for c in y_values], dtype=np.float32
    )
    w_train = sample_weights[:train_end]
    w_val = sample_weights[train_end:val_end]
    # (we don’t need weights on test for evaluation)

    train_set = lgb.Dataset(
        X_train,
        label=y_train,
        weight=w_train,
        feature_name=feature_names,
    )
    val_set = lgb.Dataset(
        X_val,
        label=y_val,
        weight=w_val,
        feature_name=feature_names,
    )

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 50,
        "verbose": -1,
    }

    print("Training LightGBM classifier...")
    model = lgb.train(
        params,
        train_set,
        num_boost_round=500,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50),
        ],
    )

    print(f"Best iteration: {model.best_iteration}")

    # Eval on test set
    y_proba = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = y_proba.argmax(axis=1)

    from sklearn.metrics import classification_report

    print("Test classification report (labels 0=down,1=flat,2=up):")
    print(classification_report(y_test, y_pred, digits=3))

    model_output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_output))
    feature_path = model_output.with_suffix(".features.txt")
    feature_path.write_text("\n".join(feature_names))

    print(f"Model saved to {model_output}")
    print(f"Feature list saved to {feature_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train LightGBM classifier on direction labels."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/btcusdt_1m_2025_q3.parquet",
        help="Path to dataset parquet file.",
    )
    parser.add_argument(
        "--model_output",
        type=str,
        default="models/lightgbm_direction_btcusdt_1m.txt",
        help="Path to save trained LightGBM model.",
    )

    args = parser.parse_args()

    data_path = Path(args.data)
    model_output = Path(args.model_output)

    print(f"Loading dataset from {data_path} ...")
    X, y = load_dataset(data_path)
    print(f"Dataset shape: {X.shape}, target length: {len(y)}")

    train_lightgbm_classifier(X, y, model_output)


if __name__ == "__main__":
    main()
