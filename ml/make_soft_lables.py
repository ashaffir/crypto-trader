#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def add_direction_labels(df: pd.DataFrame, thr: float) -> pd.DataFrame:
    if "future_return" not in df.columns:
        raise ValueError("Input parquet must contain 'future_return' column.")

    fr = df["future_return"].values

    # 0 = down, 1 = flat, 2 = up
    direction = np.where(
        fr > thr,
        2,
        np.where(fr < -thr, 0, 1),
    )

    df = df.copy()
    df["direction"] = direction
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create softer direction labels from future_return."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input parquet (with future_return).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output parquet (with direction column).",
    )
    parser.add_argument(
        "--thr",
        type=float,
        default=0.0005,
        help="Threshold for up/down (e.g. 0.0005 = 0.05%% move).",
    )

    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    print(f"Loading {in_path} ...")
    df = pd.read_parquet(in_path)
    print(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")

    print(f"Adding direction labels with thr={args.thr} ...")
    df = add_direction_labels(df, args.thr)

    print(f"Class counts (0=down,1=flat,2=up): {np.bincount(df['direction'])}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved labeled dataset to {out_path}")


if __name__ == "__main__":
    main()
