"""to runt this script, use:
python ml/confusion_matrix.py \
    --data_path data/supervised_dataset.parquet \
    --model_path models/lightgbm_classifier.txt \
    --features_path models/lightgbm_classifier.features.txt
    
This will print the confusion matrix and classification report for the trained model on the dataset."""

import numpy as np
import pandas as pd
import lightgbm as lgb
import argparse
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

argparser = argparse.ArgumentParser(
    description="Evaluate LightGBM classifier on direction prediction."
)

argparser.add_argument(
    "--data_path",
    type=Path,
    required=True,
    help="Path to the dataset parquet file.",
)
argparser.add_argument(
    "--model_path",
    type=Path,
    required=True,
    help="Path to the trained LightGBM model file.",
)
argparser.add_argument(
    "--features_path",
    type=Path,
    required=True,
    help="Path to the feature names text file.",
)
args = argparser.parse_args()

data_path = args.data_path
model_path = args.model_path
features_path = args.features_path

df = pd.read_parquet(data_path)
feature_names = [l.strip() for l in features_path.read_text().splitlines() if l.strip()]

X = df[feature_names].to_numpy(dtype=np.float32)
y_dir = df["direction"].to_numpy()  # -1, 0, 1

# Support both string labels ("down"/"flat"/"up") and numeric 0/1/2
label_map = {"down": 0, "flat": 1, "up": 2}
if df["direction"].dtype == "O":  # object -> likely strings
    y_true = np.vectorize(label_map.get)(y_dir)
else:
    # already numeric 0/1/2
    y_true = y_dir.astype(int)

if np.isnan(y_true).any():
    raise ValueError(f"Unmapped labels found in direction: {set(y_dir)}")

model = lgb.Booster(model_file=str(model_path))
proba = model.predict(X)  # (n, 3)
y_pred = proba.argmax(axis=1)

print("Class counts in labels (0=down,1=flat,2=up):", np.bincount(y_true))
print("Class counts in predictions (0=down,1=flat,2=up):", np.bincount(y_pred))

print("\nConfusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y_true, y_pred))

print("\nClassification report:")
print(classification_report(y_true, y_pred, digits=3))
