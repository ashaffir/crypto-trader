from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, classification_report

data_path = Path("data/btcusdt_1m_2025_q3.parquet")
model_path = Path("models/lightgbm_direction_btcusdt_1m.txt")
features_path = Path("models/lightgbm_direction_btcusdt_1m.features.txt")

df = pd.read_parquet(data_path)
feature_names = [l.strip() for l in features_path.read_text().splitlines() if l.strip()]

X = df[feature_names].to_numpy(dtype=np.float32)
y_dir = df["direction"].to_numpy()  # -1, 0, 1

label_map = {-1: 0, 0: 1, 1: 2}
y_true = np.vectorize(label_map.get)(y_dir)

model = lgb.Booster(model_file=str(model_path))
proba = model.predict(X)  # (n, 3)
y_pred = proba.argmax(axis=1)

print("Class counts in labels (0=down,1=flat,2=up):", np.bincount(y_true))
print("Class counts in predictions (0=down,1=flat,2=up):", np.bincount(y_pred))

print("\nConfusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y_true, y_pred))

print("\nClassification report:")
print(classification_report(y_true, y_pred, digits=3))
