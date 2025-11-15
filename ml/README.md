# ML model for technical analysis

The computation is done on Google Colab.

### Prepare Data
uv run prepare_data.py \
  --symbol BTC/USDT \
  --timeframe 1m \
  --start 2025-06-01 \
  --end 2025-11-01 \
  --horizon 15 \
  --thr 0.0007 \
  --output data/btcusdt_1m_2025_q3.parquet


### Train Model
python train_model.py --data data/btcusdt_1m_2023_q1.parquet --model_output models/lightgbm_btcusdt_1m.txt

### Backtesting
python backtest_classifier.py \
  --data data/btcusdt_1m_2025_q3.parquet \
  --model models/lightgbm_direction_btcusdt_1m.txt \
  --features models/lightgbm_direction_btcusdt_1m.features.txt \
  --p_threshold 0.50 \
  --p_down_threshold 0.45 \
  --max_leverage 3.0 \
  --vol_percentile 60 \
  --tp 0.004 \
  --sl -0.003

### Realtime trading simulator
python realtime_trading.py

