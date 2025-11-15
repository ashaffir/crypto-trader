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
* With thresholds sweep
  !python backtest_model.py \
  --data data/btcusdt_1m_2025_q3.parquet \
  --model models/lightgbm_btcusdt_1m.txt \
  --features models/lightgbm_btcusdt_1m.features.txt \
  --sweep \
  --threshold 0.0005 \
  --max_leverage 3.0 \
  --transaction_cost_bps 1.0 \
  --horizon_minutes 15

* Without threhold sweep
python backtest_model.py \
  --data data/btcusdt_1m_2025_q3.parquet \
  --model models/lightgbm_btcusdt_1m.txt \
  --features models/lightgbm_btcusdt_1m.features.txt \
  --threshold 0.0005 \
  --max_leverage 3.0 \
  --transaction_cost_bps 1.0 \
  --horizon_minutes 15

### Realtime trading simulator
python realtime_trading.py

