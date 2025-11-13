# CRYPTO-TRADER

Concise user manual for running the bot, UI, and live execution.

## Requirements
- Python 3.11+
- Install deps: `pip install -r requirements.txt`

## Start services
1) Bot (data + engine):
```bash
# Option A: recommended helper (foreground)
./manage.sh dev bot

# Option B: direct Python (foreground)
export PYTHONPATH=$(pwd)
python -m src.supervisor
```

2) UI (in a new terminal):
```bash
# Option A: recommended helper
./manage.sh dev ui --port 8501

# Option B: direct Streamlit
export PYTHONPATH=$(pwd)
streamlit run ui/Home.py --server.port=8501 --server.address=0.0.0.0
```

## Bot lifecycle (start, restart, stop)

Local (foreground processes):
```bash
# Start bot (foreground)
./manage.sh dev bot

# Restart bot (Ctrl-C to stop, then start again)
#   Press Ctrl-C to stop the running bot
./manage.sh dev bot

# Stop bot (foreground)
#   Press Ctrl-C in the terminal where the bot is running
```

Docker (background services):
```bash
# Start all services in background
./manage.sh up

# Start in foreground (stream logs in terminal)
./manage.sh up fg

# Restart only the bot service
./manage.sh restart bot

# Stop all services
./manage.sh down

# View logs (last 100 lines; follow with -f)
./manage.sh logs bot -f
```

Alternative with docker compose directly:
```bash
docker compose up -d            # start in background
docker compose restart bot      # restart bot only
docker compose logs -f bot      # follow bot logs
docker compose down             # stop all
```

## Configure in the UI
- Tracked Symbols: e.g., `BTCUSDT`.
- Market Mode: `spot` or `futures` (affects collector, fees, and execution venue).
- Trader Settings: position limits, TP/SL, trailing SL, leverage caps.
- Execution Mode (right column):
  - Mode: `paper` or `live`
  - Network: `testnet` or `mainnet`
  - API Key and Secret: your Binance keys
  - Save Execution Settings.

Notes:
- Live orders only send when Mode=`live` and keys are set. Venue derives from Market Mode.
- All settings persist to `data/control/runtime_config.json` and hot-reload within ~1s.

## Tiny Order Test (Spot)
Use this to verify keys and connectivity with a tiny market order.
- In Settings → “Tiny Order Test (Spot)”, enter:
  - Symbol (e.g., `BTCUSDT`)
  - Side `BUY`/`SELL`
  - Quote Amount (e.g., `5` for 5 USDT)
  - Confirm (extra prompt on mainnet) → Place Tiny Order
- Shows full Binance response or error details (e.g., insufficient balance, min notional).

CLI alternative:
```bash
export PYTHONPATH=$(pwd)
python -m src.tiny_order BTCUSDT BUY 5
```

## Files and data
- Runtime control: `data/control/runtime_config.json`
- Logbook (parquet): `data/logbook/`
- Positions DB (SQLite): `data/control/positions.sqlite`

## Safety checklist for live
- Mode=`live`, Network as intended (`testnet` recommended first)
- Market Mode matches your intended venue (`spot` for tiny test)
- API key permissions: enable Spot & Margin Trading (or Futures for futures)
- Correct symbol, amounts respect Binance minimums
- System clock reasonably synced (we also fetch server time)

