# CRYPTO-TRADER

## Live controls

- UI writes desired bot state to `data/control/bot_control.json`.
- Strategy parameters can be changed in the Streamlit sidebar:
  - Rules: momentum and mean-reversion toggles
  - Thresholds: `imbalance`, `max_spread_bps`
  - Horizons: `scalp` (evaluation window, seconds), `ttl_s`
- Changes are persisted to `data/control/runtime_config.json` and hot-reloaded by the running bot within ~1s.

## Quickstart

1) Start the bot: `python -m src.app`

2) Launch UI: `streamlit run ui/ui_app.py`

3) Use the sidebar to toggle the bot and tune strategy parameters. Recent signals and outcomes will appear as data arrives.

## Strategy Overview

See `docs/strategy.md` for a description of the live momentum strategy, including equations for computed features, the signal emission rule, a flow schematic, and practical tuning advice.
