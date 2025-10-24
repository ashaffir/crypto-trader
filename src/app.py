from __future__ import annotations

import asyncio
import os
from typing import Dict

from loguru import logger

from src.config import load_app_config
from src.collector import SpotCollector, FuturesCollector
from src.features import FeatureEngine
from src.logger import ParquetLogbook
from src.control import Control
from src.runtime_config import RuntimeConfigManager
from src.utils.llm_client import LLMClient, LLMConfig
from src.positions import PositionStore
from src.trading import TradingEngine, TraderSettings, load_trader_settings
from src.broker import ExecutionSettings, PaperBroker, BinanceBrokerSkeleton


async def pipeline() -> None:
    cfg = load_app_config()
    out_dir = cfg.storage.get("logbook_dir", "data/logbook")
    os.makedirs(out_dir, exist_ok=True)

    queue: asyncio.Queue = asyncio.Queue(maxsize=10_000)

    # Track current symbols from config; can be hot-reloaded via runtime overrides
    current_symbols: list[str] = list(cfg.symbols)

    # Choose collector based on market mode
    collector = (
        FuturesCollector(current_symbols, vars(cfg.streams), queue)
        if str(getattr(cfg, "market", "spot")).lower() == "futures"
        else SpotCollector(current_symbols, vars(cfg.streams), queue)
    )
    features = FeatureEngine(
        symbols=current_symbols,
        vol_window_s=cfg.features.vol_1s,
        delta_window_s=cfg.features.delta_1s,
        ma_windows=cfg.features.ma,
    )
    logbook = ParquetLogbook(out_dir)
    control = Control()
    runtime_cfg = RuntimeConfigManager()
    # Trading components
    position_store = PositionStore()
    # Execution/broker setup (safe defaults)
    exec_settings = ExecutionSettings.from_overrides(None)
    broker = PaperBroker(position_store)
    engine = TradingEngine(position_store, TraderSettings(), broker)

    # ---- LLM Recommender State ----
    llm_client: LLMClient | None = None
    llm_active_name: str | None = None
    llm_window_s: int = 30
    llm_refresh_s: int = 5

    def _apply_llm_timing(overrides: Dict | None) -> None:
        nonlocal llm_window_s, llm_refresh_s
        if overrides is None:
            return
        llm = overrides.get("llm") if isinstance(overrides, dict) else None
        if not isinstance(llm, dict):
            return
        try:
            llm_window_s = int(llm.get("window_seconds", llm_window_s) or llm_window_s)
        except Exception:
            pass
        try:
            llm_refresh_s = int(
                llm.get("refresh_seconds", llm_refresh_s) or llm_refresh_s
            )
        except Exception:
            pass

    def _rebuild_llm_client_from_llm_configs(initial: bool = False) -> None:
        nonlocal llm_client, llm_active_name
        try:
            changed, doc = runtime_cfg.load_llm_configs_if_changed()
            if not changed and not initial:
                return
            active = RuntimeConfigManager.get_active_llm_config_from(doc)
            if not isinstance(active, dict):
                return
            name = active.get("name") if isinstance(active.get("name"), str) else None
            conf = LLMConfig(
                base_url=str(active.get("base_url") or ""),
                provider=(active.get("provider") or None),
                api_key=(active.get("api_key") or None),
                model=(active.get("model") or None),
                system_prompt=(active.get("system_prompt") or None),
                user_template=(active.get("user_template") or None),
            )
            if llm_active_name == name and llm_client is not None:
                return
            if llm_client is not None:
                try:
                    import anyio

                    anyio.from_thread.run(llm_client.aclose)
                except Exception:
                    pass
            llm_client = LLMClient(conf)
            # If debug_save_request enabled in runtime_config.json, set path
            try:
                overrides = runtime_cfg.read() or {}
                llm_section = (
                    overrides.get("llm") if isinstance(overrides, dict) else None
                )
                if isinstance(llm_section, dict) and llm_section.get(
                    "debug_save_request"
                ):
                    import os

                    base_dir = runtime_cfg.paths.base_dir
                    os.makedirs(base_dir, exist_ok=True)
                    llm_client.set_debug_save_path(
                        os.path.join(base_dir, "llm_last_request.json")
                    )
                else:
                    llm_client.set_debug_save_path(None)
            except Exception:
                pass
            llm_active_name = name
            logger.info("LLM client ready: from_llm_configs.json")
        except Exception as e:
            logger.warning(f"Failed to load llm_configs.json: {e}")

    def _apply_llm_debug_setting() -> None:
        """Enable/disable saving last LLM req/resp based on runtime overrides."""
        nonlocal llm_client
        if llm_client is None:
            return
        try:
            overrides = runtime_cfg.read() or {}
            llm_section = overrides.get("llm") if isinstance(overrides, dict) else None
            if isinstance(llm_section, dict) and llm_section.get("debug_save_request"):
                import os

                base_dir = runtime_cfg.paths.base_dir
                os.makedirs(base_dir, exist_ok=True)
                llm_client.set_debug_save_path(
                    os.path.join(base_dir, "llm_last_request.json")
                )
            else:
                llm_client.set_debug_save_path(None)
        except Exception:
            # Never crash the loop on debug setting updates
            pass

    async def llm_loop() -> None:
        nonlocal llm_client, llm_window_s, llm_refresh_s
        # Initialize from llm_configs.json before entering loop
        _rebuild_llm_client_from_llm_configs(initial=True)
        while True:
            try:
                # Rebuild client if llm_configs.json changed
                _rebuild_llm_client_from_llm_configs()
                if llm_client is None:
                    await asyncio.sleep(1)
                    continue
                # Apply debug save toggle live from runtime_config.json
                _apply_llm_debug_setting()
                # Refresh trader settings from runtime overrides
                try:
                    _changed, _ovr = runtime_cfg.load_if_changed()
                    settings = load_trader_settings(_ovr)
                    engine.update_settings(settings)
                    # Rebuild broker if execution settings changed
                    try:
                        new_exec = ExecutionSettings.from_overrides(_ovr)
                        nonlocal exec_settings, broker
                        if new_exec != exec_settings:
                            exec_settings = new_exec
                            if exec_settings.mode == "live":
                                broker = BinanceBrokerSkeleton(
                                    position_store, exec_settings
                                )
                            else:
                                broker = PaperBroker(position_store)
                            engine.broker = broker
                            logger.info(
                                f"Broker switched: mode={exec_settings.mode}, venue={exec_settings.venue}, network={exec_settings.network}"
                            )
                    except Exception:
                        pass
                except Exception:
                    pass
                # Aggregate per-symbol summaries and query LLM once per refresh
                summaries = []
                for sym in current_symbols:
                    summary = features.summarize_window(sym, window_s=llm_window_s)
                    if summary and summary.get("count", 0) > 0:
                        summaries.append(summary)
                if summaries:
                    variables = {
                        "symbols": list(current_symbols),
                        "window_seconds": llm_window_s,
                        "DATA_WINDOW": summaries,
                    }
                    recs = await llm_client.generate(variables)
                    if recs:
                        for rec in recs:
                            try:
                                asset = str(rec.get("asset") or "").upper()
                                # Fallback to first summary symbol if asset missing
                                if not asset and summaries:
                                    asset = str(
                                        summaries[0].get("symbol") or ""
                                    ).upper()
                                if not asset:
                                    continue
                                direction_raw = str(rec.get("direction")).lower()
                                direction = (
                                    "buy"
                                    if direction_raw in ("buy", "long")
                                    else "sell"
                                )
                                leverage = int(rec.get("leverage") or 1)
                                # Optional confidence from LLM (0..1)
                                conf_val = rec.get("confidence")
                                try:
                                    confidence = (
                                        float(conf_val)
                                        if conf_val is not None
                                        else None
                                    )
                                    if confidence is not None:
                                        if confidence < 0.0 or confidence > 1.0:
                                            confidence = None
                                except Exception:
                                    confidence = None
                                now_ms = int(__import__("time").time() * 1000)
                                logbook.append_trade_recommendation(
                                    [
                                        {
                                            "ts_ms": now_ms,
                                            "symbol": asset,
                                            "asset": asset,
                                            "direction": direction,
                                            "leverage": leverage,
                                            # Persist model used for this recommendation
                                            **(
                                                {"llm_model": str(llm_client.cfg.model)}
                                                if getattr(llm_client, "cfg", None)
                                                and getattr(
                                                    llm_client.cfg, "model", None
                                                )
                                                else {}
                                            ),
                                            **(
                                                {"confidence": confidence}
                                                if confidence is not None
                                                else {}
                                            ),
                                            "source": "llm",
                                        }
                                    ]
                                )

                                # ---- Decision Flow ----
                                # 1) Close existing if inverse or TP/SL
                                price_info = None
                                try:
                                    # Prefer the specific summary for symbol if available
                                    price_info = next(
                                        (
                                            s
                                            for s in summaries
                                            if s.get("symbol") == asset
                                        ),
                                        None,
                                    )
                                    # Map last_mid to mid for engine convenience
                                    if (
                                        isinstance(price_info, dict)
                                        and "last_mid" in price_info
                                    ):
                                        price_info = {
                                            "mid": price_info.get("last_mid"),
                                            "last_px": None,
                                        }
                                except Exception:
                                    price_info = None

                                engine.maybe_close_on_inverse_or_tp_sl(
                                    symbol=asset,
                                    recommendation_direction=direction,
                                    confidence=confidence,
                                    ts_ms=now_ms,
                                    price_info=price_info,
                                )

                                # 2) If no open or slot available and confidence ok -> open
                                engine.maybe_open_from_recommendation(
                                    symbol=asset,
                                    direction=direction,
                                    leverage=leverage,
                                    confidence=confidence,
                                    ts_ms=now_ms,
                                    price_info=price_info,
                                    llm_model=str(getattr(llm_client.cfg, "model", "")),
                                )
                            except Exception:
                                # Ignore malformed recs
                                pass
            except Exception as e:
                logger.warning(f"LLM loop error: {e}")
            finally:
                await asyncio.sleep(max(1, int(llm_refresh_s)))

    async def consumer() -> None:
        batch_snapshots: list[Dict] = []
        last_flush_ts_ms: int | None = None
        while True:
            msg = await queue.get()
            snap = features.on_message(msg)
            if not snap:
                queue.task_done()
                continue

            # Append to market_snapshot batch
            batch_snapshots.append(snap)

            # Flush periodically based on ts or batch size
            ts_ms = snap.get("ts_ms") or last_flush_ts_ms
            should_flush = False
            if ts_ms is not None and last_flush_ts_ms is not None:
                should_flush = (ts_ms - last_flush_ts_ms) >= 1000
            if last_flush_ts_ms is None:
                last_flush_ts_ms = ts_ms

            if should_flush or len(batch_snapshots) >= 2000:
                try:
                    if batch_snapshots:
                        # Group by symbol to avoid cross-partition contamination
                        snaps_by_sym: Dict[str, list[Dict]] = {}
                        for r in batch_snapshots:
                            s = str(r.get("symbol") or "").upper()
                            if not s:
                                continue
                            snaps_by_sym.setdefault(s, []).append(r)
                        for _sym, rows in snaps_by_sym.items():
                            logbook.append_market_snapshot(rows)
                        batch_snapshots.clear()
                except Exception as e:
                    logger.exception(f"Logbook write failed: {e}")
                finally:
                    last_flush_ts_ms = ts_ms

            queue.task_done()

    # Collector task is managed separately to allow restart when symbols change
    collector_task = asyncio.create_task(collector.run(), name="collector")
    tasks = [
        asyncio.create_task(consumer(), name="consumer"),
        asyncio.create_task(llm_loop(), name="llm"),
    ]

    try:
        # Periodically write a heartbeat while running
        while True:
            hb = {
                "status": "running",
                "queue_size": queue.qsize(),
                "symbols": current_symbols,
            }
            control.write_status(hb)

            # Hot-reload runtime overrides if changed (timings, thresholds, horizons)
            try:
                changed, overrides = runtime_cfg.load_if_changed()
                if changed and overrides:
                    _apply_llm_timing(overrides)

                    # Handle tracked symbols change
                    try:
                        new_syms = overrides.get("symbols")
                        if isinstance(new_syms, list):
                            norm = [str(s).upper() for s in new_syms if s]
                            if norm and norm != current_symbols:
                                # Restart collector and rebuild features atomically
                                current_symbols = norm
                                try:
                                    await collector.stop()
                                except Exception:
                                    pass
                                try:
                                    collector_task.cancel()
                                except Exception:
                                    pass
                                # Recreate collector preserving market mode
                                market = str(getattr(cfg, "market", "spot")).lower()
                                collector = (
                                    FuturesCollector(
                                        current_symbols, vars(cfg.streams), queue
                                    )
                                    if market == "futures"
                                    else SpotCollector(
                                        current_symbols, vars(cfg.streams), queue
                                    )
                                )
                                features = FeatureEngine(
                                    symbols=current_symbols,
                                    vol_window_s=cfg.features.vol_1s,
                                    delta_window_s=cfg.features.delta_1s,
                                    ma_windows=cfg.features.ma,
                                )
                                collector_task = asyncio.create_task(
                                    collector.run(), name="collector"
                                )
                    except Exception:
                        # Never crash hot-reload loop on symbol update
                        pass
            except Exception:
                # Swallow to avoid impacting main loop
                pass
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        await collector.stop()
        try:
            collector_task.cancel()
        except Exception:
            pass
        for t in tasks:
            t.cancel()


def main() -> None:
    try:
        asyncio.run(pipeline())
    except KeyboardInterrupt:
        logger.info("Shutting down")


if __name__ == "__main__":
    main()
