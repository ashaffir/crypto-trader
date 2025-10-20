from __future__ import annotations

import asyncio
import os
import uuid
from typing import Dict

from loguru import logger

from src.config import load_app_config
from src.collector import SpotCollector
from src.features import FeatureEngine
from src.signals import SignalEngine
from src.logger import ParquetLogbook
from src.evaluator import Evaluator
from src.control import Control
from src.runtime_config import RuntimeConfigManager
from src.utils.llm_client import LLMClient, LLMConfig


async def pipeline() -> None:
    cfg = load_app_config()
    out_dir = cfg.storage.get("logbook_dir", "data/logbook")
    os.makedirs(out_dir, exist_ok=True)

    queue: asyncio.Queue = asyncio.Queue(maxsize=10_000)

    # Track current symbols from config; can be hot-reloaded via runtime overrides
    current_symbols: list[str] = list(cfg.symbols)

    collector = SpotCollector(current_symbols, vars(cfg.streams), queue)
    features = FeatureEngine(
        symbols=current_symbols,
        vol_window_s=cfg.features.vol_1s,
        delta_window_s=cfg.features.delta_1s,
        ma_windows=cfg.features.ma,
    )
    signal_engine = SignalEngine(
        thresholds=vars(cfg.signal_thresholds),
        horizons=vars(cfg.horizons),
        rules=vars(cfg.rules),
    )
    logbook = ParquetLogbook(out_dir)

    evaluator = Evaluator(logbook_dir=out_dir, horizon_s=cfg.horizons.scalp)
    control = Control()
    runtime_cfg = RuntimeConfigManager()

    # ---- LLM Recommender State ----
    llm_client: LLMClient | None = None
    llm_active_name: str | None = None
    llm_window_s: int = 30
    llm_refresh_s: int = 5

    def _rebuild_llm_client(overrides: Dict | None) -> None:
        nonlocal llm_client, llm_active_name, llm_window_s, llm_refresh_s
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
        active = llm.get("active") if isinstance(llm.get("active"), str) else None
        configs = llm.get("configs") if isinstance(llm.get("configs"), dict) else None
        if active and configs and isinstance(configs.get(active), dict):
            conf = configs[active]
            new_cfg = LLMConfig(
                base_url=str(conf.get("base_url")),
                api_key=(conf.get("api_key") or None),
                model=(conf.get("model") or None),
                system_prompt=(conf.get("system_prompt") or None),
                user_template=(conf.get("user_template") or None),
            )
            # Only rebuild when active name or essential params changed
            if llm_active_name != active:
                if llm_client is not None:
                    try:
                        # Close old client
                        import anyio

                        anyio.from_thread.run(llm_client.aclose)  # best-effort
                    except Exception:
                        pass
                llm_client = LLMClient(new_cfg)
                llm_active_name = active

    async def llm_loop() -> None:
        nonlocal llm_client, llm_window_s, llm_refresh_s
        while True:
            try:
                if llm_client is None:
                    await asyncio.sleep(1)
                    continue
                # For each symbol, build a summary and query LLM
                for sym in cfg.symbols:
                    summary = features.summarize_window(sym, window_s=llm_window_s)
                    if not summary or summary.get("count", 0) == 0:
                        continue
                    variables = {
                        "symbol": sym,
                        "window_seconds": llm_window_s,
                        "summary": summary,
                    }
                    recs = await llm_client.generate(variables)
                    if not recs:
                        continue
                    for rec in recs:
                        try:
                            asset = str(rec.get("asset") or sym).upper()
                            direction_raw = str(rec.get("direction")).lower()
                            direction = (
                                "buy" if direction_raw in ("buy", "long") else "sell"
                            )
                            leverage = int(rec.get("leverage") or 1)
                            now_ms = int(__import__("time").time() * 1000)
                            logbook.append_trade_recommendation(
                                [
                                    {
                                        "ts_ms": now_ms,
                                        "symbol": asset,
                                        "asset": asset,
                                        "direction": direction,
                                        "leverage": leverage,
                                        "source": "llm",
                                    }
                                ]
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
        batch_signals: list[Dict] = []
        last_flush_ts_ms: int | None = None
        while True:
            msg = await queue.get()
            snap = features.on_message(msg)
            if not snap:
                queue.task_done()
                continue

            # Append to market_snapshot batch
            batch_snapshots.append(snap)

            sig = signal_engine.on_features(snap)
            if sig:
                # Enrich and queue
                sig_row = {
                    **sig,
                    "ts_ms": snap.get("ts_ms"),
                    "signal_id": uuid.uuid4().hex,
                }
                batch_signals.append(sig_row)

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
                    if batch_signals:
                        sigs_by_sym: Dict[str, list[Dict]] = {}
                        for r in batch_signals:
                            s = str(r.get("symbol") or "").upper()
                            if not s:
                                continue
                            sigs_by_sym.setdefault(s, []).append(r)
                        for _sym, rows in sigs_by_sym.items():
                            logbook.append_signal_emitted(rows)
                        batch_signals.clear()
                except Exception as e:
                    logger.exception(f"Logbook write failed: {e}")
                finally:
                    last_flush_ts_ms = ts_ms

            queue.task_done()

    async def evaluator_loop() -> None:
        await evaluator.run_periodic(interval_seconds=5)

    # Collector task is managed separately to allow restart when symbols change
    collector_task = asyncio.create_task(collector.run(), name="collector")
    tasks = [
        asyncio.create_task(consumer(), name="consumer"),
        asyncio.create_task(evaluator_loop(), name="evaluator"),
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

            # Hot-reload runtime overrides if changed
            try:
                changed, overrides = runtime_cfg.load_if_changed()
                if changed and overrides:
                    RuntimeConfigManager.apply_to_engines(
                        overrides,
                        signal_engine=signal_engine,
                        evaluator=evaluator,
                    )
                    _rebuild_llm_client(overrides)

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
                                collector = SpotCollector(
                                    current_symbols, vars(cfg.streams), queue
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
