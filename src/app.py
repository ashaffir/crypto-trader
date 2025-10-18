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


async def pipeline() -> None:
    cfg = load_app_config()
    out_dir = cfg.storage.get("logbook_dir", "data/logbook")
    os.makedirs(out_dir, exist_ok=True)

    queue: asyncio.Queue = asyncio.Queue(maxsize=10_000)

    collector = SpotCollector(cfg.symbols, vars(cfg.streams), queue)
    features = FeatureEngine(
        symbols=cfg.symbols,
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
                        logbook.append_market_snapshot(batch_snapshots)
                        batch_snapshots.clear()
                    if batch_signals:
                        logbook.append_signal_emitted(batch_signals)
                        batch_signals.clear()
                except Exception as e:
                    logger.exception(f"Logbook write failed: {e}")
                finally:
                    last_flush_ts_ms = ts_ms

            queue.task_done()

    async def evaluator_loop() -> None:
        await evaluator.run_periodic(interval_seconds=5)

    tasks = [
        asyncio.create_task(collector.run(), name="collector"),
        asyncio.create_task(consumer(), name="consumer"),
        asyncio.create_task(evaluator_loop(), name="evaluator"),
    ]

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        await collector.stop()
        for t in tasks:
            t.cancel()


def main() -> None:
    try:
        asyncio.run(pipeline())
    except KeyboardInterrupt:
        logger.info("Shutting down")


if __name__ == "__main__":
    main()
