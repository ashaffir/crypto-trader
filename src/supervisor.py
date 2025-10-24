from __future__ import annotations

import asyncio
import time as _t
from loguru import logger

from src.control import Control
from src.app import pipeline
from src.positions import PositionStore
from src.runtime_config import RuntimeConfigManager


def should_stop_by_time_limit(
    start_ts_ms: int | None, max_run_minutes: int | None, now_ts_ms: int | None
) -> bool:
    """Return True if a positive max_run_minutes has elapsed since start_ts_ms.

    Any None or non-positive values result in False (no stop).
    """
    try:
        if start_ts_ms is None or now_ts_ms is None:
            return False
        m = int(max_run_minutes or 0)
        if m <= 0:
            return False
        deadline = int(start_ts_ms) + int(m) * 60_000
        return int(now_ts_ms) >= deadline
    except Exception:
        return False


async def run_supervisor(poll_interval_s: float = 1.0) -> None:
    control = Control()
    runtime_cfg = RuntimeConfigManager()

    task: asyncio.Task | None = None
    desired = None

    # Track run timing for auto-stop
    start_ts_ms: int | None = None
    max_run_minutes: int = 0

    while True:
        try:
            new_desired = control.get_desired_state()
        except Exception:
            new_desired = "stopped"

        # Reload runtime overrides if changed (to capture supervisor max runtime)
        try:
            changed, overrides = runtime_cfg.load_if_changed()
        except Exception:
            changed, overrides = (False, None)
        if changed and isinstance(overrides, dict):
            try:
                sup = overrides.get("supervisor") or {}
                m = int(sup.get("max_run_minutes", 0))
                max_run_minutes = max(0, m)
            except Exception:
                pass

        # If the task terminated on its own, clear it so we can relaunch when desired
        if task is not None and task.done():
            try:
                exc = task.exception()
                if exc is not None:
                    logger.exception(f"Pipeline task ended with error: {exc}")
            except asyncio.CancelledError:
                # Normal cancellation path
                pass
            except Exception:
                # Best-effort logging
                pass
            finally:
                task = None

        if new_desired != desired:
            desired = new_desired
            logger.info(f"Desired state changed to: {desired}")
            if desired == "running" and task is None:
                task = asyncio.create_task(pipeline(), name="pipeline")
                # Start timing window at (re)launch
                start_ts_ms = int(_t.time() * 1000)
                # Ensure latest setting is applied even if overrides not changed this tick
                try:
                    cur_ovr = runtime_cfg.read() or {}
                    sup = cur_ovr.get("supervisor") or {}
                    m = int(sup.get("max_run_minutes", max_run_minutes))
                    max_run_minutes = max(0, m)
                except Exception:
                    pass
            elif desired == "stopped" and task is not None:
                # Request cooperative cancellation
                task.cancel()
                try:
                    # Allow a short grace period for clean shutdown
                    await asyncio.wait_for(task, timeout=3.0)
                except asyncio.TimeoutError:
                    logger.warning("Pipeline did not stop within 3s; forcing stop")
                except Exception:
                    pass
                finally:
                    task = None
                # Reset timing window
                start_ts_ms = None
                # After pipeline stops, close all open positions (paper trades)
                try:
                    store = PositionStore()
                    opens = store.get_open_positions()
                    if opens:
                        now_ms = int(_t.time() * 1000)
                        from ui.lib.logbook_utils import price_at_ts as _price_at_ts

                        for p in opens:
                            try:
                                sym = str(p.get("symbol")) if p.get("symbol") else None
                            except Exception:
                                sym = None
                            px = None
                            try:
                                if sym:
                                    px = _price_at_ts(sym, now_ms)
                            except Exception:
                                px = None
                            store.close_position(
                                int(p["id"]),
                                now_ms,
                                exit_px=px,
                                pnl=None,
                                close_reason="Operation",
                            )
                        logger.info(f"Closed {len(opens)} open positions on stop")
                except Exception as _e:
                    logger.warning(f"Failed to close open positions on stop: {_e}")

        # If desired is running but task is absent (e.g., after crash), start it
        if desired == "running" and task is None:
            task = asyncio.create_task(pipeline(), name="pipeline")
            # If we didn't have a start time (e.g., after crash), set it now
            if start_ts_ms is None:
                start_ts_ms = int(_t.time() * 1000)

        # Enforce time limit if configured
        if desired == "running" and task is not None and not task.done():
            now_ms = int(_t.time() * 1000)
            if should_stop_by_time_limit(start_ts_ms, max_run_minutes, now_ms):
                logger.info(
                    f"Max run time reached ({max_run_minutes} min). Requesting stop."
                )
                try:
                    control.set_desired_state("stopped")
                except Exception:
                    # Fallback: cancel directly if control channel write fails
                    try:
                        task.cancel()
                    except Exception:
                        pass

        # Write status heartbeat for visibility; mark "stopped" when not running
        if task is not None and not task.done():
            control.write_status({"status": "running"})
        else:
            control.write_status({"status": "stopped"})

        await asyncio.sleep(poll_interval_s)


def main() -> None:
    try:
        asyncio.run(run_supervisor())
    except KeyboardInterrupt:
        logger.info("Supervisor stopped")


if __name__ == "__main__":
    main()
