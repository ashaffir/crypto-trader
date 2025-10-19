from __future__ import annotations

import asyncio
from loguru import logger

from src.control import Control
from src.app import pipeline


async def run_supervisor(poll_interval_s: float = 1.0) -> None:
    control = Control()
    task: asyncio.Task | None = None
    desired = None
    while True:
        try:
            new_desired = control.get_desired_state()
        except Exception:
            new_desired = "stopped"

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

        # If desired is running but task is absent (e.g., after crash), start it
        if desired == "running" and task is None:
            task = asyncio.create_task(pipeline(), name="pipeline")

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
