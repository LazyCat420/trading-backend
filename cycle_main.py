"""
Trading Cycle Backend — Standalone cycle worker.

Runs the full trading cycle (collect → analyze → trade) as an
autonomous process. Communicates with the frontend via shared
PostgreSQL database and optional Redis pub/sub.

Usage:
    python -m cycle_main                     # run with default .env
    python -m cycle_main --once              # run one cycle and exit
    python -m cycle_main --tickers AAPL,NVDA # override tickers

This is the entrypoint for the trading-cycle-backend Docker container.
"""

import asyncio
import argparse
import logging
import os
import signal
import sys
import time

# ── Ensure the shared codebase is importable ────────────────────
# The Docker container mounts vllm-trading-bot at /app/shared
# In dev, PYTHONPATH should include the vllm-trading-bot directory
SHARED_CODE = os.environ.get(
    "SHARED_CODEBASE_PATH",
    os.path.join(os.path.dirname(__file__), "..", "vllm-trading-bot"),
)
if os.path.isdir(SHARED_CODE) and SHARED_CODE not in sys.path:
    sys.path.insert(0, SHARED_CODE)

logger = logging.getLogger("cycle_backend")


async def run_single_cycle(
    tickers: list[str] | None = None,
    cycle_id: str = "",
    bot_id: str = "cycle-backend",
) -> dict:
    """Execute one full trading cycle and return the summary."""
    from app.cycle.core import PipelineContext
    from app.cycle.orchestration.orchestrator_core import OrchestratorCoreMixin
    from app.services.bot_manager import get_active_bot_id
    from unittest.mock import MagicMock

    if not cycle_id:
        cycle_id = f"cycle-{int(time.time())}"

    if not tickers:
        from app.cycle.orchestration.lifecycle_controller import LifecycleControllerMixin
        try:
            from app.pipeline.ticker_selector import TickerSelector
            selector = TickerSelector()
            tickers = await selector.select()
        except Exception as e:
            logger.warning("[cycle_backend] Ticker selection failed: %s", e)
            tickers = ["AAPL"]

    try:
        bot_id = get_active_bot_id()
    except Exception:
        pass

    ctx = PipelineContext(
        tickers=tickers,
        collect=True,
        analyze=True,
        trade=True,
        cycle_id=cycle_id,
    )

    # Set up the mixin state for standalone execution
    OrchestratorCoreMixin._state = {"status": "idle"}
    OrchestratorCoreMixin._cycle_summary = {}
    OrchestratorCoreMixin.emit = lambda *a, **k: logger.info(
        "[cycle] %s %s", a[1] if len(a) > 1 else "", a[2] if len(a) > 2 else ""
    )
    OrchestratorCoreMixin.save_state = lambda *a, **k: None

    logger.info(
        "[cycle_backend] Starting cycle %s | tickers=%s",
        cycle_id,
        tickers,
    )
    t0 = time.monotonic()

    try:
        await OrchestratorCoreMixin._execute_cycle(ctx)
        elapsed = time.monotonic() - t0
        summary = OrchestratorCoreMixin._cycle_summary
        summary["elapsed_s"] = round(elapsed, 1)
        logger.info(
            "[cycle_backend] Cycle %s completed in %.1fs",
            cycle_id,
            elapsed,
        )
        return summary
    except Exception as e:
        elapsed = time.monotonic() - t0
        logger.error(
            "[cycle_backend] Cycle %s failed after %.1fs: %s",
            cycle_id,
            elapsed,
            e,
        )
        return {"cycle_id": cycle_id, "status": "error", "error": str(e)}


async def run_scheduler(
    interval_minutes: int = 30,
    tickers: list[str] | None = None,
) -> None:
    """Run cycles on a schedule until shutdown is requested."""
    shutdown = asyncio.Event()

    def _request_shutdown():
        logger.info("[cycle_backend] Shutdown requested...")
        shutdown.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_shutdown)
        except NotImplementedError:
            signal.signal(sig, lambda s, f: _request_shutdown())

    cycle_num = 0
    logger.info(
        "[cycle_backend] Scheduler started | interval=%dm | tickers=%s",
        interval_minutes,
        tickers or "auto-select",
    )

    while not shutdown.is_set():
        cycle_num += 1
        cycle_id = f"sched-{int(time.time())}-{cycle_num}"

        try:
            summary = await run_single_cycle(
                tickers=tickers,
                cycle_id=cycle_id,
            )
            logger.info(
                "[cycle_backend] Cycle %d done: %s",
                cycle_num,
                summary.get("status", "unknown"),
            )
        except Exception as e:
            logger.error("[cycle_backend] Cycle %d crashed: %s", cycle_num, e)

        # Wait for the next cycle or shutdown
        try:
            await asyncio.wait_for(
                shutdown.wait(),
                timeout=interval_minutes * 60,
            )
        except asyncio.TimeoutError:
            pass  # Normal — time for the next cycle

    logger.info("[cycle_backend] Scheduler stopped after %d cycles.", cycle_num)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    ap = argparse.ArgumentParser(description="Trading Cycle Backend")
    ap.add_argument("--once", action="store_true", help="Run one cycle and exit")
    ap.add_argument("--tickers", type=str, help="Comma-separated tickers (e.g. AAPL,NVDA)")
    ap.add_argument("--interval", type=int, default=30, help="Minutes between cycles (default: 30)")
    args = ap.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",")] if args.tickers else None

    if args.once:
        result = asyncio.run(run_single_cycle(tickers=tickers))
        print(f"Result: {result}")
    else:
        asyncio.run(run_scheduler(
            interval_minutes=args.interval,
            tickers=tickers,
        ))


if __name__ == "__main__":
    main()
