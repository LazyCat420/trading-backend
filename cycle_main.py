"""
Trading Cycle Backend — Standalone cycle worker.

Runs the full trading cycle (collect → analyze → trade) as an
autonomous process. Communicates with the frontend via shared
PostgreSQL database and optional Redis pub/sub.

Usage:
    python -m cycle_main                     # run with default .env
    python -m cycle_main --once              # run one cycle and exit
    python -m cycle_main --tickers AAPL,NVDA # override tickers

This is the entrypoint for the trading-service Docker container.
"""

import asyncio
import argparse
import logging
import os
import signal
import sys
import time
import json
import uvicorn
from fastapi import FastAPI

# ── Ensure the local directory and shared codebase are importable ────────────────────
# The Docker container mounts trading-client at /app/shared
# In dev, PYTHONPATH should include the trading-client directory
local_dir = os.path.dirname(os.path.abspath(__file__))
if local_dir not in sys.path:
    sys.path.insert(0, local_dir)

SHARED_CODE = os.environ.get(
    "SHARED_CODEBASE_PATH",
    os.path.join(local_dir, "..", "trading-client"),
)
if os.path.isdir(SHARED_CODE) and SHARED_CODE not in sys.path:
    sys.path.append(SHARED_CODE)

logger = logging.getLogger("cycle_backend")

from app.cycle.orchestration.state_manager import PipelineStateMixin
from app.cycle.orchestration.orchestrator_core import OrchestratorCoreMixin

class CycleEngine(OrchestratorCoreMixin, PipelineStateMixin):
    pass


async def run_single_cycle(
    tickers: list[str] | None = None,
    cycle_id: str = "",
    bot_id: str = "cycle-backend",
) -> dict:
    """Execute one full trading cycle and return the summary."""
    from app.services.boot_service import BootService
    from app.cycle.core import PipelineContext
    from app.services.bot_manager import resolve_bot_id
    from unittest.mock import MagicMock

    if not cycle_id:
        cycle_id = f"cycle-{int(time.time())}"

    # Ensure system is fully booted before executing the cycle
    await BootService.startup()

    # Reset cycle control to ensure it's not paused when running single cycle via CLI
    from app.cycle.orchestration.cycle_control import cycle_control
    cycle_control.reset()

    try:
        if not tickers:
            from app.cycle.orchestration.lifecycle_controller import LifecycleControllerMixin
            try:
                from app.pipeline.ticker_selector import TickerSelector
                tickers = TickerSelector.select_tickers_for_cycle(requested_tickers=[], cap=50)
            except Exception as e:
                logger.warning("[cycle_backend] Ticker selection failed: %s", e)
                tickers = ["AAPL"]

        bot_id = resolve_bot_id(bot_id)

        ctx = PipelineContext(
            tickers=tickers,
            collect=True,
            analyze=True,
            trade=True,
            cycle_id=cycle_id,
            bot_id=bot_id,
        )

        # Initialize the pipeline state using the db state manager
        CycleEngine.load_state()

        logger.info(
            "[cycle_backend] Starting cycle %s | tickers=%s",
            cycle_id,
            tickers,
        )
        t0 = time.monotonic()

        try:
            await CycleEngine._execute_cycle(ctx)
            elapsed = time.monotonic() - t0
            summary = CycleEngine._cycle_summary
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
    finally:
        await BootService.shutdown()


_background_tasks = set()


def track_task(coro):
    """Start a background task and hold a strong reference to prevent GC."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


async def poll_system_commands(shutdown: asyncio.Event):
    """Background loop to pick up manual triggers from the frontend."""
    from app.db.connection import get_db
    import json
    
    logger.info("[cycle_backend] Started system commands poller.")
    
    while not shutdown.is_set():
        try:
            job_id, cmd_type, payload_val = None, None, None
            with get_db() as db:
                with db.transaction():
                    row = db.execute(
                        "SELECT id, command_type, payload FROM system_commands WHERE status = 'pending' ORDER BY created_at ASC LIMIT 1 FOR UPDATE SKIP LOCKED"
                    ).fetchone()
                    
                    if row:
                        job_id, cmd_type, payload_val = row
                        db.execute(
                            "UPDATE system_commands SET status = 'running', started_at = CURRENT_TIMESTAMP WHERE id = %s", 
                            [job_id]
                        )
            
            if job_id:
                try:
                    payload = json.loads(payload_val) if isinstance(payload_val, str) else (payload_val or {})
                    result = None
                    
                    logger.info("[cycle_backend] Processing command %s (%s)", cmd_type, job_id)
                    
                    if cmd_type == "START_CYCLE":
                        from app.services.pipeline_service import PipelineService
                        # Sync in-memory state from DB before checking guards.
                        # Without this, stale in-memory state (e.g. 'starting' from
                        # a previous failed attempt) will reject the command even
                        # though the DB has been reset to 'idle'.
                        PipelineService.load_state()
                        result = await PipelineService.start_cycle(
                            tickers=payload.get("tickers", []),
                            collect=payload.get("collect", True),
                            analyze=payload.get("analyze", True),
                            trade=payload.get("trade", True),
                            max_tickers=payload.get("max_tickers"),
                        )
                    elif cmd_type == "ANALYZE_TICKER":
                        from app.pipeline.analysis.decision_engine import analyze_ticker
                        result = await analyze_ticker(payload.get("ticker"), cycle_id="manual_run")
                    elif cmd_type == "MORNING_BRIEFING":
                        from app.pipeline.analysis.morning_briefing import generate_morning_briefing
                        result = await generate_morning_briefing()
                    elif cmd_type == "FLASH_BRIEFING":
                        from app.services.flash_briefing import generate_flash_briefing
                        result = await generate_flash_briefing()
                    elif cmd_type == "STOP_CYCLE":
                        from app.services.pipeline_service import PipelineService
                        result = await PipelineService.stop_cycle()
                    elif cmd_type == "PAUSE_CYCLE":
                        from app.services.pipeline_service import PipelineService
                        PipelineService.pause_cycle()
                        result = {"status": "paused"}
                    elif cmd_type == "RESUME_CYCLE":
                        from app.services.pipeline_service import PipelineService
                        await PipelineService.resume_cycle()
                        result = {"status": "resumed"}
                    elif cmd_type == "RESUME_INTERRUPTED":
                        from app.services.pipeline_service import PipelineService
                        result = await PipelineService.resume_interrupted_cycle()
                    elif cmd_type == "DISCARD_CHECKPOINT":
                        from app.services.pipeline_service import PipelineService
                        result = PipelineService.discard_checkpoint()
                    elif cmd_type == "FORCE_CHECKPOINT":
                        from app.services.pipeline_service import PipelineService
                        PipelineService.force_save_checkpoint()
                        result = {"status": "checkpoint_saved"}
                    elif cmd_type == "REFRESH_SCHEDULE":
                        from app.services.cycle_scheduler import SchedulerService
                        SchedulerService.refresh_job(payload.get("job_id"))
                        result = {"status": "schedule_refreshed"}
                    elif cmd_type == "AUTORESEARCH":
                        from app.services.logging import run_autoresearch
                        track_task(run_autoresearch(payload.get("cycle_id"), payload.get("cycle_summary")))
                        result = {"status": "autoresearch_started"}
                    elif cmd_type == "DEPLOY_FIX":
                        from app.cognition.evolution.deployer import deploy_fix_to_disk
                        result = deploy_fix_to_disk(payload.get("fix_id"))
                    elif cmd_type == "ROLLBACK_FIX":
                        from app.cognition.evolution.deployer import rollback_fix
                        result = rollback_fix(payload.get("fix_id"))
                    elif cmd_type == "ACTIVATE_BRAIN_GRAPH":
                        from app.cognition.ontology.ontology_builder import BrainGraph
                        ticker = payload.get("ticker")
                        max_hops = payload.get("max_hops", 3)
                        seeded = BrainGraph.seed_from_ticker_metadata(ticker)
                        graph_res = BrainGraph.spreading_activation(seed_node_ids=[ticker], max_hops=max_hops)
                        graph_res["seeded"] = seeded
                        result = graph_res
                    elif cmd_type == "EVALUATE_STRATEGY":
                        from app.cognition.evaluation.strategy_auditor import evaluate_strategy
                        track_task(evaluate_strategy(cycle_id=payload.get("cycle_id"), refresh_pending=True))
                        result = {"status": "evaluation_started"}
                    elif cmd_type == "GENERATE_MORNING_BRIEFING":
                        from app.pipeline.analysis.morning_briefing import generate_morning_briefing
                        track_task(generate_morning_briefing())
                        result = {"status": "briefing_started"}
                    elif cmd_type == "RUN_MARKET_COLLECTION":
                        from app.collectors.market_regime_collector import collect_market_data
                        from app.data.market_regime_engine import compute_market_regime, compute_sector_breadth
                        async def _do_market_collect():
                            await collect_market_data(period=payload.get("period", "6mo"))
                            await compute_market_regime()
                            await compute_sector_breadth()
                        track_task(_do_market_collect())
                        result = {"status": "market_collection_started"}
                    elif cmd_type == "RUN_FRED_COLLECTION":
                        from app.services.boot_service import BootService
                        track_task(BootService._startup_fred_refresh())
                        result = {"status": "fred_collection_started"}
                    elif cmd_type == "COLLECT_SP500_DATA":
                        from app.data.sp500_universe import load_sp500_universe
                        from app.data.sp500_price_collector import collect_sp500_prices
                        async def _do_sp500_collect():
                            await load_sp500_universe(enrich=payload.get("enrich", False))
                            await collect_sp500_prices(period=payload.get("price_period", "6mo"))
                        track_task(_do_sp500_collect())
                        result = {"status": "sp500_collection_started"}
                    elif cmd_type == "REFRESH_SECTORS":
                        from app.data.sector_aggregator import compute_sector_performance
                        from app.data.sector_correlation_engine import compute_all_correlations
                        from app.data.rotation_detector import detect_rotations
                        async def _do_refresh_sectors():
                            await compute_sector_performance()
                            await compute_all_correlations()
                            await detect_rotations()
                        track_task(_do_refresh_sectors())
                        result = {"status": "refresh_sectors_started"}

                    else:
                        logger.warning(
                            "[cycle_backend] Unknown command type '%s' (job %s) — no handler matched",
                            cmd_type, job_id,
                        )
                        result = {"status": "error", "reason": f"Unknown command type: {cmd_type}"}

                    with get_db() as db:
                        db.execute(
                            "UPDATE system_commands SET status = 'completed', completed_at = CURRENT_TIMESTAMP, result = %s WHERE id = %s", 
                            [json.dumps(result), job_id]
                        )
                    logger.info("[cycle_backend] Completed command %s", job_id)
                except BaseException as e:
                    logger.error("[cycle_backend] Command %s failed: %s", job_id, e)
                    with get_db() as db:
                        db.execute(
                            "UPDATE system_commands SET status = 'error', completed_at = CURRENT_TIMESTAMP, error_message = %s WHERE id = %s", 
                            [str(e), job_id]
                        )
                    if isinstance(e, asyncio.CancelledError):
                        raise
        except BaseException as e:
            logger.error("[cycle_backend] Poller error: %s", e)
            if isinstance(e, asyncio.CancelledError):
                raise
            
        try:
            await asyncio.wait_for(shutdown.wait(), timeout=0.2)
        except asyncio.TimeoutError:
            pass


async def run_worker(
    tickers: list[str] | None = None,
    shutdown_event: asyncio.Event | None = None,
) -> None:
    """Run the backend worker (scheduler + poller) until shutdown."""
    from app.services.boot_service import BootService
    
    if shutdown_event is None:
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
    else:
        shutdown = shutdown_event

    # Start boot sequence (including DB connection & Schema init, Reset Application State, and Scheduler Start)
    await BootService.startup()
    
    # Start the system commands poller
    poller_task = asyncio.create_task(poll_system_commands(shutdown))
    
    # Keep the main loop alive
    await shutdown.wait()
    
    # Cancel and await remaining background tasks before shutting down dependencies
    if _background_tasks:
        logger.info("[cycle_backend] Cleaning up %d background tasks on shutdown...", len(_background_tasks))
        for task in list(_background_tasks):
            if not task.done():
                task.cancel()
        await asyncio.gather(*_background_tasks, return_exceptions=True)

    # Cleanup poller task
    poller_task.cancel()
    try:
        await poller_task
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error("[cycle_backend] Error during poller task cleanup: %s", e)

    await BootService.shutdown()


async def start_health_server(shutdown_event: asyncio.Event):
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi import Depends, HTTPException, Security
    from app.config import settings

    security = HTTPBearer()

    def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
        if credentials.credentials != settings.API_SERVER_KEY:
            raise HTTPException(status_code=403, detail="Invalid API Server Key")
        return credentials.credentials

    app = FastAPI(title="Trading Cycle Backend Health")
    @app.get("/health")
    def health():
        return {"status": "ok", "service": "trading-service"}

    @app.get("/status")
    def status(summary_only: bool = False, token: str = Depends(verify_api_key)):
        from app.cycle.orchestration.state_manager import PipelineStateMixin
        return PipelineStateMixin.get_current_state(summary_only=summary_only)

    from app.services.vllm_router import router as vllm_router
    app.include_router(vllm_router)

    config = uvicorn.Config(app, host="0.0.0.0", port=8080, log_level="error")
    server = uvicorn.Server(config)
    
    async def _serve():
        try:
            await server.serve()
        except asyncio.CancelledError:
            pass

    task = asyncio.create_task(_serve())
    await shutdown_event.wait()
    server.should_exit = True
    await task


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    ap = argparse.ArgumentParser(description="Trading Cycle Backend")
    ap.add_argument("--once", action="store_true", help="Run one cycle and exit")
    ap.add_argument("--tickers", type=str, help="Comma-separated tickers (e.g. AAPL,NVDA)")
    ap.add_argument("--interval", type=int, help="Legacy interval arg, ignored")
    args = ap.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",")] if args.tickers else None

    if args.once:
        result = asyncio.run(run_single_cycle(tickers=tickers))
        print(f"Result: {result}")
    else:
        async def _run_all():
            shutdown = asyncio.Event()
            
            def _request_shutdown():
                logger.info("[cycle_backend] Shutdown requested (daemon/all mode)...")
                shutdown.set()

            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(sig, _request_shutdown)
                except NotImplementedError:
                    pass

            worker_task = asyncio.create_task(run_worker(tickers=tickers, shutdown_event=shutdown))
            health_task = asyncio.create_task(start_health_server(shutdown))
            
            await asyncio.gather(worker_task, health_task)
            
        try:
            asyncio.run(_run_all())
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    main()
