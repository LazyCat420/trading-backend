"""
Trading Cycle Backend — Standalone cycle worker.

Runs the full trading cycle (collect → analyze → trade) as an
autonomous process. Communicates with the frontend via shared
PostgreSQL database and optional Redis pub/sub.

Usage:
    python -m cycle_main                     # run with default .env
    python -m cycle_main --once              # run one cycle and exit
    python -m cycle_main --tickers AAPL,NVDA # override tickers

This is the entrypoint for the trading-backend Docker container.
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

# ── Ensure the shared codebase is importable ────────────────────
# The Docker container mounts trading-frontend at /app/shared
# In dev, PYTHONPATH should include the trading-frontend directory
SHARED_CODE = os.environ.get(
    "SHARED_CODEBASE_PATH",
    os.path.join(os.path.dirname(__file__), "..", "trading-frontend"),
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
    from app.cycle.orchestration.state_manager import PipelineStateMixin
    from app.cycle.orchestration.orchestrator_core import OrchestratorCoreMixin
    from app.services.bot_manager import get_active_bot_id
    from unittest.mock import MagicMock

    class CycleEngine(OrchestratorCoreMixin, PipelineStateMixin):
        pass

    if not cycle_id:
        cycle_id = f"cycle-{int(time.time())}"

    if not tickers:
        from app.cycle.orchestration.lifecycle_controller import LifecycleControllerMixin
        try:
            from app.pipeline.ticker_selector import TickerSelector
            tickers = TickerSelector.select_tickers_for_cycle(requested_tickers=[], cap=50)
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


async def poll_system_commands(shutdown: asyncio.Event):
    """Background loop to pick up manual triggers from the frontend."""
    from app.db.connection import get_db
    import json
    
    logger.info("[cycle_backend] Started system commands poller.")
    
    while not shutdown.is_set():
        try:
            with get_db() as db:
                row = db.execute(
                    "SELECT id, command_type, payload FROM system_commands WHERE status = 'pending' ORDER BY created_at ASC LIMIT 1"
                ).fetchone()
                
                if row:
                    job_id, cmd_type, payload_val = row
                    db.execute(
                        "UPDATE system_commands SET status = 'running', started_at = CURRENT_TIMESTAMP WHERE id = %s", 
                        [job_id]
                    )
                    
                    try:
                        payload = json.loads(payload_val) if isinstance(payload_val, str) else (payload_val or {})
                        result = None
                        
                        logger.info(f"[cycle_backend] Processing command {cmd_type} ({job_id})")
                        
                        if cmd_type == "START_CYCLE":
                            from app.services.pipeline_service import PipelineService
                            result = await PipelineService.start_cycle(tickers=payload.get("tickers", []))
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
                            from app.pipeline.analysis.autoresearch import run_autoresearch
                            asyncio.create_task(run_autoresearch(payload.get("cycle_id"), payload.get("cycle_summary")))
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
                            asyncio.create_task(evaluate_strategy(cycle_id=payload.get("cycle_id"), refresh_pending=True))
                            result = {"status": "evaluation_started"}
                        elif cmd_type == "GENERATE_MORNING_BRIEFING":
                            from app.pipeline.analysis.morning_briefing import generate_morning_briefing
                            asyncio.create_task(generate_morning_briefing())
                            result = {"status": "briefing_started"}
                            
                        db.execute(
                            "UPDATE system_commands SET status = 'completed', completed_at = CURRENT_TIMESTAMP, result = %s WHERE id = %s", 
                            [json.dumps(result), job_id]
                        )
                        logger.info(f"[cycle_backend] Completed command {job_id}")
                    except BaseException as e:
                        logger.error(f"[cycle_backend] Command {job_id} failed: {e}")
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
            await asyncio.wait_for(shutdown.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass


async def run_worker(tickers: list[str] | None = None) -> None:
    """Run the backend worker (scheduler + poller) until shutdown."""
    from app.services.cycle_scheduler import SchedulerService
    
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

    # Start APScheduler (which runs DB schedules + background jobs)
    SchedulerService.start()
    
    # Start the system commands poller
    poller_task = asyncio.create_task(poll_system_commands(shutdown))
    
    # Keep the main loop alive
    await shutdown.wait()
    
    # Cleanup
    SchedulerService.stop()
    await poller_task


async def start_health_server(shutdown_event: asyncio.Event):
    app = FastAPI(title="Trading Cycle Backend Health")
    @app.get("/health")
    def health():
        return {"status": "ok", "service": "trading-backend"}

    @app.get("/status")
    def status(summary_only: bool = False):
        from app.cycle.orchestration.state_manager import PipelineStateMixin
        return PipelineStateMixin.get_current_state(summary_only=summary_only)

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
                shutdown.set()

            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(sig, _request_shutdown)
                except NotImplementedError:
                    pass

            worker_task = asyncio.create_task(run_worker(tickers=tickers))
            health_task = asyncio.create_task(start_health_server(shutdown))
            
            await asyncio.gather(worker_task, health_task)
            
        try:
            asyncio.run(_run_all())
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    main()
