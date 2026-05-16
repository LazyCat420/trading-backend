import logging
import asyncio
from typing import Callable, Any

from app.config import settings
from app.monitoring.pipeline_profiler import profiler as pipeline_profiler

logger = logging.getLogger(__name__)


async def run_phase4_analysis(
    ctx: Any,
    bot_id: str,
    macro_memo: str | dict,
    emit: Callable,
    cycle_summary: dict,
    state: dict,
    analysis_queue: asyncio.Queue | None = None,
) -> list[dict]:
    """
    Phase 4: Analysis
    Strictly isolated and timeout-guarded worker pool.

    Supports two modes:
      - Queue mode (analysis_queue provided): Workers consume tickers from the
        queue as they arrive from collection. Sentinels (None) signal shutdown.
      - Batch mode (analysis_queue is None): Workers process ctx.tickers directly
        (legacy / resume mode).

    macro_memo can be:
      - str: static memo text (legacy / resume)
      - dict: {"memo": str} holder filled asynchronously by macro scout.
        Workers read the current value when analyzing each ticker.
    """
    if not ctx.analyze:
        return []

    emit(
        "analyzing",
        "start",
        f"Analysis Phase: evaluating {len(ctx.tickers)} tickers",
        status="running",
    )

    from app.cognition.orchestration.runner import execute_v2_pipeline

    results = []
    errors = []

    worker_count = settings.V2_TICKER_CONCURRENCY or 3

    # Determine which queue to use
    if analysis_queue is not None:
        # Queue mode: workers consume from the external queue fed by collection.
        # Sentinels are added by the orchestrator after collection finishes.
        work_queue = analysis_queue
    else:
        # Batch mode: pre-populate a local queue with all tickers + sentinels.
        work_queue = asyncio.Queue()
        for t in ctx.tickers:
            work_queue.put_nowait(t)
        for _ in range(worker_count):
            work_queue.put_nowait(None)

    analyzed_count = 0
    count_lock = asyncio.Lock()

    def _get_macro_memo() -> str:
        """Read the current macro memo, whether it's a string or a dict holder."""
        if isinstance(macro_memo, dict):
            return macro_memo.get("memo", "")
        return macro_memo or ""

    async def _worker(worker_id: int):
        nonlocal analyzed_count
        while True:
            ticker = await work_queue.get()
            if ticker is None:
                work_queue.task_done()
                break

            from app.cycle.orchestration.cycle_control import cycle_control
            await cycle_control.wait_if_paused()

            logger.info(f"[CYCLE] [Worker {worker_id}] Analyzing {ticker}")
            logger.info("[phase4] ticker=%s using v2 pipeline", ticker)

            # Check triage tier
            _triage_state = state.get("triage", {})
            _tier = "standard"
            if ticker in _triage_state.get("glance", []):
                _tier = "glance"
            elif ticker in _triage_state.get("deep", []):
                _tier = "deep"

            _is_highly_redundant = ticker in state.get("highly_redundant_tickers", [])

            # Read the macro memo at analysis time (may have been filled by scout since launch)
            current_macro_memo = _get_macro_memo()

            result = None
            try:
                # Wrap the ENTIRE ticker analysis in a strict wait_for
                # so one bad LLM call doesn't hang the worker forever.
                _ticker_timeout = float(settings.ANALYSIS_WORKER_TIMEOUT_SECONDS)
                result = await asyncio.wait_for(
                    execute_v2_pipeline(
                        ticker,
                        cycle_id=ctx.cycle_id,
                        bot_id=bot_id,
                        emit=emit,
                        macro_memo=current_macro_memo,
                        is_highly_redundant=_is_highly_redundant,
                    ),
                    timeout=_ticker_timeout,
                )

                if result:
                    async with count_lock:
                        results.append(result)
                        analyzed_count += 1

            except asyncio.TimeoutError:
                err_str = f"LLM Timeout after {settings.VLLM_FUTURE_TIMEOUT}s"
                logger.error("Analysis TIMEOUT for %s: %s", ticker, err_str)
                async with count_lock:
                    errors.append(ticker)
                    results.append(
                        {
                            "ticker": ticker,
                            "action": "HOLD",
                            "confidence": 0,
                            "error": err_str,
                            "error_type": "timeout",
                            "is_timeout_fallback": True,
                        }
                    )
            except Exception as e:
                err_str = str(e)
                logger.error("Analysis crashed for %s: %s", ticker, err_str)
                async with count_lock:
                    errors.append(ticker)
                    results.append(
                        {
                            "ticker": ticker,
                            "action": "HOLD",
                            "confidence": 0,
                            "error": err_str,
                            "error_type": type(e).__name__,
                            "is_timeout_fallback": True,
                        }
                    )
            finally:
                work_queue.task_done()

    logger.info(f"[CYCLE] Launching {worker_count} analysis worker threads")
    workers = [asyncio.create_task(_worker(i)) for i in range(worker_count)]

    # We wait for the queue to be fully processed or a global safety timeout.
    # Since individual tickers are strictly timed out, the workers should never hit this global timeout.
    try:
        async with pipeline_profiler.phase("analysis_drain"):
            # Set to cycle timeout + 5 minutes grace buffer, rather than a hardcoded 1 hour
            cycle_timeout_seconds = (int(getattr(settings, "CYCLE_TIMEOUT_MINUTES", 120)) + 5) * 60.0
            await asyncio.wait_for(work_queue.join(), timeout=cycle_timeout_seconds)
    except asyncio.TimeoutError:
        logger.critical("[CYCLE] FATAL WORKER POOL TIMEOUT. Queue failed to drain within %ss.", cycle_timeout_seconds)
        emit(
            "analyzing",
            "worker_timeout",
            "Fatal queue timeout. Proceeding with partial results.",
            status="error",
        )
        for w in workers:
            w.cancel()

    # Calculate summary metrics
    for r in results:
        action = r.get("action", "HOLD")
        if action == "BUY":
            cycle_summary["buy_count"] += 1
        elif action == "SELL":
            cycle_summary["sell_count"] += 1
        elif action == "HOLD":
            cycle_summary["hold_count"] += 1
        if r.get("human_review"):
            cycle_summary["review_count"] += 1

    cycle_summary["analysis_results_count"] = len(results)

    # ── All-Crash Detection Gate ──
    # If every ticker produced a crash-fallback (HOLD @ 0%), the pipeline itself
    # is broken (e.g. import error, variable shadowing). Abort loudly rather than
    # silently passing 30 garbage decisions downstream.
    crash_count = sum(1 for r in results if r.get("is_timeout_fallback"))
    if crash_count > 0 and crash_count == len(results):
        msg = (
            f"CRITICAL: All {crash_count} tickers crashed during analysis. "
            f"Pipeline is broken — aborting cycle. First error: "
            f"{results[0].get('error', 'unknown')}"
        )
        logger.critical("[PIPELINE] %s", msg)
        emit("analyzing", "all_crashed", msg, status="error")
        cycle_summary["status"] = "error"
        cycle_summary["primary_failure_reason"] = f"All {crash_count} tickers crashed"
        cycle_summary["no_trade_reason"] = "all_crashed"
        raise RuntimeError(msg)

    emit(
        "analyzing",
        "pipeline_done",
        "Analysis complete",
        status="ok",
        data={"analyzed": analyzed_count, "errors": errors},
    )

    # Universal hard-fail check (Moved out of legacy else-block)
    if ctx.tickers and not results:
        msg = "CRITICAL: Analysis produced ZERO results. Aborting cycle to prevent silent failure."
        logger.error("[PIPELINE] %s", msg)
        emit("analyzing", "error", msg, status="error")
        cycle_summary["status"] = "error"
        cycle_summary["primary_failure_reason"] = "Analysis produced ZERO results"
        cycle_summary["no_trade_reason"] = "zero_results"
        raise RuntimeError("Analysis produced zero results")

    return results


