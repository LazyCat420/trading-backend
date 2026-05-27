import logging
import asyncio
import time
from typing import Callable

from app.config import settings
from app.monitoring.pipeline_profiler import profiler as pipeline_profiler
from app.cycle.orchestration.cycle_control import cycle_control
from app.services.logging.cycle_auditor import auditor
from app.cycle.context import CycleContext
from app.utils.emit import noop_emit
from app.cycle.summary import tally_results

logger = logging.getLogger(__name__)


def _log_ticker_result(cycle_id: str, ticker: str, result: dict, elapsed_ms: int) -> None:
    try:
        auditor.ticker_result(cycle_id, ticker, result, elapsed_s=elapsed_ms / 1000.0)
    except Exception as ae:
        logger.error("Failed to log ticker_result to auditor: %s", ae)


async def run_phase4_analysis(
    ctx: CycleContext,
    bot_id: str,
    macro_memo: str | dict,
    emit: Callable = noop_emit,
    cycle_summary: dict = None,
    state: dict = None,
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

    from app.cognition.orchestration.runner import execute_v2_pipeline

    emit(
        "analyzing",
        "start",
        f"Analysis Phase: evaluating {len(ctx.tickers)} tickers",
        status="running",
    )

    results = []
    errors = []

    worker_count = settings.V2_TICKER_CONCURRENCY or 3

    # Determine which queue to use
    is_queue_mode = analysis_queue is not None
    if is_queue_mode:
        # Queue mode: workers consume from the external queue fed by collection.
        # Sentinels are added by the orchestrator after collection finishes.
        work_queue = analysis_queue
        emit(
            "analyzing",
            "queue_mode",
            f"Analysis workers in QUEUE mode — waiting for collection to feed tickers. "
            f"Queue depth: {work_queue.qsize()}",
            status="running",
            data={"mode": "queue", "queue_depth": work_queue.qsize(), "worker_count": worker_count},
        )
    else:
        # Batch mode: pre-populate a local queue with all tickers + sentinels.
        work_queue = asyncio.Queue()
        for t in ctx.tickers:
            work_queue.put_nowait(t)
        for _ in range(worker_count):
            work_queue.put_nowait(None)
        emit(
            "analyzing",
            "batch_mode",
            f"Analysis workers in BATCH mode — {len(ctx.tickers)} tickers pre-loaded",
            status="running",
            data={"mode": "batch", "queue_depth": work_queue.qsize(), "worker_count": worker_count},
        )

    analyzed_count = 0
    count_lock = asyncio.Lock()
    _worker_start_time = time.monotonic()
    _seen_tickers: set[str] = set()  # Dedup: prevents double-analysis when tickers arrive from multiple sources

    def _get_macro_memo() -> str:
        """Read the current macro memo, whether it's a string or a dict holder."""
        if isinstance(macro_memo, dict):
            return macro_memo.get("memo", "")
        return macro_memo or ""

    async def _await_macro_memo(timeout_s: float = 5.0) -> str:
        """Wait for the macro scout to finish (up to timeout), then read memo.

        Fix B.1: Ensures analysis workers get the COMPLETE memo instead of
        reading a partial/empty value while the scout is still writing.
        """
        if isinstance(macro_memo, dict) and "_ready" in macro_memo:
            try:
                await asyncio.wait_for(macro_memo["_ready"].wait(), timeout=timeout_s)
            except asyncio.TimeoutError:
                logger.info("[CYCLE] Macro memo not ready after %.0fs — proceeding without", timeout_s)
        return _get_macro_memo()

    async def _worker(worker_id: int):
        nonlocal analyzed_count
        logger.info("[CYCLE] [Worker %d] Started — waiting for tickers from queue", worker_id)
        emit(
            "analyzing",
            f"worker_{worker_id}_ready",
            f"Worker {worker_id}/{worker_count} started, waiting for tickers...",
            status="running",
            data={"worker_id": worker_id, "queue_depth": work_queue.qsize()},
        )

        tickers_processed = 0
        while True:
            # Log that we're waiting (helps diagnose queue starvation)
            _wait_start = time.monotonic()
            ticker = await work_queue.get()
            _wait_ms = int((time.monotonic() - _wait_start) * 1000)

            if ticker is None:
                logger.info(
                    "[CYCLE] [Worker %d] Received sentinel — shutting down after %d tickers",
                    worker_id, tickers_processed,
                )
                emit(
                    "analyzing",
                    f"worker_{worker_id}_done",
                    f"Worker {worker_id} finished: {tickers_processed} tickers processed",
                    status="ok",
                    data={"worker_id": worker_id, "tickers_processed": tickers_processed},
                )
                work_queue.task_done()
                break

            tickers_processed += 1
            _remaining = work_queue.qsize()
            logger.info(
                "[CYCLE] [Worker %d] Got %s (waited %dms, %d remaining in queue)",
                worker_id, ticker, _wait_ms, _remaining,
            )
            emit(
                "analyzing",
                f"worker_got_{ticker}",
                f"Worker {worker_id} → {ticker} (queue: {_remaining} remaining, waited {_wait_ms}ms)",
                status="running",
                data={
                    "worker_id": worker_id, "ticker": ticker,
                    "queue_depth": _remaining, "wait_ms": _wait_ms,
                },
            )

            # ── Dedup gate: skip if another worker already processed this ticker ──
            if ticker in _seen_tickers:
                logger.info(
                    "[CYCLE] [Worker %d] Skipping %s — already analyzed by another worker",
                    worker_id, ticker,
                )
                emit(
                    "analyzing",
                    f"worker_dedup_{ticker}",
                    f"Worker {worker_id}: {ticker} skipped (already analyzed)",
                    status="skipped",
                    data={"worker_id": worker_id, "ticker": ticker},
                )
                work_queue.task_done()
                continue
            _seen_tickers.add(ticker)

            await cycle_control.wait_if_paused()

            # Check triage tier
            _triage_state = state.get("triage", {})
            _tier = "standard"
            if ticker in _triage_state.get("glance", []):
                _tier = "glance"
            elif ticker in _triage_state.get("deep", []):
                _tier = "deep"

            _is_highly_redundant = ticker in state.get("highly_redundant_tickers", [])

            # Read the macro memo at analysis time — await readiness (up to 5s)
            # Fix B.1: Workers wait for scout to finish instead of reading partial memo
            current_macro_memo = await _await_macro_memo(timeout_s=5.0)

            result = None
            _ticker_start = time.monotonic()
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

                _ticker_elapsed_ms = int((time.monotonic() - _ticker_start) * 1000)
                if result:
                    async with count_lock:
                        results.append(result)
                        analyzed_count += 1
                    _action = result.get("action", "?")
                    _conf = result.get("confidence", 0)
                    logger.info(
                        "[CYCLE] [Worker %d] Completed %s → %s@%d%% in %dms (%d/%d done)",
                        worker_id, ticker, _action, _conf, _ticker_elapsed_ms,
                        analyzed_count, len(ctx.tickers),
                    )
                    _log_ticker_result(ctx.cycle_id, ticker, result, _ticker_elapsed_ms)

            except asyncio.TimeoutError:
                _ticker_elapsed_ms = int((time.monotonic() - _ticker_start) * 1000)
                err_str = f"LLM Timeout after {settings.ANALYSIS_WORKER_TIMEOUT_SECONDS}s"
                logger.error("Analysis TIMEOUT for %s: %s (worker %d, %dms)", ticker, err_str, worker_id, _ticker_elapsed_ms)
                emit(
                    "analyzing",
                    f"worker_timeout_{ticker}",
                    f"⏰ Worker {worker_id}: {ticker} TIMEOUT after {_ticker_elapsed_ms}ms",
                    status="error",
                    data={"worker_id": worker_id, "ticker": ticker, "elapsed_ms": _ticker_elapsed_ms},
                )
                fallback_result = {
                    "ticker": ticker,
                    "action": "HOLD",
                    "confidence": 0,
                    "error": err_str,
                    "error_type": "timeout",
                    "is_timeout_fallback": True,
                }
                async with count_lock:
                    errors.append(ticker)
                    results.append(fallback_result)
                _log_ticker_result(ctx.cycle_id, ticker, fallback_result, _ticker_elapsed_ms)
            except Exception as e:
                _ticker_elapsed_ms = int((time.monotonic() - _ticker_start) * 1000)
                err_str = str(e)
                logger.error("Analysis crashed for %s: %s (worker %d, %dms)", ticker, err_str, worker_id, _ticker_elapsed_ms)
                emit(
                    "analyzing",
                    f"worker_crash_{ticker}",
                    f"💥 Worker {worker_id}: {ticker} CRASHED — {err_str[:100]}",
                    status="error",
                    data={"worker_id": worker_id, "ticker": ticker, "error": err_str[:300], "elapsed_ms": _ticker_elapsed_ms},
                )
                fallback_result = {
                    "ticker": ticker,
                    "action": "HOLD",
                    "confidence": 0,
                    "error": err_str,
                    "error_type": type(e).__name__,
                    "is_timeout_fallback": True,
                }
                async with count_lock:
                    errors.append(ticker)
                    results.append(fallback_result)
                _log_ticker_result(ctx.cycle_id, ticker, fallback_result, _ticker_elapsed_ms)
            finally:
                work_queue.task_done()

    logger.info(f"[CYCLE] Launching {worker_count} analysis worker threads")
    emit(
        "analyzing",
        "workers_launching",
        f"Launching {worker_count} analysis workers (timeout={settings.ANALYSIS_WORKER_TIMEOUT_SECONDS}s/ticker)",
        status="running",
        data={"worker_count": worker_count, "timeout_per_ticker": settings.ANALYSIS_WORKER_TIMEOUT_SECONDS},
    )
    workers = [asyncio.create_task(_worker(i)) for i in range(worker_count)]

    # ── Progress heartbeat — logs status every 15 seconds while analysis is running ──
    async def _progress_heartbeat():
        """Emit periodic progress so the frontend never shows stale state."""
        while True:
            await asyncio.sleep(15)
            _elapsed = int(time.monotonic() - _worker_start_time)
            _qsize = work_queue.qsize() if work_queue else 0
            _active_workers = sum(1 for w in workers if not w.done())
            emit(
                "analyzing",
                "heartbeat",
                f"Analysis: {analyzed_count} done, {len(errors)} errors, "
                f"queue={_qsize}, workers={_active_workers}/{worker_count}, "
                f"elapsed={_elapsed}s",
                status="running",
                data={
                    "analyzed": analyzed_count,
                    "errors": len(errors),
                    "queue_depth": _qsize,
                    "active_workers": _active_workers,
                    "elapsed_s": _elapsed,
                },
            )

    heartbeat_task = asyncio.create_task(_progress_heartbeat())

    # We wait for the worker tasks to complete or a global safety timeout.
    # Since individual tickers are strictly timed out, the workers should never hit this global timeout.
    try:
        async with pipeline_profiler.phase("analysis_drain"):
            # Set to cycle timeout + 5 minutes grace buffer, rather than a hardcoded 1 hour
            cycle_timeout_seconds = (int(getattr(settings, "CYCLE_TIMEOUT_MINUTES", 120)) + 5) * 60.0
            await asyncio.wait_for(asyncio.gather(*workers), timeout=cycle_timeout_seconds)
    except asyncio.TimeoutError:
        logger.critical("[CYCLE] FATAL WORKER POOL TIMEOUT. Workers failed to complete within %ss.", cycle_timeout_seconds)
        emit(
            "analyzing",
            "worker_timeout",
            "Fatal queue timeout. Proceeding with partial results.",
            status="error",
        )
        for w in workers:
            w.cancel()
    finally:
        # Always cancel the heartbeat when analysis completes
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass

    _total_elapsed = int(time.monotonic() - _worker_start_time)
    logger.info(
        "[CYCLE] Analysis phase complete: %d analyzed, %d errors, %ds elapsed",
        analyzed_count, len(errors), _total_elapsed,
    )

    # Calculate summary metrics
    tally_results(results, cycle_summary)

    cycle_summary["analysis_results_count"] = len(results)

    # ── All-Crash Detection Gate ──
    # If every ticker produced a crash-fallback (HOLD @ 0%), the pipeline itself
    # is broken (e.g. import error, variable shadowing). Abort loudly rather than
    # silently passing 30 garbage decisions downstream.
    # FIX: Require at least 3 tickers to trigger the all-crash gate.
    # A single-ticker timeout is common (vLLM overload, slow model) and should
    # NOT abort the entire cycle. For 1-2 ticker batches, we log a warning
    # instead and let the pipeline continue with the fallback HOLD results.
    crash_count = sum(1 for r in results if r.get("is_timeout_fallback"))
    if crash_count > 0 and crash_count == len(results):
        if len(results) >= 3:
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
        else:
            msg = (
                f"WARNING: All {crash_count} ticker(s) crashed, but batch too small "
                f"to trigger all-crash abort (need ≥3). Continuing with fallback HOLD results. "
                f"First error: {results[0].get('error', 'unknown')}"
            )
            logger.warning("[PIPELINE] %s", msg)
            emit("analyzing", "small_batch_crash", msg, status="warning")

    emit(
        "analyzing",
        "pipeline_done",
        f"Analysis complete: {analyzed_count} analyzed, {len(errors)} errors, {_total_elapsed}s elapsed",
        status="ok",
        data={"analyzed": analyzed_count, "errors": errors, "elapsed_s": _total_elapsed},
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
