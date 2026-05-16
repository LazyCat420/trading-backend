"""
Scraper Worker — background consumer for the JIT scraper queue.

Polls the scraper_queue table for PENDING requests, routes each
to the appropriate collector, and marks results as RESOLVED/FAILED.

Runs as an asyncio task alongside the main pipeline. Uses the
existing api_rate_limiter for concurrency control.

Usage:
    from app.pipeline.data.scraper_worker import start_scraper_worker, stop_scraper_worker

    # In pipeline startup:
    worker_task = asyncio.create_task(start_scraper_worker(emit=cls.emit))

    # In pipeline shutdown:
    await stop_scraper_worker(worker_task)
"""

import asyncio
import logging
import time

from app.config import settings
from app.pipeline.data.scraper_queue import (
    get_pending_requests,
    mark_processing,
    mark_resolved,
    mark_failed,
    cleanup_stale,
)

logger = logging.getLogger(__name__)

# Maps data_type_requested → collector function path
# Each collector returns (rows_fetched: int) or raises on failure
_COLLECTOR_REGISTRY: dict[str, str] = {
    "news": "app.collectors.news_collector",
    "reddit": "app.collectors.reddit_collector",
    "youtube": "app.collectors.youtube_collector",
    "price": "app.collectors.yfinance_collector",
    "fundamentals": "app.collectors.yfinance_collector",
    "options": "app.collectors.yfinance_collector",
    "technicals": "app.processors.technical_calculator",
}

# Maps data_type → specific function name in the collector module
_COLLECTOR_FUNCTIONS: dict[str, str] = {
    "news": "collect_news",
    "reddit": "collect_reddit",
    "youtube": "collect_youtube_for_ticker",
    "price": "collect_price_history",
    "fundamentals": "collect_fundamentals",
    "options": "collect_options_chain",
    "technicals": "compute_technicals",
}

_running = False


async def _dispatch_request(request: dict) -> None:
    """Route a single scraper request to its collector.

    Imports the collector lazily to avoid circular imports.
    Respects Rule 6: collectors are pure data, no LLM calls.
    """
    data_type = request["data_type"]
    ticker = request["ticker"]
    request_id = request["id"]

    module_path = _COLLECTOR_REGISTRY.get(data_type)
    func_name = _COLLECTOR_FUNCTIONS.get(data_type)

    if not module_path or not func_name:
        mark_failed(request_id, f"Unknown data_type: {data_type}")
        return

    if not mark_processing(request_id):
        # Another worker claimed it
        return

    start = time.monotonic()
    try:
        # Lazy import the collector module
        import importlib

        module = importlib.import_module(module_path)
        func = getattr(module, func_name, None)

        if func is None:
            mark_failed(request_id, f"Function {func_name} not found in {module_path}")
            return

        # All collectors are async or sync — handle both
        if asyncio.iscoroutinefunction(func):
            result = await func(ticker)
        else:
            # Run sync collectors in thread pool to avoid blocking
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, func, ticker)

        elapsed_ms = int((time.monotonic() - start) * 1000)
        mark_resolved(request_id)

        logger.info(
            "[SCRAPER_WORKER] ✅ %s/%s completed in %dms (lens=%s)",
            ticker,
            data_type,
            elapsed_ms,
            request.get("requested_by_lens", "none"),
        )

    except Exception as e:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        error_msg = f"{type(e).__name__}: {str(e)[:200]}"
        mark_failed(request_id, error_msg)

        logger.warning(
            "[SCRAPER_WORKER] ❌ %s/%s failed in %dms: %s",
            ticker,
            data_type,
            elapsed_ms,
            error_msg,
        )


async def start_scraper_worker(
    emit=None,
    poll_interval: int | None = None,
) -> None:
    """Main worker loop — polls queue and dispatches requests.

    Runs indefinitely until stop_scraper_worker() is called.

    Args:
        emit: Optional event callback for pipeline progress reporting
        poll_interval: Seconds between queue polls (default from settings)
    """
    global _running
    _running = True

    if poll_interval is None:
        poll_interval = settings.SCRAPER_WORKER_POLL_SECS

    logger.info(
        "[SCRAPER_WORKER] Starting background scraper worker (poll=%ds)", poll_interval
    )

    if emit:
        emit(
            "collecting",
            "scraper_worker_start",
            "JIT Scraper Worker started",
            status="ok",
        )

    stale_check_counter = 0

    while _running:
        try:
            # Periodic stale request cleanup (every 12 polls)
            stale_check_counter += 1
            if stale_check_counter >= 12:
                stale_check_counter = 0
                cleanup_stale(timeout_minutes=10)

            # Fetch next batch of pending requests
            requests = get_pending_requests(limit=5)

            if not requests:
                await asyncio.sleep(poll_interval)
                continue

            # Process requests concurrently (limited batch)
            tasks = [_dispatch_request(req) for req in requests]
            await asyncio.gather(*tasks, return_exceptions=True)

            # Prevent infinite spin loops if requests fail instantly
            await asyncio.sleep(0.5)

        except asyncio.CancelledError:
            logger.info("[SCRAPER_WORKER] Worker cancelled, shutting down")
            break
        except Exception as e:
            logger.error("[SCRAPER_WORKER] Worker loop error: %s", e)
            await asyncio.sleep(poll_interval)

    logger.info("[SCRAPER_WORKER] Worker stopped")


async def stop_scraper_worker(task: asyncio.Task | None) -> None:
    """Gracefully stop the scraper worker task."""
    global _running
    _running = False

    if task and not task.done():
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=3.0)
        except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
            pass

    logger.info("[SCRAPER_WORKER] Worker stopped cleanly")
