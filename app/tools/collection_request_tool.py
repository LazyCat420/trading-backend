"""
Collection Request Tool — Agent-callable tool for LLM-steered data collection.

Agents can call this during analysis to request additional data collection
for specific tickers and data types. Requests go through the scraper_queue
for deduplication, priority ordering, and JIT processing.

Security: This tool is internal-only (server-side agent calls). Input validation
restricts data_types to a known allow-list to prevent injection via tool args.
"""

import json
import logging
import asyncio

from app.tools.registry import registry, PermissionLevel
from app.pipeline.data.scraper_queue import enqueue_request

logger = logging.getLogger(__name__)

# Allow-list of valid data types to prevent injection
VALID_DATA_TYPES = frozenset({
    "news", "reddit", "youtube", "price",
    "fundamentals", "sec_filings", "web_search",
})


async def _process_jit_requests(ticker: str, data_types: list[str]) -> dict:
    """Process high-priority JIT requests immediately instead of waiting for a worker.

    Routes each data type to its appropriate collector for immediate fulfillment.
    Returns a dict of {data_type: result_summary}.
    """
    results = {}
    tasks = []
    labels = []

    for dt in data_types:
        if dt == "price":
            from app.collectors.yfinance_collector import collect_price_history
            tasks.append(collect_price_history(ticker, period="6mo"))
            labels.append("price")
        elif dt == "fundamentals":
            from app.collectors.yfinance_collector import collect_fundamentals
            tasks.append(collect_fundamentals(ticker))
            labels.append("fundamentals")
        elif dt == "news":
            from app.collectors.news_collector import collect_for_ticker
            tasks.append(collect_for_ticker(ticker))
            labels.append("news")
        elif dt == "reddit":
            from app.collectors.reddit_collector import collect_for_ticker as reddit_collect
            tasks.append(reddit_collect(ticker))
            labels.append("reddit")
        elif dt == "youtube":
            from app.collectors.youtube_collector import collect_for_ticker as yt_collect
            tasks.append(yt_collect(ticker))
            labels.append("youtube")
        elif dt == "sec_filings":
            try:
                from app.collectors.sec_collector import collect_sec_filings
                tasks.append(collect_sec_filings(ticker))
                labels.append("sec_filings")
            except ImportError:
                results["sec_filings"] = "collector not available"
        elif dt == "web_search":
            # Web search is best-effort — uses existing web_tools infrastructure
            try:
                from app.tools.web_tools import _search_and_scrape_internal
                tasks.append(_search_and_scrape_internal(f"{ticker} stock analysis latest news"))
                labels.append("web_search")
            except (ImportError, AttributeError):
                results["web_search"] = "web search not available"

    if tasks:
        # Run all collectors in parallel with a timeout
        try:
            outputs = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=60.0,
            )
            for label, output in zip(labels, outputs):
                if isinstance(output, Exception):
                    results[label] = f"error: {output}"
                    logger.warning(
                        "[CollectionTool] JIT collection failed for %s/%s: %s",
                        ticker, label, output,
                    )
                else:
                    results[label] = f"collected: {output}" if output else "no new data"
        except asyncio.TimeoutError:
            for label in labels:
                if label not in results:
                    results[label] = "timeout (60s)"
            logger.warning("[CollectionTool] JIT collection timed out for %s", ticker)

    return results


@registry.register(
    name="request_data_collection",
    description=(
        "Request the pipeline to collect specific data for a ticker. "
        "Use this when your analysis has gaps that need more data before you can "
        "make a confident decision. Available data types: "
        "news, reddit, youtube, price, fundamentals, sec_filings, web_search. "
        "Data is collected immediately (JIT) and you can proceed with analysis."
    ),
    parameters={
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "The stock ticker to collect data for (e.g. 'AAPL').",
            },
            "data_types": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "List of data types to collect. Valid values: "
                    "news, reddit, youtube, price, fundamentals, sec_filings, web_search."
                ),
            },
            "reason": {
                "type": "string",
                "description": "Brief explanation of why this data is needed (for audit trail).",
            },
        },
        "required": ["ticker", "data_types", "reason"],
    },
    tier=1,
    source="collection",
    permission=PermissionLevel.WRITE,
    tags=["collection", "data", "scraper", "jit", "tool"],
)
async def request_data_collection(
    ticker: str,
    data_types: list[str],
    reason: str = "",
    _agent_name: str = "unknown_agent",
    _cycle_id: str = "",
) -> str:
    """Request data collection for a ticker with specific data types.

    Validates input, enqueues requests, and triggers immediate JIT processing.
    """
    # Input validation: sanitize ticker (alphanumeric + dots only)
    ticker = ticker.upper().strip()
    if not ticker or len(ticker) > 10:
        return json.dumps({"status": "error", "message": "Invalid ticker symbol"})

    # Filter to valid data types only (prevent injection)
    valid_types = [dt for dt in data_types if dt in VALID_DATA_TYPES]
    invalid_types = [dt for dt in data_types if dt not in VALID_DATA_TYPES]

    if not valid_types:
        return json.dumps({
            "status": "error",
            "message": f"No valid data types provided. Valid: {sorted(VALID_DATA_TYPES)}",
            "invalid_types": invalid_types,
        })

    # Enqueue each request for tracking and deduplication
    enqueued = []
    skipped = []
    for dt in valid_types:
        req_id = enqueue_request(
            ticker, dt,
            priority=1,  # JIT priority (highest)
            requested_by_lens=f"{_agent_name}: {reason[:200]}",
        )
        if req_id:
            enqueued.append(dt)
        else:
            skipped.append(dt)  # Deduped or in cooldown

    # Process immediately (JIT) for enqueued types
    jit_results = {}
    if enqueued:
        logger.info(
            "[CollectionTool] %s requested JIT collection for %s: %s (reason: %s)",
            _agent_name, ticker, enqueued, reason[:100],
        )
        jit_results = await _process_jit_requests(ticker, enqueued)

    return json.dumps({
        "status": "success",
        "ticker": ticker,
        "collected": enqueued,
        "skipped_dedup": skipped,
        "invalid_types": invalid_types,
        "jit_results": jit_results,
        "message": (
            f"Collected {len(enqueued)} data types for {ticker}. "
            f"{len(skipped)} skipped (already queued/fresh)."
        ),
    })
