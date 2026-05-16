"""
Tier-based distributed worker.

Pulls tasks from Redis queues up to max_tier.
Deploy identically to Jetson, DGX1, DGX2 — worker_config.json controls behavior.

Usage:
    python -m app.worker.worker
    python -m app.worker.worker --config /path/to/worker_config.json
"""

import argparse
import asyncio
import json
import logging
import signal
import time

logger = logging.getLogger(__name__)

# Tier queue names in Redis
TIER_QUEUES = {
    0: "queue:tier0:collect",
    1: "queue:tier1:analyze",
    2: "queue:tier2:validate",
}
RESULTS_QUEUE = "queue:results"
HEARTBEAT_KEY_PREFIX = "worker:heartbeat:"


async def _process_task(task: dict, vllm_endpoint: str) -> dict:
    """Process a single task by dispatching to the appropriate pipeline handler.

    Task types:
      - collect: Run data collection for a ticker (yfinance, news, reddit, etc.)
      - analyze: Run specialist agents in parallel (technical, fundamental, sentiment, etc.)
      - trade:   Run the decision engine for a final BUY/SELL/HOLD verdict
      - curate:  Run LLM curation pass on collected data

    The vllm_endpoint is set per-worker via worker_config.json and determines
    which vLLM server this worker talks to.
    """
    task_type = task.get("type", "unknown")
    ticker = task.get("ticker", "")
    tier = task.get("tier", 0)
    cycle_id = task.get("cycle_id", "")
    bot_id = task.get("bot_id", "default")

    logger.info(
        "[Worker] Processing task: type=%s ticker=%s tier=%d",
        task_type,
        ticker,
        tier,
    )

    result = {
        "task_id": task.get("task_id", ""),
        "ticker": ticker,
        "tier": tier,
        "status": "completed",
        "timestamp": time.time(),
        "vllm_endpoint": vllm_endpoint,
    }

    try:
        if task_type == "collect":
            # Tier 0: Data collection — fetch price, fundamentals, news
            from app.collectors.yfinance_collector import (
                collect_price_history,
                collect_fundamentals,
            )
            from app.collectors.finnhub_collector import collect_news

            await collect_price_history(ticker, period="6mo")
            await collect_fundamentals(ticker)
            await collect_news(ticker)
            result["data"] = {"collected": ["price_history", "fundamentals", "news"]}
            logger.info("[Worker] Collection complete for %s", ticker)

        elif task_type == "analyze":
            # Tier 1: Run specialist agents in parallel
            from app.pipeline.analysis.agent_execution import run_specialist_agents

            agent_results = await run_specialist_agents(
                ticker=ticker,
                cycle_id=cycle_id,
                bot_id=bot_id,
            )
            # Summarize results (don't send full LLM output over Redis)
            result["data"] = {
                "agents_run": list(agent_results.keys()),
                "total_tokens": sum(
                    r.get("tokens_used", 0)
                    for r in agent_results.values()
                    if isinstance(r, dict)
                ),
            }
            logger.info(
                "[Worker] Analysis complete for %s (%d agents)",
                ticker,
                len(agent_results),
            )

        elif task_type == "trade":
            # Tier 2: Run decision engine for final verdict
            from app.pipeline.analysis.decision_engine import run_decision_engine

            decision = await run_decision_engine(
                ticker=ticker,
                cycle_id=cycle_id,
                bot_id=bot_id,
            )
            result["data"] = {
                "decision": decision.get("decision", "HOLD")
                if isinstance(decision, dict)
                else "HOLD",
                "confidence": decision.get("confidence", 0)
                if isinstance(decision, dict)
                else 0,
            }
            logger.info(
                "[Worker] Trading decision for %s: %s",
                ticker,
                result["data"]["decision"],
            )

        elif task_type == "curate":
            # Tier 0: LLM curation pass on collected data
            from app.pipeline.analysis.curation_pass import run_curation

            await run_curation(ticker=ticker, cycle_id=cycle_id, bot_id=bot_id)
            result["data"] = {"curated": True}
            logger.info("[Worker] Curation complete for %s", ticker)

        else:
            logger.warning("[Worker] Unknown task type: %s", task_type)
            result["status"] = "skipped"
            result["error"] = f"Unknown task type: {task_type}"

    except Exception as e:
        logger.error(
            "[Worker] Task failed: type=%s ticker=%s error=%s", task_type, ticker, e
        )
        result["status"] = "error"
        result["error"] = str(e)

    return result


async def _heartbeat_loop(
    redis_client: object,
    worker_id: str,
    interval: int,
) -> None:
    """Publish periodic heartbeats so the orchestrator knows we're alive."""
    import redis.asyncio as aioredis

    r: aioredis.Redis = redis_client  # type: ignore[assignment]
    key = f"{HEARTBEAT_KEY_PREFIX}{worker_id}"

    while True:
        try:
            payload = json.dumps(
                {
                    "worker_id": worker_id,
                    "timestamp": time.time(),
                    "status": "alive",
                }
            )
            # SET with 3x interval TTL so stale heartbeats auto-expire
            await r.set(key, payload, ex=interval * 3)
            logger.debug("[Heartbeat] %s", worker_id)
        except Exception as e:
            logger.warning("[Heartbeat] Failed: %s", e)
        await asyncio.sleep(interval)


async def worker_loop(config: object) -> None:
    """Main worker event loop. Pulls from Redis, processes, escalates."""
    try:
        import redis.asyncio as aioredis
    except ImportError:
        logger.error("redis[asyncio] not installed. Run: pip install redis[hiredis]")
        return

    from app.worker.worker_config import WorkerConfig

    cfg: WorkerConfig = config  # type: ignore[assignment]

    r = aioredis.from_url(cfg.redis_url, decode_responses=True)

    # Verify Redis connectivity
    try:
        pong = await r.ping()
        if not pong:
            raise ConnectionError("Redis PING failed")
        logger.info("[Worker] Redis connected ✓ (%s)", cfg.redis_url)
    except Exception as e:
        logger.error("[Worker] Cannot connect to Redis at %s: %s", cfg.redis_url, e)
        return

    # Build queue list — highest tier first for priority routing
    queues = [TIER_QUEUES[t] for t in range(cfg.max_tier, -1, -1)]
    semaphore = asyncio.Semaphore(cfg.max_parallel_requests)
    shutdown_event = asyncio.Event()

    # Track in-flight tasks for graceful shutdown
    in_flight: set[asyncio.Task] = set()  # type: ignore[type-arg]

    async def handle(raw_queue: str, raw_task: str) -> None:
        """Process a single task with concurrency limiting."""
        async with semaphore:
            try:
                task = json.loads(raw_task)
            except json.JSONDecodeError as e:
                logger.error("[Worker] Bad JSON from %s: %s", raw_queue, e)
                return

            try:
                result = await _process_task(task, cfg.vllm_endpoint)
            except Exception as e:
                logger.error(
                    "[Worker] Task failed: %s — %s",
                    task.get("task_id", "?"),
                    e,
                )
                result = {
                    "task_id": task.get("task_id", ""),
                    "status": "error",
                    "error": str(e),
                    "timestamp": time.time(),
                }

            # Escalate to next tier or push to results
            next_tier = task.get("escalate_to_tier")
            if next_tier is not None and next_tier in TIER_QUEUES:
                escalated = {**task, "result": result, "tier": next_tier}
                await r.rpush(TIER_QUEUES[next_tier], json.dumps(escalated))
                logger.info(
                    "[Worker] Escalated %s → tier %d",
                    task.get("task_id", "?"),
                    next_tier,
                )
            else:
                final = {**task, "result": result}
                await r.rpush(RESULTS_QUEUE, json.dumps(final))

    # Signal handlers for graceful shutdown
    def _request_shutdown() -> None:
        logger.info("[Worker] Shutdown requested — draining in-flight tasks...")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    _signals_registered = False
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_shutdown)
            _signals_registered = True
        except NotImplementedError:
            # Windows doesn't support add_signal_handler for asyncio loops
            pass

    if not _signals_registered:
        # Windows fallback: use signal.signal() for Ctrl+C
        def _win_handler(signum, frame):
            _request_shutdown()

        signal.signal(signal.SIGINT, _win_handler)
        logger.info("[Worker] Using Windows signal handler (Ctrl+C to stop)")

    # Start heartbeat
    hb_task = asyncio.create_task(
        _heartbeat_loop(r, cfg.worker_id, cfg.heartbeat_interval_s)
    )

    logger.info(
        "[Worker] Online | id=%s | tiers=0-%d | parallel=%d | queues=%s",
        cfg.worker_id,
        cfg.max_tier,
        cfg.max_parallel_requests,
        queues,
    )

    try:
        while not shutdown_event.is_set():
            # blpop blocks until any queue has work, highest tier first
            try:
                raw = await r.blpop(queues, timeout=5)
            except Exception as e:
                logger.warning("[Worker] Redis blpop error: %s", e)
                await asyncio.sleep(2)
                continue

            if raw is None:
                # Timeout — no work available, loop back
                continue

            queue_name, task_data = raw
            t = asyncio.create_task(handle(queue_name, task_data))
            in_flight.add(t)
            t.add_done_callback(in_flight.discard)

            # Clean up completed tasks periodically
            done = {t for t in in_flight if t.done()}
            in_flight -= done

    except asyncio.CancelledError:
        logger.info("[Worker] Cancelled")
    finally:
        # Graceful drain
        if in_flight:
            logger.info(
                "[Worker] Draining %d in-flight tasks (timeout=%ds)...",
                len(in_flight),
                cfg.drain_timeout_s,
            )
            done, pending = await asyncio.wait(in_flight, timeout=cfg.drain_timeout_s)
            if pending:
                logger.warning(
                    "[Worker] %d tasks still pending after drain timeout",
                    len(pending),
                )
                for t in pending:
                    t.cancel()

        hb_task.cancel()
        await r.close()
        logger.info("[Worker] Shutdown complete.")


def main() -> None:
    """CLI entry point for the worker."""
    from app.worker.worker_config import load_config

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    ap = argparse.ArgumentParser(description="Tier-based distributed worker")
    ap.add_argument("--config", help="Path to worker_config.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    logger.info("[Worker] Config: %s", cfg.model_dump_json(indent=2))

    asyncio.run(worker_loop(cfg))


if __name__ == "__main__":
    main()
