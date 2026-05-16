"""
Cycle Benchmark — Compute and persist per-cycle and per-ticker timing.

Reads the in-memory event log from pipeline.py at cycle completion,
aggregates timing data, and writes to cycle_benchmarks + cycle_ticker_benchmarks.

This lets us compare run 1 (cold, all API calls) vs run 2 (warm, cache hits)
and see exactly how much time/work the caching system saves.
"""

import logging
from datetime import datetime

from app.db.connection import get_db

logger = logging.getLogger(__name__)


def persist_benchmark(cycle_state: dict) -> dict | None:
    """Extract timing from cycle events and write to DB.

    Args:
        cycle_state: The _cycle_state dict from pipeline.py
            Must have: cycle_id, started_at, finished_at, events, tickers

    Returns:
        Summary dict with computed stats, or None on error.
    """
    cycle_id = cycle_state.get("cycle_id")
    if not cycle_id:
        return None

    events = cycle_state.get("events", [])
    tickers = cycle_state.get("tickers", [])
    status = cycle_state.get("status", "unknown")
    requested_version = cycle_state.get("requested_version", "v2")
    effective_version = cycle_state.get("effective_version", "v2")
    benchmark_group = cycle_state.get("benchmark_group", "baseline")
    execution_mode = cycle_state.get("execution_mode", "production")
    v2_stage = cycle_state.get("v2_stage", 0)

    try:
        # ── Compute cycle-level stats ──
        started_at = cycle_state.get("started_at")
        finished_at = cycle_state.get("finished_at")

        total_ms = 0
        if started_at and finished_at:
            try:
                t0 = datetime.fromisoformat(started_at)
                t1 = datetime.fromisoformat(finished_at)
                total_ms = int((t1 - t0).total_seconds() * 1000)
            except Exception:
                pass

        # Phase timing from events
        collect_ms = _sum_phase_ms(events, "collecting")
        analyze_ms = _sum_phase_ms(events, "analyzing")
        trade_ms = _sum_phase_ms(events, "trading")

        # Step counts
        steps_total = 0
        steps_skipped = 0
        steps_ok = 0
        steps_error = 0
        total_tokens = 0

        # Collector steps for cache hit calculation
        collector_steps = 0
        collector_skipped = 0

        for evt in events:
            s = evt.get("status", "")
            if s in ("ok", "skipped", "error"):
                steps_total += 1
            if s == "skipped":
                steps_skipped += 1
            if s == "ok":
                steps_ok += 1
            if s == "error":
                steps_error += 1
            # Sum tokens from event data
            data = evt.get("data", {})
            if isinstance(data, dict):
                total_tokens += data.get("tokens", 0)

            # Track collector cache hits
            if evt.get("phase") == "collecting" and s in ("ok", "skipped"):
                step = evt.get("step", "")
                # Per-ticker collector steps (yfinance_X, finnhub_X, reddit_X, etc.)
                if _is_ticker_collector_step(step):
                    collector_steps += 1
                    if s == "skipped":
                        collector_skipped += 1

        cache_hit_pct = (
            round(collector_skipped / collector_steps * 100, 1)
            if collector_steps > 0
            else 0.0
        )

        avg_ticker_ms = int(total_ms / max(1, len(tickers)))

        # ── Compute per-ticker stats ──
        ticker_stats = {}
        for ticker in tickers:
            ticker_stats[ticker] = _compute_ticker_stats(events, ticker)

        # ── Persist to DB ──
        with get_db() as db:
            db.execute(
                """
                INSERT INTO cycle_benchmarks
                (cycle_id, requested_version, effective_version, benchmark_group, execution_mode, v2_stage,
                 started_at, finished_at, total_ms, avg_ticker_ms,
                 collect_ms, analyze_ms, trade_ms,
                 ticker_count, steps_total, steps_skipped,
                 steps_ok, steps_error, total_tokens,
                 cache_hit_pct, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (cycle_id) DO NOTHING
            """,
                [
                    cycle_id,
                    requested_version,
                    effective_version,
                    benchmark_group,
                    execution_mode,
                    v2_stage,
                    started_at,
                    finished_at,
                    total_ms,
                    avg_ticker_ms,
                    collect_ms,
                    analyze_ms,
                    trade_ms,
                    len(tickers),
                    steps_total,
                    steps_skipped,
                    steps_ok,
                    steps_error,
                    total_tokens,
                    cache_hit_pct,
                    status,
                ],
            )

            for ticker, stats in ticker_stats.items():
                db.execute(
                    """
                    INSERT INTO cycle_ticker_benchmarks
                    (cycle_id, ticker, collect_ms, analyze_ms, total_ms,
                     steps_skipped, steps_ok, tokens_used,
                     action, confidence)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (cycle_id, ticker) DO NOTHING
                """,
                    [
                        cycle_id,
                        ticker,
                        stats["collect_ms"],
                        stats["analyze_ms"],
                        stats["total_ms"],
                        stats["steps_skipped"],
                        stats["steps_ok"],
                        stats["tokens_used"],
                        stats["action"],
                        stats["confidence"],
                    ],
                )

            summary = {
                "cycle_id": cycle_id,
                "requested_version": requested_version,
                "effective_version": effective_version,
                "benchmark_group": benchmark_group,
                "execution_mode": execution_mode,
                "v2_stage": v2_stage,
                "total_ms": total_ms,
                "collect_ms": collect_ms,
                "analyze_ms": analyze_ms,
                "trade_ms": trade_ms,
                "ticker_count": len(tickers),
                "steps_total": steps_total,
                "steps_skipped": steps_skipped,
                "cache_hit_pct": cache_hit_pct,
                "total_tokens": total_tokens,
                "ticker_stats": ticker_stats,
            }
            logger.info(
                "[BENCHMARK] Persisted: %s | %dms total | %d/%d steps skipped (%.1f%% cache)",
                cycle_id,
                total_ms,
                steps_skipped,
                steps_total,
                cache_hit_pct,
            )
            return summary

    except Exception as e:
        logger.error("[PIPELINE] [BENCHMARK] Failed to persist: %s", e)
        return None


def aggregate_benchmarks_by_execution_mode() -> list[dict]:
    """Return aggregated benchmark rows grouped by runtime lane."""
    with get_db() as db:
        rows = db.execute(
            """
            SELECT
                effective_version,
                benchmark_group,
                execution_mode,
                v2_stage,
                COUNT(*) AS cycles,
                AVG(total_ms) AS avg_total_ms,
                AVG(total_tokens) AS avg_total_tokens,
                AVG(cache_hit_pct) AS avg_cache_hit_pct,
                SUM(CASE WHEN status = 'done' THEN 1 ELSE 0 END) AS done_cycles
            FROM cycle_benchmarks
            GROUP BY effective_version, benchmark_group, execution_mode, v2_stage
            ORDER BY cycles DESC, effective_version ASC
            """
        ).fetchall()
        return [
            {
                "effective_version": row[0],
                "benchmark_group": row[1],
                "execution_mode": row[2],
                "v2_stage": row[3],
                "cycles": row[4],
                "avg_total_ms": float(row[5] or 0),
                "avg_total_tokens": float(row[6] or 0),
                "avg_cache_hit_pct": float(row[7] or 0),
                "done_cycles": row[8],
            }
            for row in rows
        ]


def _sum_phase_ms(events: list[dict], phase: str) -> int:
    """Sum elapsed_ms for all events in a given phase."""
    total = 0
    for evt in events:
        if evt.get("phase") == phase:
            total += evt.get("elapsed_ms", 0)
    return total


def _is_ticker_collector_step(step: str) -> bool:
    """Check if a step name is a per-ticker collector step.

    Matches patterns like: yfinance_NVDA, finnhub_AAPL, reddit_GOOGL, etc.
    Excludes global steps like: pass1_global, discovery, dedup.
    """
    prefixes = (
        "yfinance_",
        "finnhub_",
        "reddit_",
        "youtube_",
        "yf_news_",
    )
    return any(step.startswith(p) for p in prefixes)


def _extract_ticker_from_step(step: str) -> str | None:
    """Extract ticker symbol from a step name like 'yfinance_NVDA'."""
    prefixes = (
        "yfinance_",
        "finnhub_",
        "reddit_",
        "youtube_",
        "yf_news_",
        "agents_",
        "agent_technical_",
        "agent_fundamental_",
        "agent_sentiment_",
        "agent_fund_flow_",
        "agent_risk_",
        "rlm_config_c_",
        "rlm_config_d_",
        "debate_",
        "start_",
        "decision_",
        "escalation_",
        "no_escalation_",
        "data_completeness_",
    )
    for p in prefixes:
        if step.startswith(p):
            return step[len(p) :].upper()
    return None


def _compute_ticker_stats(events: list[dict], ticker: str) -> dict:
    """Compute timing stats for a single ticker from events."""
    collect_ms = 0
    analyze_ms = 0
    steps_skipped = 0
    steps_ok = 0
    tokens = 0
    action = ""
    confidence = 0

    ticker_upper = ticker.upper()
    for evt in events:
        step = evt.get("step", "")
        extracted = _extract_ticker_from_step(step)
        if extracted != ticker_upper:
            continue

        ms = evt.get("elapsed_ms", 0)
        status = evt.get("status", "")
        phase = evt.get("phase", "")

        if phase == "collecting":
            collect_ms += ms
        elif phase == "analyzing":
            analyze_ms += ms

        if status == "skipped":
            steps_skipped += 1
        elif status == "ok":
            steps_ok += 1

        data = evt.get("data", {})
        if isinstance(data, dict):
            tokens += data.get("tokens", 0)
            if data.get("action"):
                action = data["action"]
            if data.get("confidence"):
                confidence = data["confidence"]

    return {
        "collect_ms": collect_ms,
        "analyze_ms": analyze_ms,
        "total_ms": collect_ms + analyze_ms,
        "steps_skipped": steps_skipped,
        "steps_ok": steps_ok,
        "tokens_used": tokens,
        "action": action,
        "confidence": confidence,
    }
