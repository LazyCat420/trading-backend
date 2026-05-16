"""
Box Scorecard — Per-endpoint performance summary at cycle end.

Queries llm_audit_logs for the cycle and generates a structured
performance report per hardware box (Jetson, DGX Spark, Goldspark).
The scorecard is:
  1. Printed to terminal (for operator visibility)
  2. Returned as dict (for storage in cycle_run_summaries.summary_json)
  3. Available to autoresearch for self-optimization feedback
"""

import logging
from app.db.connection import get_db

logger = logging.getLogger(__name__)


def generate_box_scorecard(cycle_id: str) -> dict:
    """Generate per-endpoint performance scorecard for a cycle.

    Returns dict keyed by endpoint_name with stats, plus an
    '_aggregate' key with cycle-level totals.
    """
    scorecard = {}

    try:
        with get_db() as db:
            # Per-endpoint breakdown
            rows = db.execute(
                """
                SELECT
                    COALESCE(endpoint_name, 'unknown') as ep,
                    COUNT(*) as calls,
                    COALESCE(SUM(tokens_used), 0) as total_tokens,
                    COALESCE(SUM(prompt_tokens), 0) as total_prompt,
                    COALESCE(SUM(completion_tokens), 0) as total_completion,
                    COALESCE(AVG(execution_ms), 0) as avg_latency_ms,
                    COALESCE(MIN(execution_ms), 0) as min_latency_ms,
                    COALESCE(MAX(execution_ms), 0) as max_latency_ms,
                    COALESCE(SUM(execution_ms), 0) as total_ms,
                    COALESCE(AVG(queue_wait_ms), 0) as avg_queue_wait_ms,
                    COALESCE(AVG(tokens_per_second), 0) as avg_tok_per_sec,
                    model
                FROM llm_audit_logs
                WHERE cycle_id = %s
                GROUP BY ep, model
                ORDER BY total_tokens DESC
                """,
                [cycle_id],
            ).fetchall()

            for r in rows:
                ep_name = r[0]
                model_name = r[11] or "unknown"

                entry = {
                    "model": model_name,
                    "calls": r[1],
                    "total_tokens": r[2],
                    "prompt_tokens": r[3],
                    "completion_tokens": r[4],
                    "avg_latency_ms": round(r[5]),
                    "min_latency_ms": r[6],
                    "max_latency_ms": r[7],
                    "total_time_s": round(r[8] / 1000, 1),
                    "avg_queue_wait_ms": round(r[9]),
                    "avg_tok_per_sec": round(r[10], 1),
                    # Derived: aggregate throughput
                    "aggregate_tok_per_sec": round(r[2] / (r[8] / 1000), 1)
                    if r[8] > 0
                    else 0,
                }
                scorecard[ep_name] = entry

            # Aggregate totals
            agg = db.execute(
                """
                SELECT
                    COUNT(*) as calls,
                    COALESCE(SUM(tokens_used), 0) as total_tokens,
                    COALESCE(SUM(execution_ms), 0) as total_ms,
                    COALESCE(AVG(execution_ms), 0) as avg_ms,
                    COALESCE(AVG(queue_wait_ms), 0) as avg_queue_wait_ms
                FROM llm_audit_logs
                WHERE cycle_id = %s
                """,
                [cycle_id],
            ).fetchone()

            if agg:
                scorecard["_aggregate"] = {
                    "total_calls": agg[0],
                    "total_tokens": agg[1],
                    "total_time_s": round(agg[2] / 1000, 1) if agg[2] else 0,
                    "avg_latency_ms": round(agg[3]) if agg[3] else 0,
                    "avg_queue_wait_ms": round(agg[4]) if agg[4] else 0,
                }

            # Slowest calls (top 5)
            slow = db.execute(
                """
                SELECT agent_step, ticker, execution_ms,
                       COALESCE(endpoint_name, 'unknown') as ep
                FROM llm_audit_logs
                WHERE cycle_id = %s
                ORDER BY execution_ms DESC
                LIMIT 5
                """,
                [cycle_id],
            ).fetchall()

            scorecard["_slowest"] = [
                {
                    "agent_step": s[0],
                    "ticker": s[1],
                    "execution_ms": s[2],
                    "endpoint": s[3],
                }
                for s in slow
            ]

    except Exception as e:
        logger.error("[BOX_SCORECARD] Query failed: %s", e)
        return {}

    return scorecard


def print_box_scorecard(scorecard: dict) -> None:
    """Print a human-readable box scorecard to the logger."""
    if not scorecard:
        return

    lines = ["", "╔═══════════════════════════════════════════════════════════════╗"]
    lines.append("║              BOX PERFORMANCE SCORECARD                        ║")
    lines.append("╠═══════════════════════════════════════════════════════════════╣")

    for ep_name, stats in scorecard.items():
        if ep_name.startswith("_"):
            continue

        lines.append(f"║  {ep_name.upper():20s}  ({stats.get('model', '?')[:30]})")
        lines.append(
            f"║    Calls: {stats['calls']:>5,d}  │  Tokens: {stats['total_tokens']:>10,d}  "
            f"│  Time: {stats['total_time_s']:>6.0f}s"
        )
        lines.append(
            f"║    Prompt: {stats['prompt_tokens']:>8,d}  │  Completion: {stats['completion_tokens']:>8,d}  "
            f"│  Tok/s: {stats['aggregate_tok_per_sec']:>6.1f}"
        )
        lines.append(
            f"║    Avg Latency: {stats['avg_latency_ms']:>6,d}ms  │  Queue Wait: {stats['avg_queue_wait_ms']:>5,d}ms  "
            f"│  Min/Max: {stats['min_latency_ms']:,}/{stats['max_latency_ms']:,}ms"
        )
        lines.append("║")

    agg = scorecard.get("_aggregate", {})
    if agg:
        lines.append(
            "╠═══════════════════════════════════════════════════════════════╣"
        )
        lines.append(
            f"║  TOTAL: {agg.get('total_calls', 0):>5,d} calls  │  "
            f"{agg.get('total_tokens', 0):>12,d} tokens  │  "
            f"{agg.get('total_time_s', 0):>6.0f}s"
        )

    slowest = scorecard.get("_slowest", [])
    if slowest:
        lines.append("║  Slowest calls:")
        for s in slowest[:3]:
            lines.append(
                f"║    {s['agent_step']:30s} {s['ticker']:6s} "
                f"{s['execution_ms']:>8,d}ms ({s['endpoint']})"
            )

    lines.append("╚═══════════════════════════════════════════════════════════════╝")

    for line in lines:
        logger.info("[BOX_SCORECARD] %s", line)
