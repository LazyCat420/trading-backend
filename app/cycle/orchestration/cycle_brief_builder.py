import logging
from app.db.connection import get_db
from app.pipeline.best_result_store import load_best_result

logger = logging.getLogger(__name__)

# Hard cap on each warm-start block to prevent context bloat (issue #7)
_MAX_BLOCK_CHARS = 800


def build_warm_start_brief(ticker: str) -> str:
    """Builds the warm-start brief prefix for the LLM context.

    Each block is capped at _MAX_BLOCK_CHARS to prevent warm-start context
    from pushing actual market data past the vLLM context window.
    """

    parts = ["# WARM-START BRIEF (Historical Context)"]

    # 1. Best historical result
    best = load_best_result(ticker)
    if best:
        rationale = (best.get("rationale") or "")[:300]
        block = (
            f"## Previous Best Analysis\n"
            f"- Action: {best.get('action')}\n"
            f"- Confidence: {best.get('confidence')}%\n"
            f"- Rationale: {rationale}"
        )
        parts.append(block[:_MAX_BLOCK_CHARS])

    # 2. Debate history (last winner, TTL 7 days — issue #5)
    try:
        with get_db() as db:
            debate_row = db.execute(
                "SELECT winner, final_action, final_confidence, persona_name, key_risk "
                "FROM debate_history "
                "WHERE ticker = %s "
                "  AND created_at > NOW() - INTERVAL '7 days' "
                "ORDER BY created_at DESC LIMIT 1",
                (ticker,),
            ).fetchone()

            if debate_row:
                block = (
                    f"## Last Debate Outcome\n"
                    f"- Winner: {debate_row[0]}\n"
                    f"- Final Action: {debate_row[1]} @ {debate_row[2]}%\n"
                    f"- Persona: {debate_row[3]}\n"
                    f"- Key Risk: {(debate_row[4] or 'N/A')[:200]}"
                )
                parts.append(block[:_MAX_BLOCK_CHARS])
    except Exception as e:
        logger.debug("Failed to fetch debate history: %s", e)

    # 3. Last cycle summary (TTL 7 days)
    try:
        with get_db() as db:
            summary_row = db.execute(
                "SELECT total_tickers, buy_count, sell_count, hold_count, "
                "avg_confidence, top_ticker, lesson_summary "
                "FROM autoresearch_cycle_summaries "
                "WHERE created_at > NOW() - INTERVAL '7 days' "
                "ORDER BY created_at DESC LIMIT 1"
            ).fetchone()

            if summary_row:
                avg_conf = summary_row[4] or 0
                block = (
                    f"## Last Cycle Summary\n"
                    f"- Tickers Analyzed: {summary_row[0]}\n"
                    f"- Outcomes: {summary_row[1]} BUY / {summary_row[2]} SELL / {summary_row[3]} HOLD\n"
                    f"- Avg Confidence: {avg_conf:.0f}%\n"
                    f"- Top Pick: {summary_row[5] or 'N/A'}\n"
                    f"- Lesson: {(summary_row[6] or 'N/A')[:200]}"
                )
                parts.append(block[:_MAX_BLOCK_CHARS])
    except Exception as e:
        logger.debug("Failed to fetch cycle summary: %s", e)

    if len(parts) == 1:
        return ""  # Don't prepend if empty

    return "\n\n".join(parts)
