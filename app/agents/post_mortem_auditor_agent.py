"""
Post-Mortem Auditor Agent — Retrospective audit on closed trades.

Phase 6: Post-Mortem Auditor Agent. Runs immediately after a position is closed.
Analyzes the trade setup, exit conditions, and overall P&L, then updates the
decision outcomes database and lesson store.
"""

import logging
import json
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ValidationError

from app.agents.base_agent import run_agent
from app.utils.text_utils import parse_json_response
from app.db.connection import get_db

logger = logging.getLogger(__name__)

POST_MORTEM_SYSTEM_PROMPT = """You are the Post-Mortem Auditor Agent. Your job is to conduct a retrospective analysis on a closed stock trade.

Given the trade details (entry, exit, P&L) and recent market news during the holding period, analyze the decision.
Address the following points:
1. Was the entry thesis correct?
2. Was the exit well-timed (e.g. stop-loss triggered, target reached, or manual panic)?
3. What is the key lesson learned to improve future cycles for this stock or sector?

## OUTPUT:
Respond with JSON:
{
    "grade": "A|B|C|D|F",
    "lessons_learned": "Detailed 2-3 sentences explaining what went right/wrong and the key takeaway.",
    "thesis_validation": "Did the original trade thesis hold true? Explain.",
    "execution_rating": "Review of the exit timing and execution quality."
}

CRITICAL: Do NOT invent news or price data. Be objective and critical of past decisions to prevent repeating mistakes."""


class PostMortemResponse(BaseModel):
    grade: str = Field(..., description="Trade grade A-F")
    lessons_learned: str = Field(..., description="Key takeaway lessons")
    thesis_validation: str = Field(..., description="Thesis validation details")
    execution_rating: str = Field(..., description="Rating of the execution/exit")


async def run_post_mortem(
    ticker: str,
    entry_price: float,
    exit_price: float,
    pnl_pct: float,
    cycle_id: str,
    bot_id: str,
) -> dict:
    """Run retrospective post-mortem on a closed position.

    Queries news/history, calls the Post-Mortem agent, and writes updates.
    """
    logger.info(
        "[POST_MORTEM] Auditing closed trade for %s: entry=$%.2f exit=$%.2f P&L=%+.2f%%",
        ticker,
        entry_price,
        exit_price,
        pnl_pct,
    )

    # 1. Fetch recent news and price context for the agent to analyze
    context_lines = [
        f"## Trade Summary for {ticker}",
        f"- Entry Price: ${entry_price:.2f}",
        f"- Exit Price: ${exit_price:.2f}",
        f"- P&L percentage: {pnl_pct:+.2f}%",
    ]

    try:
        with get_db() as db:
            news_rows = db.execute(
                """
                SELECT title, publisher, published_at, summary
                FROM news_articles
                WHERE ticker = %s
                ORDER BY published_at DESC LIMIT 5
            """,
                [ticker],
            ).fetchall()
            if news_rows:
                context_lines.append("\n## Recent News Articles during/after trade:")
                for title, pub, dt, summary in news_rows:
                    context_lines.append(f"- [{pub}] {title} ({dt})")
                    if summary:
                        context_lines.append(f"  Summary: {summary[:150]}...")
    except Exception:
        pass

    news_context = "\n".join(context_lines)

    result = await run_agent(
        agent_name="post_mortem",
        ticker=ticker,
        cycle_id=cycle_id,
        bot_id=bot_id,
        system_prompt=POST_MORTEM_SYSTEM_PROMPT,
        user_prompt=(
            f"Conduct a post-mortem review of the closed trade for {ticker} using the following details:\n\n"
            f"{news_context}\n\n"
            f"Identify if the exit was optimal and formulate the key lesson learned. Return the required JSON schema."
        ),
        max_tokens=1024,
        enable_tools=True,
    )

    response_text = result.get("response", "")
    parsed_json = parse_json_response(response_text)

    if not parsed_json:
        logger.warning("[POST_MORTEM] Failed to parse agent output for %s.", ticker)
        return {}

    try:
        validated = PostMortemResponse(**parsed_json)
        lessons = validated.lessons_learned
        grade = validated.grade

        logger.info(
            "[POST_MORTEM] Audit complete for %s: Grade=%s | Lessons=%s",
            ticker,
            grade,
            lessons[:80],
        )

        # 2. Write the retrospective update back to PostgreSQL decision_outcomes
        try:
            with get_db() as db:
                db.execute(
                    """
                    UPDATE decision_outcomes
                    SET lesson_stored = %s
                    WHERE ticker = %s AND action = 'BUY' AND resolved_at IS NOT NULL
                    AND id = (SELECT id FROM decision_outcomes WHERE ticker = %s AND action = 'BUY' AND resolved_at IS NOT NULL ORDER BY resolved_at DESC LIMIT 1)
                """,
                    [f"Grade {grade}: {lessons}", ticker, ticker],
                )
                logger.info("[POST_MORTEM] Updated database outcome log for %s", ticker)
        except Exception as db_err:
            logger.warning("[POST_MORTEM] Database update failed: %s", db_err)

        # 3. Dual-write to lesson_store
        try:
            from app.cognition.lesson_store import add_lesson
            lesson_text = (
                f"[{grade}] Retrospective for {ticker}: entry=${entry_price:.2f} exit=${exit_price:.2f} "
                f"PnL={pnl_pct:+.1f}%. Thesis validation: {validated.thesis_validation}. Lesson: {lessons}"
            )
            add_lesson(
                text=lesson_text,
                metadata={
                    "session_id": f"post_mortem_{datetime.now(timezone.utc).strftime('%b%d').lower()}",
                    "round": 0,
                    "score": round(pnl_pct, 2),
                    "status": "WIN" if pnl_pct > 0.5 else "LOSS",
                    "source": "post_mortem_agent",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            logger.info("[POST_MORTEM] Lesson saved to lesson_store.")
        except Exception as store_err:
            logger.warning("[POST_MORTEM] Lesson store write failed: %s", store_err)

        return validated.model_dump()

    except ValidationError as e:
        logger.warning("[POST_MORTEM] Pydantic validation failed: %s", e)
        return {}
