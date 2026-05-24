"""
Flash Briefing Generator
Produces short intraday market summaries every 2 hours during market hours.
Summarizes the latest news headlines with source citations.
"""

import logging
from datetime import datetime, timezone

from app.db.connection import get_db
from app.services.vllm_client import llm, Priority
from app.services.prism_agent_caller import call_prism_agent

logger = logging.getLogger(__name__)


async def generate_flash_briefing() -> str | None:
    """Generate a short flash briefing from the most recently collected news."""
    logger.info("[FLASH] Generating flash briefing...")

    try:
        from app.collectors.news_collector import collect_all
        logger.info("[FLASH] Fetching fresh articles before generating briefing...")
        await collect_all(limit_feeds=10)
    except ImportError:
        logger.info("[FLASH] news_collector not available, skipping article fetch")

    with get_db() as db:
        # Get articles collected in the last 2 hours
        rows = db.execute(
            """
            SELECT title, publisher, url, ticker, published_at
            FROM news_articles
            WHERE collected_at >= NOW() - INTERVAL '2 hours'
            ORDER BY published_at DESC NULLS LAST
            LIMIT 30
            """,
        ).fetchall()

    if len(rows) < 3:
        logger.info("[FLASH] Only %d recent articles — skipping flash briefing.", len(rows))
        return None

    # Build context for the LLM
    articles_text = []
    source_urls = []
    for r in rows:
        title, publisher, url, ticker, pub_at = r
        ticker_str = f" [{ticker}]" if ticker else ""
        pub_str = pub_at.strftime("%H:%M") if pub_at else ""
        articles_text.append(f"• {title}{ticker_str} — {publisher} ({pub_str})")
        if url:
            source_urls.append(url)

    context = "\n".join(articles_text)

    system_prompt = (
        "You are a financial news desk analyst. Given the following recent headlines, "
        "write a concise 200-300 word market flash briefing. "
        "Group related stories together. Highlight the most market-moving items first. "
        "Mention specific tickers where relevant. "
        "End with a 'Sources' section listing the top 5 most important article URLs. "
        "Output in Markdown format."
    )

    response, tokens, ms = await call_prism_agent(
        agent_id="CUSTOM_FLASH_BRIEFING_AGENT",
        user_message=context,
        fallback_system_prompt=system_prompt,
        fallback_agent_name="flash_briefing",
        temperature=0.3,
        max_tokens=800,
        priority=Priority.NORMAL,
    )

    # Save to DB
    try:
        with get_db() as db:
            db.execute(
                """
                INSERT INTO flash_briefings (report_content, source_urls, article_count)
                VALUES (%s, %s, %s)
                """,
                [response, source_urls[:10], len(rows)],
            )
        logger.info("[FLASH] Saved flash briefing (%d articles summarized)", len(rows))
    except Exception as e:
        logger.error("[FLASH] Failed to save: %s", e)

    return response


def get_recent_flash_briefings(limit: int = 10) -> list[dict]:
    """Fetch the most recent flash briefings."""
    from app.utils.tz import utc_iso
    try:
        with get_db() as db:
            rows = db.execute(
                """
                SELECT id, created_at, report_content, source_urls, article_count
                FROM flash_briefings
                ORDER BY created_at DESC
                LIMIT %s
                """,
                [limit],
            ).fetchall()

            return [
                {
                    "id": r[0],
                    "created_at": utc_iso(r[1]),
                    "report_content": r[2],
                    "source_urls": r[3] or [],
                    "article_count": r[4] or 0,
                }
                for r in rows
            ]
    except Exception as e:
        logger.error("[FLASH] Failed to fetch flash briefings: %s", e)
        return []
