"""
Database Curator — Background hygiene agent for unstructured data.

Responsibilities:
1. Scans `news_articles`, `reddit_posts`, `youtube_transcripts`.
2. Marks duplicates and low-effort spam as 'merged', 'duplicate', or 'spam'.
3. Generates high-density summaries for rows missing `llm_summary` using the
   Jetson collector node to ensure data is robust and detailed before debate.

Designed to be run as an independent loop or pipeline step.
"""

import asyncio
import logging
from typing import Callable

from app.db.connection import get_db
from app.services.vllm_client import llm, Priority
from app.services.prism_agent_caller import call_prism_agent
from app.utils.pipeline_utils import noop as _noop
from app.utils.text_utils import is_html, clean_html

logger = logging.getLogger(__name__)

CURATOR_PROMPT_MAP = {
    "news": "Summarize this financial news article. Highlight key catalysts, earnings guidance, lawsuits, and management tone. Be concise and data-dense.",
    "reddit": "Summarize this Reddit post analyzing a stock. Highlight sentiment (bull/bear), specific technical/fundamental claims, and whether it's primarily meme hype or serious analysis.",
    "youtube": "Summarize this YouTube transcript about a stock. Extract the core thesis, specific price targets, and key risk factors discussed by the speaker.",
}


async def generate_summary(text: str, source_type: str, max_words: int = 100) -> str:
    """Use the collector LLM to generate a robust summary."""
    sys_prompt = f"You are a strict data curator for a quantitative trading system. {CURATOR_PROMPT_MAP.get(source_type, 'Summarize this context.')} Do not exceed {max_words} words."
    try:
        summary, _, _ = await asyncio.wait_for(
            call_prism_agent(
                agent_id="CUSTOM_DATA_CURATOR_AGENT",
                user_message=text,
                fallback_system_prompt=sys_prompt,
                fallback_agent_name="data_curator",
                temperature=0.2,
                max_tokens=256,
                priority=Priority.LOW,
            ),
            timeout=90.0,
        )
        return summary.strip()
    except Exception as e:
        logger.error(f"[CURATOR] Summary generation failed: {e}")
        return ""


async def run_data_curation(emit: Callable | None = None) -> dict:
    """Main curation entry point. Runs dedup, cleanup, and summarization."""
    if emit is None:
        emit = _noop

    metrics = {"purged": 0, "summarized": 0}

    emit(
        "curator",
        "started",
        "Starting database curation (dedup & summarize)",
        status="running",
    )
    logger.info("[CURATOR] Starting active database curation...")

    # --- Phase 0: Sanitize raw HTML in database summaries ---
    try:
        with get_db() as db:
            html_rows = db.execute("""
                SELECT id, summary FROM news_articles
                WHERE (summary LIKE '<!DOCTYPE%' OR summary LIKE '%<html%' OR summary LIKE '%<script%' OR summary LIKE '%<p>%')
                  AND quality_status IS DISTINCT FROM 'discarded'
                LIMIT 100
            """).fetchall()
            
            cleaned_count = 0
            for row_id, raw_summary in html_rows:
                if raw_summary and is_html(raw_summary):
                    clean_text = clean_html(raw_summary)
                    if clean_text:
                        db.execute(
                            "UPDATE news_articles SET summary = %s WHERE id = %s",
                            [clean_text, row_id]
                        )
                        cleaned_count += 1
            if cleaned_count > 0:
                logger.info(f"[CURATOR] Cleaned up {cleaned_count} existing raw HTML summaries in news_articles.")
    except Exception as e:
        logger.error(f"[CURATOR] Database HTML sanitization failed: {e}")

    # --- Phase 1: Deduplication (News & YouTube) ---
    # Find exact duplicate news titles for the same ticker
    with get_db() as db:
        duplicate_news = db.execute("""
            SELECT ticker, title, COUNT(*), MIN(id) 
            FROM news_articles 
            WHERE quality_status IS NULL OR quality_status = ''
            GROUP BY ticker, title 
            HAVING COUNT(*) > 1
        """).fetchall()

    for row in duplicate_news:
        ticker, title, count, keep_id = row
        with get_db() as db:
            db.execute(
                """
                UPDATE news_articles 
                SET quality_status = 'duplicate', quality_reason = 'Exact title match' 
                WHERE ticker = %s AND title = %s AND id != %s
            """,
                [ticker, title, keep_id],
            )
        metrics["purged"] += count - 1

    # --- Phase 2: Spam/Low-effort removal (Reddit) ---
    # Example: Reddit posts with score <= 0 and no comments
    with get_db() as db:
        spam_reddit = db.execute("""
            SELECT id FROM reddit_posts
            WHERE score <= 0 AND comment_count = 0 AND (summary IS NULL OR summary = '')
        """).fetchall()
    for row in spam_reddit:
        with get_db() as db:
            db.execute(
                "UPDATE reddit_posts SET summary = 'LOW_EFFORT_SPAM_PURGED' WHERE id = %s",
                [row[0]],
            )
        metrics["purged"] += 1

    # --- Phase 3: High-Density Summarization (Jetson Role) ---
    # We take a small batch each run to avoid blocking forever

    # 3a. News Articles without an llm_summary
    with get_db() as db:
        unsummarized_news = db.execute("""
            SELECT id, COALESCE(summary, title) FROM news_articles 
            WHERE llm_summary IS NULL AND (quality_status IS NULL OR quality_status != 'duplicate')
            ORDER BY published_at DESC LIMIT 5
        """).fetchall()

    for uid, content in unsummarized_news:
        if is_html(content):
            content = clean_html(content)
        if len(content) > 50:
            summary = await generate_summary(content, "news")
            if not summary:
                logger.warning("[CURATOR] Summary generation failed or empty. Skipping this news article.")
                continue
            with get_db() as db:
                db.execute(
                    "UPDATE news_articles SET llm_summary = %s, summarized_at = CURRENT_TIMESTAMP WHERE id = %s",
                    [summary, uid],
                )
            metrics["summarized"] += 1

    # 3b. Reddit Posts missing a summary
    with get_db() as db:
        unsummarized_reddit = db.execute("""
            SELECT id, title || ' ' || body FROM reddit_posts 
            WHERE summary IS NULL OR summary = ''
            ORDER BY created_utc DESC LIMIT 5
        """).fetchall()

    for uid, content in unsummarized_reddit:
        if len(content.strip()) > 30 and content != "LOW_EFFORT_SPAM_PURGED":
            summary = await generate_summary(content[:4000], "reddit")
            if not summary:
                logger.warning("[CURATOR] Summary generation failed or empty. Skipping this Reddit post.")
                continue
            with get_db() as db:
                db.execute(
                    "UPDATE reddit_posts SET summary = %s, summarized_at = CURRENT_TIMESTAMP WHERE id = %s",
                    [summary, uid],
                )
            metrics["summarized"] += 1

    # 3c. YouTube Transcripts missing summary
    with get_db() as db:
        unsummarized_yt = db.execute("""
            SELECT video_id, title || ' ' || raw_transcript FROM youtube_transcripts 
            WHERE summary IS NULL OR summary = ''
            ORDER BY published_at DESC LIMIT 3
        """).fetchall()

    for vid, content in unsummarized_yt:
        if len(content.strip()) > 100:
            summary = await generate_summary(content[:6000], "youtube")
            if not summary:
                logger.warning("[CURATOR] Summary generation failed or empty. Skipping this YouTube transcript.")
                continue
            with get_db() as db:
                db.execute(
                    "UPDATE youtube_transcripts SET summary = %s, summarized_at = CURRENT_TIMESTAMP WHERE video_id = %s",
                    [summary, vid],
                )
            metrics["summarized"] += 1

    logger.info(
        f"[CURATOR] Curation complete. Purged/flagged {metrics['purged']} items. Summarized {metrics['summarized']} items."
    )
    emit(
        "curator",
        "finished",
        f"Curator finished: {metrics['purged']} purged, {metrics['summarized']} summarized.",
        status="ok",
    )

    return metrics
