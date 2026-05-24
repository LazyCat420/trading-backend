"""
Janitor Mode — Pipeline step to prune irrelevant data (noise) from the database.

Scans recently ingested news, reddit posts, and transcripts. Uses an LLM pass
to determine if the content is actually relevant to the stock market, macro
economics, or specific companies. If not, it marks them as 'discarded'.
"""

import logging
import json
import asyncio
from datetime import datetime
from typing import Callable

from app.db.connection import get_db
from app.services.vllm_client import llm, Priority
from app.services.prism_agent_caller import call_prism_agent
from app.utils.pipeline_utils import noop as _noop
from app.cognition.reflection_utils import generate_critique_prompt

logger = logging.getLogger(__name__)

JANITOR_PROMPT = """You are a strict data janitor for an institutional trading bot.
Your job is to read this text and determine if it is RELEVANT to financial markets.

RELEVANT topics (mark as "relevant" if ANY of these apply):
- Company news for ANY publicly traded company (earnings, products, lawsuits, C-suite changes)
- Macroeconomics: inflation, interest rates, employment, GDP, CPI, PPI, Fed policy, Treasury yields
- Commodities: gold, silver, oil, copper, natural gas, agricultural futures — price moves, supply/demand analysis, or macro drivers
- Global politics ONLY IF it impacts markets (e.g., tariffs, trade deals, sanctions, war, oil supply disruptions)
- Analyst or bank research notes (e.g., UBS, Goldman Sachs, JPMorgan, Morgan Stanley research)
- Stock analysis, technical signals, or sector rotation analysis
- Bond markets, currency moves, or cross-asset macro analysis

IMPORTANT: Macro and commodity news does NOT need a specific ticker from the watchlist to be relevant.
If it discusses inflation data, central bank policy, commodity prices, or analyst macro research, it IS relevant.

NOISE (Must be DISCARDED):
- Generic politics with no direct market or economic impact
- Celebrity gossip or random human interest stories
- Spam, crypto-scams, or completely unidentifiable financial relevance
- "Penny stock" hype that has zero fundamental data backing it
- Discard ONLY if the text has NO connection to financial markets, macro indicators, or commodities.

{context}

Evaluate the following text and respond ONLY in JSON:
{{
  "status": "relevant" | "discarded",
  "reason": "Short explanation why",
  "confidence": 0-100
}}

Text to evaluate:
\"\"\"
{text}
\"\"\"
"""

CRITIC_PROMPT = """You are a QA Evaluator for the Data Janitor.
The Janitor just marked the following text as '{status}'.
Reason given: {reason}

Analyze if the Janitor made the right call. Remember:
- Macro news (inflation, CPI, interest rates, GDP, Fed policy) is ALWAYS relevant even without a specific ticker.
- Commodity analysis (gold, silver, oil, copper) is ALWAYS relevant — commodities are core macro indicators.
- Analyst/bank research notes (UBS, Goldman, etc.) are ALWAYS relevant.
- Only mark INVALID if the Janitor incorrectly discarded content that fits the above, or incorrectly kept pure noise.

If the Janitor was correct, respond with 'VALID'.
If the Janitor was wrong, write a short critique.

Respond ONLY in JSON:
{{
  "verdict": "VALID" | "INVALID",
  "critique": "If INVALID, explain why."
}}

{context}

Text the Janitor evaluated:
\"\"\"
{text}
\"\"\"
"""


async def evaluate_relevance(text: str, context: str = "") -> dict:
    """Use the collector LLM to classify text relevance, with reflection loop."""
    max_retries = 3  # was 10 — reduced to prevent timeout storms (3 × timeout = max 9min, not 30min)
    current_prompt = (
        "You are a JSON-only API. Respond strictly in the requested format."
    )

    for attempt in range(max_retries):
        try:
            response, _, _ = await call_prism_agent(
                agent_id="CUSTOM_DATA_JANITOR_AGENT",
                user_message=JANITOR_PROMPT.format(text=text[:15000], context=context),
                fallback_system_prompt=current_prompt,
                fallback_agent_name="data_janitor",
                temperature=0.1,
                max_tokens=800,
                priority=Priority.LOW,
            )

            import re

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
            else:
                data = {
                    "status": "relevant",
                    "reason": "Failed to parse JSON",
                    "confidence": 50,
                }

            status = data.get("status", "relevant")
            confidence = int(data.get("confidence", 100))

            # Reflection pass if confidence is low or marked as relevant
            if status == "relevant" and confidence < 90:
                critic_res, _, _ = await call_prism_agent(
                    agent_id="CUSTOM_DATA_JANITOR_CRITIC_AGENT",
                    user_message=CRITIC_PROMPT.format(
                        status=status, reason=data.get("reason", ""), context=context, text=text[:15000]
                    ),
                    fallback_system_prompt="You are a strict QA API. Respond ONLY in JSON.",
                    fallback_agent_name="data_janitor_critic",
                    temperature=0.1,
                    max_tokens=150,
                    priority=Priority.LOW,
                )
                critic_match = re.search(r"\{.*\}", critic_res, re.DOTALL)
                if critic_match:
                    critic_data = json.loads(critic_match.group(0))
                    if critic_data.get("verdict", "VALID") == "INVALID":
                        logger.warning(
                            f"[JANITOR] Attempt {attempt + 1} Failed: {critic_data.get('critique')}"
                        )
                        current_prompt = generate_critique_prompt(
                            "You are a JSON-only API. Respond strictly in the requested format.",
                            critic_data.get("critique", "Re-evaluate carefully."),
                            attempt + 1,
                        )
                        continue  # Retry

            return data

        except Exception as e:
            logger.error(
                f"[JANITOR] Relevance evaluation failed on attempt {attempt + 1}: {e}"
            )
            if attempt == max_retries - 1:
                return {"status": "relevant", "reason": f"Error: {e}"}


async def _evaluate_and_tag(
    item_id: str, content: str, context: str, table_name: str
) -> tuple:
    """Helper to evaluate relevance and return result tuple for bulk processing."""
    eval_data = await evaluate_relevance(content, context)
    status = eval_data.get("status", "relevant")
    reason = eval_data.get("reason", "No reason provided")
    return (table_name, item_id, status, reason)


TABLE_UPDATE_MAP = {
    "news_articles": "UPDATE news_articles SET quality_status = %s, quality_reason = %s WHERE id = %s",
    "reddit_posts": "UPDATE reddit_posts SET quality_status = %s, quality_reason = %s WHERE id = %s",
    "youtube_transcripts": "UPDATE youtube_transcripts SET quality_status = %s, quality_reason = %s WHERE video_id = %s",
}


async def run_data_janitor(
    emit: Callable | None = None, tickers: list[str] | None = None
) -> dict:
    """Scan and prune irrelevant noise from recently ingested data concurrently."""
    if emit is None:
        emit = _noop

    metrics = {"scanned": 0, "discarded": 0}

    emit(
        "janitor", "started", "Starting Janitor Mode (Noise Pruning)", status="running"
    )
    logger.info("[JANITOR] Starting data pruning pass...")

    current_date = datetime.now().strftime("%Y-%m-%d")
    context = f"CURRENT DATE: {current_date}\n"
    if tickers:
        context += f"WATCHLIST: {', '.join(tickers)}\n"

    tasks = []

    # 1. Gather News Articles
    with get_db() as db:
        unflagged_news = db.execute("""
            SELECT id, COALESCE(summary, title) 
            FROM news_articles 
            WHERE quality_status IS NULL
            ORDER BY published_at DESC LIMIT 20
        """).fetchall()

    for uid, content in unflagged_news:
        if len(content) > 20:
            tasks.append(_evaluate_and_tag(uid, content, context, "news_articles"))

    # 2. Gather Reddit Posts
    with get_db() as db:
        unflagged_reddit = db.execute("""
            SELECT id, title || ' ' || body 
            FROM reddit_posts 
            WHERE quality_status IS NULL OR quality_status = ''
            ORDER BY created_utc DESC LIMIT 20
        """).fetchall()

    for uid, content in unflagged_reddit:
        if len(content) > 20 and "LOW_EFFORT_SPAM_PURGED" not in content:
            tasks.append(_evaluate_and_tag(uid, content, context, "reddit_posts"))

    # 3. Gather YouTube Transcripts
    with get_db() as db:
        unflagged_yt = db.execute("""
            SELECT video_id, COALESCE(summary, title)
            FROM youtube_transcripts 
            WHERE quality_status IS NULL OR quality_status = ''
            ORDER BY published_at DESC LIMIT 10
        """).fetchall()

    for vid, content in unflagged_yt:
        if len(content) > 20:
            tasks.append(
                _evaluate_and_tag(vid, content, context, "youtube_transcripts")
            )

    if not tasks:
        logger.info("[JANITOR] Pass complete. No data to scan.")
        emit("janitor", "finished", "Janitor found no new items", status="ok")
        return metrics

    # Execute all evaluation tasks with adaptive concurrency (8-16 dynamic limit)
    # Replaces the old unbounded gather() that would dump 30-50 requests at once
    from app.services.adaptive_concurrency import concurrency_controller
    results = await concurrency_controller.gather(tasks, label="data_janitor")

    # Apply database updates sequentially to avoid locking/leak issues
    with get_db() as db:
        for res in results:
            if isinstance(res, Exception):
                logger.error(f"[JANITOR] Task failed: {res}")
                continue
            if not res:
                continue

            table, item_id, status, reason = res
            metrics["scanned"] += 1

            update_sql = TABLE_UPDATE_MAP.get(table)
            if not update_sql:
                logger.error(f"[JANITOR] Unknown table for update: {table}")
                continue

            if status == "discarded":
                metrics["discarded"] += 1
                db.execute(update_sql, [status, reason, item_id])
                logger.info(f"[JANITOR] Discarded {table} {item_id}: {reason}")
            else:
                db.execute(update_sql, [status, None, item_id])

    logger.info(
        f"[JANITOR] Pass complete. Scanned {metrics['scanned']}, Discarded {metrics['discarded']}"
    )
    emit(
        "janitor",
        "finished",
        f"Janitor removed {metrics['discarded']} irrelevant items",
        status="ok",
    )
    return metrics
