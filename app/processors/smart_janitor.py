"""
Smart Janitor Agent.
Filters out noise/spam from scraped articles and pre-processes relevant items
into structured qualitative drafts (category, impact, relevance, facts) to save downstream tokens.
"""

import json
import logging
import datetime
from app.db.connection import get_db
from app.services.prism_agent_caller import call_prism_agent
from app.services.vllm_client import Priority

logger = logging.getLogger(__name__)

SMART_JANITOR_SYSTEM_PROMPT = """You are a qualitative data-cleansing janitor for a quantitative hedge fund.
Your task is to review a scraped news article or social media post for a given stock ticker and decide if it is signal or noise.

NOISE (Discard):
- Low-effort clickbait, speculation, generic market summaries with no stock-specific detail.
- Promotional articles, advertisements, PR teasers without factual disclosures.
- Duplicate news covering already-processed events with no new details.

SIGNAL (Keep):
- Structural events: CEO/C-suite transitions, M&A activity, long-term partnerships, factory expansions.
- Transient events: Earnings reports, material layoffs, lawsuits, product recalls, supply chain disruptions.

If it is NOISE, return a JSON object with:
{
  "decision": "discard",
  "justification": "Brief explanation of why this is noise."
}

If it is SIGNAL, return a JSON object with:
{
  "decision": "keep",
  "category": "structural" or "transient",
  "impact": "bullish" or "bearish" or "neutral",
  "suggested_theme": "A short 2-4 word theme name (e.g. 'CEO Transition', 'Workforce Restructure', 'Mexico Factory Expansion')",
  "relevance_label": "Critical Core Thesis" or "Major Strategic Catalyst" or "Significant Risk Factor" or "Moderate Operational Headwind" or "Minor Narrative Noise",
  "bullet_points": [
    "Dense factual bullet point 1 (cite dates/numbers if any)",
    "Dense factual bullet point 2"
  ],
  "justification": "Brief explanation of how this affects the company's story."
}

Ensure the output is strictly valid JSON without any markdown formatting around it."""

async def run_smart_janitor_for_article(article_id: str | int) -> bool:
    """
    Fetch a raw news article, run the Smart Janitor Agent, and save the structured draft to the database.
    """
    try:
        with get_db() as db:
            row = db.execute(
                "SELECT title, publisher, summary, url, published_at, ticker FROM news_articles WHERE id = %s",
                [article_id]
            ).fetchone()
            if not row:
                logger.warning(f"[JANITOR] News article {article_id} not found.")
                return False
                
            title, publisher, summary, url, published_at, ticker = row
            
        # Call LLM
        user_message = f"""Ticker: {ticker}
Source: News Article
Publisher: {publisher}
Published At: {published_at.isoformat() if isinstance(published_at, datetime.datetime) else published_at}
Title: {title}
Summary/Snippet: {summary or "No summary content."}
"""
        response, _, _ = await call_prism_agent(
            agent_id="CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
            user_message=user_message,
            fallback_system_prompt=SMART_JANITOR_SYSTEM_PROMPT,
            fallback_agent_name="smart_janitor",
            temperature=0.1,
            max_tokens=1024,
            priority=Priority.NORMAL,
            ticker=ticker,
        )
        
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```json")[-1].split("```")[0].strip()
            
        draft_data = json.loads(cleaned)
        
        with get_db() as db:
            db.execute(
                "UPDATE news_articles SET qualitative_draft = %s::jsonb WHERE id = %s",
                [json.dumps(draft_data), article_id]
            )
        logger.info(f"[JANITOR] Processed article {article_id} for {ticker} -> {draft_data.get('decision')}")
        return True
        
    except Exception as e:
        logger.error(f"[JANITOR] Failed to run Smart Janitor for article {article_id}: {e}", exc_info=True)
        return False

async def run_smart_janitor_for_reddit(post_id: str | int) -> bool:
    """
    Fetch a Reddit post, run the Smart Janitor Agent, and save the structured draft to the database.
    """
    try:
        with get_db() as db:
            row = db.execute(
                "SELECT title, subreddit, body, created_utc, ticker FROM reddit_posts WHERE id = %s",
                [post_id]
            ).fetchone()
            if not row:
                logger.warning(f"[JANITOR] Reddit post {post_id} not found.")
                return False
                
            title, subreddit, body, created_utc, ticker = row
            
        user_message = f"""Ticker: {ticker}
Source: Reddit Post
Subreddit: r/{subreddit}
Created At: {created_utc.isoformat() if isinstance(created_utc, datetime.datetime) else created_utc}
Title: {title}
Body: {body or "No body content."}
"""
        response, _, _ = await call_prism_agent(
            agent_id="CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
            user_message=user_message,
            fallback_system_prompt=SMART_JANITOR_SYSTEM_PROMPT,
            fallback_agent_name="smart_janitor",
            temperature=0.1,
            max_tokens=1024,
            priority=Priority.NORMAL,
            ticker=ticker,
        )
        
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```json")[-1].split("```")[0].strip()
            
        draft_data = json.loads(cleaned)
        
        with get_db() as db:
            db.execute(
                "UPDATE reddit_posts SET qualitative_draft = %s::jsonb WHERE id = %s",
                [json.dumps(draft_data), post_id]
            )
        logger.info(f"[JANITOR] Processed Reddit post {post_id} for {ticker} -> {draft_data.get('decision')}")
        return True
        
    except Exception as e:
        logger.error(f"[JANITOR] Failed to run Smart Janitor for Reddit post {post_id}: {e}", exc_info=True)
        return False
