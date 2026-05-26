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

IMPORTANT TICKER BUCKETING RULE:
- If the text has valid company signal but is about a DIFFERENT ticker or multiple tickers (e.g. it discusses Adobe (ADBE) instead of the given ticker Hims (HIMS)), do NOT discard it. Mark it as "keep", and list all relevant tickers (e.g. `["ADBE"]`) in the `actual_tickers` array.
- Discard ONLY if the text has no company-specific signal at all (e.g. generic politics, spam, celebrity gossip, or clickbait).

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
  "actual_tickers": ["TICKER1", "TICKER2"], // List all publicly traded stock tickers this text is actually about/relevant to. If it's about the given ticker, include it.
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
        
        decision = draft_data.get("decision", "keep")
        actual_tickers = draft_data.get("actual_tickers", [ticker])
        if isinstance(actual_tickers, str):
            actual_tickers = [actual_tickers]
        actual_tickers = [t.upper() for t in actual_tickers if isinstance(t, str)]

        validated_tickers = []
        if decision == "keep":
            from app.processors.ticker_extractor import get_registry, validate_unknown_tickers
            from app.trading.watchlist import is_banned
            
            registry = get_registry()
            for t in actual_tickers:
                if is_banned(t):
                    continue
                if registry.is_known(t):
                    validated_tickers.append(t)
                else:
                    try:
                        validated = await validate_unknown_tickers([t])
                        if validated.get(t):
                            validated_tickers.append(t)
                    except Exception as e:
                        logger.warning(f"[JANITOR] Failed to validate ticker {t}: {e}")

        original_ticker_in_list = ticker in validated_tickers

        for t in validated_tickers:
            if t == ticker:
                continue
            
            from app.collectors.news_collector import _get_article_id
            new_id = _get_article_id(title, t)
            new_draft = draft_data.copy()
            new_draft["actual_tickers"] = [t]
            
            try:
                with get_db() as db:
                    exists = db.execute("SELECT 1 FROM news_articles WHERE id = %s", [new_id]).fetchone()
                    if not exists:
                        db.execute(
                            """
                            INSERT INTO news_articles (id, ticker, title, publisher, url, published_at, summary, source, qualitative_draft, quality_status, collected_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, 'relevant', CURRENT_TIMESTAMP)
                            """,
                            [
                                new_id,
                                t,
                                title,
                                publisher,
                                url,
                                published_at,
                                summary,
                                "janitor_routing",
                                json.dumps(new_draft),
                            ]
                        )
                        logger.info(f"[JANITOR] Cloned/routed article {article_id} to ticker {t}")
                    else:
                        db.execute(
                            "UPDATE news_articles SET qualitative_draft = %s::jsonb, quality_status = 'relevant' WHERE id = %s",
                            [json.dumps(new_draft), new_id]
                        )
            except Exception as e:
                logger.error(f"[JANITOR] Failed to insert/route cloned article for {t}: {e}")

        if decision == "keep" and not original_ticker_in_list:
            draft_data["decision"] = "discard"
            draft_data["justification"] = f"Re-routed to: {', '.join(validated_tickers)}"

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
                """
                SELECT title, subreddit, body, created_utc, ticker,
                       score, upvote_ratio, comment_count, flair,
                       sentiment_score, award_count, comment_velocity
                FROM reddit_posts WHERE id = %s
                """,
                [post_id]
            ).fetchone()
            if not row:
                logger.warning(f"[JANITOR] Reddit post {post_id} not found.")
                return False
                
            (title, subreddit, body, created_utc, ticker,
             score, upvote_ratio, comment_count, flair,
             sentiment_score, award_count, comment_velocity) = row
            
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
            fallback_agent_name="smart_janior",
            temperature=0.1,
            max_tokens=1024,
            priority=Priority.NORMAL,
            ticker=ticker,
        )
        
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```json")[-1].split("```")[0].strip()
            
        draft_data = json.loads(cleaned)
        
        decision = draft_data.get("decision", "keep")
        actual_tickers = draft_data.get("actual_tickers", [ticker])
        if isinstance(actual_tickers, str):
            actual_tickers = [actual_tickers]
        actual_tickers = [t.upper() for t in actual_tickers if isinstance(t, str)]

        validated_tickers = []
        if decision == "keep":
            from app.processors.ticker_extractor import get_registry, validate_unknown_tickers
            from app.trading.watchlist import is_banned
            
            registry = get_registry()
            for t in actual_tickers:
                if is_banned(t):
                    continue
                if registry.is_known(t):
                    validated_tickers.append(t)
                else:
                    try:
                        validated = await validate_unknown_tickers([t])
                        if validated.get(t):
                            validated_tickers.append(t)
                    except Exception as e:
                        logger.warning(f"[JANITOR] Failed to validate ticker {t}: {e}")

        original_ticker_in_list = ticker in validated_tickers

        if "_" in str(post_id):
            raw_post_id = str(post_id).rsplit("_", 1)[0]
        else:
            raw_post_id = post_id

        for t in validated_tickers:
            if t == ticker:
                continue
            
            new_id = f"{raw_post_id}_{t}"
            new_draft = draft_data.copy()
            new_draft["actual_tickers"] = [t]
            
            try:
                with get_db() as db:
                    exists = db.execute("SELECT 1 FROM reddit_posts WHERE id = %s", [new_id]).fetchone()
                    if not exists:
                        db.execute(
                            """
                            INSERT INTO reddit_posts (id, ticker, subreddit, title, body, score, upvote_ratio, comment_count, flair, sentiment_score, award_count, comment_velocity, created_utc, qualitative_draft, quality_status, collected_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, 'relevant', CURRENT_TIMESTAMP)
                            """,
                            [
                                new_id,
                                t,
                                subreddit,
                                title,
                                body,
                                score,
                                upvote_ratio,
                                comment_count,
                                flair,
                                sentiment_score,
                                award_count,
                                comment_velocity,
                                created_utc,
                                json.dumps(new_draft),
                            ]
                        )
                        logger.info(f"[JANITOR] Cloned/routed Reddit post {post_id} to ticker {t}")
                    else:
                        db.execute(
                            "UPDATE reddit_posts SET qualitative_draft = %s::jsonb, quality_status = 'relevant' WHERE id = %s",
                            [json.dumps(new_draft), new_id]
                        )
            except Exception as e:
                logger.error(f"[JANITOR] Failed to insert/route cloned Reddit post for {t}: {e}")

        if decision == "keep" and not original_ticker_in_list:
            draft_data["decision"] = "discard"
            draft_data["justification"] = f"Re-routed to: {', '.join(validated_tickers)}"

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
