"""
Morning Briefing — Compare and contrast recent stock reports at the start of the trading day.
"""

import logging
from datetime import datetime, timezone
import json

from app.db.connection import get_db
from app.trading.portfolio import get_current_state
from app.trading.watchlist import get_active
from app.pipeline.analysis.thesis_store import get_thesis
from app.services.vllm_client import llm, Priority
from app.services.prism_agent_caller import call_prism_agent

logger = logging.getLogger(__name__)

async def generate_morning_briefing() -> str:
    """Generate a morning briefing comparing recent stock analyses."""
    logger.info("[MORNING BRIEFING] Generating morning briefing...")
    
    # 1. Gather target universe (Portfolio + Watchlist)
    state = get_current_state()
    portfolio_tickers = [p["ticker"] for p in state.get("positions", [])]
    
    watchlist_tickers = [w["ticker"] for w in get_active()]
    
    target_tickers = list(set(portfolio_tickers + watchlist_tickers))
    logger.info(f"[MORNING BRIEFING] Target universe: {len(target_tickers)} tickers")
    
    # 2. Extract recent theses
    context_parts = []
    evaluated_tickers = []
    
    for ticker in target_tickers:
        thesis = get_thesis(ticker)
        if thesis:
            # Check if thesis is relatively recent (e.g., within the last 5 days)
            age_days = (datetime.now(timezone.utc) - thesis.updated_at).total_seconds() / (3600 * 24)
            if age_days <= 5:
                context_parts.append(
                    f"## {ticker}\n"
                    f"Verdict: {thesis.verdict}\n"
                    f"Confidence: {thesis.confidence}%\n"
                    f"Updated At: {thesis.updated_at.strftime('%Y-%m-%d %H:%M UTC')}\n"
                    f"Summary: {thesis.summary}\n"
                )
                evaluated_tickers.append(ticker)
    
    if not context_parts:
        logger.warning("[MORNING BRIEFING] No recent theses found for the target universe.")
        return "No recent data available to generate a morning briefing."
        
    context = "\n".join(context_parts)
    
    system_prompt = (
        "You are the Head of Strategy at a trading desk. It is the start of the trading day. "
        "Review the following recent analysis reports (theses) for our tracked stocks. "
        "Compare and contrast them. "
        "1. Identify any sector-wide trends, correlations, or divergences.\n"
        "2. Rank the top 2 BUY candidates and highlight the top 2 SELL/risk candidates.\n"
        "3. Highlight any conflicting signals or macro risks affecting multiple tickers.\n"
        "Output a cohesive, highly readable morning briefing in Markdown."
    )
    
    logger.info(f"[MORNING BRIEFING] Running LLM analysis on {len(evaluated_tickers)} theses...")
    
    # 3. Run LLM
    response, tokens, ms = await call_prism_agent(
        agent_id="CUSTOM_MORNING_BRIEFING_AGENT",
        user_message=context,
        fallback_system_prompt=system_prompt,
        fallback_agent_name="morning_briefing_analyst",
        temperature=0.3,
        max_tokens=1500,
        priority=Priority.HIGH,
    )
    
    # 4. Save to DB
    try:
        with get_db() as db:
            db.execute(
                """
                INSERT INTO morning_briefings (report_content, tickers_evaluated)
                VALUES (%s, %s)
                """,
                [response, evaluated_tickers]
            )
        logger.info("[MORNING BRIEFING] Saved to database successfully.")
    except Exception as e:
        logger.error(f"[MORNING BRIEFING] Failed to save to DB: {e}")
        
    return response

from app.utils.tz import utc_iso

def get_latest_morning_briefing() -> dict | None:
    """Fetch the most recent morning briefing from the database."""
    try:
        with get_db() as db:
            row = db.execute(
                """
                SELECT id, created_at, report_content, tickers_evaluated
                FROM morning_briefings
                ORDER BY created_at DESC
                LIMIT 1
                """
            ).fetchone()
            
            if row:
                return {
                    "id": row[0],
                    "created_at": utc_iso(row[1]),
                    "report_content": row[2],
                    "tickers_evaluated": row[3]
                }
    except Exception as e:
        logger.error(f"[MORNING BRIEFING] Failed to fetch latest briefing: {e}")
    return None
