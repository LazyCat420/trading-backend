import logging
import json
import asyncio
import datetime
from typing import Callable

from app.db.connection import get_db
from app.services.vllm_client import llm, Priority
from app.utils.pipeline_utils import noop as _noop

logger = logging.getLogger(__name__)

CONSENSUS_PROMPT = """You are a senior news editor for a quantitative trading firm.
Your task is to analyze a batch of recent news articles about a specific ticker.
1. Synthesize the core CONSENSUS (what everyone agrees on, main events).
2. Identify OUTLIERS: Articles that present data/rumors/claims that are unique to them, or contradict the consensus.
   - For each outlier, return its exact Article ID.
   - Explain why it is an outlier (e.g. 'Only source claiming a merger', 'Contradicts earnings consensus').

Return ONLY a valid JSON object in this format:
{
  "consensus": "The general market consensus is that...",
  "outliers": [
    {"article_id": "id123", "reason": "Claims an unverified lawsuit..."}
  ]
}
"""

async def generate_consensus(ticker: str, articles: list[dict]) -> dict:
    """Pass articles to LLM and generate consensus JSON."""
    
    # Build payload
    payload = f"TICKER: {ticker}\n\n"
    for a in articles:
        content = a["llm_summary"] if a.get("llm_summary") else a.get("summary") or a.get("title")
        payload += f"--- ARTICLE ID: {a['id']} ---\n"
        payload += f"PUBLISHER: {a['publisher']}\n"
        payload += f"CONTENT: {content}\n\n"
        
    try:
        response, _, _ = await llm.chat(
            system=CONSENSUS_PROMPT,
            user=payload,
            temperature=0.2,
            max_tokens=2048,
            priority=Priority.NORMAL,
            agent_name="consensus_engine",
        )
        
        import re
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL | re.IGNORECASE)
        if match:
            response = match.group(1)
        else:
            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1:
                response = response[start:end+1]
                
        return json.loads(response)
    except Exception as e:
        logger.error(f"[CONSENSUS] LLM generation failed for {ticker}: {e}")
        return {}


async def run_consensus_engine(emit: Callable | None = None) -> dict:
    """Main consensus entry point. Groups recent articles and generates consensus/outliers."""
    if emit is None:
        emit = _noop

    metrics = {"tickers_processed": 0, "outliers_flagged": 0}

    emit(
        "consensus",
        "started",
        "Starting news consensus and outlier detection",
        status="running",
    )
    logger.info("[CONSENSUS] Starting News Consensus Engine...")

    # Fetch recent valid articles (last 7 days)
    seven_days_ago = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=7)
    
    with get_db() as db:
        rows = db.execute("""
            SELECT id, ticker, publisher, title, summary, llm_summary 
            FROM news_articles 
            WHERE published_at >= %s 
            AND (quality_status IS NULL OR quality_status = 'ok')
            ORDER BY published_at DESC
        """, [seven_days_ago]).fetchall()

    if not rows:
        return metrics

    # Group by ticker
    ticker_groups = {}
    for r in rows:
        t = r[1]
        if t not in ticker_groups:
            ticker_groups[t] = []
        ticker_groups[t].append({
            "id": r[0],
            "publisher": r[2],
            "title": r[3],
            "summary": r[4],
            "llm_summary": r[5]
        })

    # Run consensus for tickers with at least 3 articles concurrently
    tasks = []
    for ticker, articles in ticker_groups.items():
        if len(articles) < 3:
            continue
            
        # We cap at top 15 most recent to not blow context window
        recent_articles = articles[:15]
        tasks.append((ticker, recent_articles))

    if not tasks:
        logger.info("[CONSENSUS] Complete. No tickers had enough articles.")
        emit("consensus", "finished", "Consensus finished: 0 processed.", status="ok")
        return metrics

    async def process_ticker(ticker, recent_articles):
        result = await generate_consensus(ticker, recent_articles)
        if not result:
            return None
        return ticker, result.get("consensus", ""), result.get("outliers", [])

    results = await asyncio.gather(*(process_ticker(t, a) for t, a in tasks))
    
    with get_db() as db:
        for res in results:
            if not res or not res[1]:
                continue
            
            ticker, consensus_text, outliers = res
            
            # 1. Update consensus
            db.execute("""
                INSERT INTO ticker_consensus (ticker, consensus, last_updated)
                VALUES (%s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (ticker) 
                DO UPDATE SET consensus = EXCLUDED.consensus, last_updated = CURRENT_TIMESTAMP
            """, [ticker, consensus_text])
            
            # 2. Flag outliers for Human-In-The-Loop review
            for out in outliers:
                aid = out.get("article_id")
                reason = out.get("reason", "Flagged by consensus engine")
                if aid:
                    db.execute("""
                        UPDATE news_articles 
                        SET quality_status = 'pending_review', quality_reason = %s
                        WHERE id = %s
                    """, [reason, aid])
                    metrics["outliers_flagged"] += 1
                    
            metrics["tickers_processed"] += 1
        
    logger.info(
        f"[CONSENSUS] Complete. Processed {metrics['tickers_processed']} tickers, flagged {metrics['outliers_flagged']} outliers."
    )
    emit(
        "consensus",
        "finished",
        f"Consensus finished: {metrics['tickers_processed']} processed, {metrics['outliers_flagged']} flagged.",
        status="ok",
    )

    return metrics
