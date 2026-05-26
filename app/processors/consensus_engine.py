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
Real-world news often covers multiple unrelated stories. Do NOT force a single umbrella consensus if the articles cover distinct topics.

1. Group the articles into distinct stories/themes (e.g., "Land Acquisition", "New Retail Partnership", "Earnings Report").
2. Extract the key structured facts from each story (numbers, locations, partners, products, etc.).
3. Classify each story's signal (bullish/bearish/neutral) and explain why.
4. Flag NOISE: Articles that are generic profiles, mention the ticker in passing without a specific event, or have no actionable financial content.
5. Flag true OUTLIERS: Articles that genuinely contradict the facts of other articles or make unverified unique claims that conflict with the consensus.
6. TIME HORIZONS: You MUST clearly distinguish between short-term volatility/noise (e.g., daily price swings, temporary bottlenecks, immediate news catalysts) and long-term fundamental trends (e.g., YTD gains, structural market shifts, multi-year thesis). Do not conflate the two.

Return ONLY a valid JSON object in exactly this format:
{
  "stories": [
    {
      "theme": "Brief description of the story/theme",
      "article_ids": ["id123", "id456"],
      "key_facts": ["Specific fact 1", "Specific fact 2"],
      "time_horizon": "short-term OR long-term OR both",
      "signal": "bullish",
      "signal_reason": "Why this story is bullish/bearish/neutral, explicitly separating short-term catalysts from long-term trends if applicable"
    }
  ],
  "noise": [
    {"article_id": "id789", "reason": "Generic retail strategy profile, no specific event"}
  ],
  "outliers": [
    {"article_id": "id999", "reason": "Only source claiming a merger, contradicts others"}
  ],
  "consensus_summary": "A 1-2 sentence overall summary of the valid, non-noise news."
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
        from app.services.prism_agent_caller import call_prism_agent
        response, _, _ = await call_prism_agent(
            agent_id="CUSTOM_CONSENSUS_ENGINE_AGENT",
            user_message=payload,
            fallback_system_prompt=CONSENSUS_PROMPT,
            fallback_agent_name="consensus_engine",
            temperature=0.2,
            max_tokens=2048,
            priority=Priority.NORMAL,
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


async def run_consensus_engine(emit: Callable | None = None, ticker: str | None = None) -> dict:
    """Main consensus entry point. Groups recent articles and generates consensus/outliers."""
    if emit is None:
        emit = _noop

    metrics = {"tickers_processed": 0, "outliers_flagged": 0}

    emit(
        "consensus",
        "started",
        f"Starting news consensus and outlier detection{f' for ticker {ticker}' if ticker else ''}",
        status="running",
    )
    logger.info(f"[CONSENSUS] Starting News Consensus Engine...{f' for ticker {ticker}' if ticker else ''}")

    # Fetch recent valid articles (last 7 days)
    seven_days_ago = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=7)
    
    with get_db() as db:
        if ticker:
            ticker = ticker.upper()
            # Freshness check: skip if consensus was updated within last 10 mins
            fresh_consensus = db.execute("""
                SELECT last_updated FROM ticker_consensus
                WHERE ticker = %s AND last_updated >= CURRENT_TIMESTAMP - INTERVAL '10 minutes'
            """, [ticker]).fetchone()
            if fresh_consensus:
                logger.info(f"[CONSENSUS] Skipping {ticker}: consensus is already fresh (updated at {fresh_consensus[0]})")
                emit(
                    "consensus",
                    "finished",
                    f"Consensus for {ticker} is already fresh.",
                    status="ok",
                )
                return metrics

            rows = db.execute("""
                SELECT id, ticker, publisher, title, summary, llm_summary 
                FROM news_articles 
                WHERE published_at >= %s AND ticker = %s
                AND (quality_status IS NULL OR quality_status IN ('ok', 'accepted', 'relevant'))
                ORDER BY published_at DESC
            """, [seven_days_ago, ticker]).fetchall()
        else:
            rows = db.execute("""
                SELECT id, ticker, publisher, title, summary, llm_summary 
                FROM news_articles 
                WHERE published_at >= %s 
                AND (quality_status IS NULL OR quality_status IN ('ok', 'accepted', 'relevant'))
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

    from app.collectors.news_collector import _is_article_relevant_to_ticker
    
    # Run consensus for tickers with at least 3 relevant articles concurrently
    tasks = []
    for ticker, articles in ticker_groups.items():
        if not ticker:
            continue
        # Freshness check: skip if consensus was updated within last 10 mins
        with get_db() as db:
            fresh_consensus = db.execute("""
                SELECT last_updated FROM ticker_consensus
                WHERE ticker = %s AND last_updated >= CURRENT_TIMESTAMP - INTERVAL '10 minutes'
            """, [ticker.upper()]).fetchone()
            if fresh_consensus:
                logger.debug(f"[CONSENSUS] Skipping {ticker}: consensus is already fresh")
                continue
        # Relevance pre-filter
        relevant_articles = []
        for a in articles:
            full_text = f"{a['title']} {a['summary']} {a['llm_summary'] or ''}"
            if _is_article_relevant_to_ticker(ticker, full_text):
                relevant_articles.append(a)
        
        dropped = len(articles) - len(relevant_articles)
        if dropped > 0:
            logger.debug(f"[CONSENSUS] Dropped {dropped} irrelevant articles for {ticker}")

        if len(relevant_articles) < 3:
            if len(articles) >= 3:
                logger.info(f"[CONSENSUS] Skipping {ticker}: after relevance filtering, only {len(relevant_articles)}/{len(articles)} articles remain (need 3+)")
            continue
            
        # We cap at top 15 most recent to not blow context window
        recent_articles = relevant_articles[:15]
        tasks.append((ticker, recent_articles))

    if not tasks:
        logger.info("[CONSENSUS] Complete. No tickers had enough articles.")
        emit("consensus", "finished", "Consensus finished: 0 processed.", status="ok")
        return metrics

    async def process_ticker(ticker, recent_articles):
        result = await generate_consensus(ticker, recent_articles)
        if not result:
            return None
        return ticker, result

    results = await asyncio.gather(*(process_ticker(t, a) for t, a in tasks))
    
    with get_db() as db:
        for res in results:
            if not res or not res[1]:
                continue
            
            ticker, consensus_data = res
            
            # Dump the full structured JSON into the consensus text field
            consensus_text = json.dumps(consensus_data)
            
            # 1. Update consensus
            db.execute("""
                INSERT INTO ticker_consensus (ticker, consensus, last_updated)
                VALUES (%s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (ticker) 
                DO UPDATE SET consensus = EXCLUDED.consensus, last_updated = CURRENT_TIMESTAMP
            """, [ticker, consensus_text])
            
            # 2. Flag noise to exclude from future analysis
            noise_list = consensus_data.get("noise", [])
            for n in noise_list:
                aid = n.get("article_id")
                reason = n.get("reason", "Flagged as noise by consensus engine")
                if aid:
                    db.execute("""
                        UPDATE news_articles 
                        SET quality_status = 'noise', quality_reason = %s
                        WHERE id = %s
                    """, [reason, aid])
            
            # 3. Flag outliers for Human-In-The-Loop review
            outliers = consensus_data.get("outliers", [])
            for out in outliers:
                aid = out.get("article_id")
                reason = out.get("reason", "Flagged as outlier by consensus engine")
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
