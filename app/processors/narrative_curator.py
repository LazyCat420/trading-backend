"""
Narrative Curator Agent.
Evolves the persistent company story/narrative by synthesizing new news with the existing narrative.
"""

import json
import logging
import datetime
from app.db.connection import get_db
from app.services.prism_agent_caller import call_prism_agent
from app.services.vllm_client import Priority

logger = logging.getLogger(__name__)

NARRATIVE_SYSTEM_PROMPT = """You are a senior qualitative narrative architect for a quantitative hedge fund.
Your task is to maintain and update the "Company Story / Narrative" for the given stock ticker.

We track the company's story using two fields:
1. "story_summary": A high-level, data-dense summary (2-3 paragraphs) of the company's current core narrative, long-term catalysts, active themes, and ongoing challenges.
2. "key_themes": A list of active/ongoing or resolved storylines (e.g., CEO transition, product recalls, expansion plans, lawsuits, product launches, debt pressures).

You are given:
- The existing Narrative Summary and Key Themes.
- A list of recent news/Reddit/YouTube article summaries and social posts collected since the last update.
- The company's current financial profile (balance sheet indicators, profitability, and debt metrics).

Your job is to:
- Blend the new information into the existing narrative.
- Weigh qualitative news developments against the actual financial health of the company. For example, if the company is heavily leveraged (high debt-to-equity, low current ratio) or has high cash burn, a capital-intensive expansion plans theme should be labeled as a "Significant Risk Factor" with a bearish bias rather than a major bullish catalyst.
- Maintain and update ongoing stories. DO NOT automatically discard older stories just because they are old. Instead, evaluate if they are still part of the active company story.
- Qualitatively decay theme relevance over time. If a transient theme (e.g., an earnings miss, minor lawsuit, or temporary news spike) is no longer mentioned in the new batch and time has passed:
  * Move its status to "decaying" or "resolved".
  * Demote its "market_relevance_label" (e.g. from "Significant Risk Factor" to "Minor Narrative Noise").
  * Once a theme is resolved or decays to negligible importance, archive it by removing it from the active "key_themes" list and reflecting any long-term consequences in the "story_summary" paragraph.
- Return ONLY a valid JSON object matching this schema:
{
  "story_summary": "Evolving 2-3 paragraph summary of the company's active narrative.",
  "key_themes": [
    {
      "theme": "Theme Name",
      "category": "structural" or "transient",
      "status": "active" or "decaying" or "resolved",
      "impact": "bullish" or "bearish" or "neutral",
      "market_relevance_label": "Critical Core Thesis" or "Major Strategic Catalyst" or "Significant Risk Factor" or "Moderate Operational Headwind" or "Minor Narrative Noise",
      "qualitative_severity_summary": "Detailed explanation explaining why this theme is weighted this way, specifically linking it to the company's financial context (e.g. debt levels, cash runway, margins).",
      "decay_profile": "persistent" or "standard_decay" or "fast_decay",
      "summary": "Brief description of the theme's current state.",
      "first_seen": "YYYY-MM-DD",
      "last_seen": "YYYY-MM-DD"
    }
  ]
}
Ensure the output is strictly valid JSON without any markdown formatting around it."""

async def update_company_narrative(ticker: str, cycle_id: str = "") -> bool:
    """
    Fetch the existing narrative, recent news/reddit/youtube summaries, and fundamentals,
    run the Narrative Curator Agent, and update the database.
    """
    ticker = ticker.upper()
    
    # 1. Fetch existing narrative from DB
    existing_summary = ""
    existing_themes = []
    last_update = datetime.datetime.min.replace(tzinfo=datetime.UTC)
    
    try:
        with get_db() as db:
            row = db.execute(
                "SELECT story_summary, key_themes, updated_at FROM company_narratives WHERE ticker = %s",
                [ticker]
            ).fetchone()
            if row:
                existing_summary = row[0]
                existing_themes = row[1] if isinstance(row[1], list) else json.loads(row[1] or "[]")
                if row[2]:
                    last_update = row[2]
    except Exception as e:
        logger.warning(f"[NARRATIVE] Failed to load existing narrative for {ticker}: {e}")
        
    # 1.5 Fetch latest fundamentals for financial context
    fundamentals_summary = ""
    try:
        with get_db() as db:
            fund_row = db.execute(
                "SELECT * FROM fundamentals WHERE ticker = %s ORDER BY snapshot_date DESC LIMIT 1",
                [ticker]
            ).fetchone()
            if fund_row:
                cols = [d[0] for d in db.execute("SELECT * FROM fundamentals LIMIT 0").description]
                fund_dict = dict(zip(cols, fund_row))
                fund_lines = []
                for k, v in fund_dict.items():
                    if k not in ("ticker", "snapshot_date", "source") and v is not None:
                        fund_lines.append(f"- {k}: {v}")
                if fund_lines:
                    fundamentals_summary = "\n".join(fund_lines)
    except Exception as e:
        logger.warning(f"[NARRATIVE] Failed to fetch fundamentals for {ticker}: {e}")

    # 2. Fetch new news/social summaries collected since last_update
    new_summaries = []
    try:
        with get_db() as db:
            # News articles
            news_rows = db.execute(
                """
                SELECT qualitative_draft, published_at 
                FROM news_articles 
                WHERE ticker = %s AND qualitative_draft IS NOT NULL AND (published_at > %s OR qualitative_draft->>'decision' = 'keep')
                ORDER BY published_at DESC LIMIT 20
                """,
                [ticker, last_update]
            ).fetchall()
            for row in news_rows:
                draft, dt = row
                draft_dict = draft if isinstance(draft, dict) else json.loads(draft or "{}")
                if draft_dict.get("decision") == "keep":
                    dt_str = dt.strftime("%Y-%m-%d") if dt else "unknown"
                    new_summaries.append(
                        f"[News - {dt_str}] Theme: {draft_dict.get('suggested_theme')}\n"
                        f"  Relevance: {draft_dict.get('relevance_label')} (Impact: {draft_dict.get('impact')})\n"
                        f"  Facts: {', '.join(draft_dict.get('bullet_points', []))}\n"
                        f"  Justification: {draft_dict.get('justification')}"
                    )

            # Reddit posts
            reddit_rows = db.execute(
                """
                SELECT qualitative_draft, created_utc 
                FROM reddit_posts 
                WHERE ticker = %s AND qualitative_draft IS NOT NULL AND qualitative_draft->>'decision' = 'keep'
                ORDER BY created_utc DESC LIMIT 10
                """,
                [ticker]
            ).fetchall()
            for row in reddit_rows:
                draft, dt = row
                draft_dict = draft if isinstance(draft, dict) else json.loads(draft or "{}")
                dt_str = dt.strftime("%Y-%m-%d") if dt else "unknown"
                new_summaries.append(
                    f"[Reddit - {dt_str}] Theme: {draft_dict.get('suggested_theme')}\n"
                    f"  Relevance: {draft_dict.get('relevance_label')} (Impact: {draft_dict.get('impact')})\n"
                    f"  Facts: {', '.join(draft_dict.get('bullet_points', []))}\n"
                    f"  Justification: {draft_dict.get('justification')}"
                )
                
    except Exception as e:
        logger.warning(f"[NARRATIVE] Failed to fetch news/social summaries for {ticker}: {e}")

    if not new_summaries and existing_summary:
        logger.info(f"[NARRATIVE] No new news/social context for {ticker}. Evolving skipped.")
        return True

    # 3. Call LLM to update the narrative
    user_message = f"""Ticker: {ticker}
Current Date: {datetime.date.today().isoformat()}

Company Financial Indicators:
{fundamentals_summary or "No fundamental indicators available."}

Existing Narrative Summary:
{existing_summary or "No existing narrative."}

Existing Key Themes:
{json.dumps(existing_themes, indent=2) if existing_themes else "No existing key themes."}

New Recent News & Social Developments:
{chr(10).join(new_summaries) if new_summaries else "No new articles or posts."}
"""
    try:
        response, _, _ = await call_prism_agent(
            agent_id="CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
            user_message=user_message,
            fallback_system_prompt=NARRATIVE_SYSTEM_PROMPT,
            fallback_agent_name="narrative_curator",
            temperature=0.3,
            max_tokens=2048,
            priority=Priority.NORMAL,
            ticker=ticker,
            cycle_id=cycle_id,
        )
        
        from app.utils.text_utils import parse_json_response
        data = parse_json_response(response)
        story_summary = data.get("story_summary", "").strip()
        key_themes = data.get("key_themes", [])
        
        if not story_summary:
            logger.warning(f"[NARRATIVE] LLM returned empty story_summary for {ticker}")
            return False
            
        # 4. Save back to DB
        with get_db() as db:
            db.execute(
                """
                INSERT INTO company_narratives (ticker, story_summary, key_themes, updated_at)
                VALUES (%s, %s, %s::jsonb, CURRENT_TIMESTAMP)
                ON CONFLICT (ticker) DO UPDATE SET
                story_summary = EXCLUDED.story_summary,
                key_themes = EXCLUDED.key_themes,
                updated_at = CURRENT_TIMESTAMP
                """,
                [ticker, story_summary, json.dumps(key_themes)]
            )
            logger.info(f"[NARRATIVE] Successfully updated narrative for {ticker} (themes: {len(key_themes)})")
        return True
        
    except Exception as e:
        logger.error(f"[NARRATIVE] Failed to run Narrative Curator for {ticker}: {e}", exc_info=True)
        return False
