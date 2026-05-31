"""
Packet Builder.
Main orchestrator for Evidence Fusion layer.
"""

import logging
import asyncio
import json
from typing import List
from datetime import datetime, timezone

from .normalizer import (
    NormalizedDocument,
    normalize_news,
    normalize_reddit,
    normalize_youtube,
    normalize_structured_row,
)
from .claim_extractor import extract_claims
from .llm_enrichment import enrich_claims_with_llm
from .clustering import cluster_claims
from .contradiction_detector import detect_contradictions
from .source_scorer import score_sources
from ..contracts.retrieval import (
    RetrievalContext,
    StructuredFact,
    SourceDocRef,
    FreshnessSummary,
)
from ..contracts.evidence import EvidencePacket

# Import real queries
from app.db.connection import get_db
from app.trading.scoring_engine import build_hierarchical_pillar_profiles
from app.services.prism_agent_caller import call_prism_agent
from app.services.vllm_client import Priority
from app.services.adaptive_concurrency import concurrency_controller

logger = logging.getLogger(__name__)

PILLAR_ADJUSTER_SYSTEM_PROMPT = """You are a quantitative analyst and qualitative risk manager for a hedge fund.
Your task is to review the quantitative "Pillar Profiles" of a stock ticker and qualitatively adjust their scores based on the company's "Story & Narrative".

You are given:
- The stock ticker.
- The base scores, active outlier drivers, and veto flags computed deterministically for three pillars: Edge, Risk, and Regime Fit.
- The company's persistent narrative summary and active story themes (e.g. CEO transitions, expansions, lawsuits).

Your job is to:
- Review each pillar's base score and decide if the qualitative narrative warrants adjusting it.
- You can adjust each score up or down by a MAXIMUM of 2.0 points (e.g., from 6.0 to 8.0, or from 5.0 to 3.0).
- If the company has high debt or low cash runway in its narrative, penalize any capital-intensive expansion theme by lowering the Edge score or the Risk score.
- For each pillar, write:
  1. The "adjusted_score" (clamped between 1.0 and 10.0).
  2. The "adjustment_rationale" explaining why the score was adjusted or left unchanged, citing specific qualitative themes and financial context.
  3. A refined "profile_label" capturing the combined quant + qual setup.

Return ONLY a valid JSON object matching this schema:
{
  "pillars": {
    "edge": {
      "adjusted_score": float,
      "profile_label": "string",
      "adjustment_rationale": "string"
    },
    "risk": {
      "adjusted_score": float,
      "profile_label": "string",
      "adjustment_rationale": "string"
    },
    "regime": {
      "adjusted_score": float,
      "profile_label": "string",
      "adjustment_rationale": "string"
    }
  }
}
Ensure the output is strictly valid JSON without any markdown formatting around it."""


async def build_evidence_packet(
    entity_id: str, context: RetrievalContext = None
) -> EvidencePacket:
    """
    Fuses DB data into an EvidencePacket.
    Acts as the main Dev 2 interface to Dev 1/Dev 3.
    """
    ticker = entity_id.upper()

    def _fetch_db_docs():
        documents: List[NormalizedDocument] = []
        fund_dict = {}
        tech_dict = {}
        with get_db() as db:
            # 1. Fetch & Normalize Data (Mocking RetrievalContext pipeline integration here via Direct DB fallback)
            # Ideally, RetrievalContext is provided by graph/orchestrator. We backfill if it's empty.

            # -- 1.1 Structural facts (Prices, fundamentals)
            try:
                price_row = db.execute(
                    "SELECT date, close FROM price_history WHERE ticker = %s ORDER BY date DESC LIMIT 1",
                    [ticker],
                ).fetchone()
                if price_row:
                    d = normalize_structured_row(
                        "price_history",
                        "price",
                        f"{ticker}_{price_row[0]}",
                        price_row[1],
                        price_row[0]
                        if isinstance(price_row[0], datetime)
                        else datetime.fromisoformat(str(price_row[0])),
                    )
                    documents.append(d)
            except Exception as e:
                logger.warning(f"[PACKET] Failed to fetch prices for {ticker}: {e}")

            try:
                fund_row = db.execute(
                    "SELECT * FROM fundamentals WHERE ticker = %s ORDER BY snapshot_date DESC LIMIT 1",
                    [ticker],
                ).fetchone()
                if fund_row:
                    cols = [
                        d[0]
                        for d in db.execute(
                            "SELECT * FROM fundamentals LIMIT 0"
                        ).description
                    ]
                    fund_dict_temp = dict(zip(cols, fund_row))
                    fund_dict.update(fund_dict_temp)
                    fund_date = fund_dict_temp.get(
                        "snapshot_date", datetime.now(timezone.utc)
                    )
                    if not isinstance(fund_date, datetime):
                        fund_date = datetime.fromisoformat(str(fund_date))

                    for key, val in fund_dict_temp.items():
                        if (
                            key not in ("ticker", "snapshot_date", "source")
                            and val is not None
                        ):
                            d = normalize_structured_row(
                                "fundamentals",
                                key,
                                f"{ticker}_fund_{key}",
                                val,
                                fund_date,
                            )
                            documents.append(d)
            except Exception as e:
                logger.warning(
                    f"[PACKET] Failed to fetch fundamentals for {ticker}: {e}"
                )

            try:
                tech_row = db.execute(
                    "SELECT * FROM technicals WHERE ticker = %s ORDER BY date DESC LIMIT 1",
                    [ticker],
                ).fetchone()
                if tech_row:
                    cols = [
                        d[0]
                        for d in db.execute(
                            "SELECT * FROM technicals LIMIT 0"
                        ).description
                    ]
                    tech_dict_temp = dict(zip(cols, tech_row))
                    tech_dict.update(tech_dict_temp)
                    tech_date = tech_dict_temp.get("date", datetime.now(timezone.utc))
                    if not isinstance(tech_date, datetime):
                        tech_date = datetime.fromisoformat(str(tech_date))

                    for key, val in tech_dict_temp.items():
                        if key not in ("ticker", "date") and val is not None:
                            d = normalize_structured_row(
                                "technical_data",
                                key,
                                f"{ticker}_tech_{key}",
                                val,
                                tech_date,
                            )
                            documents.append(d)
            except Exception as e:
                logger.warning(f"[PACKET] Failed to fetch technicals for {ticker}: {e}")

            try:
                fin_rows = db.execute(
                    "SELECT * FROM financial_history WHERE ticker = %s ORDER BY period_end DESC LIMIT 4",
                    [ticker],
                ).fetchall()
                if fin_rows:
                    cols = [
                        d[0]
                        for d in db.execute(
                            "SELECT * FROM financial_history LIMIT 0"
                        ).description
                    ]
                    for i, row in enumerate(fin_rows):
                        fin_dict = dict(zip(cols, row))
                        fin_date = fin_dict.get(
                            "period_end", datetime.now(timezone.utc)
                        )
                        if not isinstance(fin_date, datetime):
                            fin_date = datetime.fromisoformat(str(fin_date))

                        for key, val in fin_dict.items():
                            if (
                                key not in ("ticker", "period_end", "period_type")
                                and val is not None
                            ):
                                d = normalize_structured_row(
                                    "fundamentals",
                                    key,
                                    f"{ticker}_fin_{i}_{key}",
                                    val,
                                    fin_date,
                                )
                                documents.append(d)
            except Exception as e:
                logger.warning(f"[PACKET] Failed to fetch financials for {ticker}: {e}")

            # -- 1.2 Unstructured facts (News, Reddit, YouTube)
            try:
                cols = [
                    "id",
                    "title",
                    "publisher",
                    "url",
                    "published_at",
                    "summary",
                    "llm_summary",
                ]
                q_cols = ", ".join(
                    [
                        c
                        if c not in ("llm_summary")
                        else "COALESCE(llm_summary, summary) as best_summary"
                        for c in cols
                    ]
                )
                news_rows = db.execute(
                    f"SELECT {q_cols} FROM news_articles WHERE ticker = %s AND (quality_status IS NULL OR quality_status != 'discarded') ORDER BY published_at DESC LIMIT 5",
                    [ticker],
                ).fetchall()
                for r in news_rows:
                    doc = normalize_news(r, cols)
                    if doc:
                        documents.append(doc)
            except Exception as e:
                logger.warning(f"[PACKET] Failed to fetch news for {ticker}: {e}")

            try:
                cols = [
                    "id",
                    "ticker",
                    "subreddit",
                    "title",
                    "body",
                    "score",
                    "comment_count",
                    "created_utc",
                ]
                q_cols = ",".join(cols)
                reddit_rows = db.execute(
                    f"SELECT {q_cols} FROM reddit_posts WHERE ticker = %s AND (quality_status IS NULL OR quality_status != 'discarded') ORDER BY created_utc DESC LIMIT 5",
                    [ticker],
                ).fetchall()
                for r in reddit_rows:
                    doc = normalize_reddit(r, cols)
                    if doc:
                        documents.append(doc)
            except Exception as e:
                logger.warning(f"[PACKET] Failed to fetch reddit for {ticker}: {e}")

            try:
                cols = [
                    "video_id",
                    "ticker",
                    "title",
                    "channel",
                    "raw_transcript",
                    "published_at",
                    "summary",
                    "tickers_mentioned",
                ]
                q_cols = ",".join(cols)
                yt_rows = db.execute(
                    f"SELECT {q_cols} FROM youtube_transcripts WHERE ticker = %s AND (quality_status IS NULL OR quality_status != 'discarded') ORDER BY published_at DESC LIMIT 5",
                    [ticker],
                ).fetchall()
                for r in yt_rows:
                    doc = normalize_youtube(r, cols)
                    if doc:
                        documents.append(doc)
            except Exception as e:
                logger.warning(f"[PACKET] Failed to fetch youtube for {ticker}: {e}")

            # -- 1.3 Company Story / Narrative
            story_summary = None
            key_themes = []
            try:
                narrative_row = db.execute(
                    "SELECT story_summary, key_themes FROM company_narratives WHERE ticker = %s",
                    [ticker],
                ).fetchone()
                if narrative_row:
                    story_summary = narrative_row[0]
                    key_themes = narrative_row[1] if isinstance(narrative_row[1], list) else json.loads(narrative_row[1] or "[]")
            except Exception as e:
                logger.warning(f"[PACKET] Failed to fetch company narrative for {ticker}: {e}")

        return documents, fund_dict, tech_dict, story_summary, key_themes

    documents, fund_dict, tech_dict, story_summary, key_themes = await asyncio.to_thread(_fetch_db_docs)

    # 2. Extract Claims
    claims = []
    for doc in documents:
        # Deterministic stage
        doc_claims = extract_claims(doc, entity_id=ticker)
        claims.extend(doc_claims)

    # Secondary stage: LLM enrichment (concurrent)
    if documents:
        enrichment_tasks = [
            enrich_claims_with_llm(doc, ticker, claims) for doc in documents
        ]
        enrichment_results = await concurrency_controller.gather(enrichment_tasks, label="packet_builder", return_exceptions=True)
        for res in enrichment_results:
            if not isinstance(res, Exception) and res:
                claims.extend(res)
            elif isinstance(res, Exception):
                logger.error(f"[packet_builder] Enrichment task failed: {res}")

    # 3. Cluster Claims
    clusters = cluster_claims(claims)

    # 4. Contradiction Detection
    contradictions = detect_contradictions(clusters)

    # Flatten claims (consensus/clustered represent primary output)
    final_claims = claims  # Full set inside evidence packet

    # 5. Check missing fields
    missing_fields = []
    if not any(
        d.metadata.get("fact_type") == "price"
        for d in documents
        if d.source_type == "structured"
    ):
        missing_fields.append("price")

    has_fundamentals = any(
        d.metadata.get("fact_type") in ("market_cap", "forward_pe", "profit_margin")
        for d in documents
        if d.source_type == "structured"
    )
    if not has_fundamentals:
        missing_fields.append("pe_ratio")

    # 6. Score Sources
    source_quality = score_sources(
        documents, final_claims, contradictions, missing_fields
    )

    # Compile SourceDocRefs
    source_summaries = [
        SourceDocRef(
            source_type=d.source_type,
            source_id=d.source_ref,
            summary=d.content[:200] + "...",
            timestamp=d.timestamp,
            url=d.metadata.get("url"),
            metadata=d.metadata,
        )
        for d in documents
        if d.source_type != "structured"
    ]

    # Compile StructuredFacts
    structured_facts = [
        StructuredFact(
            fact_type=d.metadata["fact_type"], value=d.content, timestamp=d.timestamp
        )
        for d in documents
        if d.source_type == "structured"
    ]

    now = datetime.now(timezone.utc)
    
    valid_timestamps = [
        d.timestamp.replace(tzinfo=timezone.utc) if d.timestamp and d.timestamp.tzinfo is None else (d.timestamp or now)
        for d in documents
    ]
    
    oldest_ts = min(valid_timestamps) if valid_timestamps else now
    newest_ts = max(valid_timestamps) if valid_timestamps else now

    freshness = FreshnessSummary(
        oldest_data_age_hours=source_quality.stale_data_severity,
        newest_data_age_hours=min(
            [
                (now - ts).total_seconds() / 3600
                for ts in valid_timestamps
            ]
        )
        if valid_timestamps
        else 0.0,
        is_stale=source_quality.stale_data_severity > 72.0,
        oldest_timestamp=oldest_ts,
        newest_timestamp=newest_ts,
    )

    tool_cache = {}
    if fund_dict:
        tool_cache["get_finviz_fundamentals"] = json.dumps(
            {
                "pe": fund_dict.get("pe_ratio"),
                "eps": fund_dict.get("eps"),
                "market_cap": fund_dict.get("market_cap"),
                "52w_high": fund_dict.get("week_52_high"),
                "52w_low": fund_dict.get("week_52_low"),
            }
        )
    if tech_dict:
        tool_cache["get_technicals"] = json.dumps(
            {
                "rsi": tech_dict.get("rsi"),
                "sma_20": tech_dict.get("sma20"),
                "sma_50": tech_dict.get("sma50"),
                "macd": tech_dict.get("macd"),
                "volume": tech_dict.get("volume"),
            }
        )

    # ── Compute and adjust Hierarchical Pillar Profiles ──
    profiles = build_hierarchical_pillar_profiles(ticker)
    
    # Initialize adjusted scores to base scores
    for pk in ["edge", "risk", "regime"]:
        if pk in profiles["pillars"]:
            profiles["pillars"][pk]["adjusted_score"] = profiles["pillars"][pk]["base_score"]
            profiles["pillars"][pk]["adjustment_rationale"] = "No qualitative adjustment applied (base quant score)."
            
    # Run qualitative adjustment if narrative is present
    if story_summary:
        try:
            user_message = f"""Ticker: {ticker}
Pillar Profiles (Quantitative Base):
{json.dumps(profiles["pillars"], indent=2)}

Company Qualitative Narrative Story:
{story_summary}

Active Narrative Themes:
{json.dumps(key_themes, indent=2)}
"""
            response, _, _ = await asyncio.wait_for(
                call_prism_agent(
                    agent_id="CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
                    user_message=user_message,
                    fallback_system_prompt=PILLAR_ADJUSTER_SYSTEM_PROMPT,
                    fallback_agent_name="pillar_adjuster",
                    temperature=0.2,
                    max_tokens=2048,
                    priority=Priority.NORMAL,
                    ticker=ticker,
                ),
                timeout=60.0,
            )
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```json")[-1].split("```")[0].strip()
                
            adj_data = json.loads(cleaned)
            adj_pillars = adj_data.get("pillars", {})
            for pk in ["edge", "risk", "regime"]:
                if pk in adj_pillars:
                    profiles["pillars"][pk]["adjusted_score"] = adj_pillars[pk].get("adjusted_score", profiles["pillars"][pk]["base_score"])
                    profiles["pillars"][pk]["profile_label"] = adj_pillars[pk].get("profile_label", profiles["pillars"][pk]["profile_label"])
                    profiles["pillars"][pk]["adjustment_rationale"] = adj_pillars[pk].get("adjustment_rationale", profiles["pillars"][pk]["adjustment_rationale"])
        except Exception as e:
            logger.warning(f"[PACKET] Qualitative pillar adjustment failed for {ticker}: {e}")

    return EvidencePacket(
        entity_id=ticker,
        claims=final_claims,
        structured_facts=structured_facts,
        source_summaries=source_summaries,
        contradictions=contradictions,
        missing_fields=missing_fields,
        tool_cache=tool_cache,
        freshness_summary=freshness,
        source_quality_summary=source_quality,
        company_story=story_summary,
        key_themes=key_themes,
        pillar_profiles=profiles,
    )


async def build_evidence_packet_partial(
    entity_id: str, refresh_tables: List[str], context: RetrievalContext = None
) -> EvidencePacket:
    """
    Stub for targeted evidence packet rebuild.
    Instead of re-querying everything, this will only refresh tables modified by tools.
    TODO: Implement targeted refresh logic based on refresh_tables.
    For now, fallback to full rebuild to maintain compatibility.
    """
    return await build_evidence_packet(entity_id, context)
