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

logger = logging.getLogger(__name__)


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
                    f"SELECT {q_cols} FROM news_articles WHERE ticker = %s ORDER BY published_at DESC LIMIT 5",
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
                    f"SELECT {q_cols} FROM reddit_posts WHERE ticker = %s ORDER BY created_utc DESC LIMIT 5",
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
                    f"SELECT {q_cols} FROM youtube_transcripts WHERE ticker = %s ORDER BY published_at DESC LIMIT 5",
                    [ticker],
                ).fetchall()
                for r in yt_rows:
                    doc = normalize_youtube(r, cols)
                    if doc:
                        documents.append(doc)
            except Exception as e:
                logger.warning(f"[PACKET] Failed to fetch youtube for {ticker}: {e}")

        return documents, fund_dict, tech_dict

    documents, fund_dict, tech_dict = await asyncio.to_thread(_fetch_db_docs)

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
        enrichment_results = await asyncio.gather(*enrichment_tasks)
        for res in enrichment_results:
            claims.extend(res)

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
    if not any(
        d.metadata.get("fact_type") == "pe_ratio"
        for d in documents
        if d.source_type == "structured"
    ):
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
