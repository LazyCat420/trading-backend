"""
RAG A/B Test Coordinator — Randomly assigns retrieval strategy and logs results.

Routes each retrieval call to either Strategy A (dense-only) or Strategy B
(hybrid BM25+dense+RRF) and logs performance metrics to rag_ab_results table.

After running the pipeline with both strategies, analyze results:
    SELECT strategy, COUNT(*), AVG(top_score), AVG(retrieval_ms)
    FROM rag_ab_results GROUP BY strategy;

Usage:
    from app.services.rag_ab_test import rag_retrieve
    chunks, strategy = rag_retrieve("NVDA", "NVDA earnings analysis")
"""

import logging
import random
import time
import uuid
from datetime import datetime, UTC
from enum import Enum

from app.db.connection import get_db
from app.services.retrieval_dense import dense_retriever, RetrievedChunk
from app.services.retrieval_hybrid import hybrid_retriever

logger = logging.getLogger(__name__)


class RAGStrategy(Enum):
    DENSE_ONLY = "dense"
    HYBRID_BM25 = "hybrid"


def rag_retrieve(
    ticker: str,
    query_text: str,
    strategy: RAGStrategy | None = None,
    top_k: int = 10,
) -> tuple[list[RetrievedChunk], RAGStrategy]:
    """Retrieve context using A/B-tested strategy.

    Args:
        ticker: Target ticker.
        query_text: Natural language query.
        strategy: Force a specific strategy. None = random 50/50.
        top_k: Number of results.

    Returns:
        Tuple of (retrieved chunks, strategy used).
    """
    # Assign strategy
    if strategy is None:
        strategy = random.choice([RAGStrategy.DENSE_ONLY, RAGStrategy.HYBRID_BM25])

    # Execute
    start_ms = time.monotonic()

    if strategy == RAGStrategy.DENSE_ONLY:
        chunks = dense_retriever.retrieve(ticker, query_text, top_k)
    else:
        chunks = hybrid_retriever.retrieve(ticker, query_text, top_k)

    elapsed_ms = int((time.monotonic() - start_ms) * 1000)

    # Log results
    _log_ab_result(ticker, query_text, strategy, chunks, elapsed_ms)

    logger.info(
        f"[rag_ab] {strategy.value} | {ticker} | {len(chunks)} chunks | {elapsed_ms}ms"
    )

    return chunks, strategy


def _log_ab_result(
    ticker: str,
    query_text: str,
    strategy: RAGStrategy,
    chunks: list[RetrievedChunk],
    elapsed_ms: int,
):
    """Log A/B test result to the database."""
    try:
        with get_db() as db:
            top_score = chunks[0].score if chunks else 0.0
            avg_score = sum(c.score for c in chunks) / len(chunks) if chunks else 0.0

            db.execute(
                """
                INSERT INTO rag_ab_results
                (id, ticker, query, strategy, chunks_returned, top_score, avg_score,
                 retrieval_ms, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
                [
                    str(uuid.uuid4()),
                    ticker,
                    query_text[:500],
                    strategy.value,
                    len(chunks),
                    round(top_score, 6),
                    round(avg_score, 6),
                    elapsed_ms,
                    datetime.now(UTC).isoformat(),
                ],
            )
    except Exception as e:
        logger.warning(f"[rag_ab] Failed to log result: {e}")


def format_rag_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a context section for the LLM prompt.

    Returns:
        Formatted text section with source attribution.
    """
    if not chunks:
        return ""

    lines = ["\n## Market Intelligence (RAG-Retrieved)"]
    lines.append(f"({len(chunks)} relevant chunks from embedded sources)\n")

    for chunk in chunks:
        source_label = _source_label(chunk.source_table, chunk.source_id)
        boost_tag = " [DECISION MEMORY]" if chunk.boosted else ""
        lines.append(f"[{source_label}{boost_tag}] (relevance: {chunk.score:.3f})")
        lines.append(f"  {chunk.content}")
        lines.append("")

    return "\n".join(lines)


def _source_label(source_table: str, source_id: str) -> str:
    """Human-readable source label."""
    labels = {
        "news_articles": "News",
        "reddit_posts": "Reddit",
        "youtube_transcripts": "YouTube",
        "analysis_results": "Decision Memory",
    }
    return labels.get(source_table, source_table)


def get_ab_summary() -> dict:
    """Get A/B test summary statistics."""
    with get_db() as db:
        try:
            rows = db.execute("""
                SELECT strategy,
                       COUNT(*) as calls,
                       ROUND(AVG(top_score), 4) as avg_top_score,
                       ROUND(AVG(avg_score), 4) as avg_avg_score,
                       ROUND(AVG(retrieval_ms), 1) as avg_latency_ms,
                       ROUND(AVG(chunks_returned), 1) as avg_chunks
                FROM rag_ab_results
                GROUP BY strategy
            """).fetchall()

            return {
                r[0]: {
                    "calls": r[1],
                    "avg_top_score": r[2],
                    "avg_avg_score": r[3],
                    "avg_latency_ms": r[4],
                    "avg_chunks": r[5],
                }
                for r in rows
            }
        except Exception as e:
            logger.warning(f"[rag_ab] Failed to get summary: {e}")
            return {}
