"""
Dense Retriever — Semantic vector search for RAG context retrieval.

Uses embedding_service + vector_store to find relevant chunks via cosine
similarity. Provides the 'Strategy A' path for the RAG A/B test.

Usage:
    from app.services.retrieval_dense import dense_retriever, RetrievedChunk
    chunks = dense_retriever.retrieve("NVDA", "NVDA earnings analysis", top_k=10)
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


QUERY_PREFIX = "Represent this financial query: "

# Minimum cosine similarity to include a chunk
MIN_SCORE = 0.3

# Score boost for decision memory chunks (bot's own past analysis)
DECISION_MEMORY_BOOST = 1.10  # +10%


@dataclass
class RetrievedChunk:
    """A single retrieved context chunk with metadata."""

    content: str = ""
    source_table: str = ""
    source_id: str = ""
    ticker: str | None = None
    score: float = 0.0
    boosted: bool = False


def _expand_query(ticker: str, query_text: str) -> str:
    """Expand bare ticker queries into natural language for better retrieval.

    e.g. "NVDA" → "NVDA NVIDIA stock trading analysis outlook"
    """
    q = query_text.strip()
    # If query is very short (just a ticker), expand it
    if len(q.split()) <= 3:
        q = f"{ticker} {q} stock market analysis trading outlook"
    return q


class DenseRetriever:
    """Dense vector retrieval using cosine similarity search."""

    def retrieve(
        self,
        ticker: str,
        query_text: str,
        top_k: int = 10,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant chunks via dense vector search.

        1. Expand and embed the query
        2. Search vector_store by cosine similarity
        3. Filter by min_score, boost decision memory, deduplicate

        Args:
            ticker: Target ticker (also retrieves ticker=NULL macro chunks).
            query_text: Natural language search query.
            top_k: Number of results to return.

        Returns:
            List of RetrievedChunk sorted by score descending.
        """
        try:
            from app.services.embedding_service import embedder
            from app.db.vector_store import vector_store

            # 1. Build and embed query
            expanded = _expand_query(ticker, query_text)
            query_vec = embedder.embed_text(expanded, prefix=QUERY_PREFIX)

            # 2. Search (retrieve extra for post-filtering headroom)
            raw = vector_store.search_cosine(
                query_vec,
                ticker=ticker,
                top_k=top_k * 3,
            )

            # 3. Convert to RetrievedChunk, apply boosting & filtering
            chunks: list[RetrievedChunk] = []
            seen: set[tuple[str, str]] = set()  # (source_table, source_id)

            for r in raw:
                score = r["score"]

                # Skip low-quality matches
                if score < MIN_SCORE:
                    continue

                key = (r["source_table"], r["source_id"])
                if key in seen:
                    continue
                seen.add(key)

                # Boost decision memory (+10%)
                boosted = False
                if r["source_table"] == "analysis_results":
                    score = min(score * DECISION_MEMORY_BOOST, 1.0)
                    boosted = True

                chunks.append(
                    RetrievedChunk(
                        content=r["content_preview"],
                        source_table=r["source_table"],
                        source_id=r["source_id"],
                        ticker=r["ticker"],
                        score=score,
                        boosted=boosted,
                    )
                )

            # Sort by score descending, take top_k
            chunks.sort(key=lambda c: c.score, reverse=True)
            chunks = chunks[:top_k]

            logger.info(
                "[dense] %s: %d chunks (from %d raw), top=%.3f",
                ticker,
                len(chunks),
                len(raw),
                chunks[0].score if chunks else 0.0,
            )
            return chunks

        except Exception as e:
            logger.warning("[dense] retrieval failed for %s: %s", ticker, e)
            return []


# Module-level singleton
dense_retriever = DenseRetriever()
"""
Global dense retriever instance:
    from app.services.retrieval_dense import dense_retriever
    chunks = dense_retriever.retrieve("NVDA", "earnings analysis")
"""
