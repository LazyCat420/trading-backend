"""
Hybrid Retriever — BM25 keyword + Dense vector search with Reciprocal Rank Fusion.

Combines keyword search (BM25) with dense vector search and merges results
using RRF for better recall. Provides 'Strategy B' for the RAG A/B test.

Usage:
    from app.services.retrieval_hybrid import hybrid_retriever
    chunks = hybrid_retriever.retrieve("NVDA", "earnings beat expectations")
"""

import logging

from app.services.retrieval_dense import (
    RetrievedChunk,
    DenseRetriever,
    _expand_query,
    MIN_SCORE,
    DECISION_MEMORY_BOOST,
    QUERY_PREFIX,
)

logger = logging.getLogger(__name__)

# RRF constant (standard value from the original RRF paper)
RRF_K = 60


class HybridRetriever:
    """Hybrid retrieval: dense vector + BM25 keyword search + RRF fusion."""

    def __init__(self):
        self._dense = DenseRetriever()

    def retrieve(
        self,
        ticker: str,
        query_text: str,
        top_k: int = 10,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks using hybrid BM25 + dense strategy.

        1. Run dense vector search (cosine similarity)
        2. Run BM25 keyword search (PostgreSQL fts)
        3. Merge results using Reciprocal Rank Fusion
        4. Apply decision memory boosting, deduplicate

        Falls back to dense-only if BM25 is unavailable.

        Args:
            ticker: Target ticker.
            query_text: Natural language search query.
            top_k: Number of results to return.

        Returns:
            List of RetrievedChunk sorted by fused score descending.
        """
        try:
            from app.db.vector_store import vector_store
            from app.services.embedding_service import embedder

            expanded = _expand_query(ticker, query_text)

            # ── 1. Dense vector search ──
            query_vec = embedder.embed_text(expanded, prefix=QUERY_PREFIX)
            dense_results = vector_store.search_cosine(
                query_vec,
                ticker=ticker,
                top_k=top_k * 3,
            )

            # ── 2. BM25 keyword search ──
            bm25_results = vector_store.search_bm25(
                expanded,
                ticker=ticker,
                top_k=top_k * 3,
            )

            # ── 3. Reciprocal Rank Fusion ──
            # Build rank maps: key → rank (1-indexed)
            def _key(r: dict) -> tuple[str, str]:
                return (r["source_table"], r["source_id"])

            dense_ranks: dict[tuple[str, str], int] = {}
            for rank, r in enumerate(dense_results, 1):
                k = _key(r)
                if k not in dense_ranks:
                    dense_ranks[k] = rank

            bm25_ranks: dict[tuple[str, str], int] = {}
            for rank, r in enumerate(bm25_results, 1):
                k = _key(r)
                if k not in bm25_ranks:
                    bm25_ranks[k] = rank

            # Collect all unique keys
            all_keys = set(dense_ranks.keys()) | set(bm25_ranks.keys())

            # Compute RRF score for each unique chunk
            rrf_scores: dict[tuple[str, str], float] = {}
            for k in all_keys:
                score = 0.0
                if k in dense_ranks:
                    score += 1.0 / (RRF_K + dense_ranks[k])
                if k in bm25_ranks:
                    score += 1.0 / (RRF_K + bm25_ranks[k])
                rrf_scores[k] = score

            # Build lookup for raw result data (prefer dense source for content)
            raw_lookup: dict[tuple[str, str], dict] = {}
            for r in bm25_results:
                raw_lookup[_key(r)] = r
            for r in dense_results:
                raw_lookup[_key(r)] = r  # dense overwrites bm25

            # Sort by RRF score
            sorted_keys = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

            # ── 4. Build RetrievedChunk list ──
            chunks: list[RetrievedChunk] = []
            for key in sorted_keys:
                r = raw_lookup[key]
                cosine_score = r["score"] if r in dense_results else 0.0

                # Use the RRF score normalized to [0, 1] approximate range
                # Max possible RRF for 2 lists = 2/(K+1) ≈ 0.033 for K=60
                # Normalize so the best score maps roughly to the cosine range
                fused = rrf_scores[key]
                max_rrf = 2.0 / (RRF_K + 1)
                normalized = min(fused / max_rrf, 1.0)

                # Apply min_score filter on the dense cosine score if available
                dense_score = 0.0
                if key in dense_ranks:
                    # Get actual cosine score from dense results
                    for dr in dense_results:
                        if _key(dr) == key:
                            dense_score = dr["score"]
                            break

                # Skip if we have a dense score and it's below threshold
                # (BM25-only results are kept if they have high RRF)
                if (
                    dense_score > 0
                    and dense_score < MIN_SCORE
                    and key not in bm25_ranks
                ):
                    continue

                # Decision memory boost
                boosted = False
                if r["source_table"] == "analysis_results":
                    normalized = min(normalized * DECISION_MEMORY_BOOST, 1.0)
                    boosted = True

                chunks.append(
                    RetrievedChunk(
                        content=r["content_preview"],
                        source_table=r["source_table"],
                        source_id=r["source_id"],
                        ticker=r["ticker"],
                        score=round(normalized, 4),
                        boosted=boosted,
                    )
                )

                if len(chunks) >= top_k:
                    break

            logger.info(
                "[hybrid] %s: %d chunks (dense=%d, bm25=%d, fused=%d), top=%.3f",
                ticker,
                len(chunks),
                len(dense_results),
                len(bm25_results),
                len(all_keys),
                chunks[0].score if chunks else 0.0,
            )
            return chunks

        except Exception as e:
            logger.warning(
                "[hybrid] retrieval failed for %s, falling back to dense: %s",
                ticker,
                e,
            )
            # Fallback to dense-only
            return self._dense.retrieve(ticker, query_text, top_k)


# Module-level singleton
hybrid_retriever = HybridRetriever()
"""
Global hybrid retriever instance:
    from app.services.retrieval_hybrid import hybrid_retriever
    chunks = hybrid_retriever.retrieve("NVDA", "earnings analysis")
"""
