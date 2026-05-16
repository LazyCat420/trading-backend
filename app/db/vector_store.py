"""
Vector Store — PostgreSQL/pgvector-backed embedding storage and similarity search.

Provides storage and retrieval of 384-dimensional embeddings using:
  - pgvector cosine distance (<=>) — always available, hardware-accelerated
  - HNSW ANN index — persistent, production-grade (created in schema_pg.sql)
  - Full-text search via tsvector/GIN — built-in PostgreSQL, no extensions needed

Usage:
    from app.db.vector_store import vector_store
    vector_store.store_embedding("news_articles", "abc123", "NVDA", "...", vec)
    results = vector_store.search_cosine(query_vec, ticker="NVDA", top_k=10)
"""

import logging
import uuid
from datetime import datetime, UTC

from app.db.connection import get_db

logger = logging.getLogger(__name__)


class VectorStore:
    """PostgreSQL/pgvector vector storage and similarity search."""

    # ─── Storage ────────────────────────────────────────────────────────

    def store_embedding(
        self,
        source_table: str,
        source_id: str,
        ticker: str | None,
        content_preview: str,
        embedding: list[float],
        embedding_id: str | None = None,
    ) -> str:
        """Store a single embedding.

        Returns the embedding ID.
        """
        with get_db() as db:
            eid = embedding_id or str(uuid.uuid4())
            now = datetime.now(UTC).isoformat()

            # PostgreSQL ON CONFLICT upsert
            db.execute(
                """
                INSERT INTO embeddings
                (id, source_table, source_id, ticker,
                 content_preview, embedding, created_at)
                VALUES (%s, %s, %s, %s, %s, %s::vector, %s)
                ON CONFLICT (id) DO UPDATE SET
                    source_table = EXCLUDED.source_table,
                    source_id = EXCLUDED.source_id,
                    ticker = EXCLUDED.ticker,
                    content_preview = EXCLUDED.content_preview,
                    embedding = EXCLUDED.embedding,
                    created_at = EXCLUDED.created_at
            """,
                [
                    eid,
                    source_table,
                    source_id,
                    ticker,
                    content_preview[:500],
                    embedding,
                    now,
                ],
            )

            return eid

    def store_batch(
        self,
        records: list[dict],
    ) -> int:
        """Store a batch of embeddings in a single transaction.

        Each record should have: source_table, source_id, ticker,
        content_preview, embedding. Optional: id.

        Returns count of records stored.
        """
        with get_db() as db:
            now = datetime.now(UTC).isoformat()
            count = 0

            for rec in records:
                eid = rec.get("id", str(uuid.uuid4()))
                db.execute(
                    """
                    INSERT INTO embeddings
                    (id, source_table, source_id, ticker,
                     content_preview, embedding, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s::vector, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        source_table = EXCLUDED.source_table,
                        source_id = EXCLUDED.source_id,
                        ticker = EXCLUDED.ticker,
                        content_preview = EXCLUDED.content_preview,
                        embedding = EXCLUDED.embedding,
                        created_at = EXCLUDED.created_at
                """,
                    [
                        eid,
                        rec["source_table"],
                        rec["source_id"],
                        rec.get("ticker"),
                        rec.get("content_preview", "")[:500],
                        rec["embedding"],
                        now,
                    ],
                )
                count += 1

            logger.info(f"[DB] Stored {count} embeddings")
            return count

    def exists(self, source_table: str, source_id: str) -> bool:
        """Check if an embedding already exists for this source."""
        with get_db() as db:
            result = db.execute(
                """
                SELECT 1 FROM embeddings
                WHERE source_table = %s AND source_id = %s
                LIMIT 1
            """,
                [source_table, source_id],
            ).fetchone()
            return result is not None

    # ─── Search: Cosine Similarity (pgvector <=> operator) ────────────

    def search_cosine(
        self,
        query_embedding: list[float],
        ticker: str | None = None,
        top_k: int = 10,
        source_filter: str | None = None,
    ) -> list[dict]:
        """Search embeddings by cosine similarity.

        Uses pgvector's <=> operator (cosine distance) with HNSW index.
        Score is converted to similarity (1 - distance) for backward compat.

        Args:
            query_embedding: 384-dim query vector.
            ticker: Optional ticker filter. If provided, returns chunks
                    where ticker matches OR ticker is NULL (macro context).
            top_k: Number of results to return.
            source_filter: Optional source_table filter (e.g., 'news_articles').

        Returns:
            List of dicts with: id, source_table, source_id, ticker,
            content_preview, score (cosine similarity 0-1).
        """
        with get_db() as db:
            # Build WHERE clause
            conditions = []
            params = []

            if ticker:
                conditions.append("(ticker = %s OR ticker IS NULL)")
                params.append(ticker)

            if source_filter:
                conditions.append("source_table = %s")
                params.append(source_filter)

            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

            query = f"""
                SELECT id, source_table, source_id, ticker, content_preview,
                       1 - (embedding <=> %s::vector) as score
                FROM embeddings
                {where_clause}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """

            # The order of params must match the query:
            # 1st %s is query_embedding in the SELECT clause
            # Next are the conditions for the WHERE clause
            # Next is query_embedding in the ORDER BY clause
            # Last is top_k in the LIMIT clause
            final_params = [query_embedding] + params + [query_embedding, top_k]

            rows = db.execute(query, final_params).fetchall()

            return [
                {
                    "id": r[0],
                    "source_table": r[1],
                    "source_id": r[2],
                    "ticker": r[3],
                    "content_preview": r[4],
                    "score": r[5],
                }
                for r in rows
            ]

    # ─── Search: HNSW ANN (always available with pgvector) ─────────────

    def search_hnsw(
        self,
        query_embedding: list[float],
        ticker: str | None = None,
        top_k: int = 10,
    ) -> list[dict]:
        """Search using HNSW approximate nearest neighbor.

        With pgvector, HNSW is always available and the index is created
        in schema_pg.sql. The planner automatically uses it when available.
        """
        return self.search_cosine(query_embedding, ticker, top_k)

    # ─── Full-Text Search: PostgreSQL tsvector (built-in) ─────────────

    def search_bm25(
        self,
        query_text: str,
        ticker: str | None = None,
        top_k: int = 30,
    ) -> list[dict]:
        """Search using PostgreSQL full-text search.

        Uses to_tsvector/plainto_tsquery for ranked keyword matching.
        """
        with get_db() as db:
            try:
                # Build WHERE clause for ticker filtering
                ticker_filter = ""
                from typing import Any

                params: list[Any] = [query_text, query_text]
                if ticker:
                    ticker_filter = "AND (ticker = %s OR ticker IS NULL)"
                    params = [query_text, query_text, ticker]

                query = f"""
                    SELECT id, source_table, source_id, ticker,
                           content_preview,
                           ts_rank(
                               to_tsvector('english', COALESCE(content_preview, '')),
                               plainto_tsquery('english', %s)
                           ) AS score
                    FROM embeddings
                    WHERE to_tsvector('english', COALESCE(content_preview, ''))
                          @@ plainto_tsquery('english', %s)
                    {ticker_filter}
                    ORDER BY score DESC
                    LIMIT %s
                """
                params.append(top_k)
                rows = db.execute(query, params).fetchall()

                return [
                    {
                        "id": r[0],
                        "source_table": r[1],
                        "source_id": r[2],
                        "ticker": r[3],
                        "content_preview": r[4],
                        "score": r[5],
                    }
                    for r in rows
                ]
            except Exception as e:
                logger.warning(f"[DB] Full-text search failed: {e}")
                return []

    # ─── Stats ────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return embedding statistics."""
        with get_db() as db:
            total = db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]

            by_source = db.execute("""
                SELECT source_table, COUNT(*) as cnt
                FROM embeddings
                GROUP BY source_table
                ORDER BY cnt DESC
            """).fetchall()

            by_ticker = db.execute("""
                SELECT ticker, COUNT(*) as cnt
                FROM embeddings
                GROUP BY ticker
                ORDER BY cnt DESC
                LIMIT 20
            """).fetchall()

            return {
                "total_embeddings": total,
                "by_source": {r[0]: r[1] for r in by_source},
                "by_ticker": {r[0]: r[1] for r in by_ticker},
                "hnsw_available": True,  # Always available with pgvector
                "fts_available": True,  # Always available with PostgreSQL
            }

    def clear(self):
        """Delete all embeddings. Use for testing."""
        with get_db() as db:
            db.execute("DELETE FROM embeddings")
            logger.warning("[DB] All embeddings cleared")


# Module-level singleton
vector_store = VectorStore()
"""
Global vector store instance. Import and use directly:

    from app.db.vector_store import vector_store
    results = vector_store.search_cosine(query_vec, ticker="AAPL", top_k=10)
"""
