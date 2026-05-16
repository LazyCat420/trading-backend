"""
Embedding Processor — Batch ingestion of multi-source content into embeddings.

Reads from source tables (news_articles, reddit_posts, youtube_transcripts,
analysis_results) and embeds their text content into the vector store.

Idempotent: skips already-embedded source records.
Non-blocking: failures log warnings but don't halt the pipeline.

Usage:
    from app.processors.embedding_processor import embed_all_sources
    stats = embed_all_sources()
    print(stats)  # {'news_articles': 150, 'reddit_posts': 45, ...}
"""

import logging

from app.db.connection import get_db
from app.db.vector_store import vector_store
from app.services.embedding_service import embedder

logger = logging.getLogger(__name__)

# Query/prefix templates per source
# BGE models perform better with instruction prefixes
_SOURCE_CONFIG = {
    "news_articles": {
        "query": """
            SELECT id, ticker, title, summary
            FROM news_articles
            ORDER BY published_at DESC
            LIMIT %s
        """,
        "content_builder": lambda r: f"{r[2] or ''}\n{r[3] or ''}".strip(),
        "id_col": 0,
        "ticker_col": 1,
        "prefix": "Represent this financial news article: ",
        "max_rows": 500,
    },
    "reddit_posts": {
        "query": """
            SELECT id, ticker, title, body
            FROM reddit_posts
            ORDER BY created_utc DESC
            LIMIT %s
        """,
        "content_builder": lambda r: f"{r[2] or ''}\n{r[3] or ''}".strip(),
        "id_col": 0,
        "ticker_col": 1,
        "prefix": "Represent this Reddit discussion about stocks: ",
        "max_rows": 300,
    },
    "youtube_transcripts": {
        "query": """
            SELECT video_id, ticker, title, raw_transcript
            FROM youtube_transcripts
            ORDER BY published_at DESC
            LIMIT %s
        """,
        "content_builder": lambda r: f"{r[2] or ''}\n{r[3] or ''}".strip(),
        "id_col": 0,
        "ticker_col": 1,
        "prefix": "Represent this YouTube financial analysis transcript: ",
        "max_rows": 200,
    },
    "analysis_results": {
        "query": """
            SELECT id, ticker, agent_name, result_json
            FROM analysis_results
            ORDER BY created_at DESC
            LIMIT %s
        """,
        "content_builder": lambda r: f"Agent {r[2] or 'unknown'}: {r[3] or ''}".strip(),
        "id_col": 0,
        "ticker_col": 1,
        "prefix": "Represent this trading decision analysis: ",
        "max_rows": 200,
    },
}


def embed_source(
    source_table: str,
    max_rows: int | None = None,
) -> int:
    """Embed all unprocessed records from a single source table.

    Args:
        source_table: Key from _SOURCE_CONFIG.
        max_rows: Override max rows to process.

    Returns:
        Number of new embeddings stored.
    """
    if source_table not in _SOURCE_CONFIG:
        logger.error(f"Unknown source table: {source_table}")
        return 0

    config = _SOURCE_CONFIG[source_table]
    limit = max_rows or config["max_rows"]
    with get_db() as db:
        logger.info(f"[embed] Processing {source_table} (limit={limit})")

        try:
            rows = db.execute(config["query"], [limit]).fetchall()
        except Exception as e:
            logger.warning(f"[embed] Failed to query {source_table}: {e}")
            return 0

        if not rows:
            logger.info(f"[embed] No rows in {source_table}")
            return 0

        # Filter out already-embedded records
        to_embed = []
        for row in rows:
            source_id = str(row[config["id_col"]])
            if not vector_store.exists(source_table, source_id):
                content = config["content_builder"](row)
                if content and len(content) > 20:  # skip near-empty content
                    to_embed.append(
                        {
                            "source_id": source_id,
                            "ticker": row[config["ticker_col"]],
                            "content": content,
                        }
                    )

        if not to_embed:
            logger.info(f"[embed] All {source_table} records already embedded")
            return 0

        logger.info(f"[embed] {len(to_embed)} new {source_table} records to embed")

        # Chunk and embed
        all_records = []
        for item in to_embed:
            chunks = embedder.chunk_text(item["content"])
            for i, chunk in enumerate(chunks):
                all_records.append(
                    {
                        "source_id": f"{item['source_id']}_chunk{i}",
                        "original_source_id": item["source_id"],
                        "ticker": item["ticker"],
                        "content": chunk,
                    }
                )

        # Batch embed all chunks
        texts = [r["content"] for r in all_records]
        prefix = config["prefix"]

        try:
            embeddings = embedder.embed_batch(texts, prefix=prefix)
        except Exception as e:
            logger.error(f"[embed] Batch embedding failed for {source_table}: {e}")
            return 0

        # Store in vector store
        batch = []
        for rec, emb in zip(all_records, embeddings):
            batch.append(
                {
                    "source_table": source_table,
                    "source_id": rec["source_id"],
                    "ticker": rec["ticker"],
                    "content_preview": rec["content"][:500],
                    "embedding": emb,
                }
            )

        stored = vector_store.store_batch(batch)
        logger.info(f"[embed] Stored {stored} embeddings for {source_table}")
        return stored


def embed_all_sources() -> dict[str, int]:
    """Embed all unprocessed records from all source tables.

    Idempotent: skips already-embedded records.

    Returns:
        Dict mapping source_table to count of new embeddings.
    """
    logger.info("[embed] Starting multi-source embedding run")
    stats = {}

    for source_table in _SOURCE_CONFIG:
        try:
            count = embed_source(source_table)
            stats[source_table] = count
        except Exception as e:
            logger.error(f"[embed] Failed to process {source_table}: {e}")
            stats[source_table] = 0

    total = sum(stats.values())
    logger.info(f"[embed] Complete. Total new embeddings: {total}")
    logger.info(f"[embed] Breakdown: {stats}")

    return stats


def get_embedding_health() -> dict:
    """Return embedding health report for monitoring."""
    with get_db() as db:
        stats = vector_store.get_stats()

        # Check source coverage
        source_counts = {}
        for table in _SOURCE_CONFIG:
            try:
                total_rows = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                embedded_chunks = stats["by_source"].get(table, 0)
                source_counts[table] = {
                    "total_source_rows": total_rows,
                    "embedded_chunks": embedded_chunks,
                    "coverage": "full" if embedded_chunks > 0 else "none",
                }
            except Exception:
                source_counts[table] = {"error": "table not found"}

        return {
            "total_embeddings": stats["total_embeddings"],
            "sources": source_counts,
            "hnsw_available": stats["hnsw_available"],
            "fts_available": stats["fts_available"],
        }
