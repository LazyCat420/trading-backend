"""
Utility Worker — The "Data Janitor"

Runs background tasks when the main analysis queue hits the high watermark limit.
Repurposes idle compute to clean data, build vectors, and deduplicate.
"""

import logging
import time

from app.config import settings
from app.db.connection import get_db
from app.cognition.remote_embedder import RemoteEmbedder

logger = logging.getLogger(__name__)


async def run_utility_cycle(emit):
    """
    Main entry point for utility mode.
    Runs one pass of all janitor tasks.
    """
    logger.info("[PIPELINE] [UTILITY] Entering Utility Mode (Data Janitor)...")
    emit(
        "utility",
        "start",
        "Entering Utility Mode to clean and embed data",
        status="running",
    )

    start_t = time.monotonic()

    # Run the janitor tasks sequentially or concurrently
    await task_embed_missing_data(emit)
    await task_deduplicate_news(emit)
    # Ontology task could go here in the future

    elapsed = time.monotonic() - start_t
    logger.info(f"[PIPELINE] [UTILITY] Utility Mode complete in {elapsed:.1f}s")
    emit("utility", "done", f"Utility Mode complete in {elapsed:.1f}s", status="ok")


async def task_embed_missing_data(emit):
    """
    Finds news articles without embeddings and calls the embed server.
    """
    logger.info("[PIPELINE] [UTILITY] Task: Embedding missing data...")
    try:
        embedder = RemoteEmbedder(settings.EMBEDDING_SERVER_URL)
        healthy = await embedder.health_check()
        if not healthy:
            logger.warning(
                "[PIPELINE] [UTILITY] Embed server unreachable. Skipping embedding."
            )
            return

        with get_db() as db:
            # We look for news articles that do not have an entry in the embeddings table
            # We limit to 50 per batch so we don't hog memory.
            query = """
                SELECT n.id, n.ticker, n.title, n.summary 
                FROM news_articles n
                LEFT JOIN embeddings e ON e.source_id = n.id AND e.source_table = 'news_articles'
                WHERE e.id IS NULL AND n.summary IS NOT NULL
                LIMIT 50
            """
            rows = db.execute(query).fetchall()

            if not rows:
                logger.info("[PIPELINE] [UTILITY] No missing embeddings found.")
                return

            texts_to_embed = []
            for r in rows:
                text = f"Title: {r[2]} | Summary: {r[3]}"
                # Ensure text is not too long for the embedder
                texts_to_embed.append(text[:512])

            logger.info(
                f"[PIPELINE] [UTILITY] Embedding {len(texts_to_embed)} articles..."
            )
            vectors = await embedder.embed(texts_to_embed)

            # Save to DB
            for i, r in enumerate(rows):
                art_id, ticker = r[0], r[1]
                vec = vectors[i]
                # Convert list of floats to pgvector string format '[0.1, 0.2, ...]'
                vec_str = "[" + ",".join(str(v) for v in vec) + "]"

                db.execute(
                    """
                    INSERT INTO embeddings (id, source_table, source_id, ticker, content_preview, embedding, created_at)
                    VALUES (gen_random_uuid(), 'news_articles', %s, %s, %s, %s, CURRENT_TIMESTAMP)
                """,
                    (art_id, ticker, texts_to_embed[i][:100], vec_str),
                )

            logger.info(
                f"[PIPELINE] [UTILITY] Successfully embedded {len(rows)} articles."
            )
            emit("utility", "embed", f"Embedded {len(rows)} news articles", status="ok")

    except Exception as e:
        logger.error(f"[PIPELINE] [UTILITY] Embedding task failed: {e}")


async def task_deduplicate_news(emit):
    """
    Uses vector similarity to find duplicate news articles.
    Flags duplicates by appending ' [DUPLICATE]' to their quality_status.
    """
    logger.info("[PIPELINE] [UTILITY] Task: Deduplicating news...")
    try:
        with get_db() as db:
            # Find pairs of embeddings for the same ticker that have cosine distance < 0.05 (i.e. similarity > 0.95)
            # <=> is the cosine distance operator in pgvector.
            query = """
                WITH duplicates AS (
                    SELECT e1.source_id as id1, e2.source_id as id2, (e1.embedding <=> e2.embedding) as dist
                    FROM embeddings e1
                    JOIN embeddings e2 ON e1.ticker = e2.ticker AND e1.source_table = 'news_articles' AND e2.source_table = 'news_articles'
                    WHERE e1.id < e2.id
                      AND (e1.embedding <=> e2.embedding) < 0.05
                )
                SELECT d.id1, d.id2, d.dist 
                FROM duplicates d
                JOIN news_articles n ON n.id = d.id2
                WHERE n.quality_status IS NULL OR n.quality_status != 'duplicate'
                LIMIT 100
            """
            rows = db.execute(query).fetchall()

            if not rows:
                logger.info("[PIPELINE] [UTILITY] No duplicates found to flag.")
                return

            dup_ids = list(
                set([r[1] for r in rows])
            )  # Flag the second article as duplicate

            # Update the quality_status
            for d_id in dup_ids:
                db.execute(
                    """
                    UPDATE news_articles 
                    SET quality_status = 'duplicate', quality_reason = 'Vector similarity > 0.95'
                    WHERE id = %s
                """,
                    (d_id,),
                )

            logger.info(
                f"[PIPELINE] [UTILITY] Flagged {len(dup_ids)} duplicate articles."
            )
            emit(
                "utility",
                "dedup",
                f"Flagged {len(dup_ids)} duplicate articles",
                status="ok",
            )

    except Exception as e:
        logger.error(f"[PIPELINE] [UTILITY] Deduplication task failed: {e}")
