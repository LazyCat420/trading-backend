"""
Evolution Lesson Store — stores and retrieves evolution lessons.

Uses the PostgreSQL embeddings table + pgvector for RAG retrieval.
Optionally offloads embedding to the remote PC server via RemoteEmbedder.
"""

import logging
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def _get_embedder():
    """Return the appropriate embedder based on config.

    Tries remote first (if configured), falls back to local.
    """
    try:
        from app.cognition.remote_embedder import RemoteEmbedder

        # Import constants - may not have EMBED_SERVER_URL yet
        try:
            from app.constants import EMBED_SERVER_URL
        except (ImportError, AttributeError):
            EMBED_SERVER_URL = ""

        if EMBED_SERVER_URL:
            return "remote", RemoteEmbedder(EMBED_SERVER_URL)
    except Exception:
        pass

    # Fallback to local embedder
    from app.services.embedding_service import embedder

    return "local", embedder


def add_lesson(text: str, metadata: dict) -> str:
    """Store an evolution lesson with its embedding in the vector store.

    Args:
        text: The lesson/analysis text from the analyzer.
        metadata: Dict with session_id, round, score, status, timestamp.

    Returns:
        The ID of the stored embedding row.
    """
    from app.db.connection import get_db

    with get_db() as db:
        lesson_id = f"evo_{uuid.uuid4().hex[:12]}"

        # Generate embedding
        mode, emb = _get_embedder()
        try:
            if mode == "remote":
                vecs = emb.embed_sync([text])
                vec = vecs[0]
            else:
                vec = emb.embed_text(text)
        except Exception as e:
            logger.warning("Remote embedder failed, falling back to local: %s", e)
            from app.services.embedding_service import embedder

            vec = embedder.embed_text(text)

        # Build content preview for search display
        session_id = metadata.get("session_id", "")
        rnd = metadata.get("round", 0)
        score = metadata.get("score")
        status = metadata.get("status", "")
        preview = f"[Evolve {session_id} R{rnd} {status} S:{score}] {text[:200]}"

        # Insert into embeddings table (same schema as existing RAG)
        db.execute(
            "INSERT INTO embeddings (id, source_table, source_id, ticker, content_preview, embedding, created_at) "
            "VALUES (%s, 'evolution_lessons', %s, %s, %s, %s::vector, CURRENT_TIMESTAMP) "
            "ON CONFLICT (id) DO NOTHING",
            [lesson_id, lesson_id, f"evo_{session_id}", preview, str(vec)],
        )

        # Store the full lesson in the dedicated table (created at schema init)
        db.execute(
            "INSERT INTO evolution_lessons (id, session_id, round, score, status, lesson_text, timestamp) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            [
                lesson_id,
                session_id,
                rnd,
                score,
                status,
                text,
                metadata.get("timestamp", datetime.now(timezone.utc).isoformat()),
            ],
        )

        logger.info(
            "[cognition] Stored evolution lesson %s (session=%s round=%d)",
            lesson_id,
            session_id,
            rnd,
        )
        return lesson_id


def retrieve_lessons(query: str, k: int = 5) -> list[dict]:
    """Retrieve the k most relevant evolution lessons via vector similarity.

    Returns a list of dicts with: id, lesson_text, score, session_id, round, status.
    """
    from app.db.connection import get_db

    with get_db() as db:
        # Generate query embedding
        mode, emb = _get_embedder()
        try:
            if mode == "remote":
                vecs = emb.embed_sync([query])
                q_vec = vecs[0]
            else:
                q_vec = emb.embed_text(query)
        except Exception:
            from app.services.embedding_service import embedder

            q_vec = embedder.embed_text(query)

        # Search via cosine similarity in embeddings table
        try:
            rows = db.execute(
                "SELECT e.source_id, e.content_preview, "
                "1 - (e.embedding <=> %s::vector) as sim "
                "FROM embeddings e "
                "WHERE e.source_table = 'evolution_lessons' "
                "ORDER BY e.embedding <=> %s::vector LIMIT %s",
                [str(q_vec), str(q_vec), k],
            ).fetchall()
        except Exception:
            return []

        if not rows:
            return []

        # Enrich with full lesson data
        results = []
        for source_id, preview, sim in rows:
            try:
                lesson_row = db.execute(
                    "SELECT id, session_id, round, score, status, lesson_text, timestamp "
                    "FROM evolution_lessons WHERE id = %s",
                    [source_id],
                ).fetchone()
                if lesson_row:
                    results.append(
                        {
                            "id": lesson_row[0],
                            "session_id": lesson_row[1],
                            "round": lesson_row[2],
                            "score": lesson_row[3],
                            "status": lesson_row[4],
                            "lesson_text": lesson_row[5],
                            "timestamp": lesson_row[6],
                            "similarity": sim,
                        }
                    )
            except Exception:
                results.append({"id": source_id, "preview": preview, "similarity": sim})

        return results
