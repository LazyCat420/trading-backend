import uuid
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class SemanticMemoryStore:
    """
    Stores durable, ticker-specific facts that don't change cycle to cycle.
    Used to inject specific thresholds, rules, or historical facts into Working Memory.
    """

    def write_semantic(
        self,
        ticker: str,
        mem_type: str,
        content: str,
        confidence: float = 0.5,
        source_agent: str = "manual",
    ) -> str:
        """Store a new piece of semantic memory."""
        # Late import to prevent circular dependencies
        from app.db.connection import get_db

        with get_db() as db:
            mem_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc).isoformat()

            db.execute(
                """
                INSERT INTO semantic_memory
                (id, ticker, type, content, confidence, source_agent, created_at, last_accessed_at, access_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
                [
                    mem_id,
                    ticker,
                    mem_type,
                    content,
                    confidence,
                    source_agent,
                    now,
                    now,
                    0,
                ],
            )

            logger.info(
                f"[SEMANTIC] Wrote '{mem_type}' for {ticker}: {content[:50]}..."
            )
            return mem_id

    def retrieve(self, ticker: str, limit: int = 6) -> list[dict]:
        """Query by ticker, ranked by confidence and last accessed."""
        from app.db.connection import get_db

        with get_db() as db:
            now = datetime.now(timezone.utc).isoformat()

            # Pull ticker-specific AND global rules
            rows = db.execute(
                """
                SELECT id, ticker, type, content, confidence
                FROM semantic_memory
                WHERE ticker = %s OR ticker = 'global'
                ORDER BY confidence DESC, last_accessed_at DESC
                LIMIT %s
            """,
                [ticker, limit],
            ).fetchall()

            results = []
            if rows:
                # Update access tracking to show frequency/recency of use
                ids = [r[0] for r in rows]
                id_list = ",".join(f"'{i}'" for i in ids)
                db.execute(
                    f"""
                    UPDATE semantic_memory
                    SET access_count = access_count + 1,
                        last_accessed_at = %s
                    WHERE id IN ({id_list})
                """,
                    [now],
                )

                for r in rows:
                    results.append(
                        {
                            "id": r[0],
                            "ticker": r[1],
                            "type": r[2],
                            "content": r[3],
                            "confidence": r[4],
                        }
                    )

            return results

    def remove(self, mem_id: str) -> bool:
        """Delete an outdated semantic memory."""
        from app.db.connection import get_db

        with get_db() as db:
            res = db.execute("DELETE FROM semantic_memory WHERE id = %s", [mem_id])
            return res.rowcount > 0


# Singleton instance
semantic_memory_store = SemanticMemoryStore()
