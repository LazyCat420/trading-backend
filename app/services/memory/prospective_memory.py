import uuid
import logging

logger = logging.getLogger(__name__)


class ProspectiveMemoryStore:
    """
    Stores intentions that should trigger in the future (reminders).
    """

    def write_prospective(
        self,
        ticker: str,
        intention: str,
        trigger_condition: str,
        priority: str = "medium",
        trigger_at: str = None,
        context: str = "",
    ) -> str:
        """Store a new prospective memory (future trigger/reminder)."""
        from app.db.connection import get_db

        with get_db() as db:
            mem_id = str(uuid.uuid4())

            db.execute(
                """
                INSERT INTO prospective_memory
                (id, ticker, intention, trigger_condition, priority, status, trigger_at, context)
                VALUES (%s, %s, %s, %s, %s, 'pending', %s, %s)
            """,
                [
                    mem_id,
                    ticker,
                    intention,
                    trigger_condition,
                    priority,
                    trigger_at,
                    context,
                ],
            )

            logger.info(f"[PROSPECTIVE] Wrote reminder for {ticker}: {intention}")
            return mem_id

    def retrieve_pending(self, ticker: str) -> list[dict]:
        """Query pending items for a ticker that should be evaluated."""
        from app.db.connection import get_db

        with get_db() as db:
            # We also might want to pull 'global' triggers
            rows = db.execute(
                """
                SELECT id, ticker, intention, trigger_condition, priority, context
                FROM prospective_memory
                WHERE (ticker = %s OR ticker = 'global') AND status = 'pending'
                ORDER BY 
                    CASE priority
                        WHEN 'critical' THEN 1
                        WHEN 'high' THEN 2
                        WHEN 'medium' THEN 3
                        WHEN 'low' THEN 4
                        ELSE 5
                    END
                LIMIT 3
            """,
                [ticker],
            ).fetchall()

            results = []
            for r in rows:
                results.append(
                    {
                        "id": r[0],
                        "ticker": r[1],
                        "intention": r[2],
                        "trigger_condition": r[3],
                        "priority": r[4],
                        "context": r[5],
                    }
                )
            return results

    def mark_triggered(self, mem_id: str):
        """Mark an item as triggered so it's no longer pending."""
        from app.db.connection import get_db

        with get_db() as db:
            db.execute(
                "UPDATE prospective_memory SET status = 'triggered' WHERE id = %s",
                [mem_id],
            )


# Singleton instance
prospective_memory_store = ProspectiveMemoryStore()
