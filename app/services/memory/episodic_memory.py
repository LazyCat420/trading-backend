import uuid
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class EpisodicMemoryStore:
    """
    Stores compressed summaries of completed trading cycles.
    Each episode captures what the agent observed, decided, and the outcome.
    """

    def write_episode(
        self,
        cycle_id: str,
        ticker: str,
        summary: str,
        key_decisions: str = "[]",
        outcome: str = "neutral",
        outcome_score: float = 0.0,
        agents_involved: str = "[]",
    ) -> str:
        """Store a new episode summarize a completed cycle."""
        from app.db.connection import get_db

        with get_db() as db:
            mem_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc).isoformat()

            db.execute(
                """
                INSERT INTO episodic_memory
                (id, cycle_id, ticker, timestamp, summary, key_decisions, outcome, outcome_score, agents_involved)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
                [
                    mem_id,
                    cycle_id,
                    ticker,
                    now,
                    summary,
                    key_decisions,
                    outcome,
                    outcome_score,
                    agents_involved,
                ],
            )

            logger.info(f"[EPISODIC] Wrote episode for {ticker} (Cycle {cycle_id})")
            return mem_id

    def retrieve(self, ticker: str, limit: int = 4) -> list[dict]:
        """Query past episodes by ticker, ranked by most successful outcomes."""
        from app.db.connection import get_db

        with get_db() as db:
            # Pull best-outcome episodes first to show the bot what worked
            # (Could also pull worst-outcome to show what didn't work)
            rows = db.execute(
                """
                SELECT id, cycle_id, timestamp, summary, outcome_score, key_decisions
                FROM episodic_memory
                WHERE ticker = %s
                ORDER BY outcome_score DESC, timestamp DESC
                LIMIT %s
            """,
                [ticker, limit],
            ).fetchall()

            results = []
            for r in rows:
                results.append(
                    {
                        "id": r[0],
                        "cycle_id": r[1],
                        "timestamp": r[2],
                        "summary": r[3],
                        "outcome_score": r[4],
                        "key_decisions": r[5],
                    }
                )
            return results


# Singleton instance
episodic_memory_store = EpisodicMemoryStore()
