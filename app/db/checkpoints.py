"""
Checkpoint Manager — Cycle state persistence for crash recovery.

CORAL Gap 3: If the bot crashes mid-cycle, resume from the last
successful sub-step instead of restarting the entire workflow.

Uses PostgreSQL for persistence. Each checkpoint records:
- cycle_id + step_name + ticker → unique checkpoint key
- state_blob (JSONB) → any partial results to restore

Usage:
    from app.db.checkpoints import checkpoint_manager

    # Save a checkpoint after a step completes
    checkpoint_manager.save("cycle-abc", "agents_complete", ticker="NVDA",
                            state={"agent_results": {...}})

    # On restart, check what was already done
    if checkpoint_manager.has_completed("cycle-abc", "agents_complete", "NVDA"):
        # Skip this step
        ...

    # Load partial state for resumption
    last = checkpoint_manager.load_latest("cycle-abc")

    # Clean up after cycle completes normally
    checkpoint_manager.clear_cycle("cycle-abc")
"""

import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# SQL for creating the checkpoints table
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS cycle_checkpoints (
    id              SERIAL PRIMARY KEY,
    cycle_id        VARCHAR NOT NULL,
    step_name       VARCHAR NOT NULL,
    ticker          VARCHAR DEFAULT '',
    state_blob      JSONB DEFAULT '{}',
    completed_at    TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (cycle_id, step_name, ticker)
);
"""


class CheckpointManager:
    """Manages cycle checkpoint persistence in PostgreSQL.

    Checkpoints are upserted (INSERT ON CONFLICT UPDATE) so that
    re-running the same step overwrites the previous checkpoint.

    Thread-safe: each call gets its own DB cursor from the pool.
    """

    def __init__(self):
        self._table_ensured = False

    def _ensure_table(self):
        """Create the checkpoints table if it doesn't exist."""
        if self._table_ensured:
            return
        try:
            from app.db.connection import get_db

            with get_db() as db:
                db.execute(_CREATE_TABLE_SQL)
                self._table_ensured = True
        except Exception as e:
            logger.warning("[CHECKPOINT] Table creation failed: %s", e)

    def save(
        self,
        cycle_id: str,
        step_name: str,
        ticker: str = "",
        state: dict | None = None,
    ) -> bool:
        """Save a checkpoint after a step completes successfully.

        Args:
            cycle_id: Current cycle identifier.
            step_name: Name of the completed step (e.g., "data_collection",
                       "agents_complete", "context_built", "debate_done").
            ticker: Optional ticker this checkpoint applies to.
            state: Optional dict of partial results to persist.

        Returns:
            True if checkpoint was saved, False on error.
        """
        self._ensure_table()
        try:
            from app.db.connection import get_db

            with get_db() as db:
                state_json = json.dumps(state or {})
                now = datetime.now(timezone.utc).isoformat()

                db.execute(
                    """
                    INSERT INTO cycle_checkpoints (cycle_id, step_name, ticker, state_blob, completed_at)
                    VALUES (%s, %s, %s, %s::jsonb, %s)
                    ON CONFLICT (cycle_id, step_name, ticker)
                    DO UPDATE SET state_blob = EXCLUDED.state_blob,
                                  completed_at = EXCLUDED.completed_at
                    """,
                    [cycle_id, step_name, ticker or "", state_json, now],
                )

                logger.debug(
                    "[CHECKPOINT] Saved: %s/%s/%s (%d bytes)",
                    cycle_id[:12],
                    step_name,
                    ticker or "*",
                    len(state_json),
                )
                return True

        except Exception as e:
            logger.warning(
                "[CHECKPOINT] Save failed for %s/%s: %s", cycle_id, step_name, e
            )
            return False

    def has_completed(
        self,
        cycle_id: str,
        step_name: str,
        ticker: str = "",
    ) -> bool:
        """Check if a step was already completed in a previous run.

        Used on resume to skip already-done work.
        """
        self._ensure_table()
        try:
            from app.db.connection import get_db

            with get_db() as db:
                row = db.execute(
                    """
                    SELECT 1 FROM cycle_checkpoints
                    WHERE cycle_id = %s AND step_name = %s AND ticker = %s
                    LIMIT 1
                    """,
                    [cycle_id, step_name, ticker or ""],
                ).fetchone()

                return row is not None

        except Exception as e:
            logger.warning("[CHECKPOINT] Check failed: %s", e)
            return False

    def load_state(
        self,
        cycle_id: str,
        step_name: str,
        ticker: str = "",
    ) -> dict | None:
        """Load the state blob for a specific checkpoint.

        Returns None if no checkpoint exists.
        """
        self._ensure_table()
        try:
            from app.db.connection import get_db

            with get_db() as db:
                row = db.execute(
                    """
                    SELECT state_blob FROM cycle_checkpoints
                    WHERE cycle_id = %s AND step_name = %s AND ticker = %s
                    """,
                    [cycle_id, step_name, ticker or ""],
                ).fetchone()

                if row and row[0]:
                    blob = row[0]
                    if isinstance(blob, str):
                        return json.loads(blob)
                    return blob  # psycopg may already parse JSONB to dict
                return None

        except Exception as e:
            logger.warning("[CHECKPOINT] Load failed: %s", e)
            return None

    def load_latest(self, cycle_id: str) -> dict | None:
        """Load the most recent checkpoint for a cycle.

        Returns dict with step_name, ticker, state, and completed_at.
        """
        self._ensure_table()
        try:
            from app.db.connection import get_db

            with get_db() as db:
                row = db.execute(
                    """
                    SELECT step_name, ticker, state_blob, completed_at
                    FROM cycle_checkpoints
                    WHERE cycle_id = %s
                    ORDER BY completed_at DESC
                    LIMIT 1
                    """,
                    [cycle_id],
                ).fetchone()

                if row:
                    blob = row[2]
                    if isinstance(blob, str):
                        blob = json.loads(blob)
                    return {
                        "step_name": row[0],
                        "ticker": row[1],
                        "state": blob or {},
                        "completed_at": str(row[3]),
                    }
                return None

        except Exception as e:
            logger.warning("[CHECKPOINT] Load latest failed: %s", e)
            return None

    def get_completed_steps(self, cycle_id: str) -> list[dict]:
        """Get all completed steps for a cycle (for resume logic).

        Returns list of {"step_name", "ticker", "completed_at"} dicts.
        """
        self._ensure_table()
        try:
            from app.db.connection import get_db

            with get_db() as db:
                rows = db.execute(
                    """
                    SELECT step_name, ticker, completed_at
                    FROM cycle_checkpoints
                    WHERE cycle_id = %s
                    ORDER BY completed_at ASC
                    """,
                    [cycle_id],
                ).fetchall()

                return [
                    {
                        "step_name": r[0],
                        "ticker": r[1],
                        "completed_at": str(r[2]),
                    }
                    for r in rows
                ]

        except Exception as e:
            logger.warning("[CHECKPOINT] Get steps failed: %s", e)
            return []

    def clear_cycle(self, cycle_id: str) -> int:
        """Delete all checkpoints for a completed cycle.

        Called after a cycle finishes cleanly. Returns count of deleted rows.
        """
        self._ensure_table()
        try:
            from app.db.connection import get_db

            with get_db() as db:
                # Count first for logging
                count_row = db.execute(
                    "SELECT COUNT(*) FROM cycle_checkpoints WHERE cycle_id = %s",
                    [cycle_id],
                ).fetchone()
                count = count_row[0] if count_row else 0

                if count > 0:
                    db.execute(
                        "DELETE FROM cycle_checkpoints WHERE cycle_id = %s",
                        [cycle_id],
                    )
                    logger.info(
                        "[CHECKPOINT] Cleared %d checkpoints for cycle %s",
                        count,
                        cycle_id[:12],
                    )

                return count

        except Exception as e:
            logger.warning("[CHECKPOINT] Clear failed for %s: %s", cycle_id, e)
            return 0

    def get_stats(self) -> dict:
        """Get checkpoint statistics for the monitoring dashboard."""
        self._ensure_table()
        try:
            from app.db.connection import get_db

            with get_db() as db:
                row = db.execute(
                    """
                    SELECT
                        COUNT(*) as total,
                        COUNT(DISTINCT cycle_id) as cycles,
                        MAX(completed_at) as last_checkpoint
                    FROM cycle_checkpoints
                    """
                ).fetchone()

                return {
                    "total_checkpoints": row[0] if row else 0,
                    "active_cycles": row[1] if row else 0,
                    "last_checkpoint": str(row[2]) if row and row[2] else None,
                }

        except Exception as e:
            logger.warning("[CHECKPOINT] Stats failed: %s", e)
            return {"error": str(e)}


# Singleton
checkpoint_manager = CheckpointManager()
