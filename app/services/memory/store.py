"""
Memory Store (DAL)
Data Access Layer handling the central brain memory storage.

DB TABLE OWNERSHIP:
- episodic_observations
- canonical_memories
"""

import json
import uuid
import logging
from datetime import datetime, timezone
from app.db.connection import get_db
from app.services.memory.repository import MemoryRepository

logger = logging.getLogger(__name__)


class MemoryStore:
    def add_episodic_observation(self, observation: dict) -> str:
        """
        Inserts a new raw episodic observation candidate.
        Expects:
          cycle_id, ticker(opt), sector(opt), source_type, observation_text, rationale_excerpt(opt),
          confidence_at_creation(opt), outcome_label(opt), outcome_score(opt)
        """
        obs_id = observation.get("id") or str(uuid.uuid4())

        # Use existing timestamp or generate now
        created_at = observation.get("created_at")
        if not created_at:
            created_at = datetime.now(timezone.utc).isoformat()

        cursor = get_db()
        cursor.execute(
            """
            INSERT INTO episodic_observations (
                id, created_at, cycle_id, ticker, sector, source_type, 
                observation_text, rationale_excerpt, confidence_at_creation, 
                outcome_label, outcome_score, promoted_to_memory
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                obs_id,
                created_at,
                observation["cycle_id"],
                observation.get("ticker"),
                observation.get("sector"),
                observation["source_type"],
                observation["observation_text"],
                observation.get("rationale_excerpt"),
                observation.get("confidence_at_creation"),
                observation.get("outcome_label"),
                observation.get("outcome_score"),
                observation.get("promoted_to_memory", False),
            ],
        )
        return obs_id

    def get_unpromoted_observations(self, limit: int = 100) -> list[dict]:
        """
        Retrieves recent candidate observations that haven't yet been promoted to canonical memory.
        """
        cursor = get_db()
        rows = cursor.execute(
            """
            SELECT id, created_at, cycle_id, ticker, sector, source_type, 
                   observation_text, rationale_excerpt, confidence_at_creation, 
                   outcome_label, outcome_score, promoted_to_memory
            FROM episodic_observations
            WHERE promoted_to_memory = FALSE
            ORDER BY created_at ASC
            LIMIT %s
            """,
            [limit],
        ).fetchall()

        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in rows]

    def mark_observation_promoted(self, obs_id: str):
        """Marks observation as having triggered or supplemented a canonical memory."""
        cursor = get_db()
        cursor.execute(
            "UPDATE episodic_observations SET promoted_to_memory = TRUE WHERE id = %s",
            [obs_id],
        )

    def add_canonical_memory(self, memory: dict) -> str:
        """
        Inserts a new canonical memory.
        Expects:
          type, ticker(opt), sector(opt), summary, tags, confidence_score, evidence_count(opt), status(opt)
        """
        mem_id = memory.get("id") or str(uuid.uuid4())

        tags = memory.get("tags") or []
        tags_json = json.dumps(tags)

        now = datetime.now(timezone.utc).isoformat()

        cursor = get_db()
        cursor.execute(
            """
            INSERT INTO canonical_memories (
                id, type, ticker, sector, summary, tags, confidence_score, 
                evidence_count, status, last_used_at, last_validated_at, created_at, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                mem_id,
                memory["type"],
                memory.get("ticker"),
                memory.get("sector"),
                memory["summary"],
                tags_json,
                memory["confidence_score"],
                memory.get("evidence_count", 1),
                memory.get("status", "tentative"),
                memory.get("last_used_at"),
                memory.get("last_validated_at"),
                memory.get("created_at", now),
                memory.get("updated_at", now),
            ],
        )
        return mem_id

    def get_memories_by_ticker(
        self, ticker: str, active_only: bool = True
    ) -> list[dict]:
        """Fetches memory rules pertinent to a specific ticker."""
        return MemoryRepository.get_memories_by_ticker(ticker, active_only)

    def update_memory_status(
        self,
        mem_id: str,
        new_confidence: float,
        new_status: str,
        validated_at: str = None,
    ):
        """
        Update the confidence decay and status flags for a canonical memory.
        """
        if not validated_at:
            validated_at = datetime.now(timezone.utc).isoformat()

        now = datetime.now(timezone.utc).isoformat()
        cursor = get_db()
        cursor.execute(
            """
            UPDATE canonical_memories 
            SET confidence_score = %s, status = %s, last_validated_at = %s, updated_at = %s
            WHERE id = %s
            """,
            [new_confidence, new_status, validated_at, now, mem_id],
        )

    def record_memory_usage(self, mem_id: str):
        """
        Touch the 'last_used_at' column when extracted for RAG context.
        """
        now = datetime.now(timezone.utc).isoformat()
        cursor = get_db()
        cursor.execute(
            """
            UPDATE canonical_memories 
            SET last_used_at = %s
            WHERE id = %s
            """,
            [now, mem_id],
        )
