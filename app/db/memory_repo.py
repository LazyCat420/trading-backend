import json
import logging
from typing import Any, Dict, List
from datetime import datetime, timezone
from app.db.connection import get_db

logger = logging.getLogger(__name__)


def _ensure_schema():
    """Ensure the schema is present if Developer 1's schema is not yet merged."""
    with get_db() as db:
        db.execute("""
        CREATE TABLE IF NOT EXISTS episodic_observations (
            id TEXT PRIMARY KEY,
            created_at TIMESTAMPTZ,
            cycle_id TEXT,
            ticker TEXT,
            sector TEXT,
            source_type TEXT,
            observation_text TEXT,
            rationale_excerpt TEXT,
            confidence_at_creation DOUBLE PRECISION,
            outcome_label TEXT,
            outcome_score DOUBLE PRECISION,
            promoted_to_memory BOOLEAN DEFAULT FALSE
        );
        """)

        db.execute("""
        CREATE TABLE IF NOT EXISTS canonical_memories (
            id TEXT PRIMARY KEY,
            type TEXT,
            ticker TEXT,
            sector TEXT,
            summary TEXT,
            tags TEXT,
            confidence_score DOUBLE PRECISION,
            evidence_count INTEGER,
            status TEXT,
            last_used_at TIMESTAMPTZ,
            last_validated_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ,
            updated_at TIMESTAMPTZ
        );
        """)

        db.execute("""
        CREATE TABLE IF NOT EXISTS consolidation_reports (
            id TEXT PRIMARY KEY,
            run_at TIMESTAMPTZ,
            ticker TEXT,
            observations_consumed INTEGER,
            memories_created INTEGER,
            memories_deprecated INTEGER
        );
        """)

        db.execute("""
        CREATE TABLE IF NOT EXISTS memory_usage_logs (
            id TEXT PRIMARY KEY,
            cycle_id TEXT,
            ticker TEXT,
            memory_id TEXT,
            budget_used_chars INTEGER,
            created_at TIMESTAMPTZ
        );
        """)


def _rows_to_dicts(cursor) -> List[Dict[str, Any]]:
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def get_unpromoted_observations(ticker: str) -> List[Dict[str, Any]]:
    _ensure_schema()
    with get_db() as db:
        db.execute(
            """
            SELECT * FROM episodic_observations 
            WHERE ticker = %s AND promoted_to_memory = FALSE
            ORDER BY created_at ASC
        """,
            [ticker],
        )
        return _rows_to_dicts(db)


def get_active_canonical_memories(ticker: str) -> List[Dict[str, Any]]:
    _ensure_schema()
    with get_db() as db:
        db.execute(
            """
            SELECT * FROM canonical_memories 
            WHERE ticker = %s AND status != 'deprecated'
        """,
            [ticker],
        )

        results = _rows_to_dicts(db)
        # Deserialize tags
        for row in results:
            if row.get("tags") and isinstance(row["tags"], str):
                try:
                    row["tags"] = json.loads(row["tags"])
                except json.JSONDecodeError:
                    row["tags"] = []
        return results


def upsert_canonical_memories(memories: List[Dict[str, Any]]):
    if not memories:
        return
    _ensure_schema()
    with get_db() as db:
        now_str = datetime.now(timezone.utc).isoformat()

        for m in memories:
            tags_str = json.dumps(m.get("tags", []))
            # PostgreSQL ON CONFLICT upsert — clean single statement
            db.execute(
                """
                INSERT INTO canonical_memories (
                    id, type, ticker, sector, summary, tags, confidence_score, evidence_count,
                    status, last_used_at, last_validated_at, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    type = EXCLUDED.type,
                    ticker = EXCLUDED.ticker,
                    sector = EXCLUDED.sector,
                    summary = EXCLUDED.summary,
                    tags = EXCLUDED.tags,
                    confidence_score = EXCLUDED.confidence_score,
                    evidence_count = EXCLUDED.evidence_count,
                    status = EXCLUDED.status,
                    last_used_at = EXCLUDED.last_used_at,
                    last_validated_at = EXCLUDED.last_validated_at,
                    updated_at = EXCLUDED.updated_at
            """,
                [
                    m["id"],
                    m.get("type"),
                    m.get("ticker"),
                    m.get("sector"),
                    m.get("summary"),
                    tags_str,
                    m.get("confidence_score"),
                    m.get("evidence_count"),
                    m.get("status", "active"),
                    m.get("last_used_at"),
                    m.get("last_validated_at"),
                    m.get("created_at", now_str),
                    now_str,
                ],
            )
        logger.info(f"Upserted {len(memories)} canonical memories.")


def deprecate_canonical_memories(memory_ids: List[str]):
    if not memory_ids:
        return
    _ensure_schema()
    with get_db() as db:
        now_str = datetime.now(timezone.utc).isoformat()
        for mid in memory_ids:
            db.execute(
                """
                UPDATE canonical_memories 
                SET status = 'deprecated', updated_at = %s
                WHERE id = %s
            """,
                [now_str, mid],
            )
        logger.info(f"Deprecated {len(memory_ids)} canonical memories.")


def mark_observations_promoted(observation_ids: List[str]):
    if not observation_ids:
        return
    _ensure_schema()
    with get_db() as db:
        for oid in observation_ids:
            db.execute(
                "UPDATE episodic_observations SET promoted_to_memory = TRUE WHERE id = %s",
                [oid],
            )
        logger.info(f"Marked {len(observation_ids)} observations as promoted.")


def log_consolidation_run(record: Dict[str, Any]):
    _ensure_schema()
    with get_db() as db:
        db.execute(
            """
            INSERT INTO consolidation_reports (
                id, run_at, ticker, observations_consumed, memories_created, memories_deprecated
            ) VALUES (%s, %s, %s, %s, %s, %s)
        """,
            [
                record.get("id"),
                record.get("run_at", datetime.now(timezone.utc).isoformat()),
                record.get("ticker"),
                record.get("observations_consumed", 0),
                record.get("memories_created", 0),
                record.get("memories_deprecated", 0),
            ],
        )


def get_memories_by_ids(memory_ids: List[str]) -> List[Dict[str, Any]]:
    if not memory_ids:
        return []
    _ensure_schema()
    with get_db() as db:
        # Build parameterized query for IN clause
        placeholders = ",".join(["?"] * len(memory_ids))
        db.execute(
            f"SELECT * FROM canonical_memories WHERE id IN ({placeholders})", memory_ids
        )
        results = _rows_to_dicts(db)
        for row in results:
            if row.get("tags") and isinstance(row["tags"], str):
                try:
                    row["tags"] = json.loads(row["tags"])
                except json.JSONDecodeError:
                    row["tags"] = []
        return results


def update_memory_validation_stats(
    memory_id: str, new_confidence: float, new_evidence_count: int, new_status: str
):
    """Specific targeted update for validation pipeline."""
    _ensure_schema()
    with get_db() as db:
        now_str = datetime.now(timezone.utc).isoformat()
        db.execute(
            """
            UPDATE canonical_memories 
            SET confidence_score = %s,
                evidence_count = %s,
                status = %s,
                last_validated_at = %s,
                updated_at = %s
            WHERE id = %s
        """,
            [
                new_confidence,
                new_evidence_count,
                new_status,
                now_str,
                now_str,
                memory_id,
            ],
        )
