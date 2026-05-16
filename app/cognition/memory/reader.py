"""
Cognition V2 — Memory Reader.

Retrieval interface for V2 memories across all four types.
Supports filtered reads by entity, memory type, tags, and status.

Usage:
    from app.cognition.memory.reader import read_memories, read_reflections
    memories = read_memories("NVDA", memory_types=["semantic", "episodic"])
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from app.cognition.memory.models import (
    MemoryEnvelope,
    MemoryStatus,
    MemoryType,
    ProceduralRule,
    ReflectionRecord,
)
from app.cognition.memory.schema import _ensure_schema
from app.db.connection import get_db

logger = logging.getLogger(__name__)


def _parse_json_field(val: Any) -> Any:
    """Safely parse a JSON string field, returning the original if not JSON."""
    if val is None:
        return []
    if isinstance(val, (list, dict)):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, ValueError):
            return val
    return val


def _rows_to_dicts(cursor) -> list[dict[str, Any]]:
    """Convert cursor results to list of dicts."""
    if cursor.description is None:
        return []
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def read_memories(
    entity_id: str,
    memory_types: list[str] | None = None,
    limit: int = 20,
    active_only: bool = True,
) -> list[MemoryEnvelope]:
    """Read memory envelopes for a given entity.

    Args:
        entity_id: Entity to retrieve memories for.
        memory_types: Filter by memory type(s). None = all types.
        limit: Max records to return.
        active_only: If True, exclude deprecated/archived memories.

    Returns:
        List of MemoryEnvelope records, ordered by recency.
    """
    _ensure_schema()
    with get_db() as db:
        query = "SELECT * FROM cognition_memory_envelopes WHERE entity_id = %s"
        params: list[Any] = [entity_id]

        if active_only:
            query += " AND status = %s"
            params.append(MemoryStatus.ACTIVE.value)

        if memory_types:
            placeholders = ",".join(["%s"] * len(memory_types))
            query += f" AND memory_type IN ({placeholders})"
            params.extend(memory_types)

        query += " ORDER BY updated_at DESC LIMIT %s"
        params.append(limit)

        rows = _rows_to_dicts(db.execute(query, params))

        envelopes = []
        for row in rows:
            try:
                env = MemoryEnvelope(
                    id=row["id"],
                    memory_type=MemoryType(row["memory_type"]),
                    entity_id=row["entity_id"],
                    content_hash=row.get("content_hash", ""),
                    status=MemoryStatus(row.get("status", "active")),
                    confidence=float(row.get("confidence", 0.5)),
                    tags=_parse_json_field(row.get("tags")),
                    ttl_days=row.get("ttl_days"),
                    payload=_parse_json_field(row.get("payload_json", "{}")),
                    created_at=str(row.get("created_at", "")),
                    updated_at=str(row.get("updated_at", "")),
                    last_accessed_at=(
                        str(row["last_accessed_at"])
                        if row.get("last_accessed_at")
                        else None
                    ),
                )
                envelopes.append(env)
            except (KeyError, ValueError) as e:
                logger.warning("[COGNITION] Skipping malformed envelope: %s", e)

        # Touch last_accessed_at for retrieved memories
        if envelopes:
            now = datetime.now(timezone.utc).isoformat()
            ids = [e.id for e in envelopes]
            placeholders = ",".join(["%s"] * len(ids))
            db.execute(
                f"UPDATE cognition_memory_envelopes SET last_accessed_at = %s WHERE id IN ({placeholders})",
                [now, *ids],
            )

        return envelopes


def read_reflections(
    entity_id: str,
    limit: int = 10,
) -> list[ReflectionRecord]:
    """Read reflection records for a given entity.

    Args:
        entity_id: Entity to retrieve reflections for.
        limit: Max records to return.

    Returns:
        List of ReflectionRecord, ordered by recency.
    """
    _ensure_schema()
    with get_db() as db:
        rows = _rows_to_dicts(
            db.execute(
                """
                SELECT * FROM cognition_reflections
                WHERE entity_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                [entity_id, limit],
            )
        )

        records = []
        for row in rows:
            try:
                rec = ReflectionRecord(
                    id=row["id"],
                    episode_id=row.get("episode_id", ""),
                    entity_id=row.get("entity_id", ""),
                    timestamp=str(row.get("timestamp", "")),
                    failure_patterns=_parse_json_field(row.get("failure_patterns")),
                    missed_signals=_parse_json_field(row.get("missed_signals")),
                    late_verifiers=_parse_json_field(row.get("late_verifiers")),
                    what_should_have_blocked=row.get("what_should_have_blocked", ""),
                    root_cause=row.get("root_cause", ""),
                    recommended_changes=_parse_json_field(
                        row.get("recommended_changes")
                    ),
                    severity=row.get("severity", "medium"),
                    expected_outcome=row.get("expected_outcome", ""),
                    actual_outcome=row.get("actual_outcome", ""),
                    outcome_delta=float(row.get("outcome_delta", 0.0)),
                )
                records.append(rec)
            except (KeyError, ValueError) as e:
                logger.warning("[COGNITION] Skipping malformed reflection: %s", e)

        return records


def read_procedural(
    tags: list[str] | None = None,
    category: str | None = None,
    active_only: bool = True,
    limit: int = 50,
) -> list[ProceduralRule]:
    """Read procedural memory rules.

    Args:
        tags: Filter by tags (OR matching).
        category: Filter by category.
        active_only: Only return active rules.
        limit: Max records to return.

    Returns:
        List of ProceduralRule, ordered by confidence DESC.
    """
    _ensure_schema()
    with get_db() as db:
        query = "SELECT * FROM cognition_procedural_memories WHERE 1=1"
        params: list[Any] = []

        if active_only:
            query += " AND active = TRUE AND status = 'active'"

        if category:
            query += " AND category = %s"
            params.append(category)

        query += " ORDER BY confidence DESC LIMIT %s"
        params.append(limit)

        rows = _rows_to_dicts(db.execute(query, params))

        rules = []
        for row in rows:
            try:
                rule = ProceduralRule(
                    id=row["id"],
                    rule_text=row.get("rule_text", ""),
                    category=row.get("category", ""),
                    applies_to=_parse_json_field(row.get("applies_to")),
                    confidence=float(row.get("confidence", 0.5)),
                    success_count=int(row.get("success_count", 0)),
                    failure_count=int(row.get("failure_count", 0)),
                    source_episodes=_parse_json_field(row.get("source_episodes")),
                    tags=_parse_json_field(row.get("tags")),
                    active=bool(row.get("active", True)),
                    created_at=str(row.get("created_at", "")),
                    updated_at=str(row.get("updated_at", "")),
                )

                # Tag filtering (OR match)
                if tags:
                    rule_tags = set(t.lower() for t in rule.tags)
                    query_tags = set(t.lower() for t in tags)
                    if not rule_tags.intersection(query_tags):
                        continue

                rules.append(rule)
            except (KeyError, ValueError) as e:
                logger.warning("[COGNITION] Skipping malformed procedural rule: %s", e)

        return rules


def read_episodic(
    entity_id: str,
    event_type: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Read raw episodic memory records.

    Args:
        entity_id: Entity to retrieve episodes for.
        event_type: Filter by event type (e.g., "pipeline_run").
        limit: Max records to return.

    Returns:
        List of episode dicts, ordered by recency.
    """
    _ensure_schema()
    with get_db() as db:
        query = "SELECT * FROM cognition_episodic_memories WHERE entity_id = %s"
        params: list[Any] = [entity_id]

        if event_type:
            query += " AND event_type = %s"
            params.append(event_type)

        query += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)

        rows = _rows_to_dicts(db.execute(query, params))

        # Parse JSON fields
        for row in rows:
            for field in ("evidence_sources", "debate_challenges", "tags"):
                row[field] = _parse_json_field(row.get(field))

        return rows
