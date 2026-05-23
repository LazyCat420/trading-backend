"""
Cognition V2 — Memory Writer.

Idempotent write interface for all four memory types.
Each write is keyed on (memory_type, content_hash) to prevent duplicates.

Usage:
    from app.cognition.memory.writer import write_episode, write_reflection
    episode_id = write_episode(run_result)
    reflection_id = write_reflection(reflection)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from app.cognition.memory.models import (
    EpisodicMemory,
    MemoryEnvelope,
    MemoryStatus,
    MemoryType,
    ProceduralRule,
    ReflectionRecord,
    SemanticFact,
    SemanticMemory,
)
from app.cognition.memory.schema import _ensure_schema
from app.db.connection import get_db

if TYPE_CHECKING:
    from app.cognition.orchestration.models import CognitionRunResult

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_duplicate(content_hash: str) -> bool:
    """Check if a memory with this content_hash already exists."""
    with get_db() as db:
        row = db.execute(
            "SELECT id FROM cognition_memory_envelopes WHERE content_hash = %s",
            [content_hash],
        ).fetchone()
        return row is not None


def _write_envelope(
    memory_id: str,
    memory_type: MemoryType,
    entity_id: str,
    content_hash: str,
    confidence: float,
    tags: list[str],
    payload: dict,
    ttl_days: int | None = None,
) -> None:
    """Write a unified memory envelope for cross-type lookups."""
    with get_db() as db:
        now = _now_iso()
        db.execute(
            """
            INSERT INTO cognition_memory_envelopes (
                id, memory_type, entity_id, content_hash, status,
                confidence, tags, ttl_days, payload_json,
                created_at, updated_at, last_accessed_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                memory_id,
                memory_type.value,
                entity_id,
                content_hash,
                MemoryStatus.ACTIVE.value,
                confidence,
                json.dumps(tags),
                ttl_days,
                json.dumps(payload),
                now,
                now,
                None,
            ],
        )


# ── Public Write Functions ─────────────────────────────────────────


def write_episode(run_result: CognitionRunResult) -> str:
    """Extract and persist an episodic memory from a cognition pipeline run.

    Idempotent: skips if a memory with the same content_hash exists.

    Args:
        run_result: Full pipeline output from orchestration.

    Returns:
        Memory ID (existing if duplicate, new if created).
    """
    _ensure_schema()
    from app.utils.text_utils import sanitize_surrogates
    run_result = sanitize_surrogates(run_result)

    # Build content key for dedup
    content_key = (
        f"{run_result.entity_id}:{run_result.cycle_id}:"
        f"{run_result.final_action}:{run_result.final_confidence}"
    )
    content_hash = MemoryEnvelope.compute_content_hash("episodic", content_key)

    if _is_duplicate(content_hash):
        logger.debug("[COGNITION] Duplicate episode skipped: %s", content_hash)
        # Return the existing ID
        with get_db() as db:
            row = db.execute(
                "SELECT id FROM cognition_memory_envelopes WHERE content_hash = %s",
                [content_hash],
            ).fetchone()
            return row[0] if row else ""

    episode = EpisodicMemory(
        entity_id=run_result.entity_id,
        cycle_id=run_result.cycle_id,
        event_type="pipeline_run",
        action=run_result.final_action,
        confidence=run_result.final_confidence,
        evidence_summary=run_result.summary or "",
        tags=run_result.tags,
    )

    with get_db() as db:
        now = _now_iso()

        db.execute(
            """
            INSERT INTO cognition_episodic_memories (
                id, entity_id, cycle_id, timestamp, event_type, action,
                confidence, evidence_summary, evidence_sources,
                debate_thesis_won, debate_persona, debate_challenges,
                outcome_label, outcome_pnl_pct, context_summary, tags,
                content_hash, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                episode.id,
                episode.entity_id,
                episode.cycle_id,
                now,
                episode.event_type,
                episode.action,
                episode.confidence,
                episode.evidence_summary,
                json.dumps(episode.evidence_sources),
                episode.debate_thesis_won,
                episode.debate_persona,
                json.dumps(episode.debate_challenges),
                episode.outcome_label,
                episode.outcome_pnl_pct,
                episode.context_summary,
                json.dumps(episode.tags),
                content_hash,
                now,
            ],
        )

        _write_envelope(
            memory_id=episode.id,
            memory_type=MemoryType.EPISODIC,
            entity_id=episode.entity_id,
            content_hash=content_hash,
            confidence=episode.confidence,
            tags=episode.tags,
            payload={"event_type": episode.event_type, "action": episode.action},
        )

        logger.info(
            "[COGNITION] Wrote episodic memory %s for %s",
            episode.id[:8],
            episode.entity_id,
        )
        return episode.id


def write_semantic(entity_id: str, fact: SemanticFact) -> str:
    """Persist a semantic fact as a memory.

    Idempotent: skips if the same subject+predicate+object exists.

    Args:
        entity_id: Entity this fact relates to.
        fact: The semantic fact to store.

    Returns:
        Memory ID.
    """
    _ensure_schema()
    from app.utils.text_utils import sanitize_surrogates
    fact = sanitize_surrogates(fact)

    content_key = f"{fact.subject}:{fact.predicate}:{fact.object}"
    content_hash = MemoryEnvelope.compute_content_hash("semantic", content_key)

    if _is_duplicate(content_hash):
        logger.debug("[COGNITION] Duplicate semantic fact skipped: %s", content_hash)
        with get_db() as db:
            row = db.execute(
                "SELECT id FROM cognition_memory_envelopes WHERE content_hash = %s",
                [content_hash],
            ).fetchone()
            return row[0] if row else ""

    mem = SemanticMemory(
        entity_id=entity_id,
        facts=[fact],
        domain_knowledge=f"{fact.subject} {fact.predicate} {fact.object}",
        tags=[fact.subject.lower(), fact.predicate.lower()],
        confidence=fact.confidence,
    )

    with get_db() as db:
        now = _now_iso()

        facts_json = json.dumps(
            [
                {
                    "subject": f.subject,
                    "predicate": f.predicate,
                    "object": f.object,
                    "source": f.source,
                    "confidence": f.confidence,
                }
                for f in mem.facts
            ]
        )

        db.execute(
            """
            INSERT INTO cognition_semantic_memories (
                id, entity_id, facts_json, domain_knowledge, tags,
                confidence, evidence_count, status, content_hash,
                created_at, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                mem.id,
                mem.entity_id,
                facts_json,
                mem.domain_knowledge,
                json.dumps(mem.tags),
                mem.confidence,
                mem.evidence_count,
                "active",
                content_hash,
                now,
                now,
            ],
        )

        _write_envelope(
            memory_id=mem.id,
            memory_type=MemoryType.SEMANTIC,
            entity_id=entity_id,
            content_hash=content_hash,
            confidence=mem.confidence,
            tags=mem.tags,
            payload={"domain_knowledge": mem.domain_knowledge},
        )

        logger.info(
            "[COGNITION] Wrote semantic memory %s: %s",
            mem.id[:8],
            mem.domain_knowledge[:60],
        )
        return mem.id


def write_reflection(reflection: ReflectionRecord) -> str:
    """Persist a reflection record from postmortem analysis.

    Idempotent: keyed on episode_id + root_cause.

    Args:
        reflection: Structured postmortem analysis.

    Returns:
        Reflection ID.
    """
    _ensure_schema()
    from app.utils.text_utils import sanitize_surrogates
    reflection = sanitize_surrogates(reflection)

    content_key = f"{reflection.episode_id}:{reflection.root_cause}"
    content_hash = MemoryEnvelope.compute_content_hash("reflective", content_key)

    if _is_duplicate(content_hash):
        logger.debug("[COGNITION] Duplicate reflection skipped: %s", content_hash)
        with get_db() as db:
            row = db.execute(
                "SELECT id FROM cognition_memory_envelopes WHERE content_hash = %s",
                [content_hash],
            ).fetchone()
            return row[0] if row else ""

    with get_db() as db:
        now = _now_iso()

        db.execute(
            """
            INSERT INTO cognition_reflections (
                id, episode_id, entity_id, timestamp,
                failure_patterns, missed_signals, late_verifiers,
                what_should_have_blocked, root_cause,
                recommended_changes, severity,
                expected_outcome, actual_outcome, outcome_delta,
                content_hash, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                reflection.id,
                reflection.episode_id,
                reflection.entity_id,
                now,
                json.dumps(reflection.failure_patterns),
                json.dumps(reflection.missed_signals),
                json.dumps(reflection.late_verifiers),
                reflection.what_should_have_blocked,
                reflection.root_cause,
                json.dumps(reflection.recommended_changes),
                reflection.severity,
                reflection.expected_outcome,
                reflection.actual_outcome,
                reflection.outcome_delta,
                content_hash,
                now,
            ],
        )

        _write_envelope(
            memory_id=reflection.id,
            memory_type=MemoryType.REFLECTIVE,
            entity_id=reflection.entity_id,
            content_hash=content_hash,
            confidence=0.5,
            tags=[reflection.severity],
            payload={"root_cause": reflection.root_cause},
        )

        logger.info(
            "[COGNITION] Wrote reflection %s for episode %s",
            reflection.id[:8],
            reflection.episode_id[:8],
        )
        return reflection.id


def write_procedural(rule: ProceduralRule) -> str:
    """Persist a process rule or heuristic.

    Idempotent: keyed on rule_text + category.

    Args:
        rule: The procedural rule to store.

    Returns:
        Rule memory ID.
    """
    _ensure_schema()
    from app.utils.text_utils import sanitize_surrogates
    rule = sanitize_surrogates(rule)

    content_key = f"{rule.category}:{rule.rule_text}"
    content_hash = MemoryEnvelope.compute_content_hash("procedural", content_key)

    if _is_duplicate(content_hash):
        logger.debug("[COGNITION] Duplicate procedural rule skipped: %s", content_hash)
        with get_db() as db:
            row = db.execute(
                "SELECT id FROM cognition_memory_envelopes WHERE content_hash = %s",
                [content_hash],
            ).fetchone()
            return row[0] if row else ""

    with get_db() as db:
        now = _now_iso()

        db.execute(
            """
            INSERT INTO cognition_procedural_memories (
                id, rule_text, category, applies_to, confidence,
                success_count, failure_count, source_episodes,
                tags, active, status, content_hash,
                created_at, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                rule.id,
                rule.rule_text,
                rule.category,
                json.dumps(rule.applies_to),
                rule.confidence,
                rule.success_count,
                rule.failure_count,
                json.dumps(rule.source_episodes),
                json.dumps(rule.tags),
                rule.active,
                "active",
                content_hash,
                now,
                now,
            ],
        )

        _write_envelope(
            memory_id=rule.id,
            memory_type=MemoryType.PROCEDURAL,
            entity_id="",
            content_hash=content_hash,
            confidence=rule.confidence,
            tags=rule.tags,
            payload={"rule_text": rule.rule_text, "category": rule.category},
        )

        logger.info(
            "[COGNITION] Wrote procedural rule %s: %s",
            rule.id[:8],
            rule.rule_text[:60],
        )
        return rule.id
