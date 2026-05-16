"""
Cognition V2 — Memory Schema.

PostgreSQL table definitions for the V2 memory subsystem.
All tables use the `cognition_` prefix to stay separate from V1.

Uses the _ensure_schema() pattern consistent with the existing codebase.
"""

import logging

from app.db.connection import get_db

logger = logging.getLogger(__name__)

_SCHEMA_INITIALIZED = False


def _ensure_schema() -> None:
    """Create V2 memory tables if they don't exist.

    Safe to call multiple times — uses CREATE TABLE IF NOT EXISTS.
    Cached after first successful run.
    """
    global _SCHEMA_INITIALIZED
    if _SCHEMA_INITIALIZED:
        return

    with get_db() as db:
        # ── Semantic memories ──
        db.execute("""
            CREATE TABLE IF NOT EXISTS cognition_semantic_memories (
                id TEXT PRIMARY KEY,
                entity_id TEXT,
                facts_json TEXT,
                domain_knowledge TEXT,
                tags TEXT,
                confidence DOUBLE PRECISION DEFAULT 0.8,
                evidence_count INTEGER DEFAULT 1,
                status TEXT DEFAULT 'active',
                content_hash TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            );
        """)

        # ── Episodic memories ──
        db.execute("""
            CREATE TABLE IF NOT EXISTS cognition_episodic_memories (
                id TEXT PRIMARY KEY,
                entity_id TEXT,
                cycle_id TEXT,
                timestamp TIMESTAMP,
                event_type TEXT,
                action TEXT,
                confidence DOUBLE PRECISION DEFAULT 0.0,
                evidence_summary TEXT,
                evidence_sources TEXT,
                debate_thesis_won BOOLEAN,
                debate_persona TEXT,
                debate_challenges TEXT,
                outcome_label TEXT,
                outcome_pnl_pct DOUBLE PRECISION,
                context_summary TEXT,
                tags TEXT,
                content_hash TEXT,
                created_at TIMESTAMP
            );
        """)

        # ── Reflective memories ──
        db.execute("""
            CREATE TABLE IF NOT EXISTS cognition_reflective_memories (
                id TEXT PRIMARY KEY,
                entity_id TEXT,
                pattern_name TEXT,
                description TEXT,
                frequency INTEGER DEFAULT 1,
                reflection_ids TEXT,
                tags TEXT,
                severity TEXT DEFAULT 'medium',
                confidence DOUBLE PRECISION DEFAULT 0.5,
                status TEXT DEFAULT 'active',
                content_hash TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            );
        """)

        # ── Reflections (individual postmortem records) ──
        db.execute("""
            CREATE TABLE IF NOT EXISTS cognition_reflections (
                id TEXT PRIMARY KEY,
                episode_id TEXT,
                entity_id TEXT,
                timestamp TIMESTAMP,
                failure_patterns TEXT,
                missed_signals TEXT,
                late_verifiers TEXT,
                what_should_have_blocked TEXT,
                root_cause TEXT,
                recommended_changes TEXT,
                severity TEXT DEFAULT 'medium',
                expected_outcome TEXT,
                actual_outcome TEXT,
                outcome_delta DOUBLE PRECISION DEFAULT 0.0,
                content_hash TEXT,
                created_at TIMESTAMP
            );
        """)

        # ── Procedural memories ──
        db.execute("""
            CREATE TABLE IF NOT EXISTS cognition_procedural_memories (
                id TEXT PRIMARY KEY,
                rule_text TEXT,
                category TEXT,
                applies_to TEXT,
                confidence DOUBLE PRECISION DEFAULT 0.5,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                source_episodes TEXT,
                tags TEXT,
                active BOOLEAN DEFAULT TRUE,
                status TEXT DEFAULT 'active',
                content_hash TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            );
        """)

        # ── Memory envelope index (unified lookup) ──
        db.execute("""
            CREATE TABLE IF NOT EXISTS cognition_memory_envelopes (
                id TEXT PRIMARY KEY,
                memory_type TEXT,
                entity_id TEXT,
                content_hash TEXT,
                status TEXT DEFAULT 'active',
                confidence DOUBLE PRECISION DEFAULT 0.5,
                tags TEXT,
                ttl_days INTEGER,
                payload_json TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                last_accessed_at TIMESTAMP
            );
        """)

        _SCHEMA_INITIALIZED = True
        logger.debug("[COGNITION] Memory schema initialized")


def reset_schema_cache() -> None:
    """Reset the schema cache — used in tests."""
    global _SCHEMA_INITIALIZED
    _SCHEMA_INITIALIZED = False
