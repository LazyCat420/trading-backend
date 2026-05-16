"""
PostgreSQL-backed graph storage for the unified Trading Brain Graph.

Uses the ``cognition`` database schema to isolate from v1 tables.

Adds ``confidence_level``, ``graph_family``, and ``version`` columns so
that hard facts are never confused with inferred relationships, and
queries can be scoped to a single logical graph family.
"""

import logging
from app.db.connection import get_db

logger = logging.getLogger(__name__)


def _ensure_schema():
    """Ensure the schema is present if Developer 1's schema is not yet merged to main schema.sql."""
    with get_db() as db:
        db.execute("CREATE SCHEMA IF NOT EXISTS cognition;")

        db.execute("""
        CREATE TABLE IF NOT EXISTS cognition.ontology_entities (
            id              TEXT PRIMARY KEY,
            entity_type     TEXT NOT NULL,
            canonical_name  TEXT NOT NULL,
            properties      JSONB,
            graph_family    TEXT,
            version         TEXT,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        db.execute("""
        CREATE TABLE IF NOT EXISTS cognition.ontology_edges (
            id               TEXT PRIMARY KEY,
            source_id        TEXT NOT NULL,
            target_id        TEXT NOT NULL,
            edge_type        TEXT NOT NULL,
            properties       JSONB,
            confidence       DOUBLE PRECISION DEFAULT 1.0,
            confidence_level TEXT DEFAULT 'fact',
            source_ref       TEXT,
            graph_family     TEXT,
            version          TEXT,
            created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        db.execute("""
        CREATE TABLE IF NOT EXISTS cognition.ontology_aliases (
            alias           TEXT NOT NULL,
            entity_id       TEXT NOT NULL,
            entity_type     TEXT,
            source          TEXT DEFAULT 'manual',
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (alias, entity_id)
        );
        """)

        db.execute("""
        CREATE TABLE IF NOT EXISTS cognition.ontology_claim_links (
            claim_id        TEXT NOT NULL,
            evidence_id     TEXT NOT NULL,
            link_type       TEXT NOT NULL,
            confidence      DOUBLE PRECISION DEFAULT 1.0,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (claim_id, evidence_id)
        );
        """)

        db.execute("""
        CREATE TABLE IF NOT EXISTS cognition.ontology_event_links (
            event_id        TEXT NOT NULL,
            entity_id       TEXT NOT NULL,
            impact_type     TEXT,
            impact_score    DOUBLE PRECISION,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (event_id, entity_id)
        );
        """)

        # ── Indexes ──
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_cog_entity_type ON cognition.ontology_entities(entity_type);"
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_cog_entity_name ON cognition.ontology_entities(canonical_name);"
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_cog_entity_family ON cognition.ontology_entities(graph_family);"
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_cog_edge_source ON cognition.ontology_edges(source_id);"
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_cog_edge_target ON cognition.ontology_edges(target_id);"
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_cog_edge_type ON cognition.ontology_edges(edge_type);"
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_cog_edge_family ON cognition.ontology_edges(graph_family);"
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_cog_edge_conf_level ON cognition.ontology_edges(confidence_level);"
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_cog_alias_name ON cognition.ontology_aliases(alias);"
        )

        # ── Migrations for existing databases ──
        _migrate_add_columns(db)


# Column definitions allowed by the migration system (allowlist).
_VALID_MIGRATION_COLUMNS = {
    ("cognition.ontology_entities", "graph_family", "TEXT"),
    ("cognition.ontology_entities", "version", "TEXT"),
    ("cognition.ontology_edges", "confidence_level", "TEXT DEFAULT 'fact'"),
    ("cognition.ontology_edges", "graph_family", "TEXT"),
    ("cognition.ontology_edges", "version", "TEXT"),
}


def _migrate_add_columns(db):
    """Add new columns to existing tables if they were created before this schema version."""
    for table, column, col_type in _VALID_MIGRATION_COLUMNS:
        _add_column_if_missing(db, table, column, col_type)


def _add_column_if_missing(db, table: str, column: str, col_type: str):
    """Safely add a column — swallow error if column already exists.

    Only accepts (table, column, col_type) tuples that are in the
    _VALID_MIGRATION_COLUMNS allowlist to prevent SQL injection.
    """
    if (table, column, col_type) not in _VALID_MIGRATION_COLUMNS:
        return
    try:
        db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type};")
    except Exception:
        pass  # Column already exists


def initialize_graph_db():
    """Manually trigger schema creation (useful for tests/boot)."""
    _ensure_schema()
