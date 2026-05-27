"""
Shared database migrations — run-once column additions.

Consolidates _ensure_summary_columns() that was duplicated in:
  - context_builder.py
  - context_builder.py
  - summarizer.py (superset with quality columns)

Usage:
    from app.utils.db_migrations import ensure_summary_columns
    ensure_summary_columns()  # safe to call multiple times
"""

import logging

from app.db.connection import get_db

logger = logging.getLogger(__name__)

_summary_columns_ensured = False


def ensure_summary_columns(db=None):
    """Add summary/quality columns if they don't exist (idempotent).

    Merges all migrations from context_builder + summarizer into one
    canonical superset. Safe to call multiple times — uses a module-level
    flag to skip after the first successful run.

    Args:
        db: Optional database connection. If None, calls get_db().
    """
    global _summary_columns_ensured
    if _summary_columns_ensured:
        return
    try:
        if db is None:
            with get_db() as new_db:
                ensure_summary_columns(new_db)
            return

        migrations = [
            # YouTube
            "ALTER TABLE youtube_transcripts ADD COLUMN IF NOT EXISTS summary VARCHAR",
            "ALTER TABLE youtube_transcripts ADD COLUMN IF NOT EXISTS tickers_mentioned VARCHAR",
            "ALTER TABLE youtube_transcripts ADD COLUMN IF NOT EXISTS summarized_at TIMESTAMP",
            "ALTER TABLE youtube_transcripts ADD COLUMN IF NOT EXISTS quality_status VARCHAR",
            "ALTER TABLE youtube_transcripts ADD COLUMN IF NOT EXISTS quality_reason VARCHAR",
            "ALTER TABLE youtube_transcripts ADD COLUMN IF NOT EXISTS quality_score INTEGER",
            # Reddit
            "ALTER TABLE reddit_posts ADD COLUMN IF NOT EXISTS summary VARCHAR",
            "ALTER TABLE reddit_posts ADD COLUMN IF NOT EXISTS summarized_at TIMESTAMP",
            "ALTER TABLE reddit_posts ADD COLUMN IF NOT EXISTS quality_status VARCHAR",
            "ALTER TABLE reddit_posts ADD COLUMN IF NOT EXISTS quality_reason VARCHAR",
            "ALTER TABLE reddit_posts ADD COLUMN IF NOT EXISTS quality_score INTEGER",
            "ALTER TABLE reddit_posts ADD COLUMN IF NOT EXISTS qualitative_draft JSONB",
            # News (base + quality columns from summarizer)
            "ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS llm_summary VARCHAR",
            "ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS summarized_at TIMESTAMP",
            "ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS quality_status VARCHAR",
            "ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS quality_reason VARCHAR",
            "ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS quality_score INTEGER",
            "ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS qualitative_draft JSONB",
            # AutoResearch V2
            """CREATE TABLE IF NOT EXISTS cycle_summaries (
                ticker VARCHAR,
                cycle_id VARCHAR,
                cycle_date TIMESTAMP,
                agent_name VARCHAR,
                action VARCHAR,
                confidence INTEGER,
                confidence_tier VARCHAR,
                rationale_summary VARCHAR,
                was_correct BOOLEAN,
                outcome_pnl DOUBLE PRECISION,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (ticker, cycle_id)
            )""",
            """CREATE TABLE IF NOT EXISTS debate_history (
                ticker VARCHAR,
                cycle_id VARCHAR,
                pro_argument VARCHAR,
                con_argument VARCHAR,
                winner VARCHAR,
                final_confidence INTEGER,
                UNIQUE (ticker, cycle_id)
            )""",
            """CREATE TABLE IF NOT EXISTS company_narratives (
                ticker VARCHAR PRIMARY KEY,
                story_summary TEXT NOT NULL,
                key_themes JSONB NOT NULL,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )""",
        ]
        for sql in migrations:
            try:
                db.execute(sql)
            except Exception:
                pass  # Column already exists or other benign error
        _summary_columns_ensured = True
    except Exception as e:
        logger.warning("[PIPELINE] Could not ensure summary columns: %s", e)
