import logging
from app.db.connection import get_db

logger = logging.getLogger(__name__)

def run_auto_migrations():
    """Run all inline DDL updates and migrations on startup."""
    with get_db() as db:
        # ── Auto-migrate: add stop_loss_pct to positions if missing ──
        try:
            cols = db.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'positions'"
            ).fetchall()
            col_names = {r[0] for r in cols}
            if "stop_loss_pct" not in col_names:
                db.execute(
                    "ALTER TABLE positions ADD COLUMN stop_loss_pct DOUBLE PRECISION DEFAULT 0.08"
                )
                logger.info("Migrated: added stop_loss_pct to positions.")
        except Exception as e:
            logger.warning("positions migration check: %s", e)

        # ── Auto-migrate: add status_reason, banned_at to watchlist if missing ──
        try:
            cols = db.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'watchlist'"
            ).fetchall()
            col_names = {r[0] for r in cols}
            if "status_reason" not in col_names:
                db.execute("ALTER TABLE watchlist ADD COLUMN status_reason VARCHAR")
                logger.info("Migrated: added status_reason to watchlist.")
            if "banned_at" not in col_names:
                db.execute("ALTER TABLE watchlist ADD COLUMN banned_at TIMESTAMP")
                logger.info("Migrated: added banned_at to watchlist.")
        except Exception as e:
            logger.warning("watchlist migration check: %s", e)

        # ── Auto-migrate: add tool_playbook table if missing ──
        try:
            db.execute("""
                CREATE TABLE IF NOT EXISTS tool_playbook (
                    id              TEXT PRIMARY KEY,
                    agent_name      TEXT,
                    tool_name       TEXT,
                    playbook_text   TEXT,
                    success_rate    DOUBLE PRECISION DEFAULT 0.0,
                    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            logger.info("Migrated: ensured tool_playbook table exists.")
        except Exception as e:
            logger.warning("tool_playbook migration check: %s", e)

        # ── Auto-migrate: add morning_briefings table if missing ──
        try:
            db.execute("""
                CREATE TABLE IF NOT EXISTS morning_briefings (
                    id                  SERIAL PRIMARY KEY,
                    created_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    report_content      TEXT NOT NULL,
                    tickers_evaluated   TEXT[] NOT NULL
                );
            """)
            logger.info("Migrated: ensured morning_briefings table exists.")
        except Exception as e:
            logger.warning("morning_briefings migration check: %s", e)

        # ── Auto-migrate: add data_archive table if missing ──
        try:
            db.execute("""
                CREATE TABLE IF NOT EXISTS data_archive (
                    id              SERIAL PRIMARY KEY,
                    source_table    TEXT NOT NULL,
                    source_id       TEXT NOT NULL,
                    ticker          TEXT,
                    title           TEXT,
                    content         TEXT,
                    original_date   TIMESTAMP WITH TIME ZONE,
                    archived_at     TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    purge_after     TIMESTAMP WITH TIME ZONE NOT NULL,
                    UNIQUE(source_table, source_id)
                );
            """)
            logger.info("Migrated: ensured data_archive table exists.")
        except Exception as e:
            logger.warning("data_archive migration check: %s", e)

        # ── Auto-migrate: add flash_briefings table if missing ──
        try:
            db.execute("""
                CREATE TABLE IF NOT EXISTS flash_briefings (
                    id              SERIAL PRIMARY KEY,
                    created_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    report_content  TEXT NOT NULL,
                    source_urls     TEXT[],
                    article_count   INTEGER DEFAULT 0
                );
            """)
            logger.info("Migrated: ensured flash_briefings table exists.")
        except Exception as e:
            logger.warning("flash_briefings migration check: %s", e)

        # ── Auto-migrate: add new columns to llm_audit_logs if missing ──
        try:
            cols = db.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'llm_audit_logs'"
            ).fetchall()
            col_names = {r[0] for r in cols}
            if "endpoint_name" not in col_names:
                db.execute("ALTER TABLE llm_audit_logs ADD COLUMN endpoint_name TEXT")
                db.execute("ALTER TABLE llm_audit_logs ADD COLUMN prompt_tokens INTEGER")
                db.execute("ALTER TABLE llm_audit_logs ADD COLUMN completion_tokens INTEGER")
                db.execute("ALTER TABLE llm_audit_logs ADD COLUMN queue_wait_ms INTEGER")
                db.execute("ALTER TABLE llm_audit_logs ADD COLUMN tokens_per_second DOUBLE PRECISION")
                logger.info("Migrated: added new metrics columns to llm_audit_logs.")
        except Exception as e:
            logger.warning("llm_audit_logs migration check: %s", e)

if __name__ == "__main__":
    run_auto_migrations()
