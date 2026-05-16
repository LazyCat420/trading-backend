"""Migration script to add avg_ticker_ms to cycle_benchmarks."""

import logging
from app.db.connection import get_db

logger = logging.getLogger(__name__)


def migrate():
    db = get_db()

    # Check if the column already exists
    try:
        columns = db.execute("PRAGMA table_info(cycle_benchmarks)").fetchall()
        column_names = [col[1] for col in columns]

        if "avg_ticker_ms" not in column_names:
            logger.info("Adding avg_ticker_ms column to cycle_benchmarks...")
            db.execute("ALTER TABLE cycle_benchmarks ADD COLUMN avg_ticker_ms INTEGER")
            logger.info(
                "Backfilling avg_ticker_ms using total_ms / GREATEST(1, ticker_count)..."
            )
            db.execute(
                "UPDATE cycle_benchmarks SET avg_ticker_ms = CAST(total_ms / GREATEST(1, ticker_count) AS INTEGER)"
            )
            logger.info("Migration successful: added avg_ticker_ms.")
        else:
            logger.info(
                "Column avg_ticker_ms already exists, checking if it needs backfill..."
            )
            db.execute(
                "UPDATE cycle_benchmarks SET avg_ticker_ms = CAST(total_ms / GREATEST(1, ticker_count) AS INTEGER) WHERE avg_ticker_ms IS NULL"
            )
            logger.info("Backfill completed.")

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise e


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    migrate()
