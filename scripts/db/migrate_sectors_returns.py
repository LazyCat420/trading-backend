import sys
import os

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.db.connection import get_db


def migrate():
    db = get_db()
    queries = [
        "ALTER TABLE sector_performance ADD COLUMN IF NOT EXISTS avg_return_60d DOUBLE PRECISION;",
        "ALTER TABLE sector_performance ADD COLUMN IF NOT EXISTS avg_return_6mo DOUBLE PRECISION;",
        "ALTER TABLE sector_performance ADD COLUMN IF NOT EXISTS avg_return_1y DOUBLE PRECISION;",
        "ALTER TABLE sector_performance ADD COLUMN IF NOT EXISTS relative_strength_1y DOUBLE PRECISION;",
    ]

    for q in queries:
        try:
            db.execute(q)
            print(f"Executed: {q}")
        except Exception as e:
            print(f"Error executing {q}: {e}")


if __name__ == "__main__":
    migrate()
    print("Migration complete.")
