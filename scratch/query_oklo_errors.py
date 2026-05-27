import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from app.db.connection import get_db

def query():
    with get_db() as db:
        print("=== Re-enable / online messages in cycle_audit_log ===")
        db.execute("""
            SELECT timestamp, cycle_id, ticker, severity, message
            FROM cycle_audit_log
            WHERE message ILIKE '%back online%' OR message ILIKE '%re-enabled%' OR message ILIKE '%models ready%'
            ORDER BY timestamp DESC
            LIMIT 30
        """)
        rows = db.fetchall()
        for r in rows:
            print(f"[{r[0]}] Cycle: {r[1]} | Ticker: {r[2]} | Severity: {r[3]} | Msg: {r[4]}")
            print("-" * 50)

if __name__ == "__main__":
    query()
