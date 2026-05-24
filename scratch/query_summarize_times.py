import os
import sys

# Insert project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.connection import get_db

def run():
    cycle_id = "cycle-1779414661"
    with get_db() as db:
        print(f"=== SUMMARIZE EVENTS FOR {cycle_id} ===")
        db.execute("""
            SELECT timestamp, step, status, elapsed_ms, detail
            FROM pipeline_events
            WHERE cycle_id = %s AND (step ILIKE '%%summarize%%' OR detail ILIKE '%%summarize%%')
            ORDER BY timestamp ASC
        """, [cycle_id])
        rows = db.fetchall()
        for r in rows:
            print(f"[{r[0]}] {r[1]} | Status: {r[2]} | Elapsed: {r[3]}ms | {r[4]}")

if __name__ == "__main__":
    run()
