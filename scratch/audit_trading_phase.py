import os
import sys

# Adjust path to import app modules
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from app.db.connection import get_db

def run():
    cycle_id = "cycle-1779695975"
    print(f"=== AUDIT FOR CYCLE {cycle_id} ===")
    with get_db() as db:
        rows = db.execute(
            "SELECT timestamp, phase, ticker, severity, message, data FROM cycle_audit_log WHERE cycle_id = %s ORDER BY timestamp ASC;",
            [cycle_id]
        ).fetchall()
        
        for r in rows:
            print(f"[{r[0]}] {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]}")

if __name__ == "__main__":
    run()
