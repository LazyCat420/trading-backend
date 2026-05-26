import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.connection import get_db

def main():
    with get_db() as db:
        cycle_id = "cycle-1779766850"
        
        # Query distinct tickers and their decisions in this cycle
        rows = db.execute(
            "SELECT timestamp, ticker, message, severity FROM cycle_audit_log WHERE cycle_id = %s AND ticker IN ('RKLB', 'GEN') ORDER BY timestamp ASC",
            [cycle_id]
        ).fetchall()
        
        print("\n=== Event Log for RKLB and GEN ===")
        for r in rows:
            print(f"[{r[0]}] {r[1]}: {r[2]}")

if __name__ == "__main__":
    main()
