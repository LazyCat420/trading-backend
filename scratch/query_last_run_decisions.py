import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.connection import get_db

def main():
    with get_db() as db:
        # Get latest cycle_id
        row = db.execute(
            "SELECT cycle_id, MAX(timestamp) FROM cycle_audit_log WHERE cycle_id LIKE 'cycle-%' GROUP BY cycle_id ORDER BY MAX(timestamp) DESC LIMIT 1"
        ).fetchone()
        if not row:
            print("No standard cycles found.")
            return
        
        cycle_id = row[0]
        timestamp = row[1]
        print(f"Latest Cycle ID: {cycle_id} at {timestamp}")
        
        # Query distinct tickers and their decisions in this cycle
        # The thesis phase logs decisions in cycle_audit_log
        rows = db.execute(
            "SELECT timestamp, ticker, message, severity FROM cycle_audit_log WHERE cycle_id = %s AND (message LIKE '%%decision%%' OR message LIKE '%%BUY%%' OR message LIKE '%%SELL%%' OR message LIKE '%%HOLD%%' OR message LIKE '%%PASS%%') ORDER BY timestamp ASC",
            [cycle_id]
        ).fetchall()
        
        print("\n=== Event Log Matching Decisions ===")
        for r in rows:
            print(f"[{r[0]}] {r[1]}: {r[2]}")

if __name__ == "__main__":
    main()
