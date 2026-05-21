import os
import sys

local_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(local_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if local_dir not in sys.path:
    sys.path.insert(0, local_dir)

from app.db.connection import get_db

def main():
    print("Inspecting cycle_audit_log entries starting with 'sched-'...")
    with get_db() as db:
        db.execute("SELECT id, cycle_id, timestamp, audit_type, phase, ticker, message FROM cycle_audit_log WHERE cycle_id LIKE 'sched-%%' ORDER BY timestamp DESC LIMIT 20")
        rows = db.fetchall()
        for r in rows:
            print(f"ID: {r[0]} | CycleID: {r[1]} | TS: {r[2]} | Type: {r[3]} | Phase: {r[4]} | Ticker: {r[5]} | Msg: {r[6]}")

if __name__ == "__main__":
    main()
