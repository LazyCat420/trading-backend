import os
import sys
import json

local_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(local_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if local_dir not in sys.path:
    sys.path.insert(0, local_dir)

from app.db.connection import get_db

def main():
    print("Checking database cycle IDs...")
    with get_db() as db:
        tables = ["cycle_audit_log", "cycle_benchmarks", "llm_audit_logs", "pipeline_state", "pipeline_events"]
        for table in tables:
            try:
                db.execute(f"SELECT DISTINCT cycle_id FROM {table} ORDER BY cycle_id DESC LIMIT 5")
                rows = db.fetchall()
                print(f"\nTable '{table}' cycle_ids:")
                for r in rows:
                    print(f"  - {r[0]}")
            except Exception as e:
                print(f"Error querying {table}: {e}")

if __name__ == "__main__":
    main()
