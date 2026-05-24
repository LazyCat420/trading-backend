import sys
import os

local_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(local_dir, ".."))

from app.db.connection import get_db

with get_db() as db:
    print("--- DEBATE FAILURES IN LAST CYCLE ---")
    db.execute(
        """
        SELECT timestamp, cycle_id, step, status, detail 
        FROM pipeline_events 
        WHERE (step LIKE '%debate_fail%' OR step LIKE '%debate_error%') 
        ORDER BY timestamp DESC LIMIT 20
        """
    )
    rows = db.fetchall()
    for row in rows:
        print(f"TS: {row[0]} | Cycle: {row[1]} | Step: {row[2]} | Status: {row[3]}")
        print(f"Detail: {row[4]}")
        print("-" * 50)
