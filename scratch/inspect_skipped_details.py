import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from app.db.connection import get_db

def run():
    cycle_id = "cycle-1779695975"
    print(f"=== SKIPPED TICKERS DETAILS FOR {cycle_id} ===")
    with get_db() as db:
        row = db.execute(
            """SELECT data_json 
               FROM pipeline_events 
               WHERE cycle_id = %s AND phase = 'trading' AND step = 'complete'""",
            [cycle_id]
        ).fetchone()
        
        if row and row[0]:
            import json
            data = json.loads(row[0]) if isinstance(row[0], str) else row[0]
            print("Data JSON keys:", data.keys() if isinstance(data, dict) else "not a dict")
            # Let's query execution_errors if any or details in skipped
            # Let's also check if there are other events with warnings/vetoes
            # We can select all events in trading phase
            events = db.execute(
                """SELECT timestamp, step, detail, data_json 
                   FROM pipeline_events 
                   WHERE cycle_id = %s AND phase = 'trading'
                   ORDER BY timestamp ASC;""",
                [cycle_id]
            ).fetchall()
            for ev in events:
                print(f"\n[{ev[0]}] Step: {ev[1]} | Detail: {ev[2]}")
                if ev[3]:
                    print("Data JSON:", json.dumps(ev[3], indent=2))
        else:
            print("No complete step found in trading phase.")

if __name__ == "__main__":
    run()
