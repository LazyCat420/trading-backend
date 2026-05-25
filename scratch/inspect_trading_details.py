import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from app.db.connection import get_db

def run():
    cycle_id = "cycle-1779695975"
    print(f"=== DETAILED TRADING PHASE AUDIT FOR {cycle_id} ===")
    with get_db() as db:
        rows = db.execute(
            """SELECT timestamp, phase, step, detail, status, data_json 
               FROM pipeline_events 
               WHERE cycle_id = %s AND (phase = 'trading' OR detail LIKE '%%VETO%%' OR detail LIKE '%%veto%%' OR detail LIKE '%%GATE%%' OR detail LIKE '%%gate%%')
               ORDER BY timestamp ASC;""",
            [cycle_id]
        ).fetchall()
        
        for r in rows:
            print(f"[{r[0]}] Phase: {r[1]} | Step: {r[2]} | Status: {r[4]} | Detail: {r[3]}")
            if r[5]:
                print(f"  Data: {r[5]}")

if __name__ == "__main__":
    run()
