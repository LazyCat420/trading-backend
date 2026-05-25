import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from app.db.connection import get_db

def run():
    print("=== MOST RECENT GLOBAL PRE_TRADE LOGS ===")
    with get_db() as db:
        rows = db.execute(
            """SELECT created_at, cycle_id, ticker, model, raw_response 
               FROM llm_audit_logs 
               WHERE agent_step = 'pre_trade' 
               ORDER BY created_at DESC LIMIT 5;"""
        ).fetchall()
        for r in rows:
            print(f"\n[{r[0]}] Cycle: {r[1]} | Ticker: {r[2]} | Model: {r[3]}")
            print("Raw Response:")
            print(r[4])

if __name__ == "__main__":
    run()
