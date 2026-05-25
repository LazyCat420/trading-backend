import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from app.db.connection import get_db

def run():
    print("=== DUMPING ALL PRE_TRADE ROWS FROM LLM_AUDIT_LOGS ===")
    with get_db() as db:
        rows = db.execute(
            """SELECT created_at, cycle_id, ticker, model, raw_response 
               FROM llm_audit_logs 
               WHERE agent_step = 'pre_trade' 
               ORDER BY created_at DESC;"""
        ).fetchall()
        print(f"Found {len(rows)} pre_trade rows:")
        for idx, r in enumerate(rows):
            print(f"\n[{idx+1}] Time: {r[0]} | Cycle: {r[1]} | Ticker: {r[2]} | Model: {r[3]}")
            print(f"Raw Response snippet: {r[4][:300]}...")

if __name__ == "__main__":
    run()
