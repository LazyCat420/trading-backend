import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.connection import get_db

def run():
    with get_db() as db:
        print("=== RECENT CYCLE AUDIT LOGS ===")
        db.execute("""
            SELECT timestamp, ticker, event_type, message
            FROM cycle_audit_log
            ORDER BY timestamp DESC
            LIMIT 50
        """)
        rows = db.fetchall()
        for r in rows:
            print(f"[{r[0]}] {r[1] or 'sys':6s} | {r[2] or '':20s} | {r[3]}")
            
        print("\n=== RECENT LLM AUDIT LOGS ===")
        db.execute("""
            SELECT created_at, ticker, agent_step, endpoint_name, model, tokens_used, execution_ms
            FROM llm_audit_logs
            ORDER BY created_at DESC
            LIMIT 30
        """)
        rows = db.fetchall()
        for r in rows:
            print(f"[{r[0]}] {r[1] or 'sys':6s} | {r[2] or '':20s} | {r[3] or '':12s} | {r[4] or '':20s} | {r[5]} tok | {r[6]} ms")

if __name__ == "__main__":
    run()
