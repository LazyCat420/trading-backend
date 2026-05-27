import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.connection import get_db

def run():
    with get_db() as db:
        print("=== OKLO CYCLE AUDIT LOGS ===")
        db.execute("""
            SELECT timestamp, cycle_id, event_type, message
            FROM cycle_audit_log
            WHERE ticker = 'OKLO'
            ORDER BY timestamp DESC
            LIMIT 50
        """)
        rows = db.fetchall()
        for r in rows:
            print(f"[{r[0]}] {r[1]:20s} | {r[2] or '':20s} | {r[3]}")
            
        print("\n=== OKLO LLM AUDIT LOGS ===")
        db.execute("""
            SELECT created_at, cycle_id, agent_step, endpoint_name, model, tokens_used, execution_ms
            FROM llm_audit_logs
            WHERE ticker = 'OKLO'
            ORDER BY created_at DESC
            LIMIT 50
        """)
        rows = db.fetchall()
        for r in rows:
            print(f"[{r[0]}] {r[1]:20s} | {r[2] or '':20s} | {r[3] or '':12s} | {r[4] or '':20s} | {r[5]} tok | {r[6]} ms")

if __name__ == "__main__":
    run()
