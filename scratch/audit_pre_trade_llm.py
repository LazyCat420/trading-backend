import os
import sys

# Adjust path to import app modules
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from app.db.connection import get_db

def run():
    cycle_id = "cycle-1779695975"
    print(f"=== CHECKING LLM AUDIT LOGS FOR CYCLE {cycle_id} ===")
    with get_db() as db:
        # Count total rows
        cnt = db.execute("SELECT COUNT(*) FROM llm_audit_logs WHERE cycle_id = %s;", [cycle_id]).fetchone()[0]
        print(f"Total audit logs for cycle: {cnt}")
        
        # Get distinct steps
        steps = db.execute("SELECT agent_step, COUNT(*) FROM llm_audit_logs WHERE cycle_id = %s GROUP BY agent_step;", [cycle_id]).fetchall()
        print("Distinct agent_steps:")
        for s in steps:
            print(f"  {s[0]}: {s[1]} rows")
            
        # Get some sample rows where step is not null
        samples = db.execute(
            "SELECT ticker, agent_step, model, raw_response FROM llm_audit_logs WHERE cycle_id = %s LIMIT 3;",
            [cycle_id]
        ).fetchall()
        print("\nSample rows:")
        for r in samples:
            print(f"Ticker: {r[0]} | Step: {r[1]} | Model: {r[2]}")
            print(f"Raw Response: {r[3][:100]}...")

if __name__ == "__main__":
    run()
