import os
import sys

# Adjust path to import app modules
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from app.db.connection import get_db

def run():
    cycle_id = "cycle-1779695975"
    print(f"=== CHECKING AGENT TRACES FOR CYCLE {cycle_id} ===")
    with get_db() as db:
        # Count total traces
        cnt = db.execute("SELECT COUNT(*) FROM agent_traces WHERE run_id = %s;", [cycle_id]).fetchone()[0]
        print(f"Total agent traces for cycle: {cnt}")
        
        # Get distinct agents in traces
        agents = db.execute("SELECT agent_name, COUNT(*) FROM agent_traces WHERE run_id = %s GROUP BY agent_name;", [cycle_id]).fetchall()
        print("Distinct agents in traces:")
        for a in agents:
            print(f"  {a[0]}: {a[1]} rows")
            
        # Get detailed traces for pre_trade agent
        traces = db.execute(
            """SELECT ticker, loop_step, tool_name, tool_args, tool_result_summary, why_tool_was_called, stop_reason, model_name 
               FROM agent_traces 
               WHERE run_id = %s AND agent_name = 'pre_trade' 
               ORDER BY ticker, loop_step;""",
            [cycle_id]
        ).fetchall()
        
        print(f"\nPre-trade traces (total: {len(traces)}):")
        for t in traces:
            print(f"\nTicker: {t[0]} | Step: {t[1]} | Tool: {t[2]}")
            print(f"  Args: {t[3]}")
            print(f"  Why: {t[5]}")
            print(f"  Result Summary: {t[4][:200]}...")
            print(f"  Stop Reason: {t[6]}")

if __name__ == "__main__":
    run()
