import os
import sys

local_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if local_dir not in sys.path:
    sys.path.insert(0, local_dir)

from app.db.connection import get_db

def run():
    with get_db() as db:
        print("=== PRE_TRADE TRACES IN LATEST CYCLE ===")
        # Get the latest cycle_id
        latest_cycle_row = db.execute(
            "SELECT cycle_id FROM analysis_results ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if not latest_cycle_row:
            print("No cycles found.")
            return
        
        cycle_id = latest_cycle_row[0]
        print(f"Latest Cycle ID: {cycle_id}\n")
        
        # Query traces
        traces = db.execute(
            """
            SELECT loop_step, tool_name, tool_args, tool_result_summary, why_tool_was_called, stop_reason
            FROM agent_traces
            WHERE run_id = %s AND agent_name = 'pre_trade'
            ORDER BY loop_step ASC
            """,
            [cycle_id]
        ).fetchall()
        
        if not traces:
            print("No pre_trade traces found in agent_traces.")
        else:
            for row in traces:
                step, tool_name, tool_args, tool_result, rationale, stop_reason = row
                print(f"  Step {step}: Tool: {tool_name}")
                print(f"    Args: {tool_args}")
                print(f"    Rationale: {rationale[:100] if rationale else None}")
                print(f"    Result Preview: {tool_result[:100] if tool_result else None}")

if __name__ == "__main__":
    run()
