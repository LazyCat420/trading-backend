import os
import sys
import json

# Add parent directory to path to allow importing app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.connection import get_db

def main():
    print("=" * 80)
    print("AUDITING CONTEXT COMPRESSION & ROLLING WINDOW SYSTEM")
    print("=" * 80)

    with get_db() as db:
        # Check if cycle_context table has any raw responses with compression markers
        print("\n--- 1. Checking cycle_context for compression/truncation markers ---")
        try:
            # We want to see if raw_response or summary contain compressed/truncated content
            rows = db.execute(
                "SELECT id, agent_name, ticker, summary FROM cycle_context WHERE raw_response LIKE '%truncated%' OR summary LIKE '%COMPRESSED%'"
            ).fetchall()
            print(f"Found {len(rows)} rows in cycle_context with compression/truncation markers.")
            for r in rows[:10]:
                print(f"  - ID: {r[0]} | Agent: {r[1]} | Ticker: {r[2]} | Summary: {r[3][:100]}...")
        except Exception as e:
            print(f"Error reading cycle_context: {e}")

        # Check agent_traces for tool result truncations
        print("\n--- 2. Checking agent_traces for truncated tool results ---")
        try:
            rows = db.execute(
                "SELECT count(*), agent_name, tool_name FROM agent_traces WHERE tool_result_summary LIKE '%truncated%' GROUP BY agent_name, tool_name"
            ).fetchall()
            print(f"Tool results truncated by agent and tool:")
            for r in rows:
                print(f"  - Count: {r[0]} | Agent: {r[1]} | Tool: {r[2]}")
            
            # Show a sample truncated trace
            sample = db.execute(
                "SELECT agent_name, tool_name, tool_result_summary FROM agent_traces WHERE tool_result_summary LIKE '%truncated%' LIMIT 1"
            ).fetchone()
            if sample:
                print(f"\nSample Truncated Result from Agent '{sample[0]}' calling Tool '{sample[1]}':")
                print("-" * 50)
                print(sample[2][:300])
                print("-" * 50)
        except Exception as e:
            print(f"Error reading agent_traces: {e}")

        # Check agent_loop_stats token distribution
        print("\n--- 3. Checking agent_loop_stats token usage distribution ---")
        try:
            stats = db.execute(
                "SELECT agent_name, COUNT(*), MIN(token_usage), AVG(token_usage)::int, MAX(token_usage) FROM agent_loop_stats GROUP BY agent_name"
            ).fetchall()
            print("Token Usage stats by Agent:")
            for s in stats:
                print(f"  - Agent: {s[0]:30} | Runs: {s[1]:3} | Min: {s[2]:6} | Avg: {s[3]:6} | Max: {s[4]:6}")
        except Exception as e:
            print(f"Error reading agent_loop_stats: {e}")

        # Check pending approvals for pauses/yields
        print("\n--- 4. Checking pending approvals (yields on limits) ---")
        try:
            rows = db.execute(
                "SELECT agent_name, command, reason, status, created_at FROM pending_approvals ORDER BY created_at DESC LIMIT 5"
            ).fetchall()
            print(f"Latest pending approvals:")
            for r in rows:
                print(f"  - Agent: {r[0]} | Status: {r[3]} | Reason: {r[2]} | Command: {r[1]}")
        except Exception as e:
            print(f"Error reading pending_approvals: {e}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
