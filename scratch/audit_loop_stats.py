import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from app.db.connection import get_db

def run():
    print("=== DUMPING AGENT LOOP STATS ===")
    with get_db() as db:
        rows = db.execute(
            """SELECT cycle_id, ticker, agent_name, loops_used, token_usage, yielded, created_at 
               FROM agent_loop_stats 
               ORDER BY created_at DESC LIMIT 20;"""
        ).fetchall()
        print(f"Found {len(rows)} rows:")
        for r in rows:
            print(f"  Cycle: {r[0]} | Ticker: {r[1]} | Agent: {r[2]} | Loops: {r[3]} | Tokens: {r[4]} | Yielded: {r[5]} | CreatedAt: {r[6]}")

if __name__ == "__main__":
    run()
