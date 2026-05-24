import sys
import os

local_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(local_dir, ".."))

from app.db.connection import get_db

with get_db() as db:
    print("--- DEBATE AUDIT LOG DETAILS ---")
    db.execute(
        """
        SELECT id, cycle_id, ticker, agent_step, model, endpoint_name, tokens_used, execution_ms, SUBSTRING(raw_response FROM 1 FOR 300) 
        FROM llm_audit_logs 
        WHERE agent_step LIKE '%bull%' OR agent_step LIKE '%bear%' OR agent_step LIKE '%judge%'
        ORDER BY created_at DESC LIMIT 30
        """
    )
    rows = db.fetchall()
    print(f"Found {len(rows)} matching rows.")
    for row in rows:
        print(f"ID: {row[0]} | Cycle: {row[1]} | Ticker: {row[2]} | Step: {row[3]} | Model: {row[4]} | EP: {row[5]}")
        print(f"Tokens: {row[6]} | Latency: {row[7]}ms")
        print(f"Response: {row[8]}")
        print("-" * 50)
