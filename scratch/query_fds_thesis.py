import os
import sys
from dotenv import load_dotenv
load_dotenv()

import psycopg

db_url = os.getenv("DATABASE_URL")

with psycopg.connect(db_url) as conn:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT l.created_at, l.ticker, l.model, l.raw_response, s_prompt.content, u_prompt.content, l.cycle_id
            FROM llm_audit_logs l
            LEFT JOIN context_blobs s_prompt ON l.system_prompt_hash = s_prompt.context_hash
            LEFT JOIN context_blobs u_prompt ON l.context_hash = u_prompt.context_hash
            WHERE l.agent_step = 'thesis_agent' AND l.ticker = 'FDS'
            ORDER BY l.created_at DESC
            LIMIT 1
        """)
        row = cur.fetchone()
        if row:
            created_at, ticker, model, raw_response, sys_prompt, user_prompt, cycle_id = row
            print(f"=== AUDITING THESIS AGENT FOR {ticker} (Cycle: {cycle_id}) ===")
            print(f"Created At: {created_at}")
            print(f"Model: {model}")
            print("\n--- SYSTEM PROMPT ---")
            print(sys_prompt[:1000])
            print("\n--- USER PROMPT (first 2000 chars) ---")
            print(user_prompt[:2000])
            print("\n--- USER PROMPT (last 2000 chars) ---")
            print(user_prompt[-2000:])
            print("\n--- RAW RESPONSE ---")
            print(raw_response)
        else:
            print("No thesis agent logs found for FDS.")
