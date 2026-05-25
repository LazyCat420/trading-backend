import sys
import os
from dotenv import load_dotenv
load_dotenv()

import psycopg

db_url = os.getenv("DATABASE_URL")

with psycopg.connect(db_url) as conn:
    with conn.cursor() as cur:
        # Find one row where raw_response is empty
        cur.execute("""
            SELECT id, cycle_id, ticker, agent_step, model, endpoint_name, tokens_used, execution_ms, raw_response, system_prompt_hash, context_hash
            FROM llm_audit_logs
            WHERE raw_response = ''
            ORDER BY created_at DESC
            LIMIT 1
        """)
        row = cur.fetchone()
        if row:
            print("ID:", row[0])
            print("CYCLE_ID:", row[1])
            print("TICKER:", row[2])
            print("STEP:", row[3])
            print("MODEL:", row[4])
            print("EP:", row[5])
            print("TOKENS:", row[6])
            print("MS:", row[7])
            print("RAW RESPONSE:", repr(row[8]))
            
            # Fetch prompts
            cur.execute("SELECT content FROM context_blobs WHERE context_hash = %s", [row[9]])
            sys_prompt = cur.fetchone()
            print("SYSTEM PROMPT:", sys_prompt[0] if sys_prompt else "None")
            
            cur.execute("SELECT content FROM context_blobs WHERE context_hash = %s", [row[10]])
            user_prompt = cur.fetchone()
            print("USER PROMPT LENGTH:", len(user_prompt[0]) if user_prompt else 0)
        else:
            print("No empty raw_response rows found")
