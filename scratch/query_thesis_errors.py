import sys
import os
from dotenv import load_dotenv
load_dotenv()

import psycopg

db_url = os.getenv("DATABASE_URL")

with psycopg.connect(db_url) as conn:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT l.created_at, l.ticker, l.model, l.raw_response, s_prompt.content
            FROM llm_audit_logs l
            LEFT JOIN context_blobs s_prompt ON l.system_prompt_hash = s_prompt.context_hash
            WHERE l.agent_step = 'thesis_agent'
            ORDER BY l.created_at DESC
            LIMIT 5
        """)
        rows = cur.fetchall()
        for i, row in enumerate(rows):
            print(f"--- ROW {i} ---")
            print("CREATED AT:", row[0])
            print("TICKER:", row[1])
            print("MODEL:", row[2])
            print("RAW RESPONSE (repr):", repr(row[3]))
            print("RAW RESPONSE:", row[3])
            print("SYSTEM PROMPT PREVIEW:", row[4][:200] if row[4] else "None")
