import sys
import os
from dotenv import load_dotenv
load_dotenv()

import psycopg

db_url = os.getenv("DATABASE_URL")

with psycopg.connect(db_url) as conn:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT l.created_at, l.ticker, l.model, l.raw_response, s_prompt.content, u_prompt.content
            FROM llm_audit_logs l
            LEFT JOIN context_blobs s_prompt ON l.system_prompt_hash = s_prompt.context_hash
            LEFT JOIN context_blobs u_prompt ON l.context_hash = u_prompt.context_hash
            WHERE l.agent_step = 'thesis_agent' AND l.ticker = 'LLY'
            ORDER BY l.created_at DESC
            LIMIT 1
        """)
        row = cur.fetchone()
        if row:
            print("USER PROMPT LENGTH:", len(row[5]))
            print("USER PROMPT:")
            print(row[5])
