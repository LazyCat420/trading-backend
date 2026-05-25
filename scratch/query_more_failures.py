import sys
import os
from dotenv import load_dotenv
load_dotenv()

import psycopg

db_url = os.getenv("DATABASE_URL")

with psycopg.connect(db_url) as conn:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT l.created_at, l.ticker, l.agent_step, l.model, l.endpoint_name, l.tokens_used, l.execution_ms, SUBSTRING(l.raw_response FROM 1 FOR 100)
            FROM llm_audit_logs l
            WHERE l.raw_response = '' OR l.raw_response IS NULL
            ORDER BY l.created_at DESC
            LIMIT 15
        """)
        rows = cur.fetchall()
        print(f"Found {len(rows)} empty raw_response rows:")
        for r in rows:
            print(f"Time: {r[0]} | Ticker: {r[1]} | Step: {r[2]} | Model: {r[3]} | EP: {r[4]} | Tokens: {r[5]} | Ms: {r[6]}")
