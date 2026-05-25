import sys
import os
from dotenv import load_dotenv
load_dotenv()

import psycopg

db_url = os.getenv("DATABASE_URL")

with psycopg.connect(db_url) as conn:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT created_at, cycle_id, phase, ticker, error_type, error_message
            FROM execution_errors
            ORDER BY created_at DESC
            LIMIT 10
        """)
        rows = cur.fetchall()
        print(f"Found {len(rows)} execution errors:")
        for r in rows:
            print(f"Time: {r[0]} | Cycle: {r[1]} | Phase: {r[2]} | Ticker: {r[3]} | Type: {r[4]}")
            print(f"Msg: {r[5]}")
            print("-" * 50)
