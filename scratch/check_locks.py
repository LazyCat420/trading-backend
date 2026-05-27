import sys
import os
import psycopg
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings

def check_locks():
    db_url = settings.DATABASE_URL
    if os.getenv("TRADING_BOT_TEST_DB") == "1":
        db_url = "postgresql://trader:trading_bot_pass@10.0.0.16:5433/trading_bot_test"
        
    print(f"Connecting to database url directly...")
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as db:
            print("=== ACTIVE QUERIES ===")
            db.execute("""
                SELECT pid, state, query, age(clock_timestamp(), query_start) as duration
                FROM pg_stat_activity
                WHERE state != 'idle' AND pid != pg_backend_pid()
                ORDER BY duration DESC
            """)
            for r in db.fetchall():
                print(f"PID: {r[0]} | State: {r[1]} | Dur: {r[3]}\nQuery: {str(r[2])[:150]}\n")
                
            print("=== BLOCKED LOCKS ===")
            db.execute("""
                SELECT
                    blocked_locks.pid     AS blocked_pid,
                    blocked_activity.query    AS blocked_statement,
                    blocking_locks.pid    AS blocking_pid,
                    blocking_activity.query   AS blocking_statement
                FROM  pg_catalog.pg_locks         blocked_locks
                JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
                JOIN pg_catalog.pg_locks         blocking_locks 
                    ON blocking_locks.locktype = blocked_locks.locktype
                    AND blocking_locks.database IS NOT DISTINCT FROM blocked_locks.database
                    AND blocking_locks.relation IS NOT DISTINCT FROM blocked_locks.relation
                    AND blocking_locks.page IS NOT DISTINCT FROM blocked_locks.page
                    AND blocking_locks.tuple IS NOT DISTINCT FROM blocked_locks.tuple
                    AND blocking_locks.virtualxid IS NOT DISTINCT FROM blocked_locks.virtualxid
                    AND blocking_locks.transactionid IS NOT DISTINCT FROM blocked_locks.transactionid
                    AND blocking_locks.classid IS NOT DISTINCT FROM blocked_locks.classid
                    AND blocking_locks.objid IS NOT DISTINCT FROM blocked_locks.objid
                    AND blocking_locks.objsubid IS NOT DISTINCT FROM blocked_locks.objsubid
                    AND blocking_locks.pid != blocked_locks.pid
                JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
                WHERE NOT blocked_locks.granted
            """)
            rows = db.fetchall()
            if not rows:
                print("No blocked locks detected.")
            for r in rows:
                print(f"Blocked PID: {r[0]} statement:\n  {str(r[1])[:120]}\n")
                print(f"Blocking PID: {r[2]} statement:\n  {str(r[3])[:120]}\n")
                print("-" * 40)

if __name__ == "__main__":
    check_locks()
