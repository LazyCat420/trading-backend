import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.connection import get_db

print("=== CHECKING PRODUCTION DATABASE (trading_bot) ===")
with get_db() as db:
    latest_cycle = db.execute("SELECT cycle_id FROM cycle_audit_log ORDER BY timestamp DESC LIMIT 1").fetchone()
    cycle_id = latest_cycle[0] if latest_cycle else None
    print(f"Latest cycle_id found: {cycle_id}")
    
    print("Latest 3 cycle summaries:")
    print(db.execute("SELECT cycle_id, buy_count, sell_count, hold_count FROM autoresearch_cycle_summaries ORDER BY cycle_id DESC LIMIT 3").fetchall())
    print("\nLatest 3 autoresearch reports:")
    print(db.execute("SELECT cycle_id, status, overall_score FROM autoresearch_reports ORDER BY cycle_id DESC LIMIT 3").fetchall())
    if cycle_id:
        print(f"\nLatest 5 cycle audit log events for {cycle_id}:")
        print(db.execute("SELECT timestamp, phase, severity, message FROM cycle_audit_log WHERE cycle_id=%s ORDER BY timestamp ASC LIMIT 5", [cycle_id]).fetchall())

print("\n=== CHECKING TEST DATABASE (trading_bot_test) ===")
os.environ['DATABASE_URL'] = 'postgresql://trader:trading_bot_pass@10.0.0.16:5433/trading_bot_test'
# Clear the pool reference to force new connection pool creation with new DATABASE_URL
import app.db.connection
app.db.connection._pool = None

with get_db() as db:
    latest_cycle = db.execute("SELECT cycle_id FROM cycle_audit_log ORDER BY timestamp DESC LIMIT 1").fetchone()
    cycle_id = latest_cycle[0] if latest_cycle else None
    print(f"Latest cycle_id found: {cycle_id}")
    
    print("Latest 3 cycle summaries:")
    print(db.execute("SELECT cycle_id, buy_count, sell_count, hold_count FROM autoresearch_cycle_summaries ORDER BY cycle_id DESC LIMIT 3").fetchall())
    print("\nLatest 3 autoresearch reports:")
    print(db.execute("SELECT cycle_id, status, overall_score FROM autoresearch_reports ORDER BY cycle_id DESC LIMIT 3").fetchall())
    if cycle_id:
        print(f"\nLatest 5 cycle audit log events for {cycle_id}:")
        print(db.execute("SELECT timestamp, phase, severity, message FROM cycle_audit_log WHERE cycle_id=%s ORDER BY timestamp ASC LIMIT 5", [cycle_id]).fetchall())
