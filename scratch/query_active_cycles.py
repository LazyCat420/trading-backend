import os
import sys

# Insert project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.connection import get_db

def run():
    with get_db() as db:
        print("=== RECENT CYCLES ===")
        db.execute("""
            SELECT DISTINCT cycle_id, MIN(timestamp) as started_at, MAX(timestamp) as last_event_at, COUNT(*) as event_count
            FROM pipeline_events
            GROUP BY cycle_id
            ORDER BY last_event_at DESC
            LIMIT 10
        """)
        rows = db.fetchall()
        for r in rows:
            print(f"Cycle: {r[0]} | Started: {r[1]} | Last Event: {r[2]} | Total Events: {r[3]}")
            
        print("\n=== RECENT PENDING / IN-PROGRESS COMMANDS ===")
        try:
            db.execute("""
                SELECT id, command_type, status, created_at, payload
                FROM system_commands
                ORDER BY created_at DESC
                LIMIT 5
            """)
            commands = db.fetchall()
            for cmd in commands:
                print(f"Cmd ID: {cmd[0]} | Type: {cmd[1]} | Status: {cmd[2]} | Created: {cmd[3]} | Payload: {cmd[4]}")
        except Exception as e:
            print(f"Error querying system_commands: {e}")

if __name__ == "__main__":
    run()
