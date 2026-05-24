import os
import sys
import uuid
import json

# Insert project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.connection import get_db

def run():
    job_id = f"job_test_{uuid.uuid4().hex[:8]}"
    payload = {
        "trade": True,
        "analyze": True,
        "collect": True,
        "tickers": ["AAPL"],
        "max_tickers": 1,
        "benchmark_group": None,
        "pipeline_version": None
    }
    
    with get_db() as db:
        print(f"Queueing START_CYCLE job: {job_id}")
        db.execute(
            "INSERT INTO system_commands (id, command_type, status, payload) VALUES (%s, %s, %s, %s)",
            (job_id, "START_CYCLE", "pending", json.dumps(payload))
        )
        print("Successfully queued command!")

if __name__ == "__main__":
    run()
