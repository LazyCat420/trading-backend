import sys
import os
import time
sys.path.append('/home/lazycat/github/projects/sun/trading-service')
from app.db.connection import get_db

max_retries = 5
for attempt in range(max_retries):
    try:
        with get_db() as db:
            db.execute("ALTER TABLE cycle_schedules ADD COLUMN IF NOT EXISTS discovered_tickers INTEGER DEFAULT 0;")
        print("Column added successfully.")
        break
    except Exception as e:
        print(f"Attempt {attempt + 1} failed: {e}")
        time.sleep(1)
else:
    print("Failed to add column after multiple attempts.")
