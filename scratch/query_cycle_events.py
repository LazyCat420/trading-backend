import sys
sys.path.append("/home/lazycat/github/rods-project/sun/trading-client")
from app.db.connection import get_db

with get_db() as db:
    # Get latest cycle
    latest_cycle = db.execute("SELECT cycle_id, status, created_at FROM cycle_runs ORDER BY created_at DESC LIMIT 5").fetchall()
    print("=== LATEST CYCLES ===")
    for c in latest_cycle:
        print(c)
        
    if latest_cycle:
        last_id = latest_cycle[0][0]
        print(f"\n=== EVENTS FOR LATEST CYCLE: {last_id} ===")
        events = db.execute(
            "SELECT stage, status, message, created_at FROM pipeline_events WHERE cycle_id = %s ORDER BY created_at ASC",
            [last_id]
        ).fetchall()
        for e in events:
            print(e)
            
        print(f"\n=== ANALYSIS RESULTS FOR LATEST CYCLE: {last_id} ===")
        results = db.execute(
            "SELECT ticker, action, confidence, trade_executed, trade_skipped FROM analysis_results WHERE cycle_id = %s",
            [last_id]
        ).fetchall()
        for r in results:
            print(r)
