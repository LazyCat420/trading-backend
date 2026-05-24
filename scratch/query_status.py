import os
import sys

# Insert project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.connection import get_db

def run():
    with get_db() as db:
        db.execute("SELECT cycle_id FROM pipeline_events ORDER BY timestamp DESC LIMIT 1")
        row = db.fetchone()
        cycle_id = row[0] if row else None
        if not cycle_id:
            print("No cycle found in database.")
            return
        print(f"=== EVENTS FOR {cycle_id} ===")
        db.execute("""
            SELECT timestamp, phase, step, status, elapsed_ms, detail
            FROM pipeline_events
            WHERE cycle_id = %s
            ORDER BY timestamp ASC
        """, [cycle_id])
        rows = db.fetchall()
        print(f"Total events found: {len(rows)}")
        
        # Print first 20 events
        print("\n--- FIRST 20 EVENTS ---")
        for r in rows[:20]:
            print(f"[{r[0]}] {r[1]} | {r[2]} | Status: {r[3]} | Elapsed: {r[4]}ms | {r[5][:80]}")
            
        # Print last 20 events
        print("\n--- LAST 20 EVENTS ---")
        for r in rows[-20:]:
            print(f"[{r[0]}] {r[1]} | {r[2]} | Status: {r[3]} | Elapsed: {r[4]}ms | {r[5][:80]}")
            
        # Let's count how many tickers were processed
        tickers_started = set()
        tickers_done = set()
        for r in rows:
            detail = r[5]
            if "for ticker" in detail:
                # e.g. "Summarizing 13 items for ticker IR"
                parts = detail.split("ticker ")
                if len(parts) > 1:
                    ticker = parts[1].split()[0].replace("...", "").strip()
                    if r[3] == "running":
                        tickers_started.add(ticker)
            if "Step: summarize" in r[2] and r[3] == "ok":
                # Find ticker name from step, wait, step is "summarize" or something?
                # Let's find from detail: e.g. "IR: Summarization complete"
                detail_lower = detail.lower()
                # Let's check detail format
                parts = detail.split(":")
                if len(parts) > 0:
                    ticker = parts[0].strip()
                    if len(ticker) <= 5 and ticker.isupper():
                        tickers_done.add(ticker)
                        
        print(f"\nTickers started summarization: {len(tickers_started)} -> {list(tickers_started)[:10]}")
        print(f"Tickers completed summarization: {len(tickers_done)} -> {list(tickers_done)[:10]}")

if __name__ == "__main__":
    run()
