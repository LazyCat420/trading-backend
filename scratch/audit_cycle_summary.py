import os
import sys
import json

# Adjust path to import app modules
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from app.db.connection import get_db

def run():
    cycle_id = "cycle-1779695975"
    print(f"=== DETAILED BUY DECISIONS FOR CYCLE {cycle_id} ===")
    with get_db() as db:
        results = db.execute("SELECT ticker, confidence, result_json FROM analysis_results WHERE cycle_id = %s;", [cycle_id]).fetchall()
        for res in results:
            ticker, conf, result_json_str = res
            try:
                result_json = json.loads(result_json_str) if isinstance(result_json_str, str) else result_json_str
                action = result_json.get("action", "HOLD")
            except Exception:
                action = "?"
                result_json = {}
                
            if action == "BUY":
                print(f"\n--- TICKER: {ticker} | Conf: {conf} ---")
                skipped = result_json.get("trade_skipped")
                if skipped:
                    print(f"  Trade Skipped Status: {skipped}")
                else:
                    print("  No trade_skipped field in result_json.")

if __name__ == "__main__":
    run()
