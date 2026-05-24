import os
import sys

# Ensure local dir is in path
local_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if local_dir not in sys.path:
    sys.path.insert(0, local_dir)

from app.db.connection import get_db

def run():
    with get_db() as db:
        print("=== PORTFOLIO STATE ===")
        # Get active bot_id
        try:
            from app.services.bot_manager import get_active_bot_id
            bot_id = get_active_bot_id()
            print(f"Active Bot ID: {bot_id}")
        except Exception as e:
            bot_id = "default"
            print(f"Could not get active bot_id (using 'default'): {e}")
            
        print("\n=== POSITIONS ===")
        pos_rows = db.execute(
            "SELECT ticker, qty, avg_entry_price, bot_id FROM positions"
        ).fetchall()
        if not pos_rows:
            print("No positions found.")
        for row in pos_rows:
            print(f"Ticker: {row[0]} | Qty: {row[1]} | Avg Price: {row[2]} | Bot: {row[3]}")

        print("\n=== LATEST ANALYSIS RESULTS ===")
        res_rows = db.execute(
            """
            SELECT cycle_id, ticker, confidence, result_json, created_at 
            FROM analysis_results 
            ORDER BY created_at DESC LIMIT 30
            """
        ).fetchall()
        if not res_rows:
            print("No analysis results found.")
        for row in res_rows:
            import json
            try:
                res_dict = json.loads(row[3])
                action = res_dict.get("action")
                trade_executed = res_dict.get("trade_executed")
                trade_skipped = res_dict.get("trade_skipped")
                estimate = res_dict.get("estimate")
                
                status_str = "NOT YET EXECUTED"
                if trade_executed:
                    status_str = f"EXECUTED: {trade_executed}"
                elif trade_skipped:
                    status_str = f"SKIPPED: {trade_skipped}"
                elif estimate:
                    status_str = f"ESTIMATED: {estimate}"
                
                print(f"Time: {row[4]} | Cycle: {row[0]} | Ticker: {row[1]} | Action: {action} | Conf: {row[2]}% | Status: {status_str}")
            except Exception as e:
                print(f"Error parsing row for {row[1]}: {e}")

if __name__ == "__main__":
    run()
