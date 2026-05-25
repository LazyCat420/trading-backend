import os
import sys
import json
import random
import subprocess
from datetime import datetime, timezone

# Add parent directory to path to allow importing app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app.db.connection import get_db
except ImportError:
    print("Error: Could not import app.db.connection. Make sure to run from project root.")
    sys.exit(1)

LAST_AUDIT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_audit.json")
HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "auto_audit_history.json")

def get_latest_cycle_info():
    """Fetches the latest cycle_id and its timestamp from the audit log."""
    query = """
        SELECT cycle_id, timestamp
        FROM cycle_audit_log
        WHERE cycle_id LIKE 'cycle-%%'
        ORDER BY timestamp DESC
        LIMIT 1
    """
    try:
        with get_db() as db:
            row = db.execute(query).fetchone()
            if row:
                return row[0], row[1]
    except Exception as e:
        print(f"Error fetching latest cycle_id: {e}")
    return None, None

def get_sp500_tickers():
    """Loads S&P 500 tickers from constituents list."""
    try:
        from app.data.sp500_constituents import SP500_TICKERS
        return [entry["ticker"] for entry in SP500_TICKERS]
    except Exception as e:
        print(f"Error loading S&P 500 constituents: {e}")
        # Return a safe fallback list if import fails
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX", "AMD", "INTC"]

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading history file: {e}")
    return {"consecutive_runs": []}

def save_history(history):
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"Error writing history file: {e}")

def run_test_cycle():
    """Runs a test cycle with 5 randomized S&P 500 tickers."""
    tickers = get_sp500_tickers()
    selected = random.sample(tickers, min(len(tickers), 5))
    tickers_str = ",".join(selected)
    print(f"No trading cycle completed in the last 2 hours. Initiating 5-stock randomized S&P 500 test run: {tickers_str}")
    
    cmd = [sys.executable, "cycle_main.py", "--once", "--tickers", tickers_str]
    try:
        # Run with a 15-minute timeout to prevent hanging
        res = subprocess.run(cmd, timeout=900, capture_output=True, text=True)
        print("--- Test Cycle Output ---")
        print(res.stdout)
        if res.stderr:
            print("--- Test Cycle Errors ---", file=sys.stderr)
            print(res.stderr, file=sys.stderr)
        
        if res.returncode != 0:
            print(f"Warning: test cycle exited with non-zero code {res.returncode}")
    except subprocess.TimeoutExpired:
        print("Error: Test cycle timed out after 15 minutes.")
    except Exception as e:
        print(f"Error running test cycle: {e}")

def check_for_bugs(cycle_id):
    """Checks the database for errors and warnings in the given cycle."""
    error_count = 0
    warning_count = 0
    
    # 1. Check execution_errors
    try:
        with get_db() as db:
            row = db.execute("SELECT COUNT(*) FROM execution_errors WHERE cycle_id = %s", [cycle_id]).fetchone()
            if row:
                error_count = row[0]
    except Exception as e:
        print(f"Error checking execution_errors: {e}")
        
    # 2. Check cycle_audit_log for warnings/criticals
    try:
        with get_db() as db:
            row = db.execute(
                "SELECT COUNT(*) FROM cycle_audit_log WHERE cycle_id = %s AND severity IN ('warning', 'critical')",
                [cycle_id]
            ).fetchone()
            if row:
                warning_count = row[0]
    except Exception as e:
        print(f"Error checking cycle_audit_log warnings: {e}")
        
    return error_count, warning_count

def main():
    print(f"Starting autonomous audit script run at {datetime.now(timezone.utc).isoformat()}...")
    
    # 1. Determine if a run occurred in the last 2 hours
    latest_cycle_id, latest_timestamp = get_latest_cycle_info()
    run_recently = False
    
    if latest_timestamp:
        # Check if latest_timestamp is within 2 hours
        now = datetime.now(latest_timestamp.tzinfo if latest_timestamp.tzinfo else timezone.utc)
        diff_seconds = (now - latest_timestamp).total_seconds()
        if diff_seconds < 7200: # 2 hours = 7200 seconds
            run_recently = True
            print(f"Found active or recent trading run: {latest_cycle_id} (elapsed: {diff_seconds / 60:.1f} mins ago).")
            
    # 2. Execute test run if no recent run exists
    run_type = "active-2h"
    if not run_recently:
        run_type = "5-stock-test"
        run_test_cycle()
        # Fetch the newly created cycle info
        latest_cycle_id, latest_timestamp = get_latest_cycle_info()
        if not latest_cycle_id:
            print("Error: Could not retrieve a cycle ID after running test cycle.")
            sys.exit(1)
            
    # 3. Perform audit
    print(f"Auditing cycle: {latest_cycle_id}...")
    errors, warnings = check_for_bugs(latest_cycle_id)
    total_bugs = errors + warnings
    print(f"Audit results for {latest_cycle_id}: {errors} execution errors, {warnings} auditor warnings/anomalies (Total: {total_bugs} issues)")
    
    # 4. Check history & bug accumulation limit
    history = load_history()
    consecutive_runs = history.get("consecutive_runs", [])
    
    # Check if we should halt (if we have >= 2 runs and bug counts are increasing)
    should_halt = False
    if len(consecutive_runs) >= 2:
        prev_run = consecutive_runs[-1]
        prev_prev_run = consecutive_runs[-2]
        # If bug count has increased compared to both of the last runs
        if total_bugs > prev_run.get("total_bugs", 0) and prev_run.get("total_bugs", 0) > prev_prev_run.get("total_bugs", 0):
            should_halt = True
            print("ALERT: Bug count has increased continuously over the last 2 runs.")
            
    # Append current run to history
    consecutive_runs.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cycle_id": latest_cycle_id,
        "run_type": run_type,
        "errors_count": errors,
        "warnings_count": warnings,
        "total_bugs": total_bugs
    })
    # Keep only the last 10 runs
    history["consecutive_runs"] = consecutive_runs[-10:]
    save_history(history)
    
    # Run the report generator
    sys.argv = [sys.argv[0], f"--cycle-id={latest_cycle_id}"]
    import scratch.audit_trading_cycle as auditor
    try:
        auditor.main()
    except Exception as e:
        print(f"Error generating audit report: {e}")
        
    # Write report status to terminal output for agent parser
    if should_halt:
        print("STATUS: HALT_BUG_ACCUMULATION")
    elif total_bugs > 0:
        print("STATUS: BUGS_DETECTED")
    else:
        print("STATUS: CLEAN")

if __name__ == "__main__":
    main()
