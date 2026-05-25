import os
import sys
import json
from datetime import datetime, timezone

# Add parent directory to path to allow importing app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app.db.connection import get_db
except ImportError:
    print("Error: Could not import app.db.connection. Make sure to run from project root.")
    sys.exit(1)

LAST_AUDIT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_audit.json")

def get_latest_cycle_id():
    query = """
        SELECT cycle_id, timestamp
        FROM cycle_audit_log
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

def load_last_audit():
    if os.path.exists(LAST_AUDIT_FILE):
        try:
            with open(LAST_AUDIT_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading last audit file: {e}")
    return {"last_cycle_id": None, "last_audit_time": None}

def save_last_audit(cycle_id):
    data = {
        "last_cycle_id": cycle_id,
        "last_audit_time": datetime.now(timezone.utc).isoformat()
    }
    try:
        with open(LAST_AUDIT_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error writing last audit file: {e}")

def main():
    print(f"Checking for audit trigger at {datetime.now(timezone.utc).isoformat()}...")
    
    latest_cycle_id, latest_timestamp = get_latest_cycle_id()
    if not latest_cycle_id:
        print("No trading cycles found in the database. Nothing to audit.")
        sys.exit(0)
        
    last_audit = load_last_audit()
    
    # Check if there is a new cycle
    is_new_cycle = latest_cycle_id != last_audit.get("last_cycle_id")
    
    # Check fallback time elapsed (90 minutes = 5400 seconds)
    time_elapsed = False
    last_audit_time_str = last_audit.get("last_audit_time")
    if last_audit_time_str:
        try:
            last_audit_time = datetime.fromisoformat(last_audit_time_str)
            seconds_since = (datetime.now(timezone.utc) - last_audit_time).total_seconds()
            if seconds_since >= 5400:
                time_elapsed = True
                print(f"Time elapsed since last audit: {seconds_since / 60:.1f} minutes (fallback trigger)")
        except Exception as e:
            print(f"Error parsing last audit time: {e}")
            time_elapsed = True
    else:
        # No record of previous audit, trigger fallback
        time_elapsed = True
        print("No previous audit timestamp found (triggering initial audit)")
        
    if is_new_cycle or time_elapsed:
        print(f"Triggering audit! New cycle={is_new_cycle}, Fallback elapsed={time_elapsed}")
        # Run audit_trading_cycle main logic or call it via import / subprocess
        # Since audit_trading_cycle has a main() function, let's import it and run it
        import scratch.audit_trading_cycle as auditor
        
        # Override args or set system argument for cycle_id if we want
        # By default auditor.main() defaults to latest if no arg, which is exactly what we want.
        sys.argv = [sys.argv[0]]
        if latest_cycle_id:
            sys.argv.append(f"--cycle-id={latest_cycle_id}")
            
        try:
            auditor.main()
            save_last_audit(latest_cycle_id)
            print("Audit completed successfully.")
            sys.exit(0)
        except Exception as e:
            print(f"Error running auditor: {e}")
            sys.exit(1)
    else:
        print(f"Audit skipped. Latest cycle {latest_cycle_id} already audited, and 90-minute fallback has not expired.")
        sys.exit(0)

if __name__ == "__main__":
    main()
