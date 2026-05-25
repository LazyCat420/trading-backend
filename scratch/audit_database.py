import os
import sys

# Add parent directory to path to allow importing app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.connection import get_db

def main():
    import sys
    with get_db() as db:
        # Get cycle ID from arguments or find latest standard cycle
        cycle_id = None
        if len(sys.argv) > 1:
            cycle_id = sys.argv[1]
        else:
            latest_cycle = db.execute(
                "SELECT cycle_id, MAX(timestamp) FROM cycle_audit_log WHERE cycle_id LIKE 'cycle-%' GROUP BY cycle_id ORDER BY MAX(timestamp) DESC LIMIT 1"
            ).fetchone()
            if latest_cycle:
                cycle_id = latest_cycle[0]
        
        if not cycle_id:
            print("No matching cycles found in cycle_audit_log.")
            return
            
        print(f"=== DATABASE AUDIT LOG FOR CYCLE: {cycle_id} ===")
        
        rows = db.execute(
            "SELECT timestamp, phase, audit_type, ticker, severity, message FROM cycle_audit_log WHERE cycle_id = %s ORDER BY timestamp ASC",
            [cycle_id]
        ).fetchall()
        
        print(f"Total events logged: {len(rows)}\n")
        
        # Group by phase
        phases = {}
        for r in rows:
            timestamp, phase, audit_type, ticker, severity, message = r
            if phase not in phases:
                phases[phase] = []
            phases[phase].append(r)
            
        for phase, events in phases.items():
            print(f"Phase: '{phase}' ({len(events)} events)")
            # Print first 2 and last 2 events for this phase
            if len(events) <= 4:
                for e in events:
                    print(f"  [{e[0]}] [{e[2]}] {e[3]} ({e[4]}): {e[5]}")
            else:
                for e in events[:2]:
                    print(f"  [{e[0]}] [{e[2]}] {e[3]} ({e[4]}): {e[5]}")
                print("  ...")
                for e in events[-2:]:
                    print(f"  [{e[0]}] [{e[2]}] {e[3]} ({e[4]}): {e[5]}")
            print()

if __name__ == "__main__":
    main()
