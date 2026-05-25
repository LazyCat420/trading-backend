import os
import sys
from datetime import datetime

# Adjust path to import app modules
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from app.db.connection import get_db

def run():
    print("=== AUDIT DATABASE START ===")
    with get_db() as db:
        # 1. Fetch recent positions
        print("\n--- POSITIONS ---")
        positions = db.execute("SELECT bot_id, ticker, qty, avg_entry_price, opened_at FROM positions ORDER BY opened_at DESC;").fetchall()
        if not positions:
            print("No active positions.")
        for p in positions:
            print(f"Bot: {p[0]} | Ticker: {p[1]} | Qty: {p[2]} | AvgPrice: {p[3]} | OpenedAt: {p[4]}")
            
        # 2. Fetch recent orders
        print("\n--- RECENT ORDERS (Last 10) ---")
        orders = db.execute("SELECT bot_id, ticker, side, qty, price, signal, created_at, filled_at, realized_pnl FROM orders ORDER BY created_at DESC LIMIT 10;").fetchall()
        for o in orders:
            print(f"Bot: {o[0]} | Ticker: {o[1]} | Side: {o[2]} | Qty: {o[3]} | Price: {o[4]} | Signal: {o[5]} | CreatedAt: {o[6]} | FilledAt: {o[7]} | PnL: {o[8]}")

        # 3. Fetch execution errors
        print("\n--- RECENT EXECUTION ERRORS (Last 10) ---")
        errors = db.execute("SELECT cycle_id, phase, ticker, error_type, error_message, created_at FROM execution_errors ORDER BY created_at DESC LIMIT 10;").fetchall()
        for e in errors:
            print(f"Cycle: {e[0]} | Phase: {e[1]} | Ticker: {e[2]} | Type: {e[3]} | Msg: {e[4]} | CreatedAt: {e[5]}")

        # 4. Fetch recent audit logs for CRWV or other skipped tickers
        print("\n--- RECENT AUDIT LOGS FOR CRWV ---")
        audit_crwv = db.execute("SELECT cycle_id, timestamp, audit_type, event_type, phase, ticker, severity, message, data FROM cycle_audit_log WHERE ticker = 'CRWV' OR message LIKE '%CRWV%' ORDER BY timestamp DESC LIMIT 20;").fetchall()
        for a in audit_crwv:
            print(f"Cycle: {a[0]} | Time: {a[1]} | AuditType: {a[2]} | Event: {a[3]} | Phase: {a[4]} | Ticker: {a[5]} | Severity: {a[6]} | Msg: {a[7]} | Data: {a[8]}")

        # 5. Fetch all cycle audit logs from last 1 hour
        print("\n--- RECENT GENERAL AUDIT LOGS (Last 30) ---")
        audit_all = db.execute("SELECT cycle_id, timestamp, audit_type, event_type, phase, ticker, severity, message FROM cycle_audit_log ORDER BY timestamp DESC LIMIT 30;").fetchall()
        for a in audit_all:
            print(f"Cycle: {a[0]} | Time: {a[1]} | AuditType: {a[2]} | Event: {a[3]} | Phase: {a[4]} | Ticker: {a[5]} | Severity: {a[6]} | Msg: {a[7]}")

if __name__ == "__main__":
    run()
