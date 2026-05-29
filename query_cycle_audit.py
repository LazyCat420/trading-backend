import psycopg

DATABASE_URL = "postgresql://trader:trading_bot_pass@10.0.0.16:5433/trading_bot"

def query_cycle_audit():
    try:
        conn = psycopg.connect(DATABASE_URL)
        cur = conn.cursor()
        
        # Get latest cycle_id
        cur.execute("SELECT cycle_id, MAX(timestamp) FROM cycle_audit_log GROUP BY cycle_id ORDER BY MAX(timestamp) DESC LIMIT 1;")
        row = cur.fetchone()
        if not row:
            print("No audit log entries found.")
            cur.close()
            conn.close()
            return
        
        latest_cycle_id = row[0]
        print(f"Latest cycle in audit log: {latest_cycle_id}")
        
        # Query audit log events for this cycle
        cur.execute("""
            SELECT timestamp, event_type, phase, ticker, severity, message 
            FROM cycle_audit_log 
            WHERE cycle_id = %s 
            ORDER BY timestamp ASC;
        """, (latest_cycle_id,))
        
        rows = cur.fetchall()
        print(f"\nFound {len(rows)} audit logs for cycle {latest_cycle_id}:")
        print("-" * 100)
        for row in rows:
            ts, ev_type, phase, ticker, severity, message = row
            print(f"[{ts}] {severity.upper()} | Ticker: {ticker} | Phase: {phase} | Event: {ev_type}")
            print(f"      {message}")
            print("-" * 100)
            
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    query_cycle_audit()
