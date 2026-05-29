import psycopg
import os

DATABASE_URL = "postgresql://trader:trading_bot_pass@10.0.0.16:5433/trading_bot"

def query_latest_errors():
    try:
        conn = psycopg.connect(DATABASE_URL)
        cur = conn.cursor()
        
        # Get latest cycle_id
        cur.execute("SELECT cycle_id, created_at FROM execution_errors ORDER BY created_at DESC LIMIT 1;")
        row = cur.fetchone()
        if not row:
            print("No execution errors found in the table.")
            cur.close()
            conn.close()
            return
        
        latest_cycle_id = row[0]
        print(f"Latest cycle with errors: {latest_cycle_id}")
        
        # Query errors for the latest cycle
        cur.execute("""
            SELECT phase, ticker, error_type, error_message, created_at 
            FROM execution_errors 
            WHERE cycle_id = %s 
            ORDER BY created_at DESC;
        """, (latest_cycle_id,))
        
        rows = cur.fetchall()
        print(f"\nFound {len(rows)} errors for cycle {latest_cycle_id}:")
        print("-" * 100)
        for i, row in enumerate(rows, 1):
            phase, ticker, err_type, err_msg, created = row
            print(f"{i}. [{created}] Phase: {phase} | Ticker: {ticker}")
            print(f"   Type: {err_type}")
            print(f"   Message: {err_msg}")
            print("-" * 100)
            
        # Let's also query the last 5 cycles with their error counts
        print("\nRecent cycles and their error counts:")
        cur.execute("""
            SELECT cycle_id, COUNT(*), MAX(created_at) 
            FROM execution_errors 
            GROUP BY cycle_id 
            ORDER BY MAX(created_at) DESC 
            LIMIT 5;
        """)
        for cycle_id, count, max_created in cur.fetchall():
            print(f"Cycle: {cycle_id} | Errors: {count} | Last error: {max_created}")
            
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error querying database: {e}")

if __name__ == "__main__":
    query_latest_errors()
