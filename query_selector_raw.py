import psycopg

DATABASE_URL = "postgresql://trader:trading_bot_pass@10.0.0.16:5433/trading_bot"

def query_selector_raw():
    try:
        conn = psycopg.connect(DATABASE_URL)
        cur = conn.cursor()
        
        cur.execute("""
            SELECT created_at, ticker, agent_step, endpoint_name, raw_response, tokens_used, execution_ms 
            FROM llm_audit_logs 
            WHERE cycle_id = 'cycle-1780050590' AND agent_step LIKE '%selector%' 
            ORDER BY created_at ASC;
        """)
        
        rows = cur.fetchall()
        print(f"\nFound {len(rows)} selector responses for cycle-1780050590:")
        print("-" * 120)
        for row in rows:
            created, ticker, agent_step, ep_name, raw, tokens, ms = row
            print(f"[{created}] Ticker: {ticker} | Agent: {agent_step} | Endpoint: {ep_name}")
            print(f"Tokens: {tokens} | MS: {ms}")
            print(f"Raw Response: {repr(raw)}")
            print("-" * 120)
            
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    query_selector_raw()
