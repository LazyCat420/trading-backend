import psycopg

DATABASE_URL = "postgresql://trader:trading_bot_pass@10.0.0.16:5433/trading_bot"

def query_llm_calls():
    try:
        conn = psycopg.connect(DATABASE_URL)
        cur = conn.cursor()
        
        # Get latest cycle_id
        cur.execute("SELECT cycle_id, MAX(created_at) FROM llm_audit_logs GROUP BY cycle_id ORDER BY MAX(created_at) DESC LIMIT 1;")
        row = cur.fetchone()
        if not row:
            print("No LLM audit logs found.")
            cur.close()
            conn.close()
            return
        
        latest_cycle_id = row[0]
        print(f"Latest cycle in LLM audit logs: {latest_cycle_id}")
        
        # Query LLM audit logs for this cycle
        cur.execute("""
            SELECT created_at, ticker, agent_step, endpoint_name, prompt_tokens, completion_tokens, execution_ms, model 
            FROM llm_audit_logs 
            WHERE cycle_id = %s 
            ORDER BY created_at ASC;
        """, (latest_cycle_id,))
        
        rows = cur.fetchall()
        print(f"\nFound {len(rows)} LLM calls for cycle {latest_cycle_id}:")
        print("-" * 120)
        for row in rows:
            created, ticker, agent_step, ep_name, p_tok, c_tok, elapsed, model = row
            print(f"[{created}] Ticker: {ticker:5s} | Agent: {agent_step:25s}")
            print(f"   Endpoint: {ep_name} | Model: {model}")
            print(f"   Tokens: {p_tok} prompt, {c_tok} completion | Elapsed: {elapsed} ms")
            print("-" * 120)
            
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    query_llm_calls()
