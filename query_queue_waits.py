import psycopg

DATABASE_URL = "postgresql://trader:trading_bot_pass@10.0.0.16:5433/trading_bot"

def query_queue_waits():
    try:
        conn = psycopg.connect(DATABASE_URL)
        cur = conn.cursor()
        
        cur.execute("""
            SELECT created_at, ticker, agent_step, endpoint_name, execution_ms, queue_wait_ms, prompt_tokens, completion_tokens
            FROM llm_audit_logs 
            WHERE cycle_id = 'cycle-1780050590'
            ORDER BY created_at ASC;
        """)
        
        rows = cur.fetchall()
        print(f"\nFound {len(rows)} LLM call details for cycle-1780050590:")
        print("-" * 120)
        for row in rows:
            created, ticker, agent_step, ep_name, exec_ms, wait_ms, p_tok, c_tok = row
            wait_str = f"{wait_ms}ms" if wait_ms else "0ms"
            print(f"[{created}] {ticker:5s} | {agent_step:25s} | {ep_name} | Wait: {wait_str} | Exec: {exec_ms}ms | p={p_tok}, c={c_tok}")
            
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    query_queue_waits()
