import psycopg
import json

def main():
    conn = psycopg.connect("postgresql://trader:trading_bot_pass@10.0.0.16:5433/trading_bot")
    cur = conn.cursor()
    try:
        # Get all tables
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        tables = [r[0] for r in cur.fetchall()]
        print("Tables in database:")
        for t in tables:
            print(f"- {t}")
            # Print columns for important tables
            if t in ["pipeline_state", "pipeline_events", "vllm_calls", "decision_evaluations", "debate_history", "news_articles"]:
                cur.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{t}'")
                cols = cur.fetchall()
                print("  Columns:")
                for c in cols:
                    print(f"    {c[0]}: {c[1]}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    main()
