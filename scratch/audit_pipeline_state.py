"""
Audit script: Check orders, fills, positions, and portfolio more broadly
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.connection import get_db

def main():
    with get_db() as db:
        # 1. Check ALL orders (not just last 7 days)
        print("=== ALL ORDERS (most recent 20) ===")
        try:
            orders = db.execute("""
                SELECT id, bot_id, ticker, side, qty, price, status, created_at, cycle_id
                FROM orders
                ORDER BY created_at DESC
                LIMIT 20
            """).fetchall()
            
            if not orders:
                print("  ❌ ZERO ORDERS IN ENTIRE TABLE!")
            for o in orders:
                print(f"  {o[3]:4s} {o[2]:8s} qty={o[4]} price=${o[5]:.2f} status={o[6]} | {o[7]} | cycle={o[8]}")
        except Exception as e:
            print(f"  Error: {e}")
        
        # 2. Check trade fills
        print("\n=== ALL TRADE FILLS (most recent 20) ===")
        try:
            fills = db.execute("""
                SELECT id, bot_id, ticker, side, qty, price, created_at
                FROM trade_fills
                ORDER BY created_at DESC
                LIMIT 20
            """).fetchall()
            
            if not fills:
                print("  ❌ ZERO TRADE FILLS IN ENTIRE TABLE!")
            for f in fills:
                print(f"  {f[3]:4s} {f[2]:8s} qty={f[4]} price=${f[5]:.2f} | {f[6]}")
        except Exception as e:
            print(f"  Error: {e}")
        
        # 3. Current positions
        print("\n=== ALL POSITIONS ===")
        try:
            positions = db.execute("SELECT bot_id, ticker, qty, avg_entry_price, created_at FROM positions").fetchall()
            if not positions:
                print("  No open positions")
            for p in positions:
                print(f"  {p[1]:8s} qty={p[2]} entry=${p[3]:.2f} | bot={p[0]} | {p[4]}")
        except Exception as e:
            print(f"  Error: {e}")
        
        # 4. Bot cash
        print("\n=== BOT PORTFOLIO STATE ===")
        try:
            bots = db.execute("SELECT id, name, cash, created_at FROM bots ORDER BY created_at DESC LIMIT 5").fetchall()
            for bot in bots:
                print(f"  Bot: {bot[0]} ({bot[1]}) cash=${bot[2]:,.2f} | {bot[3]}")
        except Exception as e:
            print(f"  Error: {e}")
        
        # 5. Debate history
        print("\n=== RECENT DEBATE HISTORY (last 14 days) ===")
        try:
            debates = db.execute("""
                SELECT ticker, cycle_id, thesis_action, thesis_confidence,
                       counter_action, counter_confidence, winner, final_action, final_confidence,
                       persona_name, key_risk, created_at
                FROM debate_history
                WHERE created_at > NOW() - INTERVAL '14 days'
                ORDER BY created_at DESC
                LIMIT 20
            """).fetchall()
            
            if not debates:
                print("  No debates found in last 14 days")
            for d in debates:
                ticker, cycle_id, thesis_act, thesis_conf, counter_act, counter_conf, winner, final_act, final_conf, persona, risk, created_at = d
                print(f"  {created_at} {ticker:8s} | Thesis: {thesis_act}@{thesis_conf}% vs Counter: {counter_act}@{counter_conf}% → Winner: {winner} → Final: {final_act}@{final_conf}%")
                print(f"           Persona: {persona} | Risk: {(risk or '')[:80]}")
        except Exception as e:
            print(f"  Error: {e}")
        
        # 6. Check the v2 cognition pipeline - any results from the last 3 days
        print("\n=== RECENT ANALYSIS: BUY ACTIONS ONLY (last 3 days) ===")
        try:
            recent_buys = db.execute("""
                SELECT cycle_id, ticker, confidence, result_json, created_at
                FROM analysis_results
                WHERE created_at > NOW() - INTERVAL '3 days'
                ORDER BY created_at DESC
            """).fetchall()
            
            buy_list = []
            sell_list = []
            hold_list = []
            for r in recent_buys:
                cycle_id, ticker, confidence, result_json_raw, created_at = r
                try:
                    rj = json.loads(result_json_raw) if isinstance(result_json_raw, str) else result_json_raw
                except:
                    rj = {}
                action = rj.get("action", "?")
                if action == "BUY":
                    buy_list.append((ticker, confidence, cycle_id, created_at))
                elif action == "SELL":
                    sell_list.append((ticker, confidence, cycle_id, created_at))
                elif action == "HOLD":
                    hold_list.append((ticker, confidence, cycle_id, created_at))
            
            print(f"  Last 3 days: BUY={len(buy_list)} SELL={len(sell_list)} HOLD={len(hold_list)}")
            if buy_list:
                print(f"  Recent BUYs:")
                for b in buy_list[:10]:
                    print(f"    {b[0]:8s} @ {b[1]:3d}% | {b[2]} | {b[3]}")
        except Exception as e:
            print(f"  Error: {e}")
        
        # 7. Check the cycle audit log for trading phase
        print("\n=== CYCLE AUDIT LOG: TRADING PHASE (last 14 days) ===")
        try:
            audit_events = db.execute("""
                SELECT timestamp, cycle_id, audit_type, ticker, severity, message
                FROM cycle_audit_log
                WHERE phase = 'trading'
                AND timestamp > NOW() - INTERVAL '14 days'
                ORDER BY timestamp DESC
                LIMIT 30
            """).fetchall()
            
            if not audit_events:
                print("  No trading phase audit events found!")
            for e in audit_events:
                print(f"  [{e[0]}] [{e[2]}] {e[3]} ({e[4]}): {e[5]}")
        except Exception as e:
            print(f"  Error: {e}")
        
        # 8. Check pipeline state to see if trading phase is even running
        print("\n=== PIPELINE STATE (from pipeline_state table) ===")
        try:
            state_rows = db.execute("""
                SELECT key, value, updated_at 
                FROM pipeline_state 
                ORDER BY updated_at DESC 
                LIMIT 10
            """).fetchall()
            for s in state_rows:
                print(f"  {s[0]}: {str(s[1])[:100]} | {s[2]}")
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    main()
