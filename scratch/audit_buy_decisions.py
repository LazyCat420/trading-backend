"""
Audit script: Trace BUY decisions through the entire pipeline.
Shows where BUY recommendations get blocked/vetoed.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.connection import get_db

def main():
    with get_db() as db:
        print("=" * 80)
        print("AUDIT: BUY DECISION PIPELINE TRACE")
        print("=" * 80)
        
        # 1. Find recent cycles
        print("\n--- RECENT CYCLES ---")
        cycles = db.execute("""
            SELECT DISTINCT cycle_id, MIN(created_at) as start, MAX(created_at) as end, COUNT(*) as num_results
            FROM analysis_results
            WHERE created_at > NOW() - INTERVAL '7 days'
            GROUP BY cycle_id
            ORDER BY MIN(created_at) DESC
            LIMIT 10
        """).fetchall()
        for c in cycles:
            print(f"  {c[0]} | {c[1]} → {c[2]} | {c[3]} results")
        
        # 2. Find ALL BUY recommendations from analysis phase
        print("\n--- ALL BUY RECOMMENDATIONS (last 7 days) ---")
        buys = db.execute("""
            SELECT cycle_id, ticker, agent_name, confidence, result_json, created_at, triage_tier
            FROM analysis_results
            WHERE created_at > NOW() - INTERVAL '7 days'
            ORDER BY created_at DESC
        """).fetchall()
        
        buy_count = 0
        sell_count = 0
        hold_count = 0
        pass_count = 0
        other_count = 0
        
        buy_tickers = []
        
        for b in buys:
            cycle_id, ticker, agent_name, confidence, result_json_raw, created_at, triage_tier = b
            try:
                rj = json.loads(result_json_raw) if isinstance(result_json_raw, str) else result_json_raw
            except:
                rj = {}
            action = rj.get("action", "?")
            
            if action == "BUY":
                buy_count += 1
                buy_tickers.append((ticker, confidence, cycle_id, created_at, rj))
                print(f"  🟢 BUY  {ticker:8s} @ {confidence:3d}% | {cycle_id} | {created_at} | tier={triage_tier}")
                # Check for estimate
                est = rj.get("estimate")
                if est:
                    print(f"         Estimate: {est}")
                # Check escalation
                escalated = rj.get("escalated", False)
                config = rj.get("config_used", "?")
                print(f"         Config: {config}, Escalated: {escalated}")
                # Check c_result and d_result
                c = rj.get("c_result", {})
                d = rj.get("d_result", {})
                if c:
                    print(f"         Config C: {c.get('action','?')} @ {c.get('confidence','?')}%")
                if d:
                    print(f"         Config D: {d.get('action','?')} @ {d.get('confidence','?')}%")
            elif action == "SELL":
                sell_count += 1
            elif action == "HOLD":
                hold_count += 1
            elif action == "PASS":
                pass_count += 1
            else:
                other_count += 1
        
        print(f"\n  Summary: BUY={buy_count} SELL={sell_count} HOLD={hold_count} PASS={pass_count} OTHER={other_count}")
        
        # 3. Check what ACTUALLY got executed in orders table
        print("\n--- EXECUTED ORDERS (last 7 days) ---")
        orders = db.execute("""
            SELECT id, bot_id, ticker, side, qty, price, status, created_at, cycle_id
            FROM orders
            WHERE created_at > NOW() - INTERVAL '7 days'
            ORDER BY created_at DESC
            LIMIT 20
        """).fetchall()
        
        if not orders:
            print("  ❌ NO ORDERS EXECUTED!")
        for o in orders:
            print(f"  {o[3]:4s} {o[2]:8s} qty={o[4]} price=${o[5]:.2f} status={o[6]} | {o[7]} | cycle={o[8]}")
        
        # 4. Check trade_fills
        print("\n--- TRADE FILLS (last 7 days) ---")
        fills = db.execute("""
            SELECT id, bot_id, ticker, side, qty, price, created_at
            FROM trade_fills
            WHERE created_at > NOW() - INTERVAL '7 days'
            ORDER BY created_at DESC
            LIMIT 20
        """).fetchall()
        
        if not fills:
            print("  ❌ NO TRADE FILLS!")
        for f in fills:
            print(f"  {f[3]:4s} {f[2]:8s} qty={f[4]} price=${f[5]:.2f} | {f[6]}")
        
        # 5. Check current positions
        print("\n--- CURRENT POSITIONS ---")
        positions = db.execute("""
            SELECT bot_id, ticker, qty, avg_entry_price, created_at
            FROM positions
            ORDER BY created_at DESC
        """).fetchall()
        
        if not positions:
            print("  No open positions")
        for p in positions:
            print(f"  {p[1]:8s} qty={p[2]} entry=${p[3]:.2f} | bot={p[0]} | {p[4]}")
        
        # 6. Check current portfolio cash
        print("\n--- PORTFOLIO STATE ---")
        try:
            bots = db.execute("SELECT id, name, cash, created_at FROM bots ORDER BY created_at DESC LIMIT 5").fetchall()
            for bot in bots:
                print(f"  Bot: {bot[0]} ({bot[1]}) cash=${bot[2]:,.2f}")
        except Exception as e:
            print(f"  Error querying bots: {e}")
        
        # 7. Check debate history
        print("\n--- RECENT DEBATE HISTORY ---")
        try:
            debates = db.execute("""
                SELECT ticker, cycle_id, thesis_action, thesis_confidence,
                       counter_action, counter_confidence, winner, final_action, final_confidence,
                       persona_name, key_risk
                FROM debate_history
                WHERE created_at > NOW() - INTERVAL '7 days'
                ORDER BY created_at DESC
                LIMIT 15
            """).fetchall()
            
            for d in debates:
                ticker, cycle_id, thesis_act, thesis_conf, counter_act, counter_conf, winner, final_act, final_conf, persona, risk = d
                print(f"  {ticker:8s} | Thesis: {thesis_act}@{thesis_conf}% vs Counter: {counter_act}@{counter_conf}% → Winner: {winner} → Final: {final_act}@{final_conf}%")
                print(f"           Persona: {persona} | Risk: {(risk or '')[:80]}")
        except Exception as e:
            print(f"  Error: {e}")
        
        # 8. Check cycle audit log for trading phase events
        print("\n--- TRADING PHASE AUDIT EVENTS (last 7 days) ---")
        try:
            audit_events = db.execute("""
                SELECT timestamp, cycle_id, phase, audit_type, ticker, severity, message
                FROM cycle_audit_log
                WHERE phase = 'trading'
                AND timestamp > NOW() - INTERVAL '7 days'
                ORDER BY timestamp DESC
                LIMIT 30
            """).fetchall()
            
            if not audit_events:
                print("  No trading phase audit events found")
            for e in audit_events:
                print(f"  [{e[0]}] {e[2]}/{e[3]} {e[4]} ({e[5]}): {e[6]}")
        except Exception as e:
            print(f"  Error: {e}")
        
        # 9. Check for blocked/vetoed trades
        print("\n--- CYCLE SUMMARIES WITH BUY ACTION (last 7 days) ---")
        try:
            summaries = db.execute("""
                SELECT ticker, cycle_id, action, confidence, confidence_tier, rationale_summary, cycle_date
                FROM cycle_summaries
                WHERE action = 'BUY'
                AND cycle_date > NOW() - INTERVAL '7 days'
                ORDER BY cycle_date DESC
                LIMIT 20
            """).fetchall()
            
            if not summaries:
                print("  No BUY cycle summaries found")
            for s in summaries:
                print(f"  {s[0]:8s} @ {s[3]:3d}% ({s[4]}) | {s[1]} | {s[6]}")
                print(f"           {(s[5] or '')[:100]}")
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    main()
