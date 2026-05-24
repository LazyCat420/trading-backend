import sys
import os
import asyncio
import json
import uuid
from datetime import datetime, timezone
import psycopg

# Add trading-service to path so we can import modules
local_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(local_dir, "..", "trading-service"))

from app.db.connection import get_db
from app.config import settings

# Output directory for audit reports
REPORTS_DIR = "/home/lazycat/.gemini/antigravity-ide/brain/cf5a2799-a60d-47e2-b3ec-1dbffab7cc5d"

async def run_strategy_evaluation(cycle_id, refresh_pending):
    print(f"Running strategy evaluation for cycle {cycle_id}...")
    try:
        from app.cognition.evaluation.strategy_auditor import evaluate_strategy
        result = await evaluate_strategy(cycle_id=cycle_id, refresh_pending=refresh_pending)
        print(f"Strategy evaluation completed. Total Score: {result.get('total_score', 'N/A')}")
        return result
    except Exception as e:
        print(f"Error during strategy evaluation: {e}")
        return None

def trigger_command(cmd_type, payload=None):
    if payload is None:
        payload = {}
    cmd_id = f"job_{uuid.uuid4().hex[:8]}"
    print(f"Triggering command {cmd_type} (ID: {cmd_id})")
    with get_db() as db:
        db.execute(
            """
            INSERT INTO system_commands (id, command_type, payload, status, created_at)
            VALUES (%s, %s, %s, 'pending', CURRENT_TIMESTAMP)
            """,
            [cmd_id, cmd_type, json.dumps(payload)]
        )
    return cmd_id

def format_timestamp(ts):
    if not ts:
        return "N/A"
    if isinstance(ts, str):
        return ts
    return ts.strftime("%Y-%m-%d %H:%M:%S")

def perform_cycle_audit(cycle_id):
    print(f"Auditing results for cycle: {cycle_id}")
    
    with get_db() as db:
        # 1. Fetch cycle metadata
        row = db.execute(
            "SELECT status, started_at, finished_at, tickers, progress, error FROM pipeline_state WHERE cycle_id = %s", 
            [cycle_id]
        ).fetchone()
        
        if not row:
            print(f"No pipeline_state found for cycle {cycle_id}")
            status, started_at, finished_at, tickers_list, progress, error = "unknown", None, None, [], "No details", None
        else:
            status, started_at, finished_at, tickers_data, progress, error = row
            # Parse tickers_data which could be json string or list
            tickers_list = []
            if tickers_data:
                if isinstance(tickers_data, list):
                    tickers_list = tickers_data
                elif isinstance(tickers_data, str):
                    try:
                        tickers_list = json.loads(tickers_data)
                    except:
                        tickers_list = [tickers_data]
        
        # Calculate duration
        duration_str = "N/A"
        if started_at:
            s_dt = started_at.replace(tzinfo=timezone.utc) if started_at.tzinfo is None else started_at
            f_dt = finished_at
            if f_dt:
                f_dt = f_dt.replace(tzinfo=timezone.utc) if f_dt.tzinfo is None else f_dt
            else:
                f_dt = datetime.now(timezone.utc)
            duration = f_dt - s_dt
            duration_str = f"{duration.total_seconds() / 60.0:.1f} minutes"

        # 2. Fetch LLM Performance Metrics
        db.execute(
            """
            SELECT 
                COALESCE(model, 'Unknown'), 
                COUNT(*), 
                SUM(tokens_used), 
                AVG(tokens_per_second), 
                AVG(queue_wait_ms)
            FROM llm_audit_logs
            WHERE cycle_id = %s
            GROUP BY model
            ORDER BY count DESC
            """,
            [cycle_id]
        )
        perf_rows = db.fetchall()
        
        perf_table_lines = []
        for p_model, p_count, p_tokens, p_tps, p_wait in perf_rows:
            p_tps_val = f"{p_tps:.1f}" if p_tps is not None else "N/A"
            p_wait_val = f"{p_wait:.1f}ms" if p_wait is not None else "N/A"
            p_tokens_val = str(p_tokens) if p_tokens is not None else "0"
            perf_table_lines.append(f"| `{p_model}` | {p_count} | {p_tokens_val} | {p_tps_val} | {p_wait_val} |")
        
        perf_table_str = "\n".join(perf_table_lines) if perf_table_lines else "| N/A | 0 | 0 | N/A | N/A |"

        # 3. Fetch LLM Decision Quality Evaluations
        db.execute(
            """
            SELECT 
                ticker, 
                judge_a_score, 
                final_quality_score, 
                red_cards, 
                first_principles_reasoning
            FROM decision_evaluations
            WHERE cycle_id = %s
            ORDER BY ticker ASC
            """,
            [cycle_id]
        )
        eval_rows = db.fetchall()
        
        eval_table_lines = []
        for ticker, j_score, f_score, red_cards, reasoning in eval_rows:
            j_score_val = f"{j_score:.1f}/5" if j_score is not None else "N/A"
            f_score_val = f"{f_score:.1f}/5" if f_score is not None else "N/A"
            
            rc_list = []
            if red_cards:
                try:
                    parsed = json.loads(red_cards)
                    if isinstance(parsed, list):
                        rc_list = parsed
                    elif isinstance(parsed, str):
                        rc_list = [parsed]
                except:
                    rc_list = [str(red_cards)]
            
            rc_str = "<br>".join(rc_list) if rc_list else "None"
            
            reasoning_val = reasoning.replace("|", "\\|").replace("\n", " ").strip() if reasoning else "N/A"
            eval_table_lines.append(f"| **{ticker}** | {j_score_val} | {f_score_val} | {rc_str} | {reasoning_val} |")
            
        eval_table_str = "\n".join(eval_table_lines) if eval_table_lines else "| N/A | N/A | N/A | None | N/A |"

        # Write Report
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        started_at_str = format_timestamp(started_at)
        finished_at_str = format_timestamp(finished_at)
        tickers_str = ", ".join(tickers_list) if tickers_list else "None"
        
        report_content = f"""# Audit Report for Trading Cycle `{cycle_id}`

Generated At: {now_str}

## 📊 Cycle Metadata
- **Status**: `{status}`
- **Started At**: {started_at_str}
- **Finished At**: {finished_at_str}
- **Duration**: {duration_str}
- **Tickers Configured**: {tickers_str}
- **Progress**: {progress}

## 🧠 LLM Endpoint Performance
| Endpoint | Calls | Total Tokens | Avg TPS | Avg Queue Wait |
|---|---|---|---|---|
{perf_table_str}

## ⚖️ LLM Decision Quality Evaluations
| Ticker | Judge Score | Final Score | Red Cards | First Principles Reasoning |
|---|---|---|---|---|
{eval_table_str}
"""
        os.makedirs(REPORTS_DIR, exist_ok=True)
        report_path = os.path.join(REPORTS_DIR, f"audit_report_{cycle_id}.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"Successfully generated audit report: {report_path}")

async def main():
    print("Starting trading cycle check...")
    with get_db() as db:
        # Fetch the latest cycle state
        db.execute(
            """
            SELECT cycle_id, status, started_at, finished_at, progress
            FROM pipeline_state
            ORDER BY started_at DESC LIMIT 1
            """
        )
        row = db.fetchone()
        
    if not row:
        print("No cycle found in pipeline_state. Triggering initial cycle.")
        trigger_command("START_CYCLE", {"tickers": []})
        return

    cycle_id, status, started_at, finished_at, progress = row
    print(f"Latest Cycle ID: {cycle_id} | Status: {status} | Started At: {started_at}")

    active_statuses = ("started", "starting", "collecting", "analyzing", "gated", "trading", "paused", "running")
    
    if status in active_statuses:
        # Check elapsed time
        if started_at:
            s_dt = started_at.replace(tzinfo=timezone.utc) if started_at.tzinfo is None else started_at
            now = datetime.now(timezone.utc)
            elapsed_hours = (now - s_dt).total_seconds() / 3600.0
            print(f"Cycle is active. Elapsed time: {elapsed_hours:.2f} hours.")
            
            if elapsed_hours > 2.0:
                print(f"Cycle {cycle_id} has been running for {elapsed_hours:.2f} hours (exceeded 2.0 hours limit). Stopping cycle...")
                trigger_command("STOP_CYCLE", {"reason": f"Cycle {cycle_id} exceeded maximum run duration of 2.0 hours."})
            else:
                # Print last 10 entries of pipeline_events
                print(f"Printing last 10 events for cycle {cycle_id}:")
                with get_db() as db:
                    db.execute(
                        """
                        SELECT timestamp, phase, step, status, elapsed_ms, detail
                        FROM pipeline_events
                        WHERE cycle_id = %s
                        ORDER BY timestamp DESC
                        LIMIT 10
                        """,
                        [cycle_id]
                    )
                    events = db.fetchall()
                    for ev in reversed(events):
                        print(f"[{ev[0]}] {ev[1]} | {ev[2]} | Status: {ev[3]} | Elapsed: {ev[4]}ms | {ev[5][:120]}")
        else:
            print("Cycle is active but started_at is NULL.")
    else:
        # Finished or idle
        print(f"Cycle {cycle_id} is idle/finished with status: {status}")
        report_file = f"audit_report_{cycle_id}.md"
        report_path = os.path.join(REPORTS_DIR, report_file)
        
        if not os.path.exists(report_path):
            print(f"Audit report for cycle {cycle_id} is missing. Auditing cycle...")
            # Run strategy evaluation
            await run_strategy_evaluation(cycle_id, refresh_pending=(status == 'done'))
            # Generate markdown report
            perform_cycle_audit(cycle_id)
            # Queue a new cycle
            trigger_command("START_CYCLE", {"tickers": []})
        else:
            print(f"Audit report for cycle {cycle_id} already exists. No action taken.")

if __name__ == "__main__":
    asyncio.run(main())