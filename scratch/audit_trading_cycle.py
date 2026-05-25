import os
import sys
import json
import argparse
from datetime import datetime, timezone

# Add parent directory to path to allow importing app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app.db.connection import get_db
except ImportError:
    print("Error: Could not import app.db.connection. Make sure to run from project root.")
    sys.exit(1)

def get_latest_cycle_id():
    query = """
        SELECT cycle_id, timestamp
        FROM cycle_audit_log
        WHERE cycle_id LIKE 'cycle-%%'
        ORDER BY timestamp DESC
        LIMIT 1
    """
    try:
        with get_db() as db:
            row = db.execute(query).fetchone()
            if row:
                return row[0], row[1]
    except Exception as e:
        print(f"Error fetching latest cycle_id: {e}")
    return None, None

def fetch_audit_log(cycle_id):
    query = """
        SELECT timestamp, audit_type, phase, ticker, severity, message, data
        FROM cycle_audit_log
        WHERE cycle_id = %s
        ORDER BY timestamp ASC
    """
    try:
        with get_db() as db:
            rows = db.execute(query, [cycle_id]).fetchall()
            return [
                {
                    "timestamp": r[0],
                    "audit_type": r[1],
                    "phase": r[2],
                    "ticker": r[3],
                    "severity": r[4],
                    "message": r[5],
                    "data": r[6] if isinstance(r[6], dict) else (json.loads(r[6]) if r[6] else {})
                }
                for r in rows
            ]
    except Exception as e:
        print(f"Error fetching audit log: {e}")
        return []

def fetch_execution_errors(cycle_id):
    query = """
        SELECT created_at, phase, ticker, error_type, error_message, stack_trace
        FROM execution_errors
        WHERE cycle_id = %s
        ORDER BY created_at ASC
    """
    try:
        with get_db() as db:
            rows = db.execute(query, [cycle_id]).fetchall()
            return [
                {
                    "created_at": r[0],
                    "phase": r[1],
                    "ticker": r[2],
                    "error_type": r[3],
                    "error_message": r[4],
                    "stack_trace": r[5]
                }
                for r in rows
            ]
    except Exception as e:
        print(f"Error fetching execution errors: {e}")
        return []

def fetch_autoresearch_report(cycle_id):
    query = """
        SELECT created_at, data_quality_score, decision_quality_score, llm_performance_score, overall_score,
               data_gaps, decision_issues, llm_issues, performance_metrics, reflection, recovery_stats
        FROM autoresearch_reports
        WHERE cycle_id = %s
        LIMIT 1
    """
    try:
        with get_db() as db:
            row = db.execute(query, [cycle_id]).fetchone()
            if row:
                cols = [
                    "created_at", "data_quality_score", "decision_quality_score", "llm_performance_score", "overall_score",
                    "data_gaps", "decision_issues", "llm_issues", "performance_metrics", "reflection", "recovery_stats"
                ]
                res = dict(zip(cols, row))
                # Deserialize JSON fields
                for f in ["data_gaps", "decision_issues", "llm_issues", "performance_metrics", "reflection", "recovery_stats"]:
                    if res[f] and isinstance(res[f], str):
                        res[f] = json.loads(res[f])
                return res
    except Exception as e:
        print(f"Error fetching autoresearch report: {e}")
    return None

def main():
    parser = argparse.ArgumentParser(description="Audit a specific or the latest trading cycle.")
    parser.add_argument("--cycle-id", type=str, help="Specific cycle_id to audit (defaults to latest)")
    args = parser.parse_args()

    cycle_id = args.cycle_id
    if not cycle_id:
        cycle_id, last_timestamp = get_latest_cycle_id()
        if not cycle_id:
            print("No cycles found in the database cycle_audit_log.")
            sys.exit(1)
        print(f"Auditing latest cycle: {cycle_id} (last event at: {last_timestamp})")
    else:
        print(f"Auditing cycle: {cycle_id}")

    # Fetch data
    audit_events = fetch_audit_log(cycle_id)
    errors = fetch_execution_errors(cycle_id)
    ar_report = fetch_autoresearch_report(cycle_id)

    print("\n" + "=" * 80)
    print(f"                  TRADING CYCLE AUDIT REPORT: {cycle_id}")
    print("=" * 80)

    # 1. Pipeline Execution Errors
    print("\n🔴 PIPELINE EXECUTION ERRORS")
    print("-" * 30)
    if not errors:
        print("✅ No pipeline execution errors logged in execution_errors.")
    else:
        print(f"⚠️ Found {len(errors)} execution error(s):")
        for i, err in enumerate(errors, 1):
            print(f"\n[{i}] Time: {err['created_at']} | Phase: {err['phase']} | Ticker: {err['ticker']}")
            print(f"    Type: {err['error_type']}")
            print(f"    Message: {err['error_message']}")
            if err['stack_trace']:
                lines = err['stack_trace'].strip().split('\n')
                snippet = '\n'.join(lines[-8:])  # print last 8 lines of traceback
                print(f"    Traceback Snippet:\n{snippet}")

    # 2. Anomalies and Warnings from Auditor
    warnings = [e for e in audit_events if e["severity"] in ("warning", "critical")]
    print("\n⚠️ AUDITOR WARNINGS & ANOMALIES")
    print("-" * 30)
    if not warnings:
        print("✅ No warning or critical events in cycle_audit_log.")
    else:
        print(f"Found {len(warnings)} warnings/anomalies:")
        for w in warnings:
            print(f" - [{w['severity'].upper()}] [{w['phase']}] {w['ticker']}: {w['message']}")

    # 3. AutoResearch Metrics
    print("\n📊 AUTORESEARCH QUALITY SCORES")
    print("-" * 30)
    if not ar_report:
        print("❌ No AutoResearch report found for this cycle.")
    else:
        print(f"Overall Quality Score:  {ar_report['overall_score']:.1f}%")
        print(f" - Data Quality Score:  {ar_report['data_quality_score']:.1f}%")
        print(f" - Decision Quality:    {ar_report['decision_quality_score']:.1f}%")
        print(f" - LLM Performance:     {ar_report['llm_performance_score']:.1f}%")

        reflection = ar_report.get("reflection", {})
        if reflection:
            print(f"\nSummary:\n  {reflection.get('summary', 'No summary available.')}")
            
            recs = reflection.get("recommendations", [])
            if recs:
                print("\nRecommendations:")
                for r in recs:
                    print(f"  * {r}")
            
            gaps = ar_report.get("data_gaps", [])
            if gaps:
                print("\nDetected Data Gaps:")
                for g in gaps:
                    print(f"  * {g.get('ticker')}: missing {g.get('missing_sources')}")

    # 4. Performance & Metrics
    print("\n⏱️ PERFORMANCE METRICS")
    print("-" * 30)
    if ar_report and ar_report.get("performance_metrics"):
        perf = ar_report["performance_metrics"]
        elapsed = perf.get("total_ms", 0) / 1000.0
        print(f"Total cycle duration:   {elapsed:.1f}s")
        print(f"Tickers analyzed:      {perf.get('tickers_analyzed', 0)}")
        print(f"Trades executed:       {perf.get('trade_executed', 0)}")
        print(f"Collector stats:       ok={perf.get('collector_ok', 0)}, error={perf.get('collector_error', 0)}")
    else:
        # Simple extraction from audit logs
        entries = [e for e in audit_events if e["audit_type"] == "phase_exit"]
        for entry in entries:
            print(f"Phase {entry['phase']}: {entry['message']}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
