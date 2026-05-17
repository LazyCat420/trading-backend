#!/usr/bin/env python3
"""
Smoke Test: Single-Ticker End-to-End Cycle
==========================================

Runs a full trading cycle for a single ticker (default: AAPL) and monitors
the pipeline_events table for progress. Validates the system can complete
a full cycle without hanging.

Usage:
    python scripts/smoke_test_cycle.py              # default AAPL
    python scripts/smoke_test_cycle.py NVDA          # custom ticker
    python scripts/smoke_test_cycle.py --timeout 600 # 10-minute timeout

Environment:
    DATABASE_URL must be set (or .env file present).
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone

# Ensure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test a single-ticker trading cycle")
    parser.add_argument("ticker", nargs="?", default="AAPL", help="Ticker symbol to test (default: AAPL)")
    parser.add_argument("--timeout", type=int, default=600, help="Max seconds to wait for cycle completion (default: 600)")
    parser.add_argument("--skip-collection", action="store_true", help="Skip data collection phase")
    parser.add_argument("--skip-trade", action="store_true", help="Skip trading phase")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print all events as they appear")
    return parser.parse_args()


async def run_smoke_test(ticker: str, timeout: int, skip_collection: bool, skip_trade: bool, verbose: bool):
    """Run a single-ticker cycle and monitor progress."""

    # Force test-friendly settings
    os.environ["MAX_CYCLE_TICKERS"] = "1"

    from app.db.connection import get_db
    from app.config import settings

    print("=" * 70)
    print(f"  SMOKE TEST: {ticker}")
    print(f"  Timeout: {timeout}s | Skip collection: {skip_collection} | Skip trade: {skip_trade}")
    print(f"  DB: {settings.DATABASE_URL.split('@')[-1] if '@' in settings.DATABASE_URL else 'local'}")
    print("=" * 70)
    print()

    # Step 1: Import and initialize the pipeline service
    print("[1/5] Initializing pipeline service...")
    from app.services.pipeline_service import PipelineService

    # Reset state if stuck
    current = PipelineService.get_current_state(summary_only=True)
    if current.get("status") not in ("idle", "done", "error", "stopped", "interrupted"):
        print(f"  ⚠️  Pipeline is in '{current.get('status')}' state. Force-resetting to idle.")
        PipelineService._state["status"] = "idle"
        PipelineService.save_state()

    # Step 2: Start the cycle
    print(f"[2/5] Starting cycle for [{ticker}]...")
    start_time = time.monotonic()

    try:
        result = await PipelineService.start_cycle(
            tickers=[ticker],
            collect=not skip_collection,
            analyze=True,
            trade=not skip_trade,
            trigger_type="smoke_test",
            max_tickers=1,
        )
        cycle_id = result.get("cycle_id", "unknown")
        print(f"  ✅ Cycle started: {cycle_id}")
    except Exception as e:
        print(f"  ❌ Failed to start cycle: {e}")
        return False

    # Step 3: Monitor progress
    print(f"[3/5] Monitoring progress (timeout={timeout}s)...")
    print()

    last_event_count = 0
    poll_interval = 3  # seconds between polls
    stale_count = 0
    max_stale = 20  # 20 * 3s = 60s of no new events → warn

    while True:
        elapsed = time.monotonic() - start_time
        if elapsed > timeout:
            print(f"\n  ❌ TIMEOUT after {int(elapsed)}s — cycle did not complete")
            _dump_final_events(cycle_id, verbose=True)
            return False

        # Check state
        state = PipelineService.get_current_state(summary_only=True)
        status = state.get("status", "unknown")
        step_count = state.get("step_count", 0)

        # Check for new events
        with get_db() as db:
            db.execute(
                "SELECT COUNT(*) FROM pipeline_events WHERE cycle_id = %s",
                [cycle_id],
            )
            event_count = db.fetchone()[0]

        if event_count > last_event_count:
            # New events arrived
            if verbose:
                with get_db() as db:
                    db.execute(
                        "SELECT phase, step, detail, status, elapsed_ms FROM pipeline_events "
                        "WHERE cycle_id = %s ORDER BY timestamp ASC OFFSET %s",
                        [cycle_id, last_event_count],
                    )
                    for row in db.fetchall():
                        _phase, _step, _detail, _status, _ms = row
                        _emoji = "✅" if _status == "ok" else "❌" if _status == "error" else "⏳" if _status == "running" else "⏭️"
                        _elapsed_str = f" ({_ms}ms)" if _ms else ""
                        print(f"  {_emoji} [{_phase}] {_step}: {_detail}{_elapsed_str}")
            else:
                new = event_count - last_event_count
                print(f"  [{int(elapsed):>4}s] {status:>12} | events: {event_count} (+{new}) | steps: {step_count}")

            last_event_count = event_count
            stale_count = 0
        else:
            stale_count += 1
            if stale_count % 10 == 0:  # every 30s
                print(f"  [{int(elapsed):>4}s] {status:>12} | events: {event_count} (no change for {stale_count * poll_interval}s)")

        # Check terminal states
        if status in ("done", "error", "stopped"):
            break

        await asyncio.sleep(poll_interval)

    elapsed = time.monotonic() - start_time

    # Step 4: Collect results
    print()
    print(f"[4/5] Cycle ended with status: {status}")

    final_state = PipelineService.get_current_state(summary_only=False)

    # Check for analysis results
    with get_db() as db:
        db.execute(
            "SELECT ticker, result_json FROM analysis_results WHERE cycle_id = %s",
            [cycle_id],
        )
        analysis_rows = db.fetchall()

    print(f"  Elapsed: {int(elapsed)}s")
    print(f"  Total events: {last_event_count}")
    print(f"  Analysis results: {len(analysis_rows)}")

    if analysis_rows:
        for row in analysis_rows:
            try:
                r = json.loads(row[1])
                action = r.get("action", "?")
                confidence = r.get("confidence", 0)
                total_time = r.get("total_time_s", 0)
                tokens = r.get("total_tokens", 0)
                print(f"    {row[0]}: {action} @ {confidence}% ({total_time:.1f}s, {tokens:,} tokens)")
            except Exception:
                print(f"    {row[0]}: (parse error)")

    # Step 5: Verdict
    print()
    print("[5/5] Verdict:")

    success = status == "done" and len(analysis_rows) > 0
    if success:
        print(f"  🟢 PASS — Cycle completed successfully in {int(elapsed)}s")
    elif status == "done" and len(analysis_rows) == 0:
        print(f"  🟡 PARTIAL — Cycle completed but no analysis results")
        success = False
    elif status == "error":
        error = final_state.get("error", "unknown")
        print(f"  🔴 FAIL — Cycle errored: {error}")
        success = False
    else:
        print(f"  🔴 FAIL — Cycle ended in unexpected state: {status}")
        success = False

    if not success:
        _dump_final_events(cycle_id, verbose=True)

    print()
    print("=" * 70)
    return success


def _dump_final_events(cycle_id: str, verbose: bool = False):
    """Print the last 20 events for debugging."""
    from app.db.connection import get_db

    print()
    print("  --- Last 20 events ---")
    with get_db() as db:
        db.execute(
            "SELECT phase, step, detail, status, elapsed_ms, timestamp "
            "FROM pipeline_events WHERE cycle_id = %s ORDER BY timestamp DESC LIMIT 20",
            [cycle_id],
        )
        rows = db.fetchall()
        for row in reversed(rows):
            _phase, _step, _detail, _status, _ms, _ts = row
            _emoji = "✅" if _status == "ok" else "❌" if _status == "error" else "⏳" if _status == "running" else "⏭️"
            _elapsed_str = f" ({_ms}ms)" if _ms else ""
            _ts_str = _ts.strftime("%H:%M:%S") if hasattr(_ts, "strftime") else str(_ts)[:8]
            print(f"  {_emoji} {_ts_str} [{_phase}] {_step}: {_detail[:80]}{_elapsed_str}")
    print("  ---")


if __name__ == "__main__":
    args = parse_args()
    success = asyncio.run(
        run_smoke_test(
            ticker=args.ticker.upper(),
            timeout=args.timeout,
            skip_collection=args.skip_collection,
            skip_trade=args.skip_trade,
            verbose=args.verbose,
        )
    )
    sys.exit(0 if success else 1)
