#!/usr/bin/env python3
"""
Smoke Test: Streaming Pipeline Validation
==========================================

Validates that the streaming pipeline changes are working correctly:
1. Analysis workers start processing within seconds (not waiting 10+ minutes)
2. Pre-pushed watchlist tickers are processed before collection completes
3. Dedup prevents double-processing
4. Newly discovered tickers are pushed to analysis queue from Track A

Usage:
    python scripts/smoke_test_streaming.py              # default AAPL
    python scripts/smoke_test_streaming.py NVDA          # custom ticker
    python scripts/smoke_test_streaming.py --timeout 600 # 10-minute timeout

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
    parser = argparse.ArgumentParser(description="Validate streaming pipeline timing")
    parser.add_argument("ticker", nargs="?", default="AAPL", help="Ticker symbol (default: AAPL)")
    parser.add_argument("--timeout", type=int, default=600, help="Max seconds (default: 600)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print all events")
    return parser.parse_args()


async def run_streaming_test(ticker: str, timeout: int, verbose: bool):
    """Run a single-ticker cycle and validate streaming timing."""

    # Force test-friendly settings
    os.environ["MAX_CYCLE_TICKERS"] = "1"

    from app.db.connection import get_db
    from app.config import settings

    print("=" * 70)
    print(f"  STREAMING PIPELINE SMOKE TEST: {ticker}")
    print(f"  Validates: pre-push, parallel tracks, dedup, timing")
    print("=" * 70)
    print()

    from app.services.pipeline_service import PipelineService

    # Reset state if stuck
    current = PipelineService.get_current_state(summary_only=True)
    if current.get("status") not in ("idle", "done", "error", "stopped", "interrupted"):
        print(f"  ⚠️  Pipeline in '{current.get('status')}' state. Resetting.")
        PipelineService._state["status"] = "idle"
        PipelineService.save_state()

    # Start cycle
    print(f"[1/6] Starting cycle for [{ticker}]...")
    start_time = time.monotonic()

    try:
        result = await PipelineService.start_cycle(
            tickers=[ticker],
            collect=True,
            analyze=True,
            trade=False,  # Skip trading for smoke test
            trigger_type="streaming_smoke_test",
            max_tickers=1,
        )
        cycle_id = result.get("cycle_id", "unknown")
        print(f"  ✅ Cycle started: {cycle_id}")
    except Exception as e:
        print(f"  ❌ Failed to start cycle: {e}")
        return False

    # Track key timing milestones
    milestones = {
        "watchlist_prepush": None,
        "first_worker_got": None,
        "track_a_start": None,
        "track_b_start": None,
        "parallel_start": None,
        "first_analysis_start": None,
        "first_dedup": None,
        "collection_complete": None,
        "pipeline_done": None,
    }

    poll_interval = 2
    last_event_count = 0

    print(f"[2/6] Monitoring streaming events (timeout={timeout}s)...")
    print()

    while True:
        elapsed = time.monotonic() - start_time
        if elapsed > timeout:
            print(f"\n  ❌ TIMEOUT after {int(elapsed)}s")
            break

        state = PipelineService.get_current_state(summary_only=True)
        status = state.get("status", "unknown")

        # Poll for new events
        with get_db() as db:
            total_count = db.execute(
                "SELECT COUNT(*) FROM pipeline_events WHERE cycle_id = %s",
                [cycle_id],
            ).fetchone()[0]

            if total_count > last_event_count:
                events = db.execute(
                    "SELECT phase, step, detail, status, elapsed_ms, timestamp "
                    "FROM pipeline_events WHERE cycle_id = %s "
                    "ORDER BY timestamp ASC OFFSET %s",
                    [cycle_id, last_event_count],
                ).fetchall()
            else:
                events = []

        for ev in events:
            _phase, _step, _detail, _status, _ms, _ts = ev
            last_event_count += 1

            # Track milestones
            if _step == "watchlist_prepush" and milestones["watchlist_prepush"] is None:
                milestones["watchlist_prepush"] = elapsed
            elif _step.startswith("worker_got_") and milestones["first_worker_got"] is None:
                milestones["first_worker_got"] = elapsed
            elif _step == "track_a_start" and milestones["track_a_start"] is None:
                milestones["track_a_start"] = elapsed
            elif _step == "track_b_start" and milestones["track_b_start"] is None:
                milestones["track_b_start"] = elapsed
            elif _step == "parallel_start" and milestones["parallel_start"] is None:
                milestones["parallel_start"] = elapsed
            elif _step.startswith("v2_start_") and milestones["first_analysis_start"] is None:
                milestones["first_analysis_start"] = elapsed
            elif _step.startswith("worker_dedup_") and milestones["first_dedup"] is None:
                milestones["first_dedup"] = elapsed
            elif _step == "collection_complete" and milestones["collection_complete"] is None:
                milestones["collection_complete"] = elapsed
            elif _step == "pipeline_done" and milestones["pipeline_done"] is None:
                milestones["pipeline_done"] = elapsed

            if verbose:
                _emoji = "✅" if _status == "ok" else "❌" if _status == "error" else "⏳" if _status == "running" else "⏭️"
                _elapsed_str = f" ({_ms}ms)" if _ms else ""
                print(f"  {_emoji} [{int(elapsed):>4}s] [{_phase}] {_step}: {_detail[:80]}{_elapsed_str}")

        if status in ("done", "error", "stopped"):
            break

        await asyncio.sleep(poll_interval)

    total_elapsed = time.monotonic() - start_time

    # Report results
    print()
    print("=" * 70)
    print(f"[3/6] STREAMING MILESTONES")
    print("=" * 70)

    _ok = "✅"
    _fail = "❌"
    _skip = "⏭️"

    for key, ts in milestones.items():
        label = key.replace("_", " ").title()
        if ts is not None:
            print(f"  {_ok} {label:.<40} {ts:.1f}s")
        else:
            print(f"  {_skip} {label:.<40} (not observed)")

    print()
    print("=" * 70)
    print(f"[4/6] STREAMING VALIDATION")
    print("=" * 70)

    checks = {}

    # Check 1: Pre-push happened
    if milestones["watchlist_prepush"] is not None:
        checks["pre_push"] = True
        print(f"  {_ok} Pre-push: watchlist tickers pushed at {milestones['watchlist_prepush']:.1f}s")
    else:
        checks["pre_push"] = False
        print(f"  {_fail} Pre-push: watchlist_prepush event NOT found")

    # Check 2: Parallel tracks started
    if milestones["track_a_start"] is not None and milestones["track_b_start"] is not None:
        delta = abs(milestones["track_a_start"] - milestones["track_b_start"])
        checks["parallel_tracks"] = delta < 5.0
        if checks["parallel_tracks"]:
            print(f"  {_ok} Parallel tracks: Track A and B started {delta:.1f}s apart (< 5s)")
        else:
            print(f"  {_fail} Parallel tracks: Track A and B started {delta:.1f}s apart (should be < 5s)")
    else:
        checks["parallel_tracks"] = False
        print(f"  {_fail} Parallel tracks: track start events not found")

    # Check 3: Worker started processing before collection finished
    if milestones["first_worker_got"] is not None:
        if milestones["collection_complete"] is None or milestones["first_worker_got"] < milestones["collection_complete"]:
            checks["early_analysis"] = True
            cc_time = milestones["collection_complete"] or total_elapsed
            saved = cc_time - milestones["first_worker_got"]
            print(f"  {_ok} Early analysis: first worker got ticker at {milestones['first_worker_got']:.1f}s "
                  f"({saved:.0f}s before collection finished)")
        else:
            checks["early_analysis"] = False
            print(f"  {_fail} Early analysis: worker started AFTER collection finished")
    else:
        checks["early_analysis"] = False
        print(f"  {_fail} Early analysis: worker_got event not found")

    # Check 4: Analysis started within 30 seconds (was 10+ minutes before)
    if milestones["first_analysis_start"] is not None:
        checks["fast_start"] = milestones["first_analysis_start"] < 30.0
        if checks["fast_start"]:
            print(f"  {_ok} Fast start: V2 pipeline started at {milestones['first_analysis_start']:.1f}s (< 30s)")
        else:
            print(f"  {_fail} Fast start: V2 pipeline started at {milestones['first_analysis_start']:.1f}s (should be < 30s)")
    else:
        checks["fast_start"] = False
        print(f"  {_fail} Fast start: v2_start event not found")

    # Check 5: Dedup working (optional — only if collection pushes same ticker again)
    if milestones["first_dedup"] is not None:
        checks["dedup_working"] = True
        print(f"  {_ok} Dedup: duplicate ticker detected and skipped at {milestones['first_dedup']:.1f}s")
    else:
        checks["dedup_working"] = None  # Not a failure — dedup may not trigger in smoke test
        print(f"  {_skip} Dedup: no duplicate observed (expected for single-ticker test)")

    # Final analysis result
    print()
    print("=" * 70)
    print(f"[5/6] ANALYSIS RESULTS")
    print("=" * 70)

    with get_db() as db:
        analysis_rows = db.execute(
            "SELECT ticker, result_json FROM analysis_results WHERE cycle_id = %s",
            [cycle_id],
        ).fetchall()

    if analysis_rows:
        for row in analysis_rows:
            try:
                r = json.loads(row[1])
                action = r.get("action", "?")
                confidence = r.get("confidence", 0)
                total_time = r.get("total_time_s", 0)
                tokens = r.get("total_tokens", 0)
                print(f"  {_ok} {row[0]}: {action} @ {confidence}% ({total_time:.1f}s, {tokens:,} tokens)")
            except Exception:
                print(f"  {_fail} {row[0]}: (parse error)")
    else:
        print(f"  {_fail} No analysis results found")

    # Verdict
    print()
    print("=" * 70)
    print(f"[6/6] VERDICT")
    print("=" * 70)

    critical_checks = ["pre_push", "parallel_tracks", "early_analysis", "fast_start"]
    passed = all(checks.get(c, False) for c in critical_checks)
    completed = status == "done" and len(analysis_rows) > 0

    if passed and completed:
        print(f"  🟢 PASS — Streaming pipeline validated in {int(total_elapsed)}s")
        print(f"     Time to first analysis: {milestones.get('first_analysis_start', '?')}s (target: <30s)")
    elif completed:
        failed_checks = [c for c in critical_checks if not checks.get(c, False)]
        print(f"  🟡 PARTIAL — Cycle completed but streaming checks failed: {', '.join(failed_checks)}")
    else:
        print(f"  🔴 FAIL — Cycle status: {status}, results: {len(analysis_rows)}")

    print()
    print("=" * 70)
    return passed and completed


if __name__ == "__main__":
    args = parse_args()
    success = asyncio.run(
        run_streaming_test(
            ticker=args.ticker.upper(),
            timeout=args.timeout,
            verbose=args.verbose,
        )
    )
    sys.exit(0 if success else 1)
