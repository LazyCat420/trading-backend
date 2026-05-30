"""
Centralized Cycle Log Manager — Single source of truth for pipeline diagnostics.

All cycle events (steps, errors, summaries, crashes) are written to a single
JSONL file per cycle in `logs_local/cycles/`. This replaces the need to
cross-reference 6+ separate logging systems.

Each log entry has the schema:
    {
        "cycle_id": str,
        "timestamp": ISO8601,
        "level": "info" | "warning" | "error" | "critical",
        "step": str,       # e.g. "v2_start", "v2_error", "cycle_summary"
        "ticker": str,     # "" for cycle-level events
        "payload": dict,
    }

Usage:
    from app.log_manager import log_manager

    log_manager.log_v2_cycle(cycle_id, "v2_start", {"ticker": "AAPL", ...})
    log_manager.log_cycle_error(cycle_id, "thesis_timeout", ticker="AAPL",
                                error="Timed out after 360s", stack_trace="...")
    log_manager.log_cycle_summary(cycle_id, summary_dict)
"""

import json
import logging
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class LogManager:
    """Manages appending logs for V2 and A/B benchmark runs.

    All writes go to `logs_local/` which is always writable (even in Docker
    where `logs/` may be a read-only volume mount).
    """

    BASE_DIR = Path("logs")
    CYCLE_DIR = BASE_DIR / "cycles"
    AB_DIR = CYCLE_DIR / "ab_results"

    def __init__(self):
        try:
            self.CYCLE_DIR.mkdir(parents=True, exist_ok=True)
            # Test write access
            test_file = self.CYCLE_DIR / ".write_test"
            test_file.touch()
            test_file.unlink()
        except (PermissionError, OSError):
            # Fallback to local logs directory if default logs dir is not writable
            self.BASE_DIR = Path("logs_local")
            self.CYCLE_DIR = self.BASE_DIR / "cycles"
            self.AB_DIR = self.CYCLE_DIR / "ab_results"
            self.CYCLE_DIR.mkdir(parents=True, exist_ok=True)

        self.AB_DIR.mkdir(parents=True, exist_ok=True)

    # ── Core Write ───────────────────────────────────────────────────────

    @staticmethod
    def _write_jsonl(path: Path, data: dict):
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, default=str) + "\n")
        except Exception as e:
            # LogManager must NEVER crash the pipeline
            logger.debug("[LogManager] Write failed for %s: %s", path.name, e)

    def _cycle_path(self, cycle_id: str) -> Path:
        return self.CYCLE_DIR / f"{cycle_id}.jsonl"

    # ── Step Logging (existing — unchanged interface) ────────────────────

    def log_v2_cycle(self, cycle_id: str, step_name: str, payload: dict):
        """Append a step result to the v2 cycle log."""
        ts = datetime.now(timezone.utc).isoformat()
        log_entry = {
            "cycle_id": cycle_id,
            "timestamp": ts,
            "level": "info",
            "step": step_name,
            "ticker": payload.get("ticker", ""),
            "payload": payload,
        }
        self._write_jsonl(self._cycle_path(cycle_id), log_entry)

    # ── Error Logging (NEW) ──────────────────────────────────────────────

    def log_cycle_error(
        self,
        cycle_id: str,
        error_type: str,
        *,
        ticker: str = "",
        error: str = "",
        stack_trace: str = "",
        stage: str = "",
        elapsed_ms: int = 0,
        extra: dict | None = None,
    ):
        """Log an error event to the cycle's JSONL file.

        This captures errors that previously went only to DB or stdout,
        ensuring they appear in the same file as step logs.
        """
        ts = datetime.now(timezone.utc).isoformat()
        payload = {
            "error_type": error_type,
            "error": error[:2000] if error else "",
            "stage": stage,
            "elapsed_ms": elapsed_ms,
        }
        if stack_trace:
            payload["stack_trace"] = stack_trace[:4000]
        if extra:
            payload.update(extra)
        if ticker:
            payload["ticker"] = ticker

        log_entry = {
            "cycle_id": cycle_id,
            "timestamp": ts,
            "level": "error",
            "step": f"error_{error_type}",
            "ticker": ticker,
            "payload": payload,
        }
        self._write_jsonl(self._cycle_path(cycle_id), log_entry)
        
        # --- WEBHOOK ALERT ---
        try:
            from app.services.logging.webhook_alerter import trigger_alert
            trigger_alert(f"Cycle Error: {error_type}", log_entry)
        except Exception:
            pass

    # ── Cycle Summary (NEW) ──────────────────────────────────────────────

    def log_cycle_summary(self, cycle_id: str, summary: dict):
        """Write the final cycle summary as the last entry in the JSONL.

        This is the single place to look for "what happened in this cycle?"
        """
        ts = datetime.now(timezone.utc).isoformat()
        log_entry = {
            "cycle_id": cycle_id,
            "timestamp": ts,
            "level": "info",
            "step": "cycle_summary",
            "ticker": "",
            "payload": summary,
        }
        self._write_jsonl(self._cycle_path(cycle_id), log_entry)

    # ── Crash Recovery Detection (NEW) ────────────────────────────────────

    def detect_and_log_crashed_cycles(self, max_age_hours: int = 48) -> list[dict]:
        """Scan cycle logs for incomplete cycles and write recovery entries.

        Called on startup to detect cycles that were interrupted by a
        container restart, OOM kill, or unhandled crash.

        Returns list of crashed cycle summaries for the boot log.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        crashed = []

        for path in sorted(self.CYCLE_DIR.glob("cycle-*.jsonl")):
            try:
                steps = set()
                first_entry = None
                last_entry = None
                tickers_seen = set()
                tickers_completed = set()
                tickers_errored = set()
                entry_count = 0

                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        entry_count += 1
                        step = entry.get("step", "")
                        steps.add(step)
                        ticker = entry.get("ticker", "") or entry.get("payload", {}).get("ticker", "")

                        if first_entry is None:
                            first_entry = entry
                        last_entry = entry

                        if step == "v2_start" and ticker:
                            tickers_seen.add(ticker)
                        elif step == "v2_pipeline_complete" and ticker:
                            tickers_completed.add(ticker)
                        elif step in ("v2_error", "error_thesis_timeout", "error_pipeline_crash") and ticker:
                            tickers_errored.add(ticker)

                if not first_entry:
                    continue

                # Skip if cycle already has a summary or was already recovered
                if "cycle_summary" in steps or "crash_recovery" in steps:
                    continue

                # Skip if cycle has a proper completion marker
                has_completion = "cycle_summary" in steps
                has_all_done = tickers_seen and tickers_seen == (tickers_completed | tickers_errored)
                if has_completion or has_all_done:
                    continue

                # Check age — don't flag cycles that might still be running
                ts_str = first_entry.get("timestamp", "")
                try:
                    cycle_start = datetime.fromisoformat(ts_str)
                    if cycle_start > cutoff:
                        continue  # Too recent
                except Exception:
                    continue

                # Determine last timestamp
                last_ts = last_entry.get("timestamp", ts_str) if last_entry else ts_str
                last_step = last_entry.get("step", "?") if last_entry else "?"
                last_ticker = (
                    last_entry.get("ticker", "")
                    or last_entry.get("payload", {}).get("ticker", "")
                    if last_entry else ""
                )

                cycle_id = first_entry.get("cycle_id", path.stem)
                tickers_abandoned = tickers_seen - tickers_completed - tickers_errored

                crash_info = {
                    "cycle_id": cycle_id,
                    "started_at": ts_str,
                    "last_activity_at": last_ts,
                    "last_step": last_step,
                    "last_ticker": last_ticker,
                    "total_tickers": len(tickers_seen),
                    "completed": len(tickers_completed),
                    "errored": len(tickers_errored),
                    "abandoned": sorted(tickers_abandoned),
                    "total_log_entries": entry_count,
                    "steps_seen": sorted(steps),
                }

                # Write the crash recovery entry to the SAME log file
                recovery_entry = {
                    "cycle_id": cycle_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "level": "critical",
                    "step": "crash_recovery",
                    "ticker": "",
                    "payload": {
                        "message": (
                            f"CRASH DETECTED: Cycle {cycle_id} was interrupted. "
                            f"Last activity was '{last_step}' for {last_ticker} at {last_ts}. "
                            f"{len(tickers_abandoned)}/{len(tickers_seen)} tickers abandoned: "
                            f"{', '.join(sorted(tickers_abandoned)[:10])}"
                        ),
                        **crash_info,
                    },
                }
                self._write_jsonl(path, recovery_entry)
                crashed.append(crash_info)

                # --- WEBHOOK ALERT ---
                try:
                    from app.services.logging.webhook_alerter import trigger_alert
                    trigger_alert(f"Crash Recovery: Cycle {cycle_id} interrupted", recovery_entry)
                except Exception:
                    pass

                logger.warning(
                    "[LogManager] CRASH RECOVERY: %s — last step '%s' for %s, "
                    "%d/%d tickers abandoned",
                    cycle_id, last_step, last_ticker,
                    len(tickers_abandoned), len(tickers_seen),
                )
            except Exception as e:
                logger.debug("[LogManager] Error scanning %s: %s", path.name, e)

        return crashed

    # ── Read-back (NEW) ──────────────────────────────────────────────────

    def list_all_cycles(self) -> list[dict]:
        """List all available cycles based on JSONL files in the cycles directory."""
        cycles = []
        try:
            if not self.cycles_dir.exists():
                return cycles
            
            for path in self.cycles_dir.glob("*.jsonl"):
                cycle_id = path.stem
                stat = path.stat()
                cycles.append({
                    "cycle_id": cycle_id,
                    "last_modified": stat.st_mtime,
                    "size_bytes": stat.st_size
                })
            # Sort newest first
            cycles.sort(key=lambda x: x["last_modified"], reverse=True)
        except Exception as e:
            logger.error("[LogManager] Failed to list cycles: %s", e)
            
        return cycles

    def get_cycle_log(self, cycle_id: str) -> list[dict]:
        """Read all entries from a cycle's JSONL file.

        Returns a list of dicts, one per log entry, ordered by timestamp.
        Useful for debugging a specific cycle.
        """
        path = self._cycle_path(cycle_id)
        if not path.exists():
            return []

        entries = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.debug("[LogManager] Failed to read %s: %s", path.name, e)

        return entries

    def get_cycle_errors(self, cycle_id: str) -> list[dict]:
        """Read only error/critical entries from a cycle log."""
        entries = self.get_cycle_log(cycle_id)
        return [e for e in entries if e.get("level") in ("error", "critical")]

    def get_cycle_stats(self, cycle_id: str) -> dict:
        """Compute quick stats from a cycle log without reading the full thing."""
        entries = self.get_cycle_log(cycle_id)
        if not entries:
            return {"cycle_id": cycle_id, "status": "not_found"}

        steps = set()
        tickers = set()
        errors = 0
        for e in entries:
            steps.add(e.get("step", ""))
            t = e.get("ticker", "") or e.get("payload", {}).get("ticker", "")
            if t:
                tickers.add(t)
            if e.get("level") in ("error", "critical"):
                errors += 1

        first_ts = entries[0].get("timestamp", "")
        last_ts = entries[-1].get("timestamp", "")
        has_summary = "cycle_summary" in steps
        has_crash = "crash_recovery" in steps

        return {
            "cycle_id": cycle_id,
            "status": "complete" if has_summary else ("crashed" if has_crash else "incomplete"),
            "started_at": first_ts,
            "last_activity_at": last_ts,
            "total_entries": len(entries),
            "tickers": len(tickers),
            "errors": errors,
            "steps": sorted(steps),
        }

    # ── Legacy Methods (preserved) ───────────────────────────────────────

    def log_ab_result(
        self,
        cycle_id: str,
        ab_chosen: str,
        v1_result: dict,
        v2_result: dict,
        context: dict = None,
    ):
        """Append an A/B benchmark result to logs/ab_results/ab_{cycle_id}.jsonl"""
        ts = datetime.now(timezone.utc).isoformat()
        log_entry = {
            "cycle_id": cycle_id,
            "timestamp": ts,
            "ab_chosen": ab_chosen,
            "v1_result": v1_result,
            "v2_result": v2_result,
            "context": context or {},
        }
        file_path = self.AB_DIR / f"ab_{cycle_id}.jsonl"
        self._write_jsonl(file_path, log_entry)

    def detect_abandoned_cycles(self, max_age_hours: int = 24) -> list[dict]:
        """Scan v2 cycle logs and identify abandoned cycles.

        An abandoned cycle has v2_start but neither v2_pipeline_complete
        nor v2_error. Returns a list of dicts with cycle_id, ticker,
        start_time, and last_step for monitoring.
        """
        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        abandoned = []

        for path in sorted(self.CYCLE_DIR.glob("*.jsonl")):
            if path.name.startswith("ab_"):
                continue  # Skip A/B result files

            steps = set()
            first_entry = None
            last_entry = None
            ticker = "?"

            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        steps.add(entry.get("step", ""))
                        if first_entry is None:
                            first_entry = entry
                        last_entry = entry
                        if entry.get("payload", {}).get("ticker"):
                            ticker = entry["payload"]["ticker"]
            except Exception:
                continue

            if not first_entry:
                continue

            # Skip if cycle completed or has error logged
            if "v2_pipeline_complete" in steps or "v2_error" in steps:
                continue

            # Only flag if cycle is old enough (not still running)
            ts_str = first_entry.get("timestamp", "")
            try:
                cycle_start = datetime.fromisoformat(ts_str)
                if cycle_start > cutoff:
                    continue  # Too recent, might still be running
            except Exception:
                continue

            abandoned.append({
                "cycle_id": first_entry.get("cycle_id", path.stem),
                "ticker": ticker,
                "start_time": ts_str,
                "last_step": last_entry.get("step", "?") if last_entry else "?",
                "file": str(path),
            })

        return abandoned

    # ── Log Rotation / Cleanup (NEW) ─────────────────────────────────────

    def cleanup_old_logs(self, max_age_days: int = 14) -> dict:
        """Delete old cycle JSONL files and debate audit files.

        Keeps logs from the last `max_age_days` days.
        Returns a summary of what was cleaned up.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        removed = {"cycle_logs": 0, "audit_logs": 0, "bytes_freed": 0}

        # 1. Clean old cycle JSONL files
        for path in sorted(self.CYCLE_DIR.glob("cycle-*.jsonl")):
            try:
                # Read first line to get cycle start timestamp
                with open(path, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                if not first_line:
                    continue
                entry = json.loads(first_line)
                ts_str = entry.get("timestamp", "")
                cycle_start = datetime.fromisoformat(ts_str)
                if cycle_start < cutoff:
                    size = path.stat().st_size
                    path.unlink()
                    removed["cycle_logs"] += 1
                    removed["bytes_freed"] += size
            except Exception:
                continue

        # 2. Clean old debate audit files
        audit_dir = self.BASE_DIR / "audit"
        if audit_dir.exists():
            for path in sorted(audit_dir.glob("debate_audit_*.jsonl")):
                try:
                    mtime = datetime.fromtimestamp(
                        path.stat().st_mtime, tz=timezone.utc
                    )
                    if mtime < cutoff:
                        size = path.stat().st_size
                        path.unlink()
                        removed["audit_logs"] += 1
                        removed["bytes_freed"] += size
                except Exception:
                    continue

        if removed["cycle_logs"] or removed["audit_logs"]:
            logger.info(
                "[LogManager] Cleanup: removed %d cycle logs + %d audit logs "
                "(freed %.1f KB)",
                removed["cycle_logs"],
                removed["audit_logs"],
                removed["bytes_freed"] / 1024,
            )

        return removed


log_manager = LogManager()
