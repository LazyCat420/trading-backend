"""
Pipeline Profiler — Phase-level timing instrumentation for cycle diagnosis.

Captures wall-clock time for every pipeline phase and produces:
- Structured JSON timing breakdown
- ASCII Gantt chart showing sequential vs parallel phases
- Database persistence for cross-cycle comparison

Usage:
    from app.monitoring.pipeline_profiler import profiler

    async with profiler.phase("global_collectors"):
        await run_collectors()

    # At end of cycle:
    report = profiler.get_report()
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class PhaseRecord:
    """Timing record for a single pipeline phase."""

    name: str
    started_at: float  # monotonic
    finished_at: float = 0.0  # monotonic
    wall_ms: int = 0
    status: str = "running"  # running | ok | error
    detail: str = ""
    parent: str = ""  # for nested phases
    ticker: str = ""  # for per-ticker phases

    def finish(self, status: str = "ok", detail: str = ""):
        self.finished_at = time.monotonic()
        self.wall_ms = int((self.finished_at - self.started_at) * 1000)
        self.status = status
        self.detail = detail

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "wall_ms": self.wall_ms,
            "wall_s": round(self.wall_ms / 1000, 2),
            "status": self.status,
            "detail": self.detail,
            "parent": self.parent,
            "ticker": self.ticker,
        }


class PipelineProfiler:
    """Singleton profiler that tracks timing for a single cycle."""

    def __init__(self):
        self._phases: list[PhaseRecord] = []
        self._cycle_id: str = ""
        self._cycle_start: float = 0.0
        self._cycle_end: float = 0.0
        self._active_phases: dict[str, PhaseRecord] = {}
        self._active_workers: dict[str, str] = {}
        self._max_concurrent_workers: int = 0
        self._lock = asyncio.Lock()
        # Previous cycle report for API access
        self._last_report: dict = {}

    def start_cycle(self, cycle_id: str):
        """Reset and start timing for a new cycle."""
        self._phases = []
        self._cycle_id = cycle_id
        self._cycle_start = time.monotonic()
        self._cycle_end = 0.0
        self._active_phases = {}
        logger.info("[PROFILER] Started timing for cycle %s", cycle_id)

    def end_cycle(self):
        """Finalize the cycle and build the report."""
        self._cycle_end = time.monotonic()
        total_ms = int((self._cycle_end - self._cycle_start) * 1000)
        logger.info(
            "[PROFILER] Cycle %s total: %dms (%.1fs)",
            self._cycle_id,
            total_ms,
            total_ms / 1000,
        )

        # Build final report and store it
        self._last_report = self._build_report()

        # Persist to database
        self._persist_to_db()

        # Log the ASCII Gantt
        gantt = self.gantt_chart()
        if gantt:
            logger.info("[PROFILER] Phase Timeline:\n%s", gantt)

        return self._last_report

    @asynccontextmanager
    async def phase(self, name: str, parent: str = "", ticker: str = ""):
        """Async context manager to time a pipeline phase."""
        record = PhaseRecord(
            name=name,
            started_at=time.monotonic(),
            parent=parent,
            ticker=ticker,
        )
        key = f"{name}:{ticker}" if ticker else name
        self._active_phases[key] = record

        try:
            yield record
            record.finish("ok")
        except asyncio.CancelledError:
            record.finish("cancelled")
            raise
        except Exception as e:
            record.finish("error", str(e)[:200])
            raise
        finally:
            self._active_phases.pop(key, None)
            self._phases.append(record)
            if record.wall_ms > 1000:  # Only log phases > 1s
                logger.info(
                    "[PROFILER] %s%s: %dms (%.1fs) [%s]",
                    f"{parent}/" if parent else "",
                    name,
                    record.wall_ms,
                    record.wall_ms / 1000,
                    record.status,
                )

    @contextmanager
    def phase_sync(self, name: str, parent: str = "", ticker: str = ""):
        """Sync context manager for non-async phases."""
        record = PhaseRecord(
            name=name,
            started_at=time.monotonic(),
            parent=parent,
            ticker=ticker,
        )
        key = f"{name}:{ticker}" if ticker else name
        self._active_phases[key] = record

        try:
            yield record
            record.finish("ok")
        except Exception as e:
            record.finish("error", str(e)[:200])
            raise
        finally:
            self._active_phases.pop(key, None)
            self._phases.append(record)

    def mark(self, name: str, detail: str = ""):
        """Record a zero-duration milestone marker."""
        now = time.monotonic()
        record = PhaseRecord(
            name=name,
            started_at=now,
            finished_at=now,
            wall_ms=0,
            status="marker",
            detail=detail,
        )
        self._phases.append(record)

    def log_active_worker(self, worker_id: str, task_name: str):
        """Register a worker as active performing a task."""
        self._active_workers[worker_id] = task_name
        current_count = len(self._active_workers)
        if current_count > self._max_concurrent_workers:
            self._max_concurrent_workers = current_count
        logger.info(
            f"[PROFILER] 🟢 {worker_id} started: {task_name} | Active: {current_count}"
        )

    def clear_active_worker(self, worker_id: str):
        """Unregister an active worker."""
        task_name = self._active_workers.pop(worker_id, "unknown")
        logger.info(
            f"[PROFILER] 🔴 {worker_id} finished: {task_name} | Active: {len(self._active_workers)}"
        )

    def get_active_workers(self) -> dict:
        """Get dictionary of currently active workers and their tasks."""
        return self._active_workers.copy()

    def _build_report(self) -> dict:
        """Build the structured timing report."""
        total_ms = int((self._cycle_end - self._cycle_start) * 1000)

        # Group phases
        top_level = [p for p in self._phases if not p.parent and not p.ticker]
        per_ticker = {}
        for p in self._phases:
            if p.ticker:
                per_ticker.setdefault(p.ticker, []).append(p)

        # Find the critical path (longest sequential chain)
        sorted_phases = sorted(top_level, key=lambda p: p.wall_ms, reverse=True)

        # Compute idle time (time not covered by any phase)
        # This reveals gaps where nothing is happening
        idle_ms = self._compute_idle_time(top_level)

        # Bottleneck identification
        bottlenecks = []
        for p in sorted_phases[:5]:
            pct = (p.wall_ms / total_ms * 100) if total_ms > 0 else 0
            if pct > 10:  # Only flag phases taking >10% of cycle time
                bottlenecks.append(
                    {
                        "phase": p.name,
                        "wall_ms": p.wall_ms,
                        "pct_of_total": round(pct, 1),
                        "status": p.status,
                    }
                )

        return {
            "cycle_id": self._cycle_id,
            "total_ms": total_ms,
            "total_s": round(total_ms / 1000, 1),
            "idle_ms": idle_ms,
            "idle_pct": round(idle_ms / total_ms * 100, 1) if total_ms > 0 else 0,
            "phase_count": len(self._phases),
            "max_concurrent_workers": self._max_concurrent_workers,
            "phases": [p.to_dict() for p in top_level],
            "bottlenecks": bottlenecks,
            "per_ticker_summary": {
                ticker: {
                    "total_ms": sum(p.wall_ms for p in phases),
                    "phase_count": len(phases),
                    "phases": [p.to_dict() for p in phases],
                }
                for ticker, phases in per_ticker.items()
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _compute_idle_time(self, phases: list[PhaseRecord]) -> int:
        """Compute time not covered by any phase (gaps = idle)."""
        if not phases or self._cycle_start == 0:
            return 0

        # Build intervals relative to cycle start
        intervals = []
        for p in phases:
            if p.started_at > 0 and p.finished_at > 0:
                s = p.started_at - self._cycle_start
                e = p.finished_at - self._cycle_start
                intervals.append((s, e))

        if not intervals:
            return 0

        # Merge overlapping intervals
        intervals.sort()
        merged = [intervals[0]]
        for s, e in intervals[1:]:
            if s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        # Total covered time
        covered = sum(e - s for s, e in merged)
        total = self._cycle_end - self._cycle_start
        idle = max(0, total - covered)
        return int(idle * 1000)

    def gantt_chart(self, width: int = 60) -> str:
        """Generate an ASCII Gantt chart of phase timings."""
        if not self._phases or self._cycle_start == 0:
            return ""

        total = self._cycle_end - self._cycle_start
        if total <= 0:
            return ""

        # Only show top-level phases (no per-ticker detail in Gantt)
        phases = [
            p for p in self._phases if not p.ticker and p.wall_ms > 500
        ]  # Skip sub-second phases
        phases.sort(key=lambda p: p.started_at)

        lines = []
        lines.append(f"{'Phase':<28} {'Time':>8}  {'Timeline'}")
        lines.append(f"{'─' * 28} {'─' * 8}  {'─' * width}")

        for p in phases:
            rel_start = (p.started_at - self._cycle_start) / total
            rel_end = (p.finished_at - self._cycle_start) / total
            bar_start = int(rel_start * width)
            bar_end = max(bar_start + 1, int(rel_end * width))

            bar = (
                " " * bar_start + "█" * (bar_end - bar_start) + " " * (width - bar_end)
            )
            time_str = f"{p.wall_ms / 1000:.1f}s"
            name = p.name[:28]
            status_char = (
                "✓" if p.status == "ok" else "✗" if p.status == "error" else "·"
            )

            lines.append(f"{name:<28} {time_str:>8}  |{bar}| {status_char}")

        # Add timeline scale
        total_s = total
        marks = [0, total_s * 0.25, total_s * 0.5, total_s * 0.75, total_s]
        scale = " " * 28 + " " * 10
        for i, m in enumerate(marks):
            pos = int(i * width / 4)
            scale += f"{m:.0f}s".ljust(width // 4) if i < 4 else f"{m:.0f}s"
        lines.append(f"{'':28} {'':8}  {'─' * (width + 2)}")
        lines.append(scale)

        return "\n".join(lines)

    def get_report(self) -> dict:
        """Get the current or last cycle report."""
        if self._cycle_end > 0:
            return self._last_report
        # Cycle still running — build a partial report
        if self._cycle_start > 0:
            self._cycle_end = time.monotonic()
            report = self._build_report()
            self._cycle_end = 0  # Reset since cycle isn't actually done
            return report
        return self._last_report or {"status": "no_data"}

    def get_active_phases(self) -> list[dict]:
        """Get currently running phases (for live monitoring)."""
        now = time.monotonic()
        return [
            {
                "name": r.name,
                "running_ms": int((now - r.started_at) * 1000),
                "ticker": r.ticker,
                "parent": r.parent,
            }
            for r in self._active_phases.values()
        ]

    def _persist_to_db(self):
        """Save the timing report to PostgreSQL for cross-cycle comparison."""
        try:
            from app.db.connection import get_db
            import json

            with get_db() as db:
                # Ensure table exists
                db.execute("""
                    CREATE TABLE IF NOT EXISTS pipeline_profiler (
                        id VARCHAR PRIMARY KEY,
                        cycle_id VARCHAR,
                        total_ms INTEGER,
                        idle_ms INTEGER,
                        phase_count INTEGER,
                        bottlenecks_json VARCHAR,
                        phases_json VARCHAR,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                report = self._last_report
                db.execute(
                    """
                    INSERT INTO pipeline_profiler
                    (id, cycle_id, total_ms, idle_ms, phase_count, bottlenecks_json, phases_json, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                    [
                        str(uuid.uuid4()),
                        report.get("cycle_id", ""),
                        report.get("total_ms", 0),
                        report.get("idle_ms", 0),
                        report.get("phase_count", 0),
                        json.dumps(report.get("bottlenecks", [])),
                        json.dumps(report.get("phases", [])),
                        datetime.now(timezone.utc).isoformat(),
                    ],
                )
                logger.info("[PROFILER] Timing data persisted to database")
        except Exception as e:
            logger.warning("[PROFILER] Failed to persist timing data: %s", e)

    def get_history(self, limit: int = 10) -> list[dict]:
        """Get timing reports from previous cycles."""
        try:
            from app.db.connection import get_db
            import json

            with get_db() as db:
                rows = db.execute(
                    """
                    SELECT cycle_id, total_ms, idle_ms, phase_count,
                           bottlenecks_json, phases_json, created_at
                    FROM pipeline_profiler
                    ORDER BY created_at DESC
                    LIMIT %s
                """,
                    [limit],
                ).fetchall()

                return [
                    {
                        "cycle_id": r[0],
                        "total_ms": r[1],
                        "total_s": round(r[1] / 1000, 1),
                        "idle_ms": r[2],
                        "idle_pct": round(r[2] / r[1] * 100, 1) if r[1] > 0 else 0,
                        "phase_count": r[3],
                        "bottlenecks": json.loads(r[4]) if r[4] else [],
                        "phases": json.loads(r[5]) if r[5] else [],
                        "created_at": str(r[6]),
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.warning("[PROFILER] Failed to load history: %s", e)
            return []


# Singleton
profiler = PipelineProfiler()
