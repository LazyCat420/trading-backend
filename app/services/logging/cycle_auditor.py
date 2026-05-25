"""
Cycle Auditor — Inline pipeline diagnostics for real-time cycle debugging.
Moved to app/services/logging/cycle_auditor.py as part of logging consolidation.
"""

import json
import logging
import time
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class CycleAuditor:
    """Stateless audit checkpoint emitter for pipeline diagnostics."""

    def _write(
        self,
        cycle_id: str,
        audit_type: str,
        phase: str = "",
        ticker: str = "",
        severity: str = "info",
        message: str = "",
        data: dict | None = None,
    ):
        """Write a single audit record. Fire-and-forget, never raises."""
        try:
            from app.db.connection import get_db

            row_id = f"aud_{uuid.uuid4().hex[:12]}"
            with get_db() as db:
                db.execute(
                    "INSERT INTO cycle_audit_log "
                    "(id, cycle_id, timestamp, audit_type, phase, ticker, severity, message, data) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)",
                    (
                        row_id,
                        cycle_id,
                        datetime.now(timezone.utc),
                        audit_type,
                        phase,
                        ticker,
                        severity,
                        message[:2000],
                        json.dumps(data or {}),
                    ),
                )
        except Exception as e:
            # Auditor must NEVER crash the pipeline
            logger.debug("[AUDITOR] Write failed (non-fatal): %s", e)

    # ── Phase Entry ──────────────────────────────────────────────────

    def phase_entry(
        self, cycle_id: str, phase_name: str, ticker_count: int = 0, **extra
    ):
        """Log the start of a pipeline phase."""
        self._write(
            cycle_id=cycle_id,
            audit_type="phase_entry",
            phase=phase_name,
            severity="info",
            message=f"Entering phase '{phase_name}' with {ticker_count} tickers",
            data={"ticker_count": ticker_count, "timestamp": time.monotonic(), **extra},
        )
        logger.info("[AUDITOR] Phase ENTRY: %s (%d tickers)", phase_name, ticker_count)

    # ── Phase Exit ───────────────────────────────────────────────────

    def phase_exit(
        self,
        cycle_id: str,
        phase_name: str,
        results_count: int = 0,
        errors_count: int = 0,
        elapsed_s: float = 0,
        **extra,
    ):
        """Log the end of a pipeline phase with success metrics."""
        total = results_count + errors_count
        success_rate = (results_count / total * 100) if total > 0 else 0
        severity = "info"
        if errors_count > 0 and success_rate < 50:
            severity = "critical"
        elif errors_count > 0:
            severity = "warning"

        self._write(
            cycle_id=cycle_id,
            audit_type="phase_exit",
            phase=phase_name,
            severity=severity,
            message=(
                f"Exiting phase '{phase_name}': {results_count} ok, "
                f"{errors_count} errors ({success_rate:.0f}% success) in {elapsed_s:.1f}s"
            ),
            data={
                "results_count": results_count,
                "errors_count": errors_count,
                "success_rate": round(success_rate, 1),
                "elapsed_s": round(elapsed_s, 2),
                **extra,
            },
        )
        logger.info(
            "[AUDITOR] Phase EXIT: %s — %d ok, %d errors (%.0f%%, %.1fs)",
            phase_name,
            results_count,
            errors_count,
            success_rate,
            elapsed_s,
        )

    # ── Per-Ticker Result Audit ──────────────────────────────────────

    def ticker_result(
        self, cycle_id: str, ticker: str, result: dict, elapsed_s: float = 0
    ):
        """Validate and log a single ticker's analysis result."""
        issues = []

        # Check for empty result
        if not result:
            issues.append("Result is empty/None")

        action = result.get("action", "")
        confidence = result.get("confidence")
        rationale = result.get("rationale", "")

        # Check for missing required keys
        if not action:
            issues.append("Missing 'action' key")
        elif action not in ("BUY", "SELL", "HOLD"):
            issues.append(f"Invalid action: '{action}'")

        if confidence is None:
            issues.append("Missing 'confidence' key")
        elif not isinstance(confidence, (int, float)):
            issues.append(f"Confidence is not numeric: {type(confidence).__name__}")
        elif confidence < 0 or confidence > 100:
            issues.append(f"Confidence out of range: {confidence}")

        if not rationale:
            issues.append("Empty rationale")

        # Check for __THINK__ marker contamination
        if "__THINK__" in str(rationale):
            issues.append("__THINK__ marker found in rationale (streaming leak)")

        # Check for error in result
        if result.get("error"):
            issues.append(f"Error in result: {result['error'][:200]}")

        severity = "info"
        if issues:
            severity = "warning" if len(issues) <= 2 else "critical"

        self._write(
            cycle_id=cycle_id,
            audit_type="ticker_result",
            phase="analyzing",
            ticker=ticker,
            severity=severity,
            message=(
                f"{ticker}: {action} @ {confidence}% "
                f"({elapsed_s:.1f}s)"
                + (f" — ISSUES: {'; '.join(issues)}" if issues else " — OK")
            ),
            data={
                "action": action,
                "confidence": confidence,
                "elapsed_s": round(elapsed_s, 2),
                "issues": issues,
                "has_error": bool(result.get("error")),
                "config_used": result.get("config_used", ""),
                "escalated": result.get("escalated", False),
                "total_tokens": result.get("total_tokens", 0),
            },
        )

        if issues:
            logger.warning(
                "[AUDITOR] %s: %d issues — %s", ticker, len(issues), "; ".join(issues)
            )
        return issues

    # ── LLM Response Audit ───────────────────────────────────────────

    def llm_response(
        self, cycle_id: str, ticker: str, agent_name: str, raw_response: str
    ):
        """Real-time check of vLLM output for common kill patterns."""
        issues = []
        severity = "info"

        if not raw_response:
            issues.append("Empty response from vLLM")
            severity = "critical"

        elif raw_response.startswith("__THINK__"):
            issues.append(
                "Response starts with __THINK__ marker (streaming leak into pipeline)"
            )
            severity = "critical"

        elif "__THINK__" in raw_response:
            issues.append("__THINK__ marker found mid-response")
            severity = "warning"

        # Check for unclosed <think> tag
        if "<think>" in raw_response and "</think>" not in raw_response:
            issues.append("Unclosed <think> tag — response may contain raw CoT")
            severity = "warning"

        # Check if response doesn't start with { (non-JSON)
        stripped = raw_response.strip()
        if (
            stripped
            and not stripped.startswith("{")
            and not stripped.startswith("<think>")
        ):
            # Could be valid if it's wrapped in markdown code block
            if "```" not in stripped[:50]:
                issues.append(f"Response doesn't start with JSON: '{stripped[:80]}...'")
                if severity == "info":
                    severity = "warning"

        if issues:
            self._write(
                cycle_id=cycle_id,
                audit_type="llm_response",
                phase="analyzing",
                ticker=ticker,
                severity=severity,
                message=f"{ticker}/{agent_name}: {'; '.join(issues)}",
                data={
                    "agent_name": agent_name,
                    "issues": issues,
                    "response_length": len(raw_response),
                    "response_preview": raw_response[:300],
                },
            )
            logger.warning(
                "[AUDITOR] LLM response issue for %s/%s: %s",
                ticker,
                agent_name,
                "; ".join(issues),
            )

        return issues

    # ── Anomaly Detection ────────────────────────────────────────────

    def anomaly(
        self,
        cycle_id: str,
        phase: str,
        message: str,
        ticker: str = "",
        data: dict | None = None,
    ):
        """Log an unexpected condition detected during the cycle."""
        self._write(
            cycle_id=cycle_id,
            audit_type="anomaly",
            phase=phase,
            ticker=ticker,
            severity="critical",
            message=message,
            data=data,
        )
        logger.error("[AUDITOR] ANOMALY in %s: %s", phase, message)

    # ── Retrieve Audit Trail ─────────────────────────────────────────

    def get_audit_trail(self, cycle_id: str) -> list[dict]:
        """Retrieve all audit records for a cycle, ordered by timestamp."""
        try:
            from app.db.connection import get_db

            with get_db() as db:
                rows = db.execute(
                    "SELECT id, timestamp, audit_type, phase, ticker, severity, message, data "
                    "FROM cycle_audit_log WHERE cycle_id = %s "
                    "ORDER BY timestamp ASC",
                    (cycle_id,),
                ).fetchall()
                return [
                    {
                        "id": r[0],
                        "timestamp": str(r[1]),
                        "audit_type": r[2],
                        "phase": r[3],
                        "ticker": r[4],
                        "severity": r[5],
                        "message": r[6],
                        "data": json.loads(r[7])
                        if isinstance(r[7], str)
                        else (r[7] or {}),
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.error("[AUDITOR] Failed to retrieve audit trail: %s", e)
            return []

    def get_anomalies(self, cycle_id: str) -> list[dict]:
        """Get only critical/warning audit records for a cycle."""
        trail = self.get_audit_trail(cycle_id)
        return [r for r in trail if r["severity"] in ("warning", "critical")]


# Module-level singleton
auditor = CycleAuditor()
