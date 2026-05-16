"""
Recovery Engine — Central failure handler for CORAL self-healing.

Receives FailureEvents and decides the appropriate RecoveryAction based
on the failure taxonomy, per-cycle failure counters, and agent registry.

Architecture:
    resilience.py (detects failure) → FailureEvent → RecoveryEngine.handle()
                                                        ↓
                                              RecoveryResult (action + context)
                                                        ↓
                                    caller executes: retry / reroute / skip

The engine is a singleton — one instance per process, reset at cycle start.

Usage:
    from app.recovery import recovery_engine

    result = recovery_engine.handle(FailureEvent(
        failure_type=FailureType.TRANSIENT,
        agent_name="sentiment_agent",
        step_name="analyze",
        ticker="NVDA",
    ))

    if result.action == RecoveryAction.RETRY:
        pass
"""

import logging
from collections import defaultdict

from app.recovery.failure_types import (
    FailureType,
    FailureEvent,
    RecoveryAction,
    RecoveryResult,
)

logger = logging.getLogger(__name__)

# After this many failures of the same step in one cycle,
# auto-promote to FATAL regardless of the original classification.
# This prevents infinite retry/reroute loops.
MAX_SAME_STEP_FAILURES = 3


class RecoveryEngine:
    """Centralized failure classification and recovery routing.

    Maintains per-cycle failure counters to prevent infinite loops.
    Emits structured events via PipelineService.emit() for monitoring.
    """

    def __init__(self):
        # Per-cycle failure counter: key → count
        self._failure_counter: dict[str, int] = defaultdict(int)
        # History of all failure events (ring buffer for this cycle)
        self._history: list[FailureEvent] = []
        self._cycle_id: str = ""

    def reset_cycle(self, cycle_id: str = ""):
        """Clear all counters at the start of a new cycle.

        Called by pipeline_service.py at cycle start.
        """
        self._failure_counter.clear()
        self._history.clear()
        self._cycle_id = cycle_id
        logger.info("[RECOVERY] Engine reset for cycle %s", cycle_id)

    def handle(self, event: FailureEvent) -> RecoveryResult:
        """Main entry point: classify failure and decide recovery action.

        Args:
            event: Structured failure event with all context.

        Returns:
            RecoveryResult with the action to take and any needed context.
        """
        # Track this failure
        self._failure_counter[event.key] += 1
        self._history.append(event)
        count = self._failure_counter[event.key]

        # ── Circuit breaker: too many failures of the same step ──
        if count >= MAX_SAME_STEP_FAILURES:
            logger.warning(
                "[RECOVERY] %s has failed %d times this cycle — forcing SKIP",
                event.key,
                count,
            )
            result = RecoveryResult(
                action=RecoveryAction.SKIP,
                failure_event=event,
                reason=f"Circuit breaker: {count} failures of {event.key}",
            )
            self._emit_recovery_event(event, result)
            return result

        # ── Route based on failure type ──
        if event.failure_type == FailureType.TRANSIENT:
            result = self._handle_transient(event)
        elif event.failure_type == FailureType.DEGRADED:
            result = self._handle_degraded(event)
        else:  # FATAL
            result = self._handle_fatal(event)

        self._emit_recovery_event(event, result)
        return result

    def _handle_transient(self, event: FailureEvent) -> RecoveryResult:
        """TRANSIENT: network blip, timeout, rate limit → RETRY.

        The resilience decorator already handles the actual retry with backoff.
        We just confirm the action and log it.
        """
        logger.info(
            "[RECOVERY] TRANSIENT %s (attempt %d/%d): %s — will retry",
            event.agent_name,
            event.attempt,
            event.max_attempts,
            event.exception_type,
        )
        return RecoveryResult(
            action=RecoveryAction.RETRY,
            failure_event=event,
            reason=f"Transient {event.exception_type}, retry with backoff",
        )

    def _handle_degraded(self, event: FailureEvent) -> RecoveryResult:
        """DEGRADED: LLM output was empty/invalid → RETRY_DEGRADED or REPAIR.

        Signals the caller to retry with a simplified prompt or stripped context.
        If it's a JSON parse error, we trigger REPAIR.
        """
        if event.step_name == "json_parse" or event.exception_type == "JSONDecodeError":
            logger.warning(
                "[RECOVERY] DEGRADED %s.%s for %s: %s — triggering REPAIR for invalid JSON",
                event.agent_name,
                event.step_name,
                event.ticker or "N/A",
                event.exception_msg[:100],
            )
            return RecoveryResult(
                action=RecoveryAction.REPAIR,
                failure_event=event,
                reason=f"Invalid JSON: {event.exception_type}. Triggering in-flight repair prompt.",
            )

        logger.warning(
            "[RECOVERY] DEGRADED %s.%s for %s: %s — will retry with reduced context",
            event.agent_name,
            event.step_name,
            event.ticker or "N/A",
            event.exception_msg[:100],
        )
        return RecoveryResult(
            action=RecoveryAction.RETRY_DEGRADED,
            failure_event=event,
            reason=f"Degraded output: {event.exception_type}. Retry with simplified context.",
            degraded_context={"strategy": "strip_to_price_only"},
        )

    def _handle_fatal(self, event: FailureEvent) -> RecoveryResult:
        """FATAL: unrecoverable error → SKIP and log everything."""
        logger.error(
            "[RECOVERY] FATAL %s.%s for %s: %s — skipping",
            event.agent_name,
            event.step_name,
            event.ticker or "N/A",
            event.exception_msg[:200],
        )
        return RecoveryResult(
            action=RecoveryAction.SKIP,
            failure_event=event,
            reason=f"Fatal: {event.exception_type}: {event.exception_msg[:100]}",
        )

    def _emit_recovery_event(self, event: FailureEvent, result: RecoveryResult):
        """Emit a structured event to PipelineService for dashboard visibility.

        Non-blocking — if PipelineService is not available, we just log.
        """
        try:
            from app.services.pipeline_service import PipelineService

            PipelineService.emit(
                "recovery",
                f"recovery_{event.agent_name}_{event.step_name}",
                f"Recovery: {event.agent_name}.{event.step_name} "
                f"[{event.failure_type.value}] → {result.action.value}",
                status="warning" if result.action != RecoveryAction.SKIP else "error",
                data={
                    "failure_type": event.failure_type.value,
                    "action": result.action.value,
                    "agent": event.agent_name,
                    "step": event.step_name,
                    "ticker": event.ticker,
                    "attempt": event.attempt,
                    "reason": result.reason,
                    "error_type": event.exception_type,
                    "error_msg": event.exception_msg[:100],
                },
            )
        except Exception:
            pass  # Monitoring must never break the recovery path

    # ── Introspection ──────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return current cycle's failure stats for the monitoring dashboard."""
        total = len(self._history)
        by_type = {}
        for event in self._history:
            t = event.failure_type.value
            by_type[t] = by_type.get(t, 0) + 1

        by_agent = {}
        for event in self._history:
            by_agent[event.agent_name] = by_agent.get(event.agent_name, 0) + 1

        return {
            "cycle_id": self._cycle_id,
            "total_failures": total,
            "by_type": by_type,
            "by_agent": by_agent,
            "circuit_breakers_tripped": sum(
                1
                for count in self._failure_counter.values()
                if count >= MAX_SAME_STEP_FAILURES
            ),
            "active_failure_counters": dict(self._failure_counter),
        }

    def get_history(self, limit: int = 20) -> list[dict]:
        """Return recent failure events for debugging."""
        events = self._history[-limit:]
        return [
            {
                "key": e.key,
                "failure_type": e.failure_type.value,
                "agent": e.agent_name,
                "step": e.step_name,
                "ticker": e.ticker,
                "attempt": e.attempt,
                "error": e.exception_msg[:100],
                "timestamp": e.timestamp,
            }
            for e in events
        ]


# Singleton
recovery_engine = RecoveryEngine()
