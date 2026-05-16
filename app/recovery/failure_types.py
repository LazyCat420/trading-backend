"""
Failure Types — Shared types for the CORAL recovery system.

Defines the failure taxonomy and data structures used by both
the resilience decorator (app.utils.resilience) and the recovery
engine (app.recovery.engine).
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

# Re-export FailureType from resilience (single source of truth)
from app.utils.resilience import FailureType


class RecoveryAction(str, Enum):
    """What the recovery engine tells the caller to do next.

    RETRY           — Simple retry (already handled by resilience decorator).
    RETRY_DEGRADED  — Retry with simplified context/prompt.
    SKIP            — Give up on this step, continue with remaining work.
    """

    RETRY = "retry"
    RETRY_DEGRADED = "retry_degraded"
    SKIP = "skip"
    REPAIR = "repair"


@dataclass
class FailureEvent:
    """Structured failure report passed to the RecoveryEngine.

    Contains all context needed to decide the recovery action.
    """

    failure_type: FailureType
    agent_name: str
    step_name: str
    ticker: str = ""
    cycle_id: str = ""
    attempt: int = 1
    max_attempts: int = 3
    exception_type: str = ""
    exception_msg: str = ""
    context_snapshot: dict = field(default_factory=dict)
    timestamp: float = field(
        default_factory=lambda: datetime.now(timezone.utc).timestamp()
    )

    @property
    def key(self) -> str:
        """Unique key for counting repeated failures of the same step."""
        return f"{self.agent_name}:{self.step_name}:{self.ticker}"


@dataclass
class RecoveryResult:
    """What the RecoveryEngine returns after handling a failure.

    Contains the action to take and any additional context the
    caller needs (e.g., which fallback agent to use for REROUTE).
    """

    action: RecoveryAction
    failure_event: FailureEvent
    reason: str = ""
    degraded_context: dict | None = None
