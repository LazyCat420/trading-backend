"""
CORAL Recovery System — Self-healing failure handling.

Public API:
    from app.recovery import recovery_engine, FailureEvent, RecoveryAction, RecoveryResult

    result = recovery_engine.handle(FailureEvent(...))
"""

from app.recovery.failure_types import (
    FailureType,
    FailureEvent,
    RecoveryAction,
    RecoveryResult,
)
from app.recovery.engine import recovery_engine

__all__ = [
    "recovery_engine",
    "FailureType",
    "FailureEvent",
    "RecoveryAction",
    "RecoveryResult",
]
