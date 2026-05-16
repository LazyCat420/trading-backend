"""
Shared pipeline utilities -- callbacks and timing helpers.

Consolidates duplicated pipeline plumbing that was copy-pasted
across data_phase.py and decision_engine.py.

Usage:
    from app.utils.pipeline_utils import noop, elapsed_ms
"""

import time


def noop(*a, **kw):
    """No-op callback for when no emit function is provided.

    Previously duplicated in:
        - data_phase.py (_noop)
        - decision_engine.py (_noop)
    """
    pass


def elapsed_ms(start: float) -> int:
    """Calculate elapsed milliseconds since a monotonic start time.

    Replaces the repeated pattern:
        ms = int((time.monotonic() - t0) * 1000)

    which appeared ~25 times in data_phase.py alone.

    Args:
        start: Value from time.monotonic() at the start of timing

    Returns:
        Integer milliseconds elapsed
    """
    return int((time.monotonic() - start) * 1000)
