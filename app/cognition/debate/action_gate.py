"""
Action Gate — Normalizes trading actions based on position state.

The bot should never produce HOLD for a ticker it doesn't own.
This module provides a single reusable function that enforces this
invariant at every layer of the pipeline.

Rules:
    Not held → HOLD is meaningless → remap to SELL (conservative: don't enter)
    Held → BUY is meaningless → remap to HOLD (already own it)
    PASS → conservative default based on position state
"""

import logging

logger = logging.getLogger(__name__)


VALID_ACTIONS_NOT_HELD = frozenset({"BUY", "SELL"})
VALID_ACTIONS_HELD = frozenset({"HOLD", "SELL", "BUY"})

# Conservative defaults: don't take the risky action on failure
_DEFAULT_NOT_HELD = "SELL"   # Don't enter uncertain positions
_DEFAULT_HELD = "HOLD"      # Don't exit without analysis


def gate_action(raw_action: str, held: bool) -> str:
    """Normalize a trading action based on position state.

    Args:
        raw_action: The raw action string from LLM output (BUY/SELL/HOLD/PASS).
        held: Whether the bot currently holds this position.

    Returns:
        A valid action string for the given position state.
    """
    action = raw_action.strip().upper()

    if held:
        if action in VALID_ACTIONS_HELD:
            return action
        logger.info("[GATE] Invalid action '%s' for held → HOLD", action)
        return _DEFAULT_HELD

    # Not held
    if action == "HOLD":
        logger.info("[GATE] Remapped HOLD → SELL (not held, conservative)")
        return "SELL"
    if action in VALID_ACTIONS_NOT_HELD:
        return action
    logger.info("[GATE] Invalid action '%s' for not-held → SELL", action)
    return _DEFAULT_NOT_HELD


def get_allowed_actions_str(held: bool) -> str:
    """Return the action options string for LLM prompts.

    Args:
        held: Whether the bot currently holds this position.

    Returns:
        A pipe-separated string of valid actions for prompt injection.
    """
    if held:
        return "BUY|HOLD|SELL"
    return "BUY|SELL"
