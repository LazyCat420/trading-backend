import pytest

from app.cognition.debate.action_gate import gate_action, get_allowed_actions_str

def test_gate_action_not_held():
    """When not held, HOLD should be remapped to SELL."""
    assert gate_action("HOLD", held=False) == "SELL"
    assert gate_action("BUY", held=False) == "BUY"
    assert gate_action("SELL", held=False) == "SELL"
    assert gate_action("PASS", held=False) == "SELL"  # Invalid action remaps to default
    assert gate_action("UNKNOWN", held=False) == "SELL"
    
def test_gate_action_held():
    """When held, BUY should be remapped to HOLD."""
    assert gate_action("BUY", held=True) == "HOLD"
    assert gate_action("HOLD", held=True) == "HOLD"
    assert gate_action("SELL", held=True) == "SELL"
    assert gate_action("PASS", held=True) == "HOLD"  # Invalid action remaps to default
    assert gate_action("UNKNOWN", held=True) == "HOLD"

def test_get_allowed_actions_str():
    """Prompt string should exclude HOLD when not held."""
    assert get_allowed_actions_str(held=False) == "BUY|SELL"
    assert get_allowed_actions_str(held=True) == "HOLD|SELL"
