import pytest
from app.agents.agent_loop import ToolCallScorecard

def test_tool_call_scorecard_empty_structures():
    """
    Regression lock for TICKET 1: ToolCallScorecard empty-structure detection.
    Ensure that the scorecard correctly flags empty JSON structures, lists, and common error strings.
    """
    scorecard = ToolCallScorecard()
    
    # Test valid JSON list
    scorecard.record('[{"data": "value"}]')
    assert scorecard.succeeded == 1
    assert scorecard.empty == 0
    assert scorecard.errored == 0
    assert scorecard.consecutive_empty == 0
    
    # Test empty JSON list
    scorecard.record('[]')
    assert scorecard.empty == 1
    assert scorecard.succeeded == 1  # From previous test
    assert scorecard.consecutive_empty == 1
    
    # Test empty JSON dict
    scorecard.record('{}')
    assert scorecard.empty == 2
    assert scorecard.consecutive_empty == 2
    
    # Test dictionary with empty values
    scorecard.record('{"result": null, "data": []}')
    assert scorecard.empty == 3
    assert scorecard.consecutive_empty == 3
    
    # Test valid string
    scorecard.record('This is a successful response.')
    assert scorecard.succeeded == 2
    assert scorecard.empty == 3
    assert scorecard.consecutive_empty == 0
    
    # Test error string
    scorecard.record('Exception: rate limit exceeded')
    assert scorecard.errored == 1
    assert scorecard.consecutive_empty == 1
    
    # Test quality ratio
    # total made = 6
    # succeeded = 2
    # expected quality_ratio = 2/6 = 0.333...
    assert abs(scorecard.quality_ratio - (2/6)) < 0.001
