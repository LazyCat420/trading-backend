import pytest
from app.agents.progressive_summarizer import summarize_opponent_turn, compress_tool_research_block

def test_summarize_opponent_turn_short_text():
    text = "Short text."
    result = summarize_opponent_turn(text, max_chars=100)
    assert result == "Short text."

def test_summarize_opponent_turn_extracts_claims():
    text = '''This is a long response.
    Here is some extra text.
    "claims": [
        "Claim 1",
        "Claim 2"
    ]
    ''' * 100 # Make it long enough to exceed max_chars

    result = summarize_opponent_turn(text, side="bull", max_chars=500)
    
    assert "[BULL CLAIMS]" in result
    assert "1. Claim 1" in result
    assert "2. Claim 2" in result

def test_summarize_opponent_turn_extracts_key_argument_and_confidence():
    text = '''This is a long response.
    "key_argument": "Company is growing",
    "confidence": 85
    ''' * 100

    result = summarize_opponent_turn(text, side="bear", max_chars=500)
    
    assert "[KEY ARGUMENT] Company is growing" in result
    assert "[CONFIDENCE] 85/100" in result

def test_summarize_opponent_turn_extracts_cited_evidence():
    text = '''This is a long response.
    Revenue is up [financials:10M].
    Profit margin decreased [margin:5%].
    ''' * 100

    result = summarize_opponent_turn(text, side="bear", max_chars=500)
    
    assert "[BEAR CITED EVIDENCE]" in result
    assert "Revenue is up [financials:10M]." in result
    assert "Profit margin decreased [margin:5%]." in result

def test_summarize_opponent_turn_fallback_truncation():
    text = "A" * 1000
    result = summarize_opponent_turn(text, side="bull", max_chars=100)
    
    assert result.startswith("A" * 100)
    assert "... [bull output truncated from 1,000 chars]" in result

def test_compress_tool_research_block_empty():
    assert compress_tool_research_block([]) == "No tools used."

def test_compress_tool_research_block_short():
    history = [
        "### Tool Call: get_price()\nPrice is 100."
    ]
    result = compress_tool_research_block(history, max_total_chars=100)
    assert result == "[get_price] Price is 100."

def test_compress_tool_research_block_long_truncation():
    history = [
        "### Tool Call: get_financials()\n" + "A" * 1000
    ]
    result = compress_tool_research_block(history, max_total_chars=100)
    
    assert result.startswith("[get_financials] " + "A" * 80)
    assert "... [research block truncated]" in result

def test_compress_tool_research_block_multi_tool():
    history = [
        "### Tool Call: tool_1()\nResult 1",
        "### Tool Call: tool_2()\nResult 2"
    ]
    result = compress_tool_research_block(history, max_total_chars=500)
    assert "[tool_1] Result 1" in result
    assert "[tool_2] Result 2" in result

def test_ps04_summarize_opponent_turn_preserves_keywords():
    """PS-04: Keyword preservation."""
    text = "A" * 1000 + "\nTherefore we should BUY because of STRONG EARNINGS."
    result = summarize_opponent_turn(text, side="bull", max_chars=100)
    
    assert "[BULL KEYWORDS]" in result
    assert "BUY" in result
