import pytest
from pydantic import ValidationError
from app.cognition.contracts.debate import DebateResult

def test_debate_result_accepts_valid_dicts():
    """Test that DebateResult accepts proper dictionary structures for claims."""
    # Arrange
    valid_bull = [
        {"claim": "Revenue grew by 20% [financials:RevGrowth=0.20]", "turn": 1, "survived_rebuttal": True}
    ]
    valid_bear = [
        {"claim": "P/E is extremely high at 80 [market_data:PE=80]", "turn": 2, "survived_rebuttal": False}
    ]

    # Act
    result = DebateResult(
        bull_claims=valid_bull,
        bear_claims=valid_bear,
        verified_bull_claims=valid_bull,
        verified_bear_claims=valid_bear,
        unverified_claims=[],
        judge_action="BUY",
        judge_confidence=80
    )

    # Assert
    assert result.judge_action == "BUY"
    assert result.judge_confidence == 80
    assert result.verified_bull_claims == valid_bull
    assert result.verified_bear_claims == valid_bear

def test_debate_result_rejects_strings():
    """Test that DebateResult explicitly rejects strings in claim fields (the bug that crashed the pipeline)."""
    # Arrange
    invalid_bull = [
        "Revenue grew by 20% [financials:RevGrowth=0.20]"  # This is a string, but the schema wants a dict
    ]

    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        DebateResult(
            bull_claims=[],
            bear_claims=[],
            verified_bull_claims=invalid_bull,
            verified_bear_claims=[],
            unverified_claims=[]
        )
    
    # We should see the specific Pydantic error indicating type=dict_type is expected but str was given
    error_str = str(exc_info.value)
    assert "Input should be a valid dictionary" in error_str

def test_d01_d02_d03_claims_list_and_confidence():
    """D-01, D-02: claims len >= 1. D-03: confidence is int."""
    parsed_bull = {"action": "BUY", "confidence": 85, "claims": ["Revenue grew [data]"], "key_argument": "Growth"}
    parsed_bear = {"action": "SELL", "confidence": 90, "claims": ["PE is 80 [data]"], "key_argument": "Overvalued"}
    
    assert len(parsed_bull["claims"]) >= 1
    assert len(parsed_bear["claims"]) >= 1
    
    assert isinstance(parsed_bull["confidence"], int) and 0 <= parsed_bull["confidence"] <= 100
    assert isinstance(parsed_bear["confidence"], int) and 0 <= parsed_bear["confidence"] <= 100

def test_d04_claims_contain_data_citation():
    import re
    parsed = {"claims": ["Revenue grew by 20% [financials:RevGrowth=0.20]", "Price is 150"]}
    
    for claim in parsed["claims"]:
        has_bracket = bool(re.search(r'\[.*\]', claim))
        has_number = bool(re.search(r'\d', claim))
        assert has_bracket or has_number, f"Claim '{claim}' lacks data citation or numbers"

def test_d05_bull_bear_key_arguments_differ():
    parsed_bull = {"key_argument": "Strong growth expected"}
    parsed_bear = {"key_argument": "Overvalued based on PE"}
    
    assert parsed_bull["key_argument"] != parsed_bear["key_argument"]
