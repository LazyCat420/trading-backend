import pytest
from app.pipeline.analysis.swarm_consensus import run_swarm_pipeline

@pytest.fixture
def mock_gather():
    from unittest.mock import patch
    with patch("app.pipeline.analysis.swarm_consensus.gather_data_parallel") as m:
        m.return_value = {"technical": "", "fundamental": "", "macro": ""}
        yield m

@pytest.fixture
def mock_predict():
    from unittest.mock import patch
    with patch("app.pipeline.analysis.swarm_consensus.generate_predictions") as m:
        m.return_value = {
            "quant_26B": {"action": "BUY", "confidence": 80},
            "macro_35B": {"action": "BUY", "confidence": 75},
            "cio_120B": {"action": "BUY", "confidence": 90}
        }
        yield m

@pytest.fixture
def mock_debate():
    from unittest.mock import patch
    with patch("app.pipeline.analysis.swarm_consensus.debate_full_participation") as m:
        m.return_value = {
            "action": "BUY",
            "confidence": 85,
            "rationale": "Strong buy",
            "dissenting_model": "None",
            "method": "swarm_v2_consensus",
            "debate_rounds": 2
        }
        yield m

@pytest.fixture
def mock_scorecard():
    from unittest.mock import patch
    with patch("app.pipeline.analysis.swarm_consensus.log_to_scorecard") as m:
        yield m

@pytest.mark.asyncio
async def test_j01_j02_j03_swarm_output_schema(mock_gather, mock_predict, mock_debate, mock_scorecard):
    """
    J-01: Output always contains action
    J-02: Output always contains confidence as int
    J-03: Output always contains integrity_status
    """
    result = await run_swarm_pipeline("AAPL")
    
    assert "action" in result
    assert result["action"] in ["BUY", "SELL", "HOLD", "PASS"]
    
    assert "confidence" in result
    assert isinstance(result["confidence"], int)
    assert 0 <= result["confidence"] <= 100
    
    assert "integrity_status" in result
    assert result["integrity_status"] in ["HIGH", "MEDIUM", "LOW_INTEGRITY"]

@pytest.mark.asyncio
async def test_j04_low_integrity_on_errors(mock_gather, mock_predict, mock_debate, mock_scorecard):
    """
    J-04: LOW_INTEGRITY is set when >= 2 specialist agents returned errors
    """
    mock_predict.return_value = {
        "quant_26B": {"raw": "Error: Timeout", "model_id": "unknown"},
        "macro_35B": {"raw": "Error: Connection", "model_id": "unknown"},
        "cio_120B": {"action": "BUY", "confidence": 90}
    }
    
    result = await run_swarm_pipeline("AAPL")
    
    assert result["integrity_status"] == "LOW_INTEGRITY"

@pytest.mark.asyncio
async def test_j07_all_fail_returns_hold_low_integrity(mock_gather, mock_predict, mock_debate, mock_scorecard):
    """
    J-07: When ALL specialist agents fail, judge returns HOLD with LOW_INTEGRITY
    """
    mock_predict.return_value = {
        "quant_26B": {"raw": "Error: Timeout", "model_id": "unknown"},
        "macro_35B": {"raw": "Error: Connection", "model_id": "unknown"},
        "cio_120B": {"raw": "Error: Bad Gateway", "model_id": "unknown"}
    }
    
    result = await run_swarm_pipeline("AAPL")
    
    assert result["action"] == "HOLD"
    assert result["integrity_status"] == "LOW_INTEGRITY"

@pytest.mark.asyncio
async def test_j05_scorecard_weights_sum_to_1():
    """
    J-05: scorecard weights sum to 1.0
    """
    from app.pipeline.analysis.swarm_scorecard import AGENT_WEIGHTS
    
    total_weight = sum(AGENT_WEIGHTS.values())
    assert abs(total_weight - 1.0) < 0.001

@pytest.mark.asyncio
async def test_j06_output_persisted_to_thesis_store(mock_gather, mock_predict, mock_debate, mock_scorecard):
    """
    J-06: Judge output persisted to thesis_store
    """
    await run_swarm_pipeline("AAPL")
    mock_scorecard.assert_called()
