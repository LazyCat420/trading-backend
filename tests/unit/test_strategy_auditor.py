import pytest
from unittest.mock import MagicMock
from app.cognition.evaluation.strategy_auditor import compute_agent_metrics

def test_compute_agent_metrics_without_cycle_id():
    db = MagicMock()
    # Mock return value of db.execute().fetchall()
    # Each row is (model, agent_step, rc_json, evidence_json)
    db.execute.return_value.fetchall.return_value = [
        ("model-a", "step-1", '{"red_cards": 1}', '{"evidence": []}')
    ]
    
    metrics = compute_agent_metrics(db)
    
    db.execute.assert_called_once()
    called_query = db.execute.call_args[0][0]
    assert "WHERE e.cycle_id" not in called_query
    
    # Since cycle_id is None, no parameters should be passed (or an empty list)
    if len(db.execute.call_args[0]) > 1:
        called_args = db.execute.call_args[0][1]
        assert not called_args or len(called_args) == 0

def test_compute_agent_metrics_with_cycle_id():
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = [
        ("model-a", "step-1", '{"red_cards": 1}', '{"evidence": []}')
    ]
    
    metrics = compute_agent_metrics(db, cycle_id="cycle-123")
    
    db.execute.assert_called_once()
    called_query = db.execute.call_args[0][0]
    called_args = db.execute.call_args[0][1]
    
    assert "WHERE e.cycle_id = %s" in called_query
    assert called_args == ["cycle-123"]
