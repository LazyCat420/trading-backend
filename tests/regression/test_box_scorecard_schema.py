import pytest
from app.monitoring.box_scorecard import generate_box_scorecard
from app.db.connection import get_db

def test_box_scorecard_schema_compatibility(patch_get_db, mock_db, monkeypatch):
    """
    Regression lock for llm_audit_logs schema mismatch.
    The original bug was that box_scorecard.py was querying for a non-existent 'model' column
    in the llm_audit_logs table, leading to a SQL exception and no scorecard being generated.
    """
    from contextlib import contextmanager
    @contextmanager
    def mock_get_db_cm():
        yield mock_db
        
    monkeypatch.setattr("app.monitoring.box_scorecard.get_db", mock_get_db_cm)
    
    test_cycle_id = "test_schema_cycle"
    
    # Mock the return values for the database queries
    # First query: per-endpoint breakdown
    mock_db.fetchall.side_effect = [
        [
            ("test_endpoint", 1, 100, 50, 50, 500, 500, 500, 500, 10, 200.0, "test_model")
        ],
        [
            ("test_agent", "TEST", 500, "test_endpoint")
        ]
    ]
    
    # Second query: aggregate totals
    mock_db.fetchone.return_value = (1, 100, 500, 500, 10)
    
    scorecard = generate_box_scorecard(test_cycle_id)
    
    # It should successfully generate the scorecard without an exception
    assert scorecard is not None
    assert "test_endpoint" in scorecard
    assert scorecard["test_endpoint"]["total_tokens"] == 100
    assert scorecard["test_endpoint"]["prompt_tokens"] == 50
    assert scorecard["test_endpoint"]["completion_tokens"] == 50
    assert scorecard["test_endpoint"]["model"] == "test_model"
    assert scorecard["_aggregate"]["total_calls"] == 1

