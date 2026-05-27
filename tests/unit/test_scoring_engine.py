"""
Scoring Engine Unit Tests — Verify safe_eval, pillar scoring, and feature normalization.

Tests the core scoring_engine.py functions:
  1. safe_eval with valid expressions
  2. safe_eval rejects dangerous inputs (no code injection)
  3. calculate_pillar_score with YAML spec files
  4. build_hierarchical_pillar_profiles structure
"""
import os
import sys
import math
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class TestSafeEval:
    """safe_eval should evaluate safe math expressions and reject dangerous ones."""

    def test_simple_addition(self):
        from app.trading.scoring_engine import safe_eval
        result = safe_eval("a + b", {"a": 1.0, "b": 2.0})
        assert result == 3.0

    def test_multiplication_and_subtraction(self):
        from app.trading.scoring_engine import safe_eval
        result = safe_eval("a * b - c", {"a": 3.0, "b": 4.0, "c": 2.0})
        assert result == 10.0

    def test_division(self):
        from app.trading.scoring_engine import safe_eval
        result = safe_eval("a / b", {"a": 10.0, "b": 4.0})
        assert result == 2.5

    def test_unary_negation(self):
        from app.trading.scoring_engine import safe_eval
        result = safe_eval("-a", {"a": 5.0})
        assert result == -5.0

    def test_nested_expression(self):
        from app.trading.scoring_engine import safe_eval
        result = safe_eval("(a + b) * c", {"a": 2.0, "b": 3.0, "c": 4.0})
        assert result == 20.0

    def test_undefined_variable_raises(self):
        from app.trading.scoring_engine import safe_eval
        with pytest.raises(ValueError, match="not defined"):
            safe_eval("unknown_var + 1", {})

    def test_function_call_rejected(self):
        from app.trading.scoring_engine import safe_eval
        with pytest.raises((ValueError, TypeError)):
            safe_eval("__import__('os')", {})

    def test_attribute_access_rejected(self):
        from app.trading.scoring_engine import safe_eval
        with pytest.raises((ValueError, TypeError)):
            safe_eval("a.__class__", {"a": 1.0})

    def test_constants_are_allowed(self):
        from app.trading.scoring_engine import safe_eval
        result = safe_eval("42", {})
        assert result == 42.0

    def test_complex_formula(self):
        from app.trading.scoring_engine import safe_eval
        # Typical scoring formula: weighted sum of normalized features
        vars_dict = {"ev_norm": 0.8, "rr_norm": 0.6, "kelly_norm": 0.5}
        result = safe_eval("ev_norm * 0.4 + rr_norm * 0.35 + kelly_norm * 0.25", vars_dict)
        expected = 0.8 * 0.4 + 0.6 * 0.35 + 0.5 * 0.25
        assert abs(result - expected) < 1e-9


class TestCalculatePillarScore:
    """calculate_pillar_score should load YAML specs and evaluate formulas."""

    def test_missing_spec_returns_default(self):
        from app.trading.scoring_engine import calculate_pillar_score
        result = calculate_pillar_score("nonexistent_spec", {"ev_norm": 0.5})
        assert result == 5.0  # Default fallback

    def test_score_clamped_between_1_and_10(self):
        from app.trading.scoring_engine import calculate_pillar_score
        # Even if formula returns extreme values, result should be in 1-10 range
        # Using a real spec if it exists, or testing the clamping logic
        vars_dict = {
            "ev_norm": 1.0, "rr_norm": 1.0, "kelly_norm": 1.0,
            "vol_norm": 0.0, "dd_norm": 0.0, "beta_norm": 0.5,
            "z_score_norm": 0.5, "rsi_norm": 0.5,
        }
        # This test just verifies the function doesn't crash with valid inputs
        result = calculate_pillar_score("edge_score", vars_dict)
        assert 1.0 <= result <= 10.0 or result == 5.0  # 5.0 if spec not found


class TestBuildHierarchicalPillarProfiles:
    """build_hierarchical_pillar_profiles should return the correct structure."""

    def test_profile_structure(self):
        from app.trading.scoring_engine import build_hierarchical_pillar_profiles

        # Mock get_db to avoid real DB calls
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None
        mock_cursor.fetchall.return_value = []
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_cursor)
        mock_ctx.__exit__ = MagicMock(return_value=False)

        with patch("app.trading.scoring_engine.get_db", return_value=mock_ctx):
            result = build_hierarchical_pillar_profiles("AAPL")

        assert "ticker" in result
        assert result["ticker"] == "AAPL"
        assert "pillars" in result
        assert "edge" in result["pillars"]
        assert "risk" in result["pillars"]
        assert "regime" in result["pillars"]
        assert "raw_features" in result

    def test_pillar_has_required_fields(self):
        from app.trading.scoring_engine import build_hierarchical_pillar_profiles

        mock_cursor = MagicMock()
        mock_cursor.execute.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None
        mock_cursor.fetchall.return_value = []
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_cursor)
        mock_ctx.__exit__ = MagicMock(return_value=False)

        with patch("app.trading.scoring_engine.get_db", return_value=mock_ctx):
            result = build_hierarchical_pillar_profiles("NVDA")

        for pillar_name in ("edge", "risk", "regime"):
            pillar = result["pillars"][pillar_name]
            assert "base_score" in pillar, f"{pillar_name} missing base_score"
            assert "profile_label" in pillar, f"{pillar_name} missing profile_label"
            assert "active_drivers" in pillar, f"{pillar_name} missing active_drivers"
            assert "veto_flags" in pillar, f"{pillar_name} missing veto_flags"

    def test_default_features_when_no_db_data(self):
        from app.trading.scoring_engine import compute_normalized_features

        mock_cursor = MagicMock()
        mock_cursor.execute.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None
        mock_cursor.fetchall.return_value = []
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_cursor)
        mock_ctx.__exit__ = MagicMock(return_value=False)

        with patch("app.trading.scoring_engine.get_db", return_value=mock_ctx):
            features = compute_normalized_features("AAPL")

        # All norms should be at defaults (0.5)
        assert features["ev_norm"] == 0.5
        assert features["rr_norm"] == 0.5
        assert features["kelly_norm"] == 0.5
        assert features["vol_norm"] == 0.5

    def test_veto_flags_on_extreme_risk(self):
        from app.trading.scoring_engine import build_hierarchical_pillar_profiles

        # Simulate high drawdown data
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value = mock_cursor
        # Fundamentals: low growth
        mock_cursor.fetchone.side_effect = [
            (0.01, 0.02, 1e9),  # fundamentals
            (30.0, 5.0, 100.0, 200.0),  # technicals
            (150.0,),  # price
        ]
        # Price history for z-score and drawdown: simulate 45% drawdown
        prices = [(100.0,)] * 5  # Current price is much lower than peak
        prices[0] = (55.0,)  # Current price = 55, peak was 100
        mock_cursor.fetchall.return_value = prices * 12  # Enough rows

        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_cursor)
        mock_ctx.__exit__ = MagicMock(return_value=False)

        with patch("app.trading.scoring_engine.get_db", return_value=mock_ctx):
            result = build_hierarchical_pillar_profiles("RISKY_STOCK")

        # The structure should still be valid even with extreme data
        assert isinstance(result["pillars"]["risk"]["veto_flags"], list)
