"""
Unit tests for the outcome-based decision quality scoring formula.

Tests the new `_audit_decisions()` function which scores based on actual
trade outcomes (WIN/LOSS/FLAT) rather than the broken action-ratio formula.

Unit tests use mocks — no NAS DB connection needed.
Integration tests use the NAS DB via conftest fixtures.
"""

import pytest
from unittest.mock import patch, MagicMock
from contextlib import contextmanager


# ── Helper to mock get_db for autoresearch module ──

def _mock_get_db_factory(confidence_rows=None, outcome_rows=None):
    """Build a mock get_db that returns canned data for the two queries
    _audit_decisions makes: (1) confidence query, (2) outcomes query.
    """
    call_count = 0

    @contextmanager
    def fake_get_db():
        nonlocal call_count
        conn = MagicMock()

        def execute_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            cursor = MagicMock()
            if call_count == 1:
                # First query: confidence from analysis_results
                cursor.fetchall.return_value = confidence_rows or []
            else:
                # Second query: resolved outcomes from decision_outcomes
                cursor.fetchall.return_value = outcome_rows or []
            return cursor

        conn.execute.side_effect = execute_side_effect
        yield conn

    return fake_get_db


class TestDecisionScoringFormula:
    """Test the decision quality scoring algorithm edge cases."""

    def _make_summary(self, buy=0, sell=0, hold=0):
        return {"buy_count": buy, "sell_count": sell, "hold_count": hold}

    def test_zero_decisions_returns_zero(self):
        """No decisions at all should score 0 with a critical issue."""
        with patch("app.pipeline.analysis.autoresearch.get_db", _mock_get_db_factory()):
            from app.pipeline.analysis.autoresearch import _audit_decisions
            result = _audit_decisions("test_cycle", self._make_summary(0, 0, 0))

        assert result["score"] == 0
        assert any(i["severity"] == "critical" for i in result["issues"])
        assert any("No decisions" in i["issue"] for i in result["issues"])

    def test_cold_start_defaults_to_neutral(self):
        """With < 3 resolved trades, score should default to ~0.5 (not 0.23)."""
        confs = [(60,), (70,), (45,)]
        outcomes = [("BUY", 70, 2.5, "WIN")]  # Only 1 trade — cold start

        with patch("app.pipeline.analysis.autoresearch.get_db", _mock_get_db_factory(confs, outcomes)):
            from app.pipeline.analysis.autoresearch import _audit_decisions
            result = _audit_decisions("test_cycle", self._make_summary(2, 1, 8))

        assert result["score"] >= 0.4, f"Cold start score {result['score']} too low"
        assert result["score"] <= 0.6, f"Cold start score {result['score']} too high"
        assert result.get("outcome_stats", {}).get("scoring_method") in ("cold_start", "fallback_error")

    def test_all_hold_mild_penalty_in_cold_start(self):
        """100% HOLD cycle should get ~0.4 in cold start (mild, not 0%)."""
        confs = [(50,), (45,), (55,)]
        outcomes = []  # No resolved trades

        with patch("app.pipeline.analysis.autoresearch.get_db", _mock_get_db_factory(confs, outcomes)):
            from app.pipeline.analysis.autoresearch import _audit_decisions
            result = _audit_decisions("test_cycle", self._make_summary(0, 0, 10))

        assert result["score"] >= 0.3, f"All-HOLD score {result['score']} is too punishing"
        assert result["score"] <= 0.5

    def test_high_win_rate_scores_well(self):
        """60% win rate with good risk mgmt should score high."""
        confs = [(70,), (65,), (80,), (55,), (75,)]
        # 3 wins, 2 losses → 60% win rate
        outcomes = [
            ("BUY", 80, 5.0, "WIN"),
            ("BUY", 70, 3.0, "WIN"),
            ("BUY", 75, 4.0, "WIN"),
            ("BUY", 55, -2.0, "LOSS"),
            ("BUY", 60, -1.5, "LOSS"),
        ]

        with patch("app.pipeline.analysis.autoresearch.get_db", _mock_get_db_factory(confs, outcomes)):
            from app.pipeline.analysis.autoresearch import _audit_decisions
            result = _audit_decisions("test_cycle", self._make_summary(5, 0, 6))

        assert result["score"] >= 0.55, f"60% win rate scored {result['score']}, should be >= 0.55"
        stats = result.get("outcome_stats", {})
        assert stats.get("scoring_method") == "outcome_based"
        assert stats.get("win_rate") == pytest.approx(0.6, abs=0.01)

    def test_low_win_rate_scores_poorly(self):
        """20% win rate should score low and flag critical issue."""
        confs = [(70,), (65,), (80,), (55,), (75,)]
        # 1 win, 4 losses → 20% win rate
        outcomes = [
            ("BUY", 80, 2.0, "WIN"),
            ("BUY", 70, -3.0, "LOSS"),
            ("BUY", 75, -4.0, "LOSS"),
            ("BUY", 55, -2.0, "LOSS"),
            ("BUY", 60, -1.5, "LOSS"),
        ]

        with patch("app.pipeline.analysis.autoresearch.get_db", _mock_get_db_factory(confs, outcomes)):
            from app.pipeline.analysis.autoresearch import _audit_decisions
            result = _audit_decisions("test_cycle", self._make_summary(5, 0, 6))

        assert result["score"] < 0.4, f"20% win rate scored {result['score']}, should be < 0.4"
        assert any("win rate" in i["issue"].lower() for i in result["issues"])

    def test_outcome_stats_always_present(self):
        """The result dict should always include outcome_stats key."""
        with patch("app.pipeline.analysis.autoresearch.get_db", _mock_get_db_factory()):
            from app.pipeline.analysis.autoresearch import _audit_decisions
            result = _audit_decisions("test_cycle", self._make_summary(1, 0, 5))

        assert "outcome_stats" in result
        assert "scoring_method" in result["outcome_stats"]

    def test_db_error_falls_back_gracefully(self):
        """If DB query fails, score should fallback to 0.5 not crash."""
        @contextmanager
        def broken_db():
            conn = MagicMock()
            conn.execute.side_effect = Exception("Connection refused")
            yield conn

        with patch("app.pipeline.analysis.autoresearch.get_db", broken_db):
            from app.pipeline.analysis.autoresearch import _audit_decisions
            result = _audit_decisions("test_cycle", self._make_summary(2, 1, 8))

        assert result["score"] == 0.5
        assert result["outcome_stats"]["scoring_method"] == "fallback_error"


@pytest.mark.integration
class TestScoringIntegration:
    """Integration tests using the NAS database.

    These use the `patch_real_get_db` fixture from conftest.py which
    points to the NAS at 10.0.0.16:5433.
    """

    @pytest.fixture(autouse=True)
    def _skip_if_no_nas(self, real_test_db_engine):
        """Skip if NAS DB is not reachable."""
        if not real_test_db_engine:
            pytest.skip("NAS database not available")

    def test_outcome_count_matches_reality(self, patch_real_get_db):
        """Verify resolved trade count matches what's in the NAS DB."""
        actual_count = patch_real_get_db.execute(
            """
            SELECT COUNT(*) FROM decision_outcomes
            WHERE resolved_at IS NOT NULL
              AND outcome != 'CANCELED'
              AND resolved_at > CURRENT_TIMESTAMP - INTERVAL '30 days'
            """
        ).fetchone()[0]

        from app.pipeline.analysis.autoresearch import _audit_decisions
        result = _audit_decisions("integration_test", {"buy_count": 1, "sell_count": 0, "hold_count": 5})

        stats = result.get("outcome_stats", {})
        if actual_count >= 3:
            assert stats.get("scoring_method") == "outcome_based"
            assert stats.get("total_resolved") == actual_count
        else:
            assert stats.get("scoring_method") in ("cold_start", "fallback_error")

    def test_win_rate_matches_manual_sql(self, patch_real_get_db):
        """Verify computed win rate matches direct SQL query."""
        rows = patch_real_get_db.execute(
            """
            SELECT outcome FROM decision_outcomes
            WHERE resolved_at IS NOT NULL
              AND outcome != 'CANCELED'
              AND resolved_at > CURRENT_TIMESTAMP - INTERVAL '30 days'
            """
        ).fetchall()

        if len(rows) < 3:
            pytest.skip("Not enough resolved trades for win rate test")

        manual_wins = sum(1 for r in rows if r[0] == "WIN")
        manual_rate = manual_wins / len(rows)

        from app.pipeline.analysis.autoresearch import _audit_decisions
        result = _audit_decisions("integration_test", {"buy_count": 1, "sell_count": 0, "hold_count": 5})
        computed_rate = result.get("outcome_stats", {}).get("win_rate", -1)

        assert abs(computed_rate - manual_rate) < 0.01, (
            f"Win rate mismatch: computed={computed_rate}, manual={manual_rate}"
        )
