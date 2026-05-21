"""
Integration tests for the Evolution Router's graduated circuit breaker
and decision issue routing.

Validates that:
1. Circuit breaker uses graduated backoff (not permanent blocking)
2. Decision quality issues are routed to constitution_amendment debates
3. Backoff resets after 12 consecutive rejections
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timezone, timedelta


class TestGraduatedCircuitBreaker:
    """Test the graduated exponential backoff circuit breaker."""

    def _make_router(self):
        from app.pipeline.analysis.evolution_router import EvolutionRouter
        return EvolutionRouter()

    @patch("app.pipeline.analysis.evolution_router.get_db")
    def test_no_history_allows_debate(self, mock_db):
        """No prior fixes = circuit closed, debate allowed."""
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []

        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_cursor
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)

        router = self._make_router()
        assert router._is_circuit_open("prompt", "test_agent") is False

    @patch("app.pipeline.analysis.evolution_router.get_db")
    def test_two_rejections_allows_debate(self, mock_db):
        """Only 2 consecutive rejections (< threshold of 3) = debate allowed."""
        now = datetime.now(timezone.utc)
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("rejected", now),
            ("rejected", now - timedelta(hours=1)),
        ]

        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_cursor
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)

        router = self._make_router()
        assert router._is_circuit_open("prompt", "test_agent") is False

    @patch("app.pipeline.analysis.evolution_router.get_db")
    def test_three_rejections_triggers_backoff(self, mock_db):
        """3 consecutive rejections should trigger backoff (skip 2 cycles)."""
        now = datetime.now(timezone.utc)
        reject_rows = [
            ("rejected", now - timedelta(hours=i))
            for i in range(3)
        ]

        fix_cursor = MagicMock()
        fix_cursor.fetchall.return_value = reject_rows

        # The cycles_since query should return 0 (no cycles passed yet)
        cycles_cursor = MagicMock()
        cycles_cursor.fetchone.return_value = (0,)

        call_count = 0
        mock_conn = MagicMock()
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return fix_cursor
            return cycles_cursor

        mock_conn.execute.side_effect = side_effect
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)

        router = self._make_router()
        assert router._is_circuit_open("prompt", "test_agent") is True

    @patch("app.pipeline.analysis.evolution_router.get_db")
    def test_backoff_expires_after_enough_cycles(self, mock_db):
        """After 2 cycles pass (tier 1 backoff), debate should be allowed again."""
        now = datetime.now(timezone.utc)
        reject_rows = [
            ("rejected", now - timedelta(hours=i))
            for i in range(3)
        ]

        fix_cursor = MagicMock()
        fix_cursor.fetchall.return_value = reject_rows

        # 2 cycles have passed since the last rejection
        cycles_cursor = MagicMock()
        cycles_cursor.fetchone.return_value = (2,)

        call_count = 0
        mock_conn = MagicMock()
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return fix_cursor
            return cycles_cursor

        mock_conn.execute.side_effect = side_effect
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)

        router = self._make_router()
        assert router._is_circuit_open("prompt", "test_agent") is False

    @patch("app.pipeline.analysis.evolution_router.get_db")
    def test_twelve_rejections_resets_backoff(self, mock_db):
        """After 12+ consecutive rejections, the backoff resets (fresh start)."""
        now = datetime.now(timezone.utc)
        reject_rows = [
            ("rejected", now - timedelta(hours=i))
            for i in range(12)
        ]

        fix_cursor = MagicMock()
        fix_cursor.fetchall.return_value = reject_rows

        mock_conn = MagicMock()
        mock_conn.execute.return_value = fix_cursor
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)

        router = self._make_router()
        # 12 rejections = reset, should allow debate
        assert router._is_circuit_open("prompt", "test_agent") is False

    @patch("app.pipeline.analysis.evolution_router.get_db")
    def test_non_rejection_breaks_chain(self, mock_db):
        """A 'pending' or 'approved' fix breaks the consecutive rejection chain."""
        now = datetime.now(timezone.utc)
        # 2 rejections, then a success, then 2 more rejections
        rows = [
            ("rejected", now),
            ("rejected", now - timedelta(hours=1)),
            ("approved", now - timedelta(hours=2)),  # Breaks chain
            ("rejected", now - timedelta(hours=3)),
            ("rejected", now - timedelta(hours=4)),
        ]

        fix_cursor = MagicMock()
        fix_cursor.fetchall.return_value = rows

        mock_conn = MagicMock()
        mock_conn.execute.return_value = fix_cursor
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)

        router = self._make_router()
        # Only 2 consecutive rejections (chain broken by 'approved'), should allow
        assert router._is_circuit_open("prompt", "test_agent") is False

    def test_backoff_tiers_defined(self):
        """Verify the backoff tiers are correctly configured."""
        router = self._make_router()
        assert router.BACKOFF_TIERS == {3: 2, 6: 4, 9: 8}
        assert router.MAX_CONSECUTIVE_REJECTIONS == 3


class TestDecisionIssueRouting:
    """Test that decision quality issues are now routed to constitution debates."""

    @pytest.mark.asyncio
    @patch("app.pipeline.analysis.evolution_router.council")
    @patch("app.pipeline.analysis.evolution_router.get_db")
    async def test_critical_decision_issues_spawn_debate(self, mock_db, mock_council):
        """Critical decision issues should trigger a constitution_amendment debate."""
        from app.pipeline.analysis.evolution_router import EvolutionRouter

        # Mock DB: report with critical decision issues
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (
            "[]",  # data_gaps
            '[{"issue": "Low win rate: 35%", "severity": "critical"}]',  # decision_issues
            "[]",  # llm_issues
        )
        mock_cursor.fetchall.return_value = []  # No prior fixes (circuit breaker check)

        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_cursor
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)

        # Mock council debate
        mock_council.run_debate = AsyncMock(return_value={"status": "rejected"})

        router = EvolutionRouter()
        await router.run_router("test_cycle_123")

        # Verify a debate was spawned for decision issues
        mock_council.run_debate.assert_called()
        call_args = mock_council.run_debate.call_args
        assert call_args.kwargs.get("target_type") == "constitution_amendment"
        assert call_args.kwargs.get("target_name") == "decision_quality"

    @pytest.mark.asyncio
    @patch("app.pipeline.analysis.evolution_router.council")
    @patch("app.pipeline.analysis.evolution_router.get_db")
    async def test_info_only_issues_skip_debate(self, mock_db, mock_council):
        """Info-level decision issues should NOT trigger a debate."""
        from app.pipeline.analysis.evolution_router import EvolutionRouter

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (
            "[]",  # data_gaps
            '[{"issue": "Uniform confidence (45-55)", "severity": "info"}]',  # decision_issues
            "[]",  # llm_issues
        )
        mock_cursor.fetchall.return_value = []

        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_cursor
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)

        mock_council.run_debate = AsyncMock()

        router = EvolutionRouter()
        await router.run_router("test_cycle_456")

        # No debate should have been spawned
        mock_council.run_debate.assert_not_called()
