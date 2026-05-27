"""
Meta-Agent Judge Unit Tests — Verify prompt lifecycle management.

Tests:
  1. Disabled gate returns early
  2. No prompts with enough trades → no changes
  3. Low win-rate prompt gets benched
  4. High win-rate candidate gets promoted
  5. Active prompt already good — no changes
  6. Cap reached — skip generation
  7. record_prompt_outcome updates stats correctly
"""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from contextlib import contextmanager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ============================================================================
# Helpers
# ============================================================================

def _make_mock_cursor():
    """Create a fresh mock cursor with execute/fetchall/fetchone."""
    cursor = MagicMock()
    cursor.execute.return_value = cursor
    cursor.fetchone.return_value = None
    cursor.fetchall.return_value = []
    return cursor


def _make_get_db(cursor):
    """Create a get_db function that yields the given cursor."""
    @contextmanager
    def fake_get_db():
        yield cursor
    return fake_get_db


# ============================================================================
# TEST: Disabled gate
# ============================================================================

class TestDisabledGate:
    """When META_AGENT_ENABLED is False, the agent should return early."""

    @pytest.mark.asyncio
    async def test_disabled_returns_status_disabled(self):
        with patch("app.config.settings") as mock_settings:
            mock_settings.META_AGENT_ENABLED = False

            from app.agents.meta_agent_judge import run_meta_agent_judge
            result = await run_meta_agent_judge("test-cycle-1")

        assert result["status"] == "disabled"


# ============================================================================
# TEST: No prompts with enough trades
# ============================================================================

class TestNoEligiblePrompts:
    """When no prompts have enough trades, no changes should be made."""

    @pytest.mark.asyncio
    async def test_no_eligible_prompts_returns_zero(self):
        cursor = _make_mock_cursor()
        cursor.fetchall.return_value = []

        with patch("app.config.settings") as mock_settings, \
             patch("app.db.connection.get_db", _make_get_db(cursor)):
            mock_settings.META_AGENT_ENABLED = True
            mock_settings.MAX_ACTIVE_GENERATED_PROMPTS = 20

            from app.agents.meta_agent_judge import run_meta_agent_judge
            result = await run_meta_agent_judge("test-cycle-2")

        assert result["evaluated"] == 0
        assert result["benched"] == []
        assert result["promoted"] == []


# ============================================================================
# TEST: Bench underperformer
# ============================================================================

class TestBenchUnderperformer:
    """Active prompts with low win rate should be benched."""

    @pytest.mark.asyncio
    async def test_low_win_rate_gets_benched(self):
        cursor = _make_mock_cursor()

        # (id, sector, action_type, status, total_trades, wins, losses, win_rate, avg_pnl, generation)
        cursor.fetchall.side_effect = [
            [("pt-bad", "technology", "BUY", "active", 25, 8, 17, 0.32, -1.5, 1)],  # evaluate
            [],  # sector counts for generation
        ]
        cursor.fetchone.side_effect = [
            (5,),  # total active generated prompts
        ]

        with patch("app.config.settings") as mock_settings, \
             patch("app.db.connection.get_db", _make_get_db(cursor)):
            mock_settings.META_AGENT_ENABLED = True
            mock_settings.MAX_ACTIVE_GENERATED_PROMPTS = 20

            from app.agents.meta_agent_judge import run_meta_agent_judge
            result = await run_meta_agent_judge("test-cycle-3")

        assert len(result["benched"]) == 1
        assert result["benched"][0]["id"] == "pt-bad"
        assert result["benched"][0]["win_rate"] == 0.32


# ============================================================================
# TEST: Promote high performer
# ============================================================================

class TestPromoteCandidate:
    """Candidate prompts with high win rate should be promoted."""

    @pytest.mark.asyncio
    async def test_high_win_rate_candidate_gets_promoted(self):
        cursor = _make_mock_cursor()

        cursor.fetchall.side_effect = [
            [("pt-good", "energy", "BUY", "candidate", 22, 14, 8, 0.64, 3.2, 2)],  # evaluate
            [],  # sector counts
        ]
        cursor.fetchone.side_effect = [
            (3,),  # total active
        ]

        with patch("app.config.settings") as mock_settings, \
             patch("app.db.connection.get_db", _make_get_db(cursor)):
            mock_settings.META_AGENT_ENABLED = True
            mock_settings.MAX_ACTIVE_GENERATED_PROMPTS = 20

            from app.agents.meta_agent_judge import run_meta_agent_judge
            result = await run_meta_agent_judge("test-cycle-4")

        assert len(result["promoted"]) == 1
        assert result["promoted"][0]["id"] == "pt-good"
        assert result["promoted"][0]["win_rate"] == 0.64


# ============================================================================
# TEST: No changes for mediocre prompt
# ============================================================================

class TestNoChanges:
    """An active prompt with mediocre win rate (40-55%) should not be touched."""

    @pytest.mark.asyncio
    async def test_mediocre_prompt_not_benched_or_promoted(self):
        cursor = _make_mock_cursor()

        cursor.fetchall.side_effect = [
            [("pt-mid", "consumer", "BUY", "active", 30, 14, 16, 0.47, 0.5, 1)],
            [],  # sector counts
        ]
        cursor.fetchone.side_effect = [
            (2,),  # total active
        ]

        with patch("app.config.settings") as mock_settings, \
             patch("app.db.connection.get_db", _make_get_db(cursor)):
            mock_settings.META_AGENT_ENABLED = True
            mock_settings.MAX_ACTIVE_GENERATED_PROMPTS = 20

            from app.agents.meta_agent_judge import run_meta_agent_judge
            result = await run_meta_agent_judge("test-cycle-5")

        assert result["evaluated"] == 1
        assert len(result["benched"]) == 0
        assert len(result["promoted"]) == 0


# ============================================================================
# TEST: record_prompt_outcome
# ============================================================================

class TestRecordPromptOutcome:
    """record_prompt_outcome should update stats atomically."""

    def test_win_updates_stats(self):
        cursor = _make_mock_cursor()

        with patch("app.db.connection.get_db", _make_get_db(cursor)):
            from app.agents.meta_agent_judge import record_prompt_outcome
            record_prompt_outcome("pt-test", is_win=True, pnl_pct=5.0)

        cursor.execute.assert_called_once()
        call_args = cursor.execute.call_args
        assert "UPDATE prompt_templates" in call_args[0][0]

    def test_loss_updates_stats(self):
        cursor = _make_mock_cursor()

        with patch("app.db.connection.get_db", _make_get_db(cursor)):
            from app.agents.meta_agent_judge import record_prompt_outcome
            record_prompt_outcome("pt-test2", is_win=False, pnl_pct=-3.0)

        cursor.execute.assert_called_once()

    def test_db_failure_does_not_crash(self):
        cursor = _make_mock_cursor()
        cursor.execute.side_effect = Exception("DB down")

        with patch("app.db.connection.get_db", _make_get_db(cursor)):
            from app.agents.meta_agent_judge import record_prompt_outcome
            # Should not raise
            record_prompt_outcome("pt-test3", is_win=True, pnl_pct=2.0)
