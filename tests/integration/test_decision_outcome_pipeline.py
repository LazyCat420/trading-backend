"""
Integration tests for the AutoResearch decision-outcome pipeline.

Tests the complete flow:
  1. Record decisions via outcome_tracker
  2. Resolve outcomes with actual PnL
  3. Verify _audit_decisions() scores correctly from DB ground truth
  4. Verify the reflection prompt receives outcome data
  5. Verify the evolution router routes decision issues

Uses the NAS test database via conftest fixtures.
"""

import pytest
from unittest.mock import patch, AsyncMock
from contextlib import contextmanager
import uuid
import datetime


@contextmanager
def _redirect_get_db(real_db):
    """Helper to redirect all get_db calls to the real test DB."""
    @contextmanager
    def fake_get_db():
        yield real_db
    yield fake_get_db


class TestOutcomeTrackerIntegration:
    """Verify decisions are recorded and resolved in the NAS database."""

    @pytest.mark.asyncio
    async def test_record_and_resolve_buy(self, real_db):
        """Record a BUY decision, then resolve it as a WIN."""
        with _redirect_get_db(real_db) as fake_db:
            with patch("app.pipeline.analysis.outcome_tracker.get_db", fake_db):
                from app.pipeline.analysis.outcome_tracker import record_decision, resolve_outcome

                # Seed a price history entry for the ticker (ignore if exists from previous run)
                real_db.execute(
                    "INSERT INTO price_history (ticker, date, open, high, low, close, volume, source) "
                    "VALUES ('TEST_WIN', '2026-05-01', 100, 105, 99, 100, 10000, 'test') "
                    "ON CONFLICT DO NOTHING"
                )

                # Record a BUY decision
                outcome_id = record_decision(
                    cycle_id="test_cycle_win",
                    ticker="TEST_WIN",
                    action="BUY",
                    confidence=75,
                    entry_price=100.0,
                    lesson="Testing positive outcome",
                )
                assert outcome_id is not None

                # Verify it's in the DB
                row = real_db.execute(
                    "SELECT ticker, action, confidence, entry_price, outcome FROM decision_outcomes WHERE id = %s",
                    [outcome_id],
                ).fetchone()
                assert row[0] == "TEST_WIN"
                assert row[1] == "BUY"
                assert row[2] == 75
                assert row[3] == 100.0
                assert row[4] is None  # Not yet resolved

                # Resolve: exit at $110 → WIN
                result = resolve_outcome("TEST_WIN", exit_price=110.0)
                assert result is not None
                assert result["outcome"] == "WIN"
                assert result["pnl_pct"] == 10.0

                # Verify DB was updated
                resolved = real_db.execute(
                    "SELECT outcome, pnl_pct, exit_price, resolved_at FROM decision_outcomes WHERE id = %s",
                    [outcome_id],
                ).fetchone()
                assert resolved[0] == "WIN"
                assert resolved[1] == 10.0
                assert resolved[2] == 110.0
                assert resolved[3] is not None  # resolved_at set

    @pytest.mark.asyncio
    async def test_resolve_loss(self, real_db):
        """Record a BUY, resolve as LOSS."""
        with _redirect_get_db(real_db) as fake_db:
            with patch("app.pipeline.analysis.outcome_tracker.get_db", fake_db):
                from app.pipeline.analysis.outcome_tracker import record_decision, resolve_outcome

                record_decision(
                    cycle_id="test_cycle_loss",
                    ticker="TEST_LOSS",
                    action="BUY",
                    confidence=60,
                    entry_price=100.0,
                )

                result = resolve_outcome("TEST_LOSS", exit_price=95.0)
                assert result is not None
                assert result["outcome"] == "LOSS"
                assert result["pnl_pct"] == -5.0

    @pytest.mark.asyncio
    async def test_hold_not_recorded(self, real_db):
        """HOLD decisions should not be recorded in decision_outcomes."""
        with _redirect_get_db(real_db) as fake_db:
            with patch("app.pipeline.analysis.outcome_tracker.get_db", fake_db):
                from app.pipeline.analysis.outcome_tracker import record_decision

                result = record_decision(
                    cycle_id="test_cycle_hold",
                    ticker="TEST_HOLD",
                    action="HOLD",
                    confidence=50,
                )
                assert result is None

    @pytest.mark.asyncio
    async def test_cancel_outcome(self, real_db):
        """Canceled trades should be marked CANCELED with 0% PnL."""
        with _redirect_get_db(real_db) as fake_db:
            with patch("app.pipeline.analysis.outcome_tracker.get_db", fake_db):
                from app.pipeline.analysis.outcome_tracker import record_decision, cancel_outcome

                record_decision(
                    cycle_id="test_cycle_cancel",
                    ticker="TEST_CANCEL",
                    action="BUY",
                    confidence=80,
                    entry_price=50.0,
                )

                canceled = cancel_outcome("TEST_CANCEL", reason="Portfolio gate blocked")
                assert canceled is True

                row = real_db.execute(
                    "SELECT outcome, pnl_pct FROM decision_outcomes WHERE ticker = 'TEST_CANCEL'"
                ).fetchone()
                assert row[0] == "CANCELED"
                assert row[1] == 0.0


class TestScoringFromRealOutcomes:
    """Verify _audit_decisions scores correctly using seeded decision_outcomes."""

    @pytest.mark.asyncio
    async def test_outcome_based_scoring_with_seeded_data(self, real_db):
        """Seed 5 trades (3W/2L), verify outcome-based scoring is triggered."""
        now = datetime.datetime.now(datetime.UTC)

        # Clean slate — remove any leftover data from prior runs
        real_db.execute("DELETE FROM decision_outcomes")
        real_db.execute("DELETE FROM analysis_results")

        with _redirect_get_db(real_db) as fake_db:
            # Seed 5 resolved trades: 3 wins, 2 losses
            trades = [
                ("BUY", 80, 5.0, "WIN"),
                ("BUY", 70, 3.0, "WIN"),
                ("BUY", 75, 4.0, "WIN"),
                ("BUY", 55, -2.0, "LOSS"),
                ("BUY", 60, -1.5, "LOSS"),
            ]
            for action, conf, pnl, outcome in trades:
                real_db.execute(
                    """INSERT INTO decision_outcomes
                    (id, cycle_id, ticker, action, confidence, entry_price, exit_price,
                     pnl_pct, outcome, created_at, resolved_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    [
                        str(uuid.uuid4()), "seed_cycle", "SEED",
                        action, conf, 100.0, 100 + pnl,
                        pnl, outcome, now, now,
                    ],
                )

            # Also seed analysis_results for confidence distribution check
            real_db.execute(
                "INSERT INTO analysis_results (id, cycle_id, bot_id, ticker, agent_name, confidence, created_at) "
                "VALUES (%s, 'score_test', 'bot1', 'SEED', 'test_agent', 70, %s)",
                [str(uuid.uuid4()), now],
            )

            with patch("app.pipeline.analysis.autoresearch.get_db", fake_db):
                from app.pipeline.analysis.autoresearch import _audit_decisions

                result = _audit_decisions("score_test", {
                    "buy_count": 3, "sell_count": 0, "hold_count": 8,
                })

                assert result["score"] > 0.4, f"Score {result['score']} too low for 60% win rate"
                stats = result.get("outcome_stats", {})
                assert stats.get("scoring_method") == "outcome_based"
                assert stats.get("wins") == 3
                assert stats.get("losses") == 2
                assert stats.get("win_rate") == pytest.approx(0.6, abs=0.01)

    @pytest.mark.asyncio
    async def test_empty_outcomes_triggers_cold_start(self, real_db):
        """No resolved trades → cold start scoring at 0.5."""
        # Clean slate
        real_db.execute("DELETE FROM decision_outcomes")
        real_db.execute("DELETE FROM analysis_results")

        with _redirect_get_db(real_db) as fake_db:
            with patch("app.pipeline.analysis.autoresearch.get_db", fake_db):
                from app.pipeline.analysis.autoresearch import _audit_decisions

                result = _audit_decisions("empty_test", {
                    "buy_count": 1, "sell_count": 0, "hold_count": 5,
                })

                assert result["score"] >= 0.4
                assert result["score"] <= 0.6
                stats = result.get("outcome_stats", {})
                assert stats.get("scoring_method") in ("cold_start", "fallback_error")


class TestReflectionPromptIntegration:
    """Verify outcome data flows through to the reflection LLM prompt."""

    @pytest.mark.asyncio
    async def test_reflection_prompt_includes_prediction_scorecard(self, real_db):
        """When outcome data exists, the reflection prompt should include WIN/LOSS stats."""
        now = datetime.datetime.now(datetime.UTC)

        with _redirect_get_db(real_db) as fake_db:
            # Seed trades
            for i, (pnl, outcome) in enumerate([(5.0, "WIN"), (3.0, "WIN"), (-2.0, "LOSS"), (-1.0, "LOSS")]):
                real_db.execute(
                    """INSERT INTO decision_outcomes
                    (id, cycle_id, ticker, action, confidence, entry_price, exit_price,
                     pnl_pct, outcome, created_at, resolved_at)
                    VALUES (%s, %s, %s, 'BUY', 70, 100, %s, %s, %s, %s, %s)""",
                    [str(uuid.uuid4()), "reflect_test", f"T{i}", 100 + pnl, pnl, outcome, now, now],
                )

            # Seed analysis_result
            real_db.execute(
                "INSERT INTO analysis_results (id, cycle_id, bot_id, ticker, agent_name, confidence, created_at) "
                "VALUES (%s, 'reflect_test', 'bot1', 'T0', 'agent', 70, %s)",
                [str(uuid.uuid4()), now],
            )

            # Capture the LLM prompt
            captured_prompts = []

            async def mock_chat(**kwargs):
                captured_prompts.append(kwargs.get("user", ""))
                return ('{"summary": "test", "recommendations": [], "system_health": "healthy"}', 100, 100)

            with patch("app.pipeline.analysis.autoresearch.get_db", fake_db), \
                 patch("app.services.vllm_client.llm") as mock_llm:
                mock_llm.chat = mock_chat

                from app.pipeline.analysis.autoresearch import _reflect, _audit_decisions

                dec_q = _audit_decisions("reflect_test", {
                    "buy_count": 2, "sell_count": 0, "hold_count": 4,
                })

                audit_bundle = {
                    "cycle_id": "reflect_test",
                    "tickers": ["T0", "T1"],
                    "data_quality": {"avg_score": 0.8, "gaps": []},
                    "decision_quality": dec_q,
                    "llm_analysis": {"score": 0.9, "total_calls": 10, "failed_calls": 0, "issues": []},
                    "performance": {"total_ms": 5000},
                    "recovery": {"total_failures": 0},
                    "schedule_health": {"active_count": 1, "avg_interval_hours": 4, "issues": []},
                    "execution_errors": [],
                }

                await _reflect(audit_bundle)

                # The prompt should include the prediction accuracy block
                assert len(captured_prompts) >= 1
                prompt = captured_prompts[0]
                assert "PREDICTION ACCURACY" in prompt
                assert "Win rate" in prompt


class TestEvolutionRouterDecisionRouting:
    """Verify that decision issues are routed to constitution_amendment debates."""

    @pytest.mark.asyncio
    async def test_critical_issues_routed_to_debate(self, real_db):
        """Critical decision issues in the autoresearch report should spawn debates."""
        import json

        with _redirect_get_db(real_db) as fake_db:
            # Seed an autoresearch report with critical decision issues
            report_id = str(uuid.uuid4())
            real_db.execute(
                """INSERT INTO autoresearch_reports
                (id, cycle_id, status, phase, decision_issues, data_gaps, llm_issues)
                VALUES (%s, %s, 'done', 'done', %s, '[]', '[]')""",
                [
                    report_id, "evo_test_cycle",
                    json.dumps([
                        {"issue": "Low win rate: 30%", "severity": "critical"},
                        {"issue": "Avg loss > avg win", "severity": "warning"},
                    ]),
                ],
            )

            debate_calls = []

            async def mock_debate(**kwargs):
                debate_calls.append(kwargs)
                return {"status": "rejected", "fix_id": None}

            with patch("app.pipeline.analysis.evolution_router.get_db", fake_db), \
                 patch("app.pipeline.analysis.evolution_router.council") as mock_council:
                mock_council.run_debate = mock_debate

                from app.pipeline.analysis.evolution_router import EvolutionRouter
                router = EvolutionRouter()
                await router.run_router("evo_test_cycle")

                # Verify a constitution_amendment debate was spawned
                assert len(debate_calls) >= 1
                call = debate_calls[0]
                assert call["target_type"] == "constitution_amendment"
                assert call["target_name"] == "decision_quality"
                assert "Low win rate" in call["issue_description"]
