"""
TDD tests for the V2 cognition pipeline runner.

These tests verify that execute_v2_pipeline actually works end-to-end
with properly mocked dependencies, rather than mocking the function itself.

Key bugs caught:
- is_highly_redundant NameError (was crashing ALL analysis)
- Missing parameter passing between phase4 workers and runner
"""

import pytest
import asyncio
from contextlib import ExitStack
from unittest.mock import patch, MagicMock, AsyncMock
from dataclasses import dataclass, field


# ── Minimal stubs ──

@dataclass
class FakeSourceQuality:
    teaser_artifact_risk: float = 0.0
    source_diversity: int = 3

@dataclass
class FakeFreshness:
    oldest_timestamp: None = None
    newest_timestamp: None = None

@dataclass
class FakeFact:
    field_name: str = "price"
    value: float = 150.0
    source: str = "yfinance"

@dataclass
class FakePacket:
    claims: list = field(default_factory=lambda: [{"text": "Revenue up"}])
    structured_facts: list = field(default_factory=lambda: [FakeFact()])
    source_summaries: list = field(default_factory=lambda: [{"source": "yf"}])
    missing_fields: list = field(default_factory=list)
    source_quality_summary: FakeSourceQuality = field(default_factory=FakeSourceQuality)
    freshness_summary: FakeFreshness = field(default_factory=FakeFreshness)

@dataclass
class FakeSufficiency:
    status: str = "sufficient"
    blockers: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

@dataclass
class FakeThesis:
    action: str = "BUY"
    confidence: int = 75
    rationale: str = "Strong fundamentals"
    core_claims: list = field(default_factory=lambda: ["Revenue growth"])
    weaknesses: list = field(default_factory=list)

@dataclass
class FakeDebateResult:
    judge_action: str = "BUY"
    judge_confidence: int = 80
    winning_side: str = "bull"
    integrity_status: str = "HIGH"
    judge_rationale: str = "Bull case stronger"
    key_deciding_factor: str = "Revenue growth"
    bull_claims: list = field(default_factory=lambda: ["Revenue up"])
    bear_claims: list = field(default_factory=lambda: ["Debt high"])
    verified_bull_claims: list = field(default_factory=lambda: ["Revenue up"])
    verified_bear_claims: list = field(default_factory=list)
    unverified_claims: list = field(default_factory=list)
    persona_outcomes: dict = field(default_factory=dict)
    transcript: str = ""
    total_tokens: int = 500


def _mock_runner_deps():
    """Return a dict of patch targets for execute_v2_pipeline."""
    mock_cc = MagicMock()
    mock_cc.wait_if_paused = AsyncMock()
    mock_cc.is_paused = False

    mock_ont = MagicMock()
    mock_ont.return_value.execute = AsyncMock(
        return_value={"ontology_nodes": [], "ontology_context": ""}
    )

    return {
        "app.pipeline.orchestration.cycle_control.cycle_control": mock_cc,
        "app.cycle.orchestration.cycle_control.cycle_control": mock_cc,
        "app.pipeline.data.data_completeness.check_and_fill": AsyncMock(
            return_value={"filled": []}
        ),
        "app.cognition.ontology.ontology_builder.OntologyBuilder": mock_ont,
        "app.cognition.evidence.packet_builder.build_evidence_packet": AsyncMock(
            return_value=FakePacket()
        ),
        "app.cognition.verification.sufficiency_gate.check_data_sufficiency": MagicMock(
            return_value=FakeSufficiency()
        ),
        "app.tools.portfolio_tools.get_position_context": MagicMock(
            return_value={"held": False}
        ),
        "app.cognition.orchestration.meta_orchestrator.MetaOrchestrator.orchestrate": AsyncMock(
            return_value=({"sentiment": "Bullish"}, 200)
        ),
        "app.cognition.debate.debate_coordinator.run_adversarial_debate": AsyncMock(
            return_value=FakeDebateResult()
        ),
        "app.cognition.debate.thesis_agent.generate_thesis": AsyncMock(
            return_value=(FakeThesis(), 300)
        ),
        "app.cognition.debate.action_gate.gate_action": MagicMock(
            side_effect=lambda a, h: a
        ),
        "app.pipeline.analysis.hallucination_checker.check_hallucinations": MagicMock(
            return_value={"rejected": False, "hallucinations": [], "rejection_reason": ""}
        ),
        "app.cognition.memory.reader.read_memories": MagicMock(return_value=[]),
        "app.cognition.memory.reader.read_procedural": MagicMock(return_value=[]),
        "app.cognition.memory.writer.write_episode": MagicMock(return_value="ep-1234"),
        "app.pipeline.analysis.decision_engine._log_decision": MagicMock(),
        "app.pipeline.orchestration.post_cycle_hooks.run_post_cycle_hooks": AsyncMock(),
        "app.pipeline.attention_tracker.record_analysis": MagicMock(),
        "app.cognition.lesson_store.retrieve_lessons": MagicMock(return_value=[]),
        "app.pipeline.trading_constitution.format_constitution_for_prompt": MagicMock(
            return_value=""
        ),
        "app.tools.portfolio_tools.format_position_context_for_prompt": MagicMock(
            return_value=""
        ),
        "app.db.connection.get_db": MagicMock(),
        "app.log_manager.log_manager": MagicMock(),
    }


@pytest.fixture
def runner_mocks():
    """Apply all runner patches via ExitStack to avoid nesting limits."""
    patches_dict = _mock_runner_deps()
    with ExitStack() as stack:
        mocks = {}
        for target, mock_obj in patches_dict.items():
            mocks[target] = stack.enter_context(patch(target, mock_obj))
        yield patches_dict
    # All patches auto-cleaned up


# ── RED/GREEN: The critical is_highly_redundant bug ──

@pytest.mark.asyncio
async def test_execute_v2_pipeline_accepts_is_highly_redundant():
    """execute_v2_pipeline MUST accept is_highly_redundant kwarg."""
    from app.cognition.orchestration.runner import execute_v2_pipeline
    import inspect

    sig = inspect.signature(execute_v2_pipeline)
    assert "is_highly_redundant" in sig.parameters, (
        "execute_v2_pipeline MUST accept is_highly_redundant. "
        "Without it, phase4 crashes with TypeError for every ticker."
    )


@pytest.mark.asyncio
async def test_execute_v2_pipeline_completes_without_crash(runner_mocks):
    """Smoke test: Full V2 pipeline completes without NameError/TypeError."""
    from app.cognition.orchestration.runner import execute_v2_pipeline

    result = await execute_v2_pipeline(
        "AAPL", cycle_id="test-001", bot_id="test-bot",
        is_highly_redundant=False,
    )

    assert result is not None
    assert result["ticker"] == "AAPL"
    assert result["action"] in ("BUY", "SELL", "HOLD", "PASS")
    assert "confidence" in result
    assert "v2_metadata" in result


@pytest.mark.asyncio
async def test_execute_v2_pipeline_redundant_flag(runner_mocks):
    """is_highly_redundant=True should reach MetaOrchestrator."""
    from app.cognition.orchestration.runner import execute_v2_pipeline

    result = await execute_v2_pipeline(
        "NVDA", cycle_id="test-002", bot_id="test-bot",
        is_highly_redundant=True,
    )

    assert result is not None
    orch = runner_mocks[
        "app.cognition.orchestration.meta_orchestrator.MetaOrchestrator.orchestrate"
    ]
    orch.assert_called_once()
    assert orch.call_args[0][5] is True


@pytest.mark.asyncio
async def test_v2_result_has_v1_compatible_shape(runner_mocks):
    """Result dict must have all keys downstream phases expect."""
    from app.cognition.orchestration.runner import execute_v2_pipeline

    result = await execute_v2_pipeline(
        "MSFT", cycle_id="test-003", bot_id="test-bot",
    )

    required_keys = [
        "ticker", "action", "confidence", "rationale",
        "config_used", "escalated", "agent_results",
        "c_result", "d_result", "human_review",
        "agent_tokens", "rlm_tokens", "total_tokens",
        "total_time_s", "timestamp", "v2_metadata",
    ]
    for key in required_keys:
        assert key in result, f"Missing V1 compat key: {key}"
    assert isinstance(result["confidence"], int)


# ── Phase 4 Worker Pool Tests ──

@pytest.mark.asyncio
async def test_phase4_passes_is_highly_redundant():
    """Phase4 workers must pass is_highly_redundant to execute_v2_pipeline."""
    from app.pipeline.phases.phase4_analysis import run_phase4_analysis
    from app.pipeline.core import PipelineContext
    from app.config import settings

    ctx = PipelineContext(
        tickers=["AAPL"], collect=True, analyze=True,
        trade=True, cycle_id="test-wkr-001",
    )
    state = {"highly_redundant_tickers": ["AAPL"], "triage": {}}
    summary = {
        "buy_count": 0, "sell_count": 0, "hold_count": 0,
        "review_count": 0, "analysis_results_count": 0,
    }
    captured = {}

    async def fake_exec(ticker, **kw):
        captured.update(kw)
        return {"ticker": ticker, "action": "BUY", "confidence": 80,
                "rationale": "t", "human_review": False}

    mock_cc = MagicMock()
    mock_cc.wait_if_paused = AsyncMock()

    with ExitStack() as stack:
        stack.enter_context(patch(
            "app.cognition.orchestration.runner.execute_v2_pipeline",
            side_effect=fake_exec,
        ))
        stack.enter_context(patch(
            "app.cycle.orchestration.cycle_control.cycle_control", mock_cc,
        ))
        stack.enter_context(patch.object(settings, "V2_TICKER_CONCURRENCY", 1))
        stack.enter_context(patch.object(settings, "ANALYSIS_WORKER_TIMEOUT_SECONDS", 30))
        stack.enter_context(patch.object(settings, "CYCLE_TIMEOUT_MINUTES", 5))

        results = await run_phase4_analysis(
            ctx, "test-bot", "memo", lambda *a, **kw: None, summary, state,
        )

    assert "is_highly_redundant" in captured
    assert captured["is_highly_redundant"] is True
    assert len(results) == 1


@pytest.mark.asyncio
async def test_phase4_all_crash_aborts():
    """If ALL tickers crash, phase4 must raise RuntimeError."""
    from app.pipeline.phases.phase4_analysis import run_phase4_analysis
    from app.pipeline.core import PipelineContext
    from app.config import settings

    ctx = PipelineContext(
        tickers=["A", "B", "C"], collect=True, analyze=True,
        trade=True, cycle_id="test-crash",
    )
    summary = {
        "buy_count": 0, "sell_count": 0, "hold_count": 0,
        "review_count": 0, "analysis_results_count": 0,
    }
    mock_cc = MagicMock()
    mock_cc.wait_if_paused = AsyncMock()

    async def crash(t, **kw):
        raise ValueError("boom")

    with ExitStack() as stack:
        stack.enter_context(patch(
            "app.cognition.orchestration.runner.execute_v2_pipeline",
            side_effect=crash,
        ))
        stack.enter_context(patch(
            "app.cycle.orchestration.cycle_control.cycle_control", mock_cc,
        ))
        stack.enter_context(patch.object(settings, "V2_TICKER_CONCURRENCY", 1))
        stack.enter_context(patch.object(settings, "ANALYSIS_WORKER_TIMEOUT_SECONDS", 5))
        stack.enter_context(patch.object(settings, "CYCLE_TIMEOUT_MINUTES", 1))

        with pytest.raises(RuntimeError, match="All.*tickers crashed"):
            await run_phase4_analysis(
                ctx, "bot", "", lambda *a, **kw: None, summary, {"triage": {}},
            )


@pytest.mark.asyncio
async def test_phase4_partial_crash_continues():
    """Partial crashes return fallback HOLD@0% alongside successful results."""
    from app.pipeline.phases.phase4_analysis import run_phase4_analysis
    from app.pipeline.core import PipelineContext
    from app.config import settings

    ctx = PipelineContext(
        tickers=["OK", "BAD"], collect=True, analyze=True,
        trade=True, cycle_id="test-partial",
    )
    summary = {
        "buy_count": 0, "sell_count": 0, "hold_count": 0,
        "review_count": 0, "analysis_results_count": 0,
    }
    mock_cc = MagicMock()
    mock_cc.wait_if_paused = AsyncMock()

    async def mixed(t, **kw):
        if t == "BAD":
            raise ValueError("boom")
        return {"ticker": t, "action": "HOLD", "confidence": 50,
                "rationale": "ok", "human_review": False}

    with ExitStack() as stack:
        stack.enter_context(patch(
            "app.cognition.orchestration.runner.execute_v2_pipeline",
            side_effect=mixed,
        ))
        stack.enter_context(patch(
            "app.cycle.orchestration.cycle_control.cycle_control", mock_cc,
        ))
        stack.enter_context(patch.object(settings, "V2_TICKER_CONCURRENCY", 1))
        stack.enter_context(patch.object(settings, "ANALYSIS_WORKER_TIMEOUT_SECONDS", 5))
        stack.enter_context(patch.object(settings, "CYCLE_TIMEOUT_MINUTES", 1))

        results = await run_phase4_analysis(
            ctx, "bot", "", lambda *a, **kw: None, summary, {"triage": {}},
        )

    assert len(results) == 2
    crash_r = [r for r in results if r.get("is_timeout_fallback")]
    assert len(crash_r) == 1
    assert crash_r[0]["action"] == "HOLD"
    assert crash_r[0]["confidence"] == 0
