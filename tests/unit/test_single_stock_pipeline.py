"""
Battle tests: Single-stock pipeline verification (AAPL baseline).

These tests verify the data contracts at each phase boundary to ensure
data flows correctly from collection → evidence → sufficiency → thesis
→ decision → trade. Every test verifies SHAPE and CONSTRAINTS of
the data passing between pipeline stages.

Test categories:
  P1: Data completeness contracts
  P2: Evidence packet contracts
  P3: Sufficiency gate behavior
  P4: V2 runner → V1-compatible result shape
  P5: Phase handoff contracts (orchestrator → phases)
  P6: Trading phase decision routing
"""

import pytest
import asyncio
from contextlib import ExitStack
from unittest.mock import patch, MagicMock, AsyncMock
from dataclasses import dataclass, field


# ── Fake stubs for V2 pipeline mocking ──

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


def _build_runner_mocks():
    """Build the full set of mocks for execute_v2_pipeline."""
    import app.pipeline.orchestration.cycle_control
    import app.pipeline.data.data_completeness
    import app.cognition.ontology.ontology_builder
    import app.cognition.evidence.packet_builder
    import app.cognition.verification.sufficiency_gate
    import app.tools.portfolio_tools
    import app.cognition.orchestration.meta_orchestrator
    import app.cognition.debate.debate_coordinator
    import app.cognition.debate.thesis_agent
    import app.cognition.debate.action_gate
    import app.pipeline.analysis.hallucination_checker
    import app.cognition.memory.reader
    import app.cognition.memory.writer
    import app.pipeline.analysis.decision_engine
    import app.pipeline.orchestration.post_cycle_hooks
    import app.pipeline.attention_tracker
    import app.cognition.lesson_store
    import app.pipeline.trading_constitution
    import app.db.connection
    import app.log_manager

    mock_cc = MagicMock()
    mock_cc.wait_if_paused = AsyncMock()
    mock_cc.is_paused = False

    mock_ont = MagicMock()
    mock_ont.return_value.execute = AsyncMock(
        return_value={"ontology_nodes": [], "ontology_context": ""}
    )

    return {
        "app.pipeline.orchestration.cycle_control.cycle_control": mock_cc,
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
    """Apply all runner patches via ExitStack."""
    patches_dict = _build_runner_mocks()
    with ExitStack() as stack:
        mocks = {}
        for target, mock_obj in patches_dict.items():
            mocks[target] = stack.enter_context(patch(target, mock_obj))
        yield patches_dict


# ═══════════════════════════════════════════════════════════════════
# P1: News Collector Import Smoke Test
# (Battle test for Breakpoint 1 — the IndentationError)
# ═══════════════════════════════════════════════════════════════════

def test_p1_news_collector_imports_cleanly():
    """Breakpoint 1: news_collector must be importable without syntax errors."""
    from app.collectors.news_collector import (
        collect_finnhub_news,
        collect_yfinance_news,
        collect_for_ticker,
        collect_feed,
        collect_all,
    )
    # All functions must be callable
    assert callable(collect_finnhub_news)
    assert callable(collect_yfinance_news)
    assert callable(collect_for_ticker)
    assert callable(collect_feed)
    assert callable(collect_all)


# ═══════════════════════════════════════════════════════════════════
# P2: Data Completeness Contract
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@patch("app.collectors.youtube_collector.collect_for_ticker")
@patch("app.collectors.reddit_collector.collect_for_ticker")
@patch("app.collectors.news_collector.collect_for_ticker")
@patch("app.pipeline.data.data_completeness._is_stale")
async def test_p2_check_and_fill_returns_valid_report(
    mock_is_stale, mock_news_collect, mock_reddit_collect, mock_yt_collect, mock_db
):
    """check_and_fill must return a dict with ticker, available, filled, missing keys."""
    # Mock all the DB count queries to return non-zero values
    mock_db.execute.return_value = mock_db
    mock_db.fetchone.return_value = (50,)  # All counts return 50
    mock_is_stale.return_value = (False, None)

    from app.pipeline.data.data_completeness import check_and_fill

    with patch("app.pipeline.data.data_completeness.get_db", return_value=mock_db):
        report = await check_and_fill("AAPL")

    # DATA CONTRACT: required keys
    assert "ticker" in report
    assert report["ticker"] == "AAPL"
    assert "available" in report
    assert "filled" in report
    assert "missing" in report
    assert isinstance(report["available"], dict)
    assert isinstance(report["filled"], list)
    assert isinstance(report["missing"], list)


@pytest.mark.asyncio
async def test_p2_data_sufficiency_gate_shape():
    """check_data_sufficiency must return dict with 'sufficient' and 'gaps'."""
    from app.pipeline.data.data_completeness import check_data_sufficiency

    # All data present
    report_full = {
        "available": {
            "price_history": 100,
            "technicals": 50,
            "fundamentals": 5,
            "news": 10,
        }
    }
    result = check_data_sufficiency(report_full)
    assert "sufficient" in result
    assert "gaps" in result
    assert result["sufficient"] is True
    assert len(result["gaps"]) == 0

    # Missing data
    report_empty = {"available": {"price_history": 0, "technicals": 0}}
    result2 = check_data_sufficiency(report_empty)
    assert result2["sufficient"] is False
    assert len(result2["gaps"]) > 0
    gap_cats = [g["category"] for g in result2["gaps"]]
    assert "price_history" in gap_cats


# ═══════════════════════════════════════════════════════════════════
# P3: V2 Runner → V1-Compatible Result Shape
# (Tests the critical data contract between Phase 4 and Phase 5)
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_p3_v2_result_v1_compatible(runner_mocks):
    """V2 runner result must have every key Phase 5 trading expects."""
    from app.cognition.orchestration.runner import execute_v2_pipeline

    result = await execute_v2_pipeline(
        "AAPL", cycle_id="battle-001", bot_id="test-bot",
    )

    # These keys are consumed by trading_phase.execute_decisions()
    required_keys = [
        "ticker", "action", "confidence", "rationale",
        "config_used", "escalated", "agent_results",
        "c_result", "d_result", "human_review",
        "agent_tokens", "rlm_tokens", "total_tokens",
        "total_time_s", "timestamp", "v2_metadata",
    ]
    for key in required_keys:
        assert key in result, f"Missing V1 compat key: {key}"

    # TYPE CONSTRAINTS
    assert isinstance(result["confidence"], int)
    assert isinstance(result["total_tokens"], int)
    assert result["action"] in ("BUY", "SELL", "HOLD", "PASS")
    assert isinstance(result["v2_metadata"], dict)

    # V2 metadata must have stages_completed
    assert "stages_completed" in result["v2_metadata"]


@pytest.mark.asyncio
async def test_p3_v2_result_for_critical_gap(runner_mocks):
    """When sufficiency returns critical_gap, runner ABSTAINs with HOLD@0%."""
    from app.cognition.orchestration.runner import execute_v2_pipeline

    # Override sufficiency to return critical_gap
    runner_mocks[
        "app.cognition.verification.sufficiency_gate.check_data_sufficiency"
    ].return_value = FakeSufficiency(
        status="critical_gap", blockers=["price_history"]
    )

    result = await execute_v2_pipeline(
        "AAPL", cycle_id="battle-gap", bot_id="test-bot",
    )

    assert result is not None
    assert result["ticker"] == "AAPL"
    assert result["action"] == "HOLD"
    assert result["confidence"] == 0


@pytest.mark.asyncio
async def test_p3_v2_hallucination_downgrade(runner_mocks):
    """When hallucination check rejects, action must be downgraded to HOLD."""
    from app.cognition.orchestration.runner import execute_v2_pipeline

    runner_mocks[
        "app.pipeline.analysis.hallucination_checker.check_hallucinations"
    ].return_value = {
        "rejected": True,
        "hallucinations": ["Fake claim"],
        "rejection_reason": "Unverifiable claim detected",
    }

    result = await execute_v2_pipeline(
        "AAPL", cycle_id="battle-hall", bot_id="test-bot",
    )

    assert result is not None
    assert result["action"] == "HOLD"
    # Hallucination gate halves confidence: max(10, 75//2) = 37
    assert result["confidence"] <= 40
    assert result["confidence"] >= 10


# ═══════════════════════════════════════════════════════════════════
# P4: Phase 4 → Phase 5 Handoff Contract
# (Ensures analysis results feed correctly into trading decisions)
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_p4_analysis_result_feeds_trading():
    """Phase 4 result dict must be consumable by execute_decisions()."""
    from app.pipeline.trading_phase import execute_decisions

    # Simulate a V2 analysis result
    analysis_result = {
        "ticker": "AAPL",
        "action": "HOLD",
        "confidence": 60,
        "rationale": "Moderate outlook",
        "human_review": False,
        "v2_metadata": {"debate": {"integrity_status": "HIGH"}},
    }

    with patch("app.cycle.trading_phase.get_portfolio") as mock_port:
        mock_port.return_value = {
            "cash": 100000.0,
            "position_count": 0,
            "positions": [],
        }

        result = await execute_decisions(
            [analysis_result], bot_id="test-bot", cycle_id="battle-004"
        )

    # DATA CONTRACT: trading result shape
    assert "bot_id" in result
    assert "executed" in result
    assert "skipped" in result
    assert "counts" in result
    assert "portfolio" in result
    assert "elapsed_s" in result

    # HOLD should be skipped, not executed
    assert len(result["executed"]) == 0
    assert result["counts"]["holds"] == 1


@pytest.mark.asyncio
async def test_p4_buy_decision_reaches_trader():
    """BUY analysis result must reach the paper trader (if not gated)."""
    from app.pipeline.trading_phase import execute_decisions

    analysis_result = {
        "ticker": "TSLA",
        "action": "BUY",
        "confidence": 85,
        "rationale": "Strong momentum",
        "human_review": False,
        "v2_metadata": {"debate": {"integrity_status": "HIGH"}},
    }

    with patch("app.cycle.trading_phase.get_portfolio") as mock_port, \
         patch("app.cycle.trading_phase.check_portfolio_gate") as mock_gate, \
         patch("app.cycle.trading_phase.buy") as mock_buy, \
         patch("app.agents.pre_trade_agent.run_pre_trade") as mock_pt, \
         patch("app.services.pipeline_service.PipelineService.emit"), \
         patch("app.pipeline.attention_tracker.record_trade"):

        mock_port.return_value = {
            "cash": 100000.0,
            "position_count": 0,
            "positions": [],
        }
        mock_gate.return_value = {"blocked": False, "warnings": []}
        mock_pt.return_value = {
            "decision": "APPROVE", "shares": 10,
            "total_cost": 1500, "risk_reward_ratio": 2.0,
        }
        mock_buy.return_value = {"qty": 10, "price": 150.0, "amount": 1500}

        result = await execute_decisions(
            [analysis_result], bot_id="test-bot", cycle_id="battle-buy"
        )

    assert result["counts"]["buy_executed"] == 1
    assert len(result["executed"]) == 1
    mock_buy.assert_called_once()


# ═══════════════════════════════════════════════════════════════════
# P5: Kelly Sizing Properties
# (Property-based test for the position sizing function)
# ═══════════════════════════════════════════════════════════════════

def test_p5_kelly_sizing_bounds():
    """get_size_pct must always return between 2% and 10%."""
    from app.pipeline.trading_phase import get_size_pct

    # Test at boundaries
    assert get_size_pct(0) == 0.02   # Below min
    assert get_size_pct(50) == 0.02  # Below min
    assert get_size_pct(70) == 0.02  # Exactly min
    assert get_size_pct(100) == 0.10 # Exactly max
    assert get_size_pct(200) == 0.10 # Above max

    # Test linearity in the valid range
    for conf in range(70, 101):
        pct = get_size_pct(conf)
        assert 0.02 <= pct <= 0.10, f"get_size_pct({conf}) = {pct} out of bounds"


def test_p5_estimate_trade_contract():
    """estimate_trade must return dict with size_pct, amount, qty, price."""
    from app.pipeline.trading_phase import estimate_trade

    result = estimate_trade(confidence=85, cash=100000, current_price=150.0)

    assert "size_pct" in result
    assert "amount" in result
    assert "qty" in result
    assert "price" in result
    assert result["price"] == 150.0
    assert result["amount"] > 0
    assert result["qty"] > 0


# ═══════════════════════════════════════════════════════════════════
# P6: Orchestrator State Transitions
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_p6_orchestrator_phases_called_in_order():
    """All 6 phases must be called in order during a full cycle."""
    from app.pipeline.orchestration.orchestrator_core import OrchestratorCoreMixin
    from app.pipeline.core import PipelineContext

    ctx = PipelineContext(
        tickers=["AAPL"], collect=True, analyze=True,
        trade=True, cycle_id="battle-orch",
    )

    call_order = []

    async def track(name, orig):
        """Track call order."""
        call_order.append(name)
        if hasattr(orig, 'return_value'):
            return orig.return_value
        return orig

    with ExitStack() as stack:
        m1 = stack.enter_context(patch(
            "app.cycle.orchestration.orchestrator_core.run_phase1_health",
            side_effect=lambda *a, **k: call_order.append("phase1"),
        ))
        m2 = stack.enter_context(patch(
            "app.cycle.orchestration.orchestrator_core.run_phase2_collection",
            side_effect=lambda *a, **k: (call_order.append("phase2"), ["AAPL"])[-1],
        ))
        m3 = stack.enter_context(patch(
            "app.cycle.orchestration.orchestrator_core.run_phase3_macro",
            side_effect=lambda *a, **k: (call_order.append("phase3"), "macro memo")[-1],
        ))
        m4 = stack.enter_context(patch(
            "app.cycle.orchestration.orchestrator_core.run_phase4_analysis",
            side_effect=lambda *a, **k: (call_order.append("phase4"), [{"ticker": "AAPL", "action": "HOLD", "confidence": 50}])[-1],
        ))
        m5 = stack.enter_context(patch(
            "app.cycle.orchestration.orchestrator_core.run_phase5_trading",
            side_effect=lambda *a, **k: (call_order.append("phase5"), {"executed": []})[-1],
        ))
        m6 = stack.enter_context(patch(
            "app.cycle.orchestration.orchestrator_core.run_phase6_post",
            side_effect=lambda *a, **k: call_order.append("phase6"),
        ))
        stack.enter_context(patch(
            "app.pipeline.analysis.autoresearch.run_autoresearch", new_callable=AsyncMock,
        ))
        stack.enter_context(patch(
            "app.cycle.orchestration.state_manager.PipelineStateDB.clear_checkpoint",
        ))

        # Set up the mixin's class-level state
        OrchestratorCoreMixin._state = {"status": "idle"}
        OrchestratorCoreMixin._cycle_summary = {}
        OrchestratorCoreMixin.emit = MagicMock()
        OrchestratorCoreMixin.save_state = MagicMock()

        await OrchestratorCoreMixin._execute_cycle(ctx)

    assert call_order == ["phase1", "phase2", "phase3", "phase4", "phase5", "phase6"], (
        f"Phases executed in wrong order: {call_order}"
    )


@pytest.mark.asyncio
async def test_p6_skip_phases_when_flags_false():
    """When collect=False / analyze=False, those phases must be skipped."""
    from app.pipeline.orchestration.orchestrator_core import OrchestratorCoreMixin
    from app.pipeline.core import PipelineContext

    ctx = PipelineContext(
        tickers=["AAPL"], collect=False, analyze=False,
        trade=True, cycle_id="battle-skip",
    )

    call_order = []

    with ExitStack() as stack:
        stack.enter_context(patch(
            "app.cycle.orchestration.orchestrator_core.run_phase1_health",
            side_effect=lambda *a, **k: call_order.append("phase1"),
        ))
        stack.enter_context(patch(
            "app.cycle.orchestration.orchestrator_core.run_phase2_collection",
            side_effect=lambda *a, **k: (call_order.append("phase2"), ["AAPL"])[-1],
        ))
        stack.enter_context(patch(
            "app.cycle.orchestration.orchestrator_core.run_phase3_macro",
            side_effect=lambda *a, **k: (call_order.append("phase3"), "")[-1],
        ))
        stack.enter_context(patch(
            "app.cycle.orchestration.orchestrator_core.run_phase4_analysis",
            side_effect=lambda *a, **k: (call_order.append("phase4"), [])[-1],
        ))
        stack.enter_context(patch(
            "app.cycle.orchestration.orchestrator_core.run_phase5_trading",
            side_effect=lambda *a, **k: (call_order.append("phase5"), {"executed": []})[-1],
        ))
        stack.enter_context(patch(
            "app.cycle.orchestration.orchestrator_core.run_phase6_post",
            side_effect=lambda *a, **k: call_order.append("phase6"),
        ))
        stack.enter_context(patch(
            "app.pipeline.analysis.autoresearch.run_autoresearch", new_callable=AsyncMock,
        ))
        stack.enter_context(patch(
            "app.cycle.orchestration.state_manager.PipelineStateDB.clear_checkpoint",
        ))

        OrchestratorCoreMixin._state = {"status": "idle"}
        OrchestratorCoreMixin._cycle_summary = {}
        OrchestratorCoreMixin.emit = MagicMock()
        OrchestratorCoreMixin.save_state = MagicMock()

        await OrchestratorCoreMixin._execute_cycle(ctx)

    # Phase 2, 3, 4 should be SKIPPED
    assert "phase2" not in call_order
    assert "phase3" not in call_order
    assert "phase4" not in call_order
    # Phase 1, 5, 6 should still run
    assert "phase1" in call_order
    assert "phase5" in call_order
    assert "phase6" in call_order
