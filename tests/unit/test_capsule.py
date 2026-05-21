"""
Unit tests for AgentCapsule dataclass and formatting functions.

Tests:
  - AgentCapsule creation and immutability
  - format_capsule_for_prompt rendering
  - format_capsule_stack budget enforcement
  - EMPTY_CAPSULE sentinel behavior
  - generate_capsule heuristic extraction (JSON and plain text)
  - write_capsule_to_db (mocked DB)
"""

import asyncio
import json
import pytest
from unittest.mock import patch, MagicMock

from app.agents.capsule import (
    AgentCapsule,
    EMPTY_CAPSULE,
    format_capsule_for_prompt,
    format_capsule_stack,
    _estimate_tokens,
)

from app.agents.context_compressor import (
    generate_capsule,
    write_capsule_to_db,
    _extract_from_json,
    _extract_from_text,
)


# ── AgentCapsule dataclass tests ──────────────────────────────────────

class TestAgentCapsule:
    def test_creation(self):
        capsule = AgentCapsule(
            agent_name="retriever",
            cycle_id="cycle-001",
            ticker="NVDA",
            summary="RSI=72, vol 3x avg, EPS surprise +8%.",
            signal="BUY",
            confidence=0.78,
            flags=["overbought"],
            source_refs=["ref:capsule:abc-123"],
            raw_id="abc-123",
            tokens_estimated=38,
        )
        assert capsule.agent_name == "retriever"
        assert capsule.signal == "BUY"
        assert capsule.confidence == 0.78
        assert capsule.ticker == "NVDA"
        assert len(capsule.flags) == 1
        assert capsule.raw_id == "abc-123"

    def test_frozen_immutability(self):
        capsule = AgentCapsule(
            agent_name="planner", cycle_id="c", ticker="T",
            summary="test", signal="HOLD", confidence=0.5,
        )
        with pytest.raises(AttributeError):
            capsule.signal = "BUY"  # type: ignore

    def test_empty_capsule_sentinel(self):
        assert EMPTY_CAPSULE.signal == "UNKNOWN"
        assert EMPTY_CAPSULE.confidence == 0.0
        assert "failure" in EMPTY_CAPSULE.flags[0]

    def test_default_fields(self):
        capsule = AgentCapsule(
            agent_name="planner", cycle_id="c", ticker="T",
            summary="test", signal="HOLD", confidence=0.5,
        )
        assert capsule.flags == []
        assert capsule.source_refs == []
        assert capsule.raw_id == ""
        assert capsule.tokens_estimated == 0


# ── format_capsule_for_prompt tests ───────────────────────────────────

class TestFormatCapsuleForPrompt:
    def test_basic_rendering(self):
        capsule = AgentCapsule(
            agent_name="retriever", cycle_id="c1", ticker="AAPL",
            summary="Price at $180, RSI 65.", signal="BUY",
            confidence=0.85, flags=["momentum"],
            source_refs=["ref:capsule:abc"],
        )
        text = format_capsule_for_prompt(capsule)
        assert "RETRIEVER" in text
        assert "signal:BUY" in text
        assert "confidence:0.85" in text
        assert "Price at $180" in text
        assert "momentum" in text
        assert "ref:capsule:abc" in text

    def test_no_flags_no_refs(self):
        capsule = AgentCapsule(
            agent_name="planner", cycle_id="c1", ticker="AAPL",
            summary="Plan to gather data.", signal="NEUTRAL",
            confidence=0.5,
        )
        text = format_capsule_for_prompt(capsule)
        assert "PLANNER" in text
        assert "⚠" not in text
        assert "Expand" not in text


# ── format_capsule_stack tests ────────────────────────────────────────

class TestFormatCapsuleStack:
    def test_empty_input(self):
        assert format_capsule_stack([]) == ""

    def test_single_capsule(self):
        capsule = AgentCapsule(
            agent_name="planner", cycle_id="c1", ticker="NVDA",
            summary="Plan to fetch price and sentiment.",
            signal="NEUTRAL", confidence=0.5,
            source_refs=["ref:capsule:abc"],
        )
        text = format_capsule_stack([capsule])
        assert "PRIOR AGENT FINDINGS" in text
        assert "PLANNER" in text
        assert "get_cycle_context" in text

    def test_multiple_capsules(self):
        capsules = [
            AgentCapsule(
                agent_name="planner", cycle_id="c1", ticker="T",
                summary="Plan step.", signal="NEUTRAL", confidence=0.5,
            ),
            AgentCapsule(
                agent_name="retriever", cycle_id="c1", ticker="T",
                summary="Data fetched.", signal="BUY", confidence=0.7,
            ),
            AgentCapsule(
                agent_name="verifier", cycle_id="c1", ticker="T",
                summary="Data verified.", signal="BUY", confidence=0.8,
            ),
        ]
        text = format_capsule_stack(capsules)
        assert "PLANNER" in text
        assert "RETRIEVER" in text
        assert "VERIFIER" in text

    def test_budget_enforcement(self):
        # Create capsules with long summaries to test truncation
        long_summary = "x" * 2000  # ~500 tokens
        capsules = [
            AgentCapsule(
                agent_name=f"agent_{i}", cycle_id="c1", ticker="T",
                summary=long_summary, signal="BUY", confidence=0.8,
                source_refs=[f"ref:capsule:{i}"],
            )
            for i in range(5)
        ]
        text = format_capsule_stack(capsules, max_tokens=200)
        # With a 200 token budget, not all 5 capsules should fit fully
        token_count = _estimate_tokens(text)
        # Allow some overhead but should be roughly within budget
        assert token_count < 300  # Some slack for truncation logic

    def test_skips_empty_capsule_singleton(self):
        """EMPTY_CAPSULE singleton is skipped entirely (no output)."""
        capsules = [
            EMPTY_CAPSULE,
            AgentCapsule(
                agent_name="retriever", cycle_id="c1", ticker="T",
                summary="Real data.", signal="BUY", confidence=0.9,
            ),
        ]
        text = format_capsule_stack(capsules)
        assert "RETRIEVER" in text
        # EMPTY_CAPSULE singleton should produce zero output
        assert "unknown" not in text.lower().split("RETRIEVER")[0].lower()

    def test_unknown_signal_shows_failure_header(self):
        """UNKNOWN-signal capsules emit a [FAILED] header instead of being silently skipped."""
        unknown_capsule = AgentCapsule(
            agent_name="planner", cycle_id="c1", ticker="T",
            summary="Agent planner failed: timeout after 30s",
            signal="UNKNOWN", confidence=0.0, flags=["agent_failure"],
        )
        good_capsule = AgentCapsule(
            agent_name="retriever", cycle_id="c1", ticker="T",
            summary="Data fetched.", signal="BUY", confidence=0.7,
        )
        text = format_capsule_stack([unknown_capsule, good_capsule])
        # Failure header should be visible
        assert "PLANNER [FAILED]" in text
        # Good capsule should still render normally
        assert "RETRIEVER" in text
        assert "signal:BUY" in text


# ── _extract_from_json tests ─────────────────────────────────────────

class TestExtractFromJson:
    def test_standard_json_response(self):
        parsed = {
            "action": "BUY",
            "confidence": 85,
            "rationale": "Strong earnings beat with 8% EPS surprise, positive momentum."
        }
        signal, summary, confidence, flags = _extract_from_json(parsed)
        assert signal == "BUY"
        assert confidence == 0.85
        assert "earnings" in summary.lower()

    def test_bullish_alias(self):
        parsed = {"signal": "BULLISH", "confidence": 0.7, "summary": "Looks good"}
        signal, _, confidence, _ = _extract_from_json(parsed)
        assert signal == "BUY"
        assert confidence == 0.7

    def test_bearish_alias(self):
        parsed = {"signal": "BEARISH", "confidence": 60, "reasoning": "Declining revenue"}
        signal, summary, confidence, _ = _extract_from_json(parsed)
        assert signal == "SELL"
        assert confidence == 0.60
        assert "revenue" in summary.lower()

    def test_flags_extraction(self):
        parsed = {
            "action": "HOLD", "confidence": 50,
            "rationale": "Mixed signals",
            "warnings": ["low volume", "divergence"]
        }
        _, _, _, flags = _extract_from_json(parsed)
        assert "low volume" in flags
        assert "divergence" in flags

    def test_missing_fields(self):
        parsed = {"random_key": "random_value"}
        signal, summary, confidence, flags = _extract_from_json(parsed)
        assert signal == "UNKNOWN"
        assert confidence == 0.0
        assert len(summary) > 0  # Fallback should extract something


# ── _extract_from_text tests ─────────────────────────────────────────

class TestExtractFromText:
    def test_buy_signal_detection(self):
        text = "Based on our analysis, we recommend a STRONG BUY with confidence: 92%"
        signal, summary, confidence, flags = _extract_from_text(text)
        assert signal == "BUY"
        assert confidence == 0.92

    def test_sell_signal_detection(self):
        text = "The stock is bearish. Sell recommendation. Confidence 75%."
        signal, _, confidence, _ = _extract_from_text(text)
        assert signal == "SELL"
        assert confidence == 0.75

    def test_flag_detection(self):
        text = "There are divergence patterns and low volume concerns with conflicting signals."
        _, _, _, flags = _extract_from_text(text)
        assert "divergence" in flags
        assert "low_volume" in flags
        assert "conflicting_signals" in flags

    def test_summary_truncation(self):
        long_text = "A" * 500
        _, summary, _, _ = _extract_from_text(long_text)
        assert len(summary) <= 310  # 300 + "..."


# ── generate_capsule tests ───────────────────────────────────────────

class TestGenerateCapsule:
    def test_json_agent_result(self):
        agent_result = {
            "response": json.dumps({
                "action": "BUY",
                "confidence": 82,
                "rationale": "Strong momentum with RSI above 70 and volume surge."
            }),
            "tokens_used": 150,
        }
        capsule = asyncio.get_event_loop().run_until_complete(
            generate_capsule(agent_result, "retriever", "cycle-001", "NVDA")
        )
        assert capsule.agent_name == "retriever"
        assert capsule.signal == "BUY"
        assert capsule.confidence == 0.82
        assert "momentum" in capsule.summary.lower()
        assert capsule.raw_id  # Should have a UUID
        assert len(capsule.source_refs) == 1
        assert capsule.source_refs[0].startswith("ref:capsule:")

    def test_tokens_estimated_uses_rendered_prompt(self):
        """tokens_estimated should reflect the full rendered prompt, not just summary."""
        agent_result = {
            "response": json.dumps({
                "action": "BUY",
                "confidence": 82,
                "rationale": "Strong momentum with RSI above 70.",
                "warnings": ["overbought", "high_volume"]
            }),
            "tokens_used": 150,
        }
        capsule = asyncio.get_event_loop().run_until_complete(
            generate_capsule(agent_result, "retriever", "cycle-001", "NVDA")
        )
        # The rendered prompt includes header line + summary + flags + refs
        rendered = format_capsule_for_prompt(capsule)
        expected_tokens = _estimate_tokens(rendered)
        assert capsule.tokens_estimated == expected_tokens
        # Should be MORE than just summary tokens
        summary_only_tokens = _estimate_tokens(capsule.summary)
        assert capsule.tokens_estimated > summary_only_tokens

    def test_non_dict_result(self):
        capsule = asyncio.get_event_loop().run_until_complete(
            generate_capsule("not a dict", "planner", "c1", "AAPL")
        )
        assert capsule.signal == "UNKNOWN"
        assert "agent_failure" in capsule.flags

    def test_empty_response(self):
        capsule = asyncio.get_event_loop().run_until_complete(
            generate_capsule({"response": ""}, "planner", "c1", "AAPL")
        )
        assert capsule.signal == "UNKNOWN"
        assert "empty_response" in capsule.flags

    def test_plain_text_fallback(self):
        agent_result = {
            "response": "I recommend buying NVDA. The stock shows strong bullish momentum. Confidence: 88%",
            "tokens_used": 50,
        }
        capsule = asyncio.get_event_loop().run_until_complete(
            generate_capsule(agent_result, "synthesizer", "c1", "NVDA")
        )
        assert capsule.signal == "BUY"
        assert capsule.confidence == 0.88


# ── write_capsule_to_db tests ────────────────────────────────────────

class TestWriteCapsuleToDb:
    @patch("app.db.connection.get_db")
    def test_writes_to_db(self, mock_get_db):
        mock_db = MagicMock()
        mock_get_db.return_value.__enter__ = MagicMock(return_value=mock_db)
        mock_get_db.return_value.__exit__ = MagicMock(return_value=False)

        capsule = AgentCapsule(
            agent_name="retriever", cycle_id="c1", ticker="NVDA",
            summary="Test summary", signal="BUY", confidence=0.8,
            flags=["test_flag"], raw_id="test-uuid-123",
            source_refs=["ref:capsule:test-uuid-123"],
        )

        asyncio.get_event_loop().run_until_complete(
            write_capsule_to_db(capsule, "Full raw response text here")
        )

        # Verify DB was called with correct params
        mock_db.execute.assert_called_once()
        call_args = mock_db.execute.call_args
        assert "cycle_context" in call_args[0][0]
        assert call_args[0][1][0] == "test-uuid-123"  # raw_id
        assert call_args[0][1][2] == "retriever"  # agent_name

    def test_skips_empty_raw_id(self):
        capsule = AgentCapsule(
            agent_name="planner", cycle_id="c1", ticker="T",
            summary="test", signal="HOLD", confidence=0.5,
            raw_id="",  # Empty — should skip DB write
        )
        # Should not raise
        asyncio.get_event_loop().run_until_complete(
            write_capsule_to_db(capsule, "some text")
        )


# ── Token estimation tests ───────────────────────────────────────────

class TestEstimateTokens:
    def test_basic_estimation(self):
        assert _estimate_tokens("") == 0
        assert _estimate_tokens("test") == 1  # 4 chars / 4 = 1
        assert _estimate_tokens("a" * 400) == 100  # 400 / 4 = 100

    def test_realistic_text(self):
        text = "RSI=72 (overbought), vol 3x avg, EPS surprise +8%."
        tokens = _estimate_tokens(text)
        assert 10 < tokens < 20  # ~50 chars → ~12 tokens

