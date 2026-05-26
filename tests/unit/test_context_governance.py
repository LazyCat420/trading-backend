"""
Tests for Context Window Governance system.

Covers:
  - Context budget registry (model-aware budgeting)
  - Tool result compression (inline summarization)
  - Progressive summarizer (debate turn + research compression)
  - Debate prompt capping
  - History compression with model-aware thresholds

All tests are pure unit tests — no DB, no LLM, no network calls.
"""


# ── Phase 1: Context Budget Registry ──────────────────────────────────


class TestContextBudget:
    """Tests for app.config.context_budget."""

    def test_default_budget_returns_valid(self):
        """Default budget should be usable even before model discovery."""
        from app.config.context_budget import get_context_budget

        budget = get_context_budget()
        assert budget.effective_context_tokens > 0
        assert budget.compressor_threshold > 0
        assert budget.data_context_chars > 0
        assert budget.tool_result_chars > 0

    def test_register_model_creates_budget(self):
        """Registering a model should create a budget accessible via get_context_budget."""
        from app.config.context_budget import register_model_context, get_context_budget

        budget = register_model_context("test-model-32k", 32768)
        assert budget.raw_context_tokens == 32768
        assert budget.effective_context_tokens == 32768

        # Should be retrievable
        fetched = get_context_budget("test-model-32k")
        assert fetched.model_id == "test-model-32k"

    def test_effective_from_raw_scaling(self):
        """Effective context should return raw context up to a 128k cap."""
        from app.config.context_budget import _effective_from_raw

        # Asserts that raw tokens are returned directly up to the 128K cap
        assert _effective_from_raw(8192) == 8192
        assert _effective_from_raw(32768) == 32768
        assert _effective_from_raw(131072) == 131072
        assert _effective_from_raw(262144) == 131072
        assert _effective_from_raw(1048576) == 131072

    def test_compressor_threshold_is_75_percent(self):
        """Compressor threshold should be 75% of effective context."""
        from app.config.context_budget import register_model_context

        budget = register_model_context("test-threshold", 32768)
        expected_effective = 32768
        expected_threshold = int(expected_effective * 0.75)
        assert budget.compressor_threshold == expected_threshold

    def test_total_allocated_fits_within_effective(self):
        """Sum of all budget slices should not exceed effective context."""
        from app.config.context_budget import register_model_context

        for raw in [8192, 16384, 32768, 65536, 131072, 262144]:
            budget = register_model_context(f"test-alloc-{raw}", raw)
            # Total allocated should be ≤ effective (with 3% margin for rounding)
            assert budget.total_allocated <= budget.effective_context_tokens, (
                f"raw={raw}: allocated={budget.total_allocated} > "
                f"effective={budget.effective_context_tokens}"
            )

    def test_estimate_tokens(self):
        """Token estimation should use 4-chars-per-token heuristic."""
        from app.config.context_budget import estimate_tokens

        assert estimate_tokens("a" * 100) == 25
        assert estimate_tokens("") == 0
        assert estimate_tokens("hello world") == 2  # 11 chars → 2 tokens

    def test_partial_model_id_match(self):
        """Should match partial model IDs for HuggingFace-style paths."""
        from app.config.context_budget import register_model_context, get_context_budget

        register_model_context("org/big-model-v2", 32768)

        # Partial match should work
        budget = get_context_budget("big-model-v2")
        assert budget.model_id == "org/big-model-v2"


# ── Phase 2: Tool Result Compression ──────────────────────────────────


class TestToolResultCompression:
    """Tests for context_compressor.summarize_tool_result."""

    def test_small_result_unchanged(self):
        """Tool results within budget should not be modified."""
        from app.agents.context_compressor import summarize_tool_result

        small = "AAPL price: $187.23, RSI: 67.4"
        result = summarize_tool_result(small, budget_tokens=1000)
        assert result == small

    def test_large_result_truncated(self):
        """Tool results exceeding budget should be truncated with marker."""
        from app.agents.context_compressor import summarize_tool_result

        large = "x" * 50000  # ~12500 tokens
        result = summarize_tool_result(large, tool_name="get_price_history", budget_tokens=500)
        assert len(result) < len(large)
        assert "truncated" in result.lower()
        assert "get_price_history" in result

    def test_truncation_preserves_head_and_tail(self):
        """Truncated results should keep head (70%) and tail (15%) data."""
        from app.agents.context_compressor import summarize_tool_result

        # Build a result with identifiable head and tail
        head_marker = "HEAD_START_DATA "
        tail_marker = " TAIL_END_DATA"
        middle = "m" * 50000
        large = head_marker + middle + tail_marker

        result = summarize_tool_result(large, budget_tokens=500)
        assert result.startswith("HEAD_START_DATA")
        assert "TAIL_END_DATA" in result

    def test_budget_uses_context_budget_default(self):
        """When no budget_tokens is specified, should use context budget default."""
        from app.agents.context_compressor import summarize_tool_result

        # A very large input should be truncated even with default budget
        huge = "data " * 100000  # 500K chars → ~125K tokens
        result = summarize_tool_result(huge, tool_name="huge_tool")
        assert len(result) < len(huge)


# ── Phase 3: Progressive Summarizer ──────────────────────────────────


class TestProgressiveSummarizer:
    """Tests for app.agents.progressive_summarizer."""

    def test_short_text_unchanged(self):
        """Text within max_chars should pass through unchanged."""
        from app.agents.progressive_summarizer import summarize_opponent_turn

        short = "Bull case: Strong earnings [fundamentals:EPS=2.5]"
        result = summarize_opponent_turn(short, max_chars=2000)
        assert result == short

    def test_json_claims_extraction(self):
        """Should extract claims from JSON structure in debate output."""
        from app.agents.progressive_summarizer import summarize_opponent_turn

        debate_output = (
            'Some rambling preamble... ' * 100 +
            '{"action": "BUY", "claims": ['
            '"EPS grew 15% YoY [fundamentals:EPS_growth=15%]", '
            '"RSI at 35 suggests oversold [technical_data:RSI=35]", '
            '"Strong institutional buying [news:inst_buying=high]"'
            '], "confidence": 72, "key_argument": "Consistent revenue growth"}'
        )

        result = summarize_opponent_turn(debate_output, side="bull", max_chars=1000)
        # Should extract structured info (claims, key_argument, or confidence)
        assert any(s in result for s in ["BULL CLAIMS", "KEY ARGUMENT", "CONFIDENCE", "BULL CITED"])
        assert "EPS" in result or "revenue" in result.lower() or "Consistent" in result
        assert len(result) <= 1000

    def test_citation_extraction_fallback(self):
        """Should extract cited lines when JSON parsing fails."""
        from app.agents.progressive_summarizer import summarize_opponent_turn

        messy_output = (
            "Analysis paragraph... " * 50 +
            "The RSI at 37.8 suggests oversold conditions [technical_data:RSI=37.8]. " +
            "P/E ratio of 15.2 is below industry average [fundamentals:PE=15.2]. " +
            "More rambling... " * 50
        )

        result = summarize_opponent_turn(messy_output, side="bear", max_chars=500)
        # Should have extracted citation-containing lines
        assert "RSI" in result or "P/E" in result or "bear" in result.lower()
        assert len(result) <= 500

    def test_fallback_to_head_truncation(self):
        """When no structure is found, should fall back to head truncation."""
        from app.agents.progressive_summarizer import summarize_opponent_turn

        plain = "Just plain text without any structure or citations. " * 200
        result = summarize_opponent_turn(plain, max_chars=500)
        assert len(result) <= 600  # Margin for truncation marker
        assert "truncated" in result.lower()

    def test_compress_tool_research_block(self):
        """Should compress multiple tool results into a dense summary."""
        from app.agents.progressive_summarizer import compress_tool_research_block

        tool_history = [
            "### Tool Call: get_price_history({\"ticker\": \"AAPL\"})\n"
            + "Price: $187.23, Change: +2.3%, Volume: 45M\n" * 100,
            "### Tool Call: get_fundamentals({\"ticker\": \"AAPL\"})\n"
            + "PE: 28.5, EPS: 6.52, Revenue: $95B\n" * 100,
        ]

        result = compress_tool_research_block(tool_history, max_total_chars=2000)
        assert "[get_price_history]" in result
        assert "[get_fundamentals]" in result
        assert len(result) <= 2020  # Small margin for markers

    def test_empty_tool_research(self):
        """Empty tool history should return a placeholder."""
        from app.agents.progressive_summarizer import compress_tool_research_block

        result = compress_tool_research_block([])
        assert result == "No tools used."


# ── Phase 4: Debate Prompt Capping ──────────────────────────────────


class TestDebatePromptCapping:
    """Tests for debate_coordinator._cap_debate_text."""

    def test_short_text_unchanged(self):
        """Text within max_chars should pass through unchanged."""
        from app.cognition.debate.debate_coordinator import _cap_debate_text

        short = "Bull argument"
        result = _cap_debate_text(short, 1000, "test")
        assert result == short

    def test_long_text_truncated_with_label(self):
        """Long text should be truncated with a labeled marker."""
        from app.cognition.debate.debate_coordinator import _cap_debate_text

        long = "x" * 5000
        result = _cap_debate_text(long, 3000, "bull_t1_quote")
        assert len(result) > 3000  # Marker adds some chars
        assert len(result) < 5000  # But much less than original
        assert "bull_t1_quote" in result
        assert "truncated" in result.lower()


# ── End-to-End Context Sizing ──────────────────────────────────────


class TestContextSizing:
    """Integration-level tests that verify context budgets are enforced."""

    def test_all_budgets_consistent(self):
        """All registered budgets should have consistent internal state."""
        from app.config.context_budget import register_model_context

        for raw in [8192, 16384, 32768, 65536, 131072, 262144]:
            b = register_model_context(f"consistency-{raw}", raw)
            # Compressor threshold < effective
            assert b.compressor_threshold < b.effective_context_tokens
            # Char conversions are consistent
            assert b.data_context_chars == b.data_context_budget * 4
            assert b.tool_result_chars == b.tool_result_budget * 4
            # Total allocated doesn't exceed effective
            assert b.total_allocated <= b.effective_context_tokens

    def test_war_context_budget_enforced(self):
        """War context chars budget should be reasonable."""
        from app.config.context_budget import get_context_budget

        budget = get_context_budget()
        # War context should be a small slice
        assert budget.war_context_chars > 0
        assert budget.war_context_chars < budget.effective_context_chars
        # Should be < 25% of total effective
        assert budget.war_context_chars < budget.effective_context_chars * 0.25
