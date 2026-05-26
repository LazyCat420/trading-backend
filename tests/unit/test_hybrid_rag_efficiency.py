"""
Unit and integration tests for the Hybrid RAG/Precision Query plan.

Tests cover:
1. Synonym resolution and metadata enrichment logic
2. Tool registration verification (tools are in registry)
3. Whitelist binding verification (tools are assigned to correct agents)
4. A/B token performance simulation
"""

import sys
import os
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

# Ensure project root is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.tools.precision_rag import (
    resolve_synonym,
    enrich_metric_metadata,
    FUNDAMENTALS_COLS,
    FINANCIAL_HISTORY_COLS,
    TECHNICALS_COLS,
    SYNONYM_MAP,
)


# ══════════════════════════════════════════════════════════════
# Phase 1: Synonym & Metadata Unit Tests
# ══════════════════════════════════════════════════════════════

class TestPrecisionRagLogic:
    """Test individual RAG tool resolution and formatting logic."""

    def test_synonym_resolution_financials(self):
        """Verify financial metric synonyms are correctly mapped to DB columns."""
        assert resolve_synonym("P/E Ratio") == "pe_ratio"
        assert resolve_synonym("pe") == "pe_ratio"
        assert resolve_synonym("Sales") == "revenue"
        assert resolve_synonym("d/e") == "debt_to_equity"
        assert resolve_synonym("FCF") == "free_cash_flow"
        assert resolve_synonym("EPS") == "eps"
        assert resolve_synonym("forward p/e") == "forward_pe"

    def test_synonym_resolution_technicals(self):
        """Verify technical indicator synonyms are correctly mapped."""
        assert resolve_synonym("rsi") == "rsi_14"
        assert resolve_synonym("relative strength index") == "rsi_14"
        assert resolve_synonym("RSI") == "rsi_14"
        assert resolve_synonym("MACD") == "macd"
        assert resolve_synonym("SMA") == "sma_20"
        assert resolve_synonym("SMA 200") == "sma_200"
        assert resolve_synonym("ATR") == "atr_14"
        assert resolve_synonym("ADX") == "adx_14"

    def test_synonym_resolution_unknown(self):
        """Verify unknown metrics return cleaned lowercase string."""
        assert resolve_synonym("Unknown Metric") == "unknown metric"
        assert resolve_synonym("RANDOM_THING") == "random_thing"

    def test_metadata_enrichment_ratio(self):
        """Verify ratio metrics are marked with correct units and multiplier."""
        meta = enrich_metric_metadata("P/E Ratio", 28.5, "2026-05-25")
        assert meta["value"] == 28.5
        assert meta["unit"] == "ratio"
        assert meta["currency"] == "N/A"
        assert meta["multiplier"] == "1"
        assert meta["period"] == "2026-05-25"

    def test_metadata_enrichment_currency(self):
        """Verify currency metrics are marked with correct scale factors."""
        meta = enrich_metric_metadata("Sales", 90750000000.0, "2026-05-25")
        assert meta["value"] == 90750000000.0
        assert meta["unit"] == "currency"
        assert meta["currency"] == "USD"
        assert meta["multiplier"] == "raw"

    def test_metadata_enrichment_rsi_status(self):
        """Verify RSI status labels are applied correctly."""
        oversold = enrich_metric_metadata("rsi", 25.0, "LIVE")
        assert oversold["status"] == "OVERSOLD"

        overbought = enrich_metric_metadata("rsi", 75.0, "LIVE")
        assert overbought["status"] == "OVERBOUGHT"

        neutral = enrich_metric_metadata("rsi", 50.0, "LIVE")
        assert neutral["status"] == "NEUTRAL"

    def test_metadata_enrichment_adx_trend(self):
        """Verify ADX trend labels are applied correctly."""
        strong = enrich_metric_metadata("adx", 30.0, "LIVE")
        assert strong["status"] == "STRONG_TREND"

        weak = enrich_metric_metadata("adx", 20.0, "LIVE")
        assert weak["status"] == "WEAK_TREND"

    def test_all_synonyms_resolve_to_valid_columns(self):
        """Verify every synonym maps to a column that exists in one of our table sets."""
        all_known_cols = FUNDAMENTALS_COLS | FINANCIAL_HISTORY_COLS | TECHNICALS_COLS
        for synonym, resolved in SYNONYM_MAP.items():
            assert resolved in all_known_cols, (
                f"Synonym '{synonym}' resolves to '{resolved}' which is not in any table column set"
            )


# ══════════════════════════════════════════════════════════════
# Phase 2: Tool Registration & Whitelist Verification
# ══════════════════════════════════════════════════════════════

class TestToolRegistration:
    """Verify precision RAG tools are properly registered in the tool registry."""

    def test_tools_registered_in_registry(self):
        """All 3 precision RAG tools must be registered in the global registry."""
        from app.tools.registry import registry

        registered_names = set(registry.tools.keys())
        assert "query_financial_metrics" in registered_names, "query_financial_metrics not registered"
        assert "query_technical_indicator" in registered_names, "query_technical_indicator not registered"
        assert "search_database_facts" in registered_names, "search_database_facts not registered"

    def test_tools_have_schemas(self):
        """All 3 tools must have JSON schemas for Prism/vLLM."""
        from app.tools.registry import registry

        schema_names = {s["function"]["name"] for s in registry.schemas}
        assert "query_financial_metrics" in schema_names
        assert "query_technical_indicator" in schema_names
        assert "search_database_facts" in schema_names

    def test_tools_have_required_parameters(self):
        """Verify tool schemas have the expected required parameters."""
        from app.tools.registry import registry

        for schema in registry.schemas:
            name = schema["function"]["name"]
            params = schema["function"]["parameters"]

            if name == "query_financial_metrics":
                assert "ticker" in params["required"]
                assert "metrics" in params["required"]

            if name == "query_technical_indicator":
                assert "ticker" in params["required"]
                assert "indicator" in params["required"]

            if name == "search_database_facts":
                assert "ticker" in params["required"]
                assert "query" in params["required"]


class TestToolWhitelists:
    """Verify precision tools are bound to the correct agent roles."""

    def test_fundamental_agent_has_financial_metrics(self):
        """Fundamental agent must have query_financial_metrics and search_database_facts."""
        from app.agents.tool_whitelists import AGENT_TOOL_WHITELISTS

        assert "query_financial_metrics" in AGENT_TOOL_WHITELISTS["fundamental"]
        assert "search_database_facts" in AGENT_TOOL_WHITELISTS["fundamental"]

    def test_technical_agent_has_technical_indicator(self):
        """Technical agent must have query_technical_indicator."""
        from app.agents.tool_whitelists import AGENT_TOOL_WHITELISTS

        assert "query_technical_indicator" in AGENT_TOOL_WHITELISTS["technical"]

    def test_sentiment_agent_has_search_facts(self):
        """Sentiment agent must have search_database_facts."""
        from app.agents.tool_whitelists import AGENT_TOOL_WHITELISTS

        assert "search_database_facts" in AGENT_TOOL_WHITELISTS["sentiment"]

    def test_fund_flow_agent_has_search_facts(self):
        """Fund flow agent must have search_database_facts."""
        from app.agents.tool_whitelists import AGENT_TOOL_WHITELISTS

        assert "search_database_facts" in AGENT_TOOL_WHITELISTS["fund_flow"]


# ══════════════════════════════════════════════════════════════
# Phase 3: Prompt Reduction Verification
# ══════════════════════════════════════════════════════════════

class TestPromptInstructions:
    """Verify the system prompt includes precision tool instructions."""

    def test_system_prompt_mentions_precision_tools(self):
        """The build_system_prompt output should include precision tool instructions."""
        from app.cognition.debate.debate_coordinator import build_system_prompt

        prompt = build_system_prompt("bull", "Fundamental analysis expert")
        assert "PRECISION QUERY TOOLS" in prompt
        assert "query_financial_metrics" in prompt
        assert "query_technical_indicator" in prompt
        assert "search_database_facts" in prompt

    def test_system_prompt_mentions_cross_verify(self):
        """The prompt should instruct agents to cross-verify claims."""
        from app.cognition.debate.debate_coordinator import build_system_prompt

        prompt = build_system_prompt("bear", "Technical analysis expert")
        assert "cross-verify" in prompt.lower()


# ══════════════════════════════════════════════════════════════
# Phase 4: A/B Token & Latency Simulation
# ══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_ab_token_performance():
    """A/B Test Simulation: Compare prompt sizes and token counts.

    A: Static upfront context prompting (dumps all financial tables and technicals)
    B: Hybrid Precision RAG (loads only narrative context, queries values on-demand)
    """

    # 1. Setup mock data payload
    financial_data_table = (
        "## FINANCIAL STATEMENTS & MULTIPLES (FY 2025/2026):\n"
        "  - P/E Ratio: 28.5 (Industry Average: 25.0)\n"
        "  - Price to Sales: 7.2\n"
        "  - Enterprise Value to EBITDA: 20.4\n"
        "  - Total Revenue: $90.75 Billion (Q1 2026)\n"
        "  - Gross Margin: 45.2%\n"
        "  - Operating Margin: 30.8%\n"
        "  - Debt to Equity Ratio: 1.2 (Leveraged but stable)\n"
        "  - Free Cash Flow: $22.40 Billion (Q1 2026)\n"
        "  - Return on Equity (ROE): 154.2%\n"
        "  - Current Ratio: 1.05\n"
        "  - Quick Ratio: 0.92\n"
        "  - Dividend Yield: 0.52%\n"
        "  - Book Value per Share: $4.35\n"
        "  - Capital Expenditure (CapEx): $2.4 Billion\n"
    )

    technical_indicators_table = (
        "## TECHNICAL INDICATORS & MOMENTUM (DAILY):\n"
        "  - Relative Strength Index (RSI): 37.8 (Neutral-Low)\n"
        "  - MACD Line: -1.45\n"
        "  - MACD Signal Line: -1.12\n"
        "  - MACD Histogram: -0.33\n"
        "  - SMA 20: 178.24 (Price is currently below SMA 20)\n"
        "  - SMA 50: 181.50\n"
        "  - SMA 200: 175.40 (Price is above long-term SMA 200)\n"
        "  - Bollinger Bands: Upper=185.2, Mid=178.5, Lower=171.8\n"
        "  - Average True Range (ATR): 3.24\n"
        "  - Average Directional Index (ADX): 22.1 (Weak trend)\n"
    )

    narrative_context = (
        "## COMPANY NEWS & NARRATIVES:\n"
        "  - AAPL shows strong services growth but hardware sales flatten in China.\n"
        "  - CEO Tim Cook sold 10,000 shares on 2026-05-12.\n"
        "  - Short-term sentiment is slightly bearish due to supply chain warnings.\n"
        "  - Long-term thesis holds up on AI-related updates and developer interest.\n"
    )

    base_system_prompt = "You are a stock analyst. Analyze the stock and provide a final verdict."

    # 2. Control Prompt (Static Context)
    control_user_prompt = f"{financial_data_table}\n{technical_indicators_table}\n{narrative_context}"
    control_total_chars = len(base_system_prompt) + len(control_user_prompt)

    # Approx 4 characters per token
    control_est_tokens = control_total_chars // 4

    # 3. Variant Prompt (Hybrid Context - Math tables removed, query tools exposed)
    variant_user_prompt = (
        f"{narrative_context}\n"
        "NOTE: Numerical financials and technical indicators are not included in this prompt. "
        "Use your precision query tools (query_financial_metrics, query_technical_indicator) "
        "if you need specific values to verify any claims."
    )

    variant_total_chars = len(base_system_prompt) + len(variant_user_prompt)
    variant_est_tokens = variant_total_chars // 4

    # 4. Measure token savings
    token_savings = control_est_tokens - variant_est_tokens
    savings_pct = (token_savings / control_est_tokens) * 100

    print("\n")
    print("=" * 60)
    print("             A/B TOKEN PERFORMANCE REPORT             ")
    print("=" * 60)
    print(f" Control (Static Context) est. input tokens: {control_est_tokens}")
    print(f" Variant (Hybrid Context) est. input tokens: {variant_est_tokens}")
    print(f" Estimated Input Token Savings:             {token_savings} tokens")
    print(f" Savings Percentage:                        {savings_pct:.1f}%")
    print("=" * 60)

    # Assert that the variant reduces input tokens by at least 40%
    assert savings_pct >= 40.0, f"RAG Variant only saved {savings_pct:.1f}% tokens. Target is >= 40%."
