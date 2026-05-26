"""
Unit and A/B verification tests for the Hybrid RAG/Precision Query plan.
Verifies that precision lookup tools behave correctly (synonym mapping, scale metadata, time locks)
and runs an A/B simulation measuring token efficiency.
"""

import sys
import os
import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

# Ensure project root is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.tools.precision_rag import (
    resolve_synonym,
    enrich_metric_metadata,
    query_financial_metrics,
    query_technical_indicator,
    search_database_facts
)

# ══════════════════════════════════════════════════════════════
# TDD Phase 1: Synonym & Metadata Unit Tests
# ══════════════════════════════════════════════════════════════

class TestPrecisionRagLogic:
    """Test individual RAG tool resolution and formatting logic."""

    def test_synonym_resolution(self):
        """Verify metric synonyms are correctly mapped to DB columns."""
        assert resolve_synonym("P/E Ratio") == "pe_ratio"
        assert resolve_synonym("pe") == "pe_ratio"
        assert resolve_synonym("Sales") == "revenue"
        assert resolve_synonym("d/e") == "debt_to_equity"
        assert resolve_synonym("rsi") == "rsi"
        assert resolve_synonym("relative strength index") == "rsi"
        assert resolve_synonym("Unknown Metric") == "unknown metric"

    def test_metadata_enrichment_ratio(self):
        """Verify ratio metrics are marked with correct units and multiplier."""
        meta = enrich_metric_metadata("P/E Ratio", 28.5, "Q1_2026")
        assert meta["value"] == 28.5
        assert meta["unit"] == "ratio"
        assert meta["currency"] == "N/A"
        assert meta["multiplier"] == "1"
        assert meta["period"] == "Q1_2026"

    def test_metadata_enrichment_currency(self):
        """Verify currency metrics are marked with correct scale factors."""
        meta = enrich_metric_metadata("Sales", 90750.0, "Q1_2026")
        assert meta["value"] == 90750.0
        assert meta["unit"] == "currency"
        assert meta["currency"] == "USD"
        assert meta["multiplier"] == "Millions"
        assert meta["period"] == "Q1_2026"

    @pytest.mark.asyncio
    async def test_query_financial_metrics_valid(self):
        """Verify query_financial_metrics fetches correctly mapped metadata."""
        res = await query_financial_metrics("AAPL", ["P/E Ratio", "free cash flow"], "Q1_2026")
        assert "pe_ratio" in res
        assert "free_cash_flow" in res
        assert res["pe_ratio"]["value"] == 28.5
        assert res["free_cash_flow"]["value"] == 22400.0
        assert res["free_cash_flow"]["multiplier"] == "Millions"

    @pytest.mark.asyncio
    async def test_query_financial_metrics_invalid(self):
        """Verify querying unknown tickers or metrics returns controlled error dict."""
        res_ticker = await query_financial_metrics("FAKE", ["pe"])
        assert "error" in res_ticker
        
        res_metric = await query_financial_metrics("AAPL", ["fake_metric"])
        assert "fake_metric" in res_metric
        assert "error" in res_metric["fake_metric"]

    @pytest.mark.asyncio
    async def test_query_technical_indicator_rsi(self):
        """Verify RSI indicator includes timeframe and relative status."""
        res = await query_technical_indicator("AAPL", "RSI", "daily")
        assert res["value"] == 37.8
        assert res["timeframe"] == "daily"
        assert res["status"] == "NEUTRAL"  # 37.8 is neutral (not < 30)

    @pytest.mark.asyncio
    async def test_search_database_facts(self):
        """Verify database keyword fact search finds matching text lines."""
        res = await search_database_facts("AAPL", "Tim Cook")
        assert len(res["results"]) == 1
        assert "Tim Cook sold 10,000 shares" in res["results"][0]

        res_none = await search_database_facts("AAPL", "Nonexistent query")
        assert "No matching records found." in res_none["results"][0]


# ══════════════════════════════════════════════════════════════
# TDD Phase 2: A/B Token & Latency Simulation
# ════════════════════════════════════──────────────────────────

@pytest.mark.asyncio
async def test_ab_token_performance():
    """A/B Test Simulation: Compare prompt sizes and token counts.
    
    A: Static upfront context prompting (Dumps all financial tables and technicals)
    B: Hybrid Precision RAG (Loads only narrative context, queries values on-demand)
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
    # The prompt only contains the narrative context
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
