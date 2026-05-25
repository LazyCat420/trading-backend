"""
Consolidated tests for the Quant Strategy Tools, Quant Research Agent whitelist,
Synthesizer whitelist, Reflector lesson injection/scoring, and Phase 6 agent launches.
"""

from unittest.mock import patch, MagicMock, AsyncMock
import pytest
import asyncio


# ─────────────────────────────────────────────────────────────
# 1. Quant Tools: Momentum Strategy
# ─────────────────────────────────────────────────────────────

@patch("app.db.connection.get_db")
@pytest.mark.asyncio
async def test_quant_tools_momentum_bullish(mock_get_db):
    """Verify execute_momentum_strategy returns BUY for strong bullish data."""
    from app.tools.quant_tools import execute_momentum_strategy

    mock_db = MagicMock()
    mock_get_db.return_value.__enter__.return_value = mock_db
    # RSI=55 (neutral-bullish), MACD=1.5 > Signal=0.5 (bullish),
    # SMA20=100, SMA50=95, SMA200=80, Close=110 (above all SMAs)
    mock_db.execute.return_value.fetchone.return_value = (55.0, 1.5, 0.5, 100.0, 95.0, 80.0, 110.0)

    result = await execute_momentum_strategy("TEST")

    assert "BUY" in result
    assert "Score:" in result
    assert "RSI is neutral-bullish" in result
    assert "MACD is bullish" in result


@patch("app.db.connection.get_db")
@pytest.mark.asyncio
async def test_quant_tools_momentum_bearish(mock_get_db):
    """Verify execute_momentum_strategy returns SELL for bearish data."""
    from app.tools.quant_tools import execute_momentum_strategy

    mock_db = MagicMock()
    mock_get_db.return_value.__enter__.return_value = mock_db
    # RSI=75 (overbought), MACD=-1.0 < Signal=0.5 (bearish),
    # SMA20=120, SMA50=125, SMA200=130, Close=90 (below all SMAs)
    mock_db.execute.return_value.fetchone.return_value = (75.0, -1.0, 0.5, 120.0, 125.0, 130.0, 90.0)

    result = await execute_momentum_strategy("TEST")

    assert "SELL" in result
    assert "overbought" in result
    assert "MACD is bearish" in result


@patch("app.db.connection.get_db")
@pytest.mark.asyncio
async def test_quant_tools_momentum_missing_data(mock_get_db):
    """Verify execute_momentum_strategy returns error when no data exists."""
    from app.tools.quant_tools import execute_momentum_strategy

    mock_db = MagicMock()
    mock_get_db.return_value.__enter__.return_value = mock_db
    mock_db.execute.return_value.fetchone.return_value = None

    result = await execute_momentum_strategy("FAKE")

    assert "Error" in result
    assert "Insufficient" in result


# ─────────────────────────────────────────────────────────────
# 2. Quant Tools: Value Strategy
# ─────────────────────────────────────────────────────────────

@patch("app.db.connection.get_db")
@pytest.mark.asyncio
async def test_quant_tools_value_attractive(mock_get_db):
    """Verify execute_value_strategy returns BUY for undervalued stock."""
    from app.tools.quant_tools import execute_value_strategy

    mock_db = MagicMock()
    mock_get_db.return_value.__enter__.return_value = mock_db
    # PE=12 (attractive), ForwardPE=10, PEG=0.8 (undervalued growth),
    # PB=1.5 (attractive), DE=0.6 (healthy)
    mock_db.execute.return_value.fetchone.return_value = (12.0, 10.0, 0.8, 1.5, 0.6)

    result = await execute_value_strategy("TEST")

    assert "BUY" in result
    assert "P/E ratio is attractive" in result
    assert "PEG ratio indicates undervalued growth" in result
    assert "Debt to Equity is healthy" in result


@patch("app.db.connection.get_db")
@pytest.mark.asyncio
async def test_quant_tools_value_overvalued(mock_get_db):
    """Verify execute_value_strategy returns HOLD/SELL for overvalued stock."""
    from app.tools.quant_tools import execute_value_strategy

    mock_db = MagicMock()
    mock_get_db.return_value.__enter__.return_value = mock_db
    # PE=45 (high), ForwardPE=40, PEG=3.5 (overvalued growth),
    # PB=8.0 (not attractive), DE=2.5 (high/risky)
    mock_db.execute.return_value.fetchone.return_value = (45.0, 40.0, 3.5, 8.0, 2.5)

    result = await execute_value_strategy("TEST")

    # Score should be negative: PE(-1) + PEG(-1) + DE(-1) = -3 → SELL
    assert "SELL" in result
    assert "P/E ratio is high" in result
    assert "overvalued growth" in result
    assert "Debt to Equity is high/risky" in result


@patch("app.db.connection.get_db")
@pytest.mark.asyncio
async def test_quant_tools_value_missing_data(mock_get_db):
    """Verify execute_value_strategy returns error when no data exists."""
    from app.tools.quant_tools import execute_value_strategy

    mock_db = MagicMock()
    mock_get_db.return_value.__enter__.return_value = mock_db
    mock_db.execute.return_value.fetchone.return_value = None

    result = await execute_value_strategy("FAKE")

    assert "Error" in result
    assert "Insufficient" in result


# ─────────────────────────────────────────────────────────────
# 3. Tool Whitelists: quant_research agent
# ─────────────────────────────────────────────────────────────

def test_quant_research_agent_whitelist():
    """Verify quant_research agent has a dedicated whitelist with web + memory tools."""
    from app.agents.tool_whitelists import AGENT_TOOL_WHITELISTS

    assert "quant_research" in AGENT_TOOL_WHITELISTS
    wl = AGENT_TOOL_WHITELISTS["quant_research"]

    # Must have web search tools
    assert "search_web" in wl
    assert "scrape_url" in wl

    # Must have memory tools
    assert "write_memory_note" in wl
    assert "read_memory_note" in wl
    assert "execute_python" in wl

    # Must NOT have destructive trading tools
    assert "buy_stock" not in wl
    assert "sell_stock" not in wl


# ─────────────────────────────────────────────────────────────
# 4. Tool Whitelists: synthesizer agent has strategy tools
# ─────────────────────────────────────────────────────────────

def test_synthesizer_whitelist_strategies():
    """Verify synthesizer agent can see strategy execution tools."""
    from app.agents.tool_whitelists import AGENT_TOOL_WHITELISTS

    assert "synthesizer" in AGENT_TOOL_WHITELISTS
    wl = AGENT_TOOL_WHITELISTS["synthesizer"]

    assert "execute_momentum_strategy" in wl
    assert "execute_value_strategy" in wl
    assert "execute_python" in wl

    # Must still have core context tools
    assert "get_cycle_context" in wl


# ─────────────────────────────────────────────────────────────
# 5. Reflector: Lesson injection into system prompt
# ─────────────────────────────────────────────────────────────

@patch("app.cognition.evolution.reflector.get_db")
def test_reflector_lesson_injection(mock_get_db):
    """Verify get_agent_lessons returns lessons from DB."""
    from app.cognition.evolution.reflector import get_agent_lessons

    mock_db = MagicMock()
    mock_get_db.return_value.__enter__.return_value = mock_db

    # Simulate 2 lessons in DB
    mock_db.execute.return_value.fetchall.return_value = [
        ("Always verify RSI before claiming momentum",),
        ("Call get_technical_indicators before any technical claim",),
    ]

    lessons = get_agent_lessons("synthesizer", limit=3)

    assert len(lessons) == 2
    assert "RSI" in lessons[0]
    assert "get_technical_indicators" in lessons[1]


@patch("app.cognition.evolution.reflector.get_db")
def test_reflector_lesson_injection_empty(mock_get_db):
    """Verify get_agent_lessons returns empty list when no lessons exist."""
    from app.cognition.evolution.reflector import get_agent_lessons

    mock_db = MagicMock()
    mock_get_db.return_value.__enter__.return_value = mock_db
    mock_db.execute.return_value.fetchall.return_value = []

    lessons = get_agent_lessons("new_agent", limit=3)

    assert lessons == []


# ─────────────────────────────────────────────────────────────
# 6. Reflector: Score adjustment
# ─────────────────────────────────────────────────────────────

@patch("app.cognition.evolution.reflector.get_db")
def test_reflector_score_adjustment_success(mock_get_db):
    """Verify adjust_lesson_score increments on success."""
    from app.cognition.evolution.reflector import adjust_lesson_score

    mock_db = MagicMock()
    mock_get_db.return_value.__enter__.return_value = mock_db

    adjust_lesson_score("synthesizer", success=True)

    # Should have called UPDATE with +0.1
    call_args = mock_db.execute.call_args
    assert "+0.1" in call_args[0][0] or "+ 0.1" in call_args[0][0]


@patch("app.cognition.evolution.reflector.get_db")
def test_reflector_score_adjustment_failure(mock_get_db):
    """Verify adjust_lesson_score decrements on failure."""
    from app.cognition.evolution.reflector import adjust_lesson_score

    mock_db = MagicMock()
    mock_get_db.return_value.__enter__.return_value = mock_db

    adjust_lesson_score("synthesizer", success=False)

    # Should have called UPDATE with -0.2
    call_args = mock_db.execute.call_args
    assert "-0.2" in call_args[0][0] or "- 0.2" in call_args[0][0]


# ─────────────────────────────────────────────────────────────
# 7. Phase 6: Both post-cycle agents fire
# ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_phase6_quant_research_launch():
    """Verify phase6_post imports and would launch both meta_audit and quant_research."""
    # We verify the import path is correct and the module has the expected function
    from app.agents.quant_research_agent import run_quant_research
    from app.agents.meta_audit_agent import run_meta_audit

    # Both must be coroutine functions (async)
    assert asyncio.iscoroutinefunction(run_quant_research)
    assert asyncio.iscoroutinefunction(run_meta_audit)


def test_phase6_post_imports_quant_research():
    """Verify phase6_post.py source code references quant_research_agent."""
    import inspect
    from app.cycle.phases import phase6_post

    source = inspect.getsource(phase6_post)
    assert "quant_research_agent" in source
    assert "run_quant_research" in source
    assert "run_meta_audit" in source
