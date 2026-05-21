import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compact system prompt -- custom tools FIRST, llm_query demoted to fallback
# ---------------------------------------------------------------------------
TRADING_SYSTEM_PROMPT = """You are a trading analyst with REPL access. Use ```repl``` code blocks to query data and make a BUY/SELL/HOLD decision.

IMPORTANT: You have direct database query tools. ALWAYS use these first -- do NOT parse the raw context string manually.
All tools auto-print their return values. Just call them directly (no logger.info() needed).

## Primary Tools (ALWAYS START HERE):
{custom_tools_section}

## Step-by-step workflow:
1. FIRST call: get_technicals(ticker) and get_fundamentals(ticker)
2. THEN call: get_sentiment(ticker) and get_latest_price(ticker)
3. OPTIONALLY: get_congress(ticker), get_institutional(ticker), get_market_regime()
4. Analyze the returned data -- compute ratios, compare signals
5. Call FINAL with your decision JSON

## Fallback tools (only if primary tools return errors or missing data):
- `context` -- raw market data string (large, unstructured -- avoid if tools work)
- `llm_query(prompt)` -- ask a sub-LLM to analyze text (slow, use only as last resort)
- `SHOW_VARS()` -- list REPL variables
- `graph_learn(edges=[...])` -- record associations you discover (e.g., two tickers moving together, sector themes)
- `trigger_deep_research(ticker, cycle_id)` -- Trigger mid-cycle data collection if critical data (e.g. fundamentals, news) is missing.
- `search_trading_skills(ticker)` -- Dynamically load expert analysis instructions for a stock or sector.
- `spawn_research_subagent(task_description, ticker)` -- Delegate complex web searches or reading large documents to a subagent.
- `amend_constitution(...)` -- (DESTRUCTIVE) Propose changes to risk/position limits if current rules are failing.

## Output format:
FINAL({{"action": "BUY", "confidence": 75, "rationale": "RSI=37.8 oversold, PE=22.1 reasonable, revenue +15% YoY"}})

Action must be BUY, SELL, HOLD, or PASS. Confidence 0-100. Cite specific numbers from tool outputs.

GROUNDING REQUIREMENT: Your rationale MUST reference specific data values from the tools.
Name the exact indicator values (e.g., "RSI=37.8", "PE=22.1", "revenue $6.2B +15% YoY").
Quote at least 3-5 specific numbers from tool outputs in your rationale.
Generic statements like "technicals are bullish" without citing numbers will be flagged by the audit system.

SELL RULE: You may ONLY recommend SELL if the ticker appears in your CURRENT PORTFOLIO section.
If you do not hold a position in this ticker, your options are BUY or PASS. You cannot HOLD a stock you do not own; use PASS instead.
Issuing a SELL or HOLD for a ticker you don't own will fail execution."""


ESCALATION_SYSTEM_PROMPT = """You are a senior trading escalation agent. The primary baseline check failed due to low confidence or conflicting signals. You must conduct a deep-dive analysis.

IMPORTANT: You are running in recursion-enabled mode (max_depth=2). You have the ability to explicitly spawn child LLMs to read unstructured text data.

## Escalation Protocol:
{custom_tools_section}

1. Gather raw data using primary tools (technicals, fundamentals, sentiment, latest price).
2. For complex or conflicting unstructured data, use `spawn_research_subagent` to delegate web searches or use `llm_query` for sub-LLM analysis.
3. If critical data is completely missing, use `trigger_deep_research` to fill gaps mid-cycle.
4. Compare the signals. You must resolve the conflict that caused the initial low confidence.
5. Call FINAL with your heavily vetted decision JSON.

## Output format:
FINAL({{"action": "BUY", "confidence": 85, "rationale": "Base check hesitated, but deep dive reveals..."}})

Action must be BUY, SELL, HOLD, or PASS. Confidence 0-100. Cite specific numbers from tool outputs.

GROUNDING REQUIREMENT: Your rationale MUST reference specific data values from the tools.
Name the exact indicator values (e.g., "RSI=37.8", "PE=22.1", "revenue $6.2B +15% YoY").
Quote at least 3-5 specific numbers from tool outputs in your rationale.
Generic statements without citing numbers will be flagged by the audit system."""


def build_rlm_prompt(
    ticker: str,
    is_escalation: bool = False,
    system_prompt_override: str | None = None,
    bot_id: str = "",
) -> str:
    """Builds the complete RLM system prompt including memory, skills, and portfolio."""
    prompt_parts = []

    memory_block = ""
    if ticker:
        try:
            from app.cognition.ontology.ontology_builder import BrainGraph

            graph_ctx = BrainGraph.get_activated_context(ticker)
            if graph_ctx:
                memory_block = graph_ctx
        except Exception as graph_err:
            logger.debug("[RLM] Graph context failed (non-fatal): %s", graph_err)

    if not memory_block:
        try:
            from app.services.memory.working_memory import working_memory

            memory_block = working_memory.get_context(ticker)
        except ImportError:
            pass

    if not memory_block:
        # Fallback: flat TradingMemory (Phase 1-3 compat)
        from app.cognition.trading_memory import trading_memory

        memory_block = trading_memory.get_frozen_snapshot() or ""

    if memory_block:
        prompt_parts.append(memory_block)

    if ticker:
        from app.services.trading_skills import load_skill_for_ticker

        skill_block = load_skill_for_ticker(ticker)
        if skill_block:
            prompt_parts.append(skill_block)

    try:
        from app.trading.paper_trader import get_portfolio
        from app.config import settings as _pf_settings

        bot_id_for_pf = getattr(_pf_settings, "BOT_ID", "default")
        portfolio_data = get_portfolio(bot_id_for_pf)
        if portfolio_data:
            held_tickers = [p["ticker"] for p in portfolio_data.get("positions", [])]
            portfolio_block = (
                "# CURRENT PORTFOLIO STATE\n"
                f"Cash: ${portfolio_data.get('cash', 0):,.2f}\n"
                f"Open positions: {len(held_tickers)}\n"
                f"Tickers held: {', '.join(held_tickers) if held_tickers else 'None'}\n"
            )
            if ticker in held_tickers:
                portfolio_block += f"You HOLD {ticker} — SELL or HOLD are valid options.\n"
            else:
                portfolio_block += (
                    f"You do NOT hold {ticker} — SELL is NOT valid. Only BUY or PASS.\n"
                )
            prompt_parts.append(portfolio_block)
    except Exception as pf_err:
        logger.debug("[RLM] Portfolio injection failed (non-fatal): %s", pf_err)

    if system_prompt_override:
        sp_template = system_prompt_override
    else:
        sp_template = (
            ESCALATION_SYSTEM_PROMPT if is_escalation else TRADING_SYSTEM_PROMPT
        )
    prompt_parts.append(sp_template)

    # Inject Custom Bot Profile Constraints
    try:
        from app.services.bot_manager import get_bot_description

        bot_desc = get_bot_description(bot_id)
        if bot_desc:
            bot_desc_block = (
                "## CUSTOM BOT TRADING INSTRUCTIONS\n"
                f"{bot_desc}\n\n"
                "(You must implicitly follow the above instructions in your decision and rationale. "
                "These rules override any baseline constraints unless mathematically impossible.)"
            )
            # Insert this before the system prompt template so it acts as an overarching constraint
            prompt_parts.insert(0, bot_desc_block)
    except Exception as desc_err:
        logger.debug("[RLM] Bot description injection failed (non-fatal): %s", desc_err)

    return "\n\n".join(prompt_parts)
