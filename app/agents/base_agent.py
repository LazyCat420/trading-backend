"""
Base agent pattern — every agent follows this exact structure.

Phase 2: Agents receive pre-computed data from processors.
Phase 3: Optional dynamic meta-prompt generates context-aware system prompts.
LLM only analyzes — never calculates.
"""

import datetime
import logging

from app.config import settings
from app.services.vllm_client import llm, Priority
from app.utils.text_utils import parse_json_response, sanitize_ascii
from app.utils.resilience import aresilient_call

logger = logging.getLogger(__name__)

# ─── Meta-prompt: generates a context-aware system prompt ───────────
AGENT_META_SYSTEM = """You are an expert at creating specialized analyst system prompts for stock market analysis.

Given an agent's role description and a preview of the market data, create an IMPROVED system prompt tailored to THIS specific analysis.

STRICT GUARDRAILS — you MUST follow these:
1. PRESERVE the exact JSON output schema from the original prompt (same keys, same value types)
2. The generated prompt must ONLY instruct the agent to analyze the data it receives — never tell it to fetch, search, or hallucinate data
3. Keep the prompt under 200 words — concise prompts produce better LLM output
4. Include "Respond with JSON:" followed by the exact schema from the original prompt
5. NEVER remove the instruction "do NOT recalculate" or "the data given is authoritative"

WHAT TO ADAPT based on the data preview:
- Identify the asset class: blue-chip stock, growth stock, penny stock, crypto, commodity, ETF
- For PENNY STOCKS (price < $5): emphasize liquidity risk, dilution risk, and pump-and-dump patterns
- For CRYPTO (BTC/ETH/XRP): skip P/E and fundamentals, focus on momentum and sentiment cycles
- For BLUE CHIPS: emphasize macro sensitivity, dividend sustainability, institutional positioning
- Reference specific data patterns you see (e.g., "RSI is oversold" or "revenue declining")
- Name the sector/industry if identifiable from the ticker or data

Respond with ONLY JSON:
{"system_prompt": "the full improved system prompt with JSON schema preserved", "focus_rationale": "1 sentence on what you adapted and why"}"""

AGENT_META_USER = """## Agent Role: {agent_name}

## Original System Prompt (template — preserve its JSON output schema exactly):
{static_prompt}

## Data Preview (first 8000 chars of what the agent will analyze):
{data_preview}

---

Create a better, more specific system prompt for this agent. You MUST preserve the exact JSON output schema from the original prompt. Adapt the analytical focus to what matters most for this specific ticker and data."""


# _parse_json_response moved to app.utils.text_utils.parse_json_response
_parse_json_response = parse_json_response

# ── Agents that receive prior trade outcome context ──
_OUTCOME_CONTEXT_AGENTS = frozenset({
    "sentiment", "technical", "fundamental", "risk", "fund_flow",
    "comparative", "retriever",
})


def get_ticker_outcome_context(ticker: str) -> str:
    """Pull resolved trade outcomes for this ticker from the DB.

    Returns a formatted string for analyst prompt injection,
    or empty string if no history exists.  Queries PostgreSQL
    (decision_outcomes table) — deterministic, bounded, no flat-file I/O.
    """
    if not ticker or ticker.startswith("_"):
        return ""  # Skip synthetic tickers like _AUDIT_
    try:
        from app.pipeline.analysis.outcome_tracker import get_past_outcomes

        outcomes = get_past_outcomes(ticker=ticker, limit=5)
        if not outcomes:
            return ""

        lines = [f"\n## PRIOR TRADE HISTORY FOR {ticker}"]
        for o in outcomes:
            lines.append(
                f"- {o['outcome']}: entry=${o.get('entry_price', 0):.2f} → "
                f"exit=${o.get('exit_price', 0):.2f} ({o.get('pnl_pct', 0):+.1f}%) "
                f"conf={o.get('confidence', 0)} [{o.get('resolved_at', '?')}]"
            )
        lines.append(
            "Use this history to calibrate your confidence — "
            "do not repeat past mistakes.\n"
        )
        return "\n".join(lines)
    except Exception:
        return ""


async def _generate_dynamic_prompt(
    agent_name: str,
    static_prompt: str,
    data_context: str,
    ticker: str,
    cycle_id: str,
    bot_id: str,
) -> tuple[str, str]:
    """Generate a context-aware system prompt via meta-prompt.

    Returns (dynamic_prompt, focus_rationale). Falls back to static_prompt on failure.
    """
    meta_user = AGENT_META_USER.format(
        agent_name=agent_name,
        static_prompt=static_prompt,
        data_preview=data_context[:8000]
        if data_context
        else "No data preview available",
    )

    try:

        @aresilient_call(
            retries=2, backoff="exponential", base_delay=1.0, max_delay=10.0
        )
        async def _meta_llm_call():
            from app.services.prism_agent_caller import call_prism_agent
            return await call_prism_agent(
                agent_id=f"CUSTOM_{agent_name.upper()}_META",
                user_message=meta_user,
                fallback_system_prompt=AGENT_META_SYSTEM,
                fallback_agent_name=f"{agent_name}_meta",
                temperature=0.5,
                max_tokens=2048,
                priority=Priority.NORMAL,
                ticker=ticker,
                cycle_id=cycle_id,
                bot_id=bot_id,
            )

        response, tokens, ms = await _meta_llm_call()
        parsed = _parse_json_response(response)
        dynamic_prompt = parsed.get("system_prompt", "")
        rationale = parsed.get("focus_rationale", "")

        if dynamic_prompt and len(dynamic_prompt) > 50:
            logger.info(
                "[META] %s: generated dynamic prompt (%d chars, %d tokens, %dms) — %s",
                agent_name,
                len(dynamic_prompt),
                tokens,
                ms,
                rationale[:80],
            )
            return dynamic_prompt, rationale

        logger.warning(
            "[META] %s: generated prompt too short, using static", agent_name
        )
        return static_prompt, "fallback: generated prompt too short"
    except Exception as e:
        logger.warning(
            "[META] %s: meta-prompt failed (%s), using static", agent_name, e
        )
        return static_prompt, f"fallback: {e}"


async def run_agent(
    agent_name: str,
    ticker: str,
    cycle_id: str,
    bot_id: str,
    system_prompt: str,
    user_prompt: str,
    data_context: str = "",
    temperature: float = 0.3,
    max_tokens: int = 1024,
    enable_dynamic_prompt: bool = False,
    endpoint_override: str | None = None,
    enable_tools: bool = False,
    response_format: dict | None = None,
) -> dict:
    """
    Generic agent runner:
    1. Optionally generate a dynamic system prompt via meta-prompt
    2. Inject data_context (pre-computed signals) into user prompt
    3. Call llm.chat() with monitoring metadata
    4. Return structured result dict

    Every specific agent builds its own prompts and calls this.
    """
    # ── Optional: generate dynamic system prompt ──
    dynamic_rationale = ""
    actual_system_prompt = system_prompt
    
    # ── Override with Generated Prompt if available and better ──
    try:
        from app.db.connection import get_db
        with get_db() as db:
            row = db.execute(
                """
                SELECT system_prompt, performance_score 
                FROM generated_agent_prompts 
                WHERE active = TRUE AND lens_type = %s
                ORDER BY performance_score DESC, win_rate DESC
                LIMIT 1
                """,
                [agent_name]
            ).fetchone()
            if row and row[0]:
                actual_system_prompt = row[0]
                logger.info(f"[BaseAgent] Using dynamically generated prompt for {agent_name} (score: {row[1]})")
    except Exception:
        pass # Table might not exist or be empty
    
    # ── Fetch Tool Playbook Rules ──
    playbook_rules = ""
    try:
        from app.db.connection import get_db
        with get_db() as db:
            cur = db.execute(
                "SELECT recommended_tool_sequence, stop_conditions, bad_patterns_to_avoid "
                "FROM tool_playbook WHERE agent_role = %s",
                (agent_name,)
            )
            rows = cur.fetchall()
            if rows:
                rules = []
                for r in rows:
                    if r[0]: rules.append(f"- Sequence: {r[0]}")
                    if r[1]: rules.append(f"- Stop Condition: {r[1]}")
                    if r[2]: rules.append(f"- Avoid: {r[2]}")
                if rules:
                    playbook_rules = "\n\n### TOOL PLAYBOOK RULES:\n" + "\n".join(rules)
    except Exception as e:
        logger.error(f"[BaseAgent] Failed to fetch playbook rules: {e}")

    if playbook_rules:
        actual_system_prompt += playbook_rules

    if enable_dynamic_prompt and data_context:
        actual_system_prompt, dynamic_rationale = await _generate_dynamic_prompt(
            agent_name=agent_name,
            static_prompt=system_prompt,
            data_context=data_context,
            ticker=ticker,
            cycle_id=cycle_id,
            bot_id=bot_id,
        )

    # ── Inject prior trade outcome context for analysis agents ──
    outcome_ctx = ""
    if agent_name in _OUTCOME_CONTEXT_AGENTS:
        outcome_ctx = get_ticker_outcome_context(ticker)

    # ── Budget-aware data truncation ──
    # Prevent any single component from blowing the context window
    from app.config.context_budget import get_context_budget

    ctx_budget = get_context_budget()
    if data_context and len(data_context) > ctx_budget.data_context_chars:
        original_len = len(data_context)
        data_context = data_context[: ctx_budget.data_context_chars]
        logger.info(
            "[BaseAgent] %s data_context truncated: %d -> %d chars (budget=%d)",
            agent_name,
            original_len,
            len(data_context),
            ctx_budget.data_context_chars,
        )

    # Inject pre-computed data before the analysis request
    if data_context:
        full_prompt = f"{outcome_ctx}{data_context}\n\n{user_prompt}"
    else:
        full_prompt = f"{outcome_ctx}{user_prompt}" if outcome_ctx else user_prompt

    # ── Verbose input logging ──
    prompt_label = (
        "DYNAMIC"
        if (enable_dynamic_prompt and actual_system_prompt != system_prompt)
        else "STATIC"
    )
    print(f"\n  {'~' * 50}")
    print(f"  AGENT INPUT: {agent_name} ({ticker}) [{prompt_label} PROMPT]")
    print(f"  {'~' * 50}")
    print(f"  System Prompt ({len(actual_system_prompt)} chars):")
    safe_sys = sanitize_ascii(actual_system_prompt)
    print(f"    {safe_sys}")
    if dynamic_rationale and prompt_label == "DYNAMIC":
        print(f"  Meta-Prompt Focus: {dynamic_rationale}")
    print(f"  User Prompt ({len(full_prompt)} chars):")
    safe_user = sanitize_ascii(full_prompt)
    print(f"    {safe_user}")
    print(f"  {'~' * 50}")

    @aresilient_call(retries=3, backoff="exponential", base_delay=1.0, max_delay=15.0)
    async def _agent_llm_call():
        from app.agents.tool_whitelists import get_agent_tools, get_agent_budget_turns

        # Per-agent tool whitelist: only show tools relevant to this agent's role
        agent_tools = get_agent_tools(agent_name) if enable_tools else None

        # Try routing via Prism agent harness first if routing is enabled and healthy
        if settings.PRISM_ENABLED and settings.PRISM_AGENT_ROUTING:
            try:
                prism_healthy = await llm.prism_client.check_health()
                if prism_healthy:
                    from app.tools.prism_agent_harness import run_prism_agent
                    logger.info("[BaseAgent] Routing %s agentic loop to Prism /agent", agent_name)
                    result = await run_prism_agent(
                        system_prompt=actual_system_prompt,
                        user_prompt=full_prompt,
                        ticker=ticker,
                        agent_name=agent_name,
                        cycle_id=cycle_id,
                        bot_id=bot_id,
                        priority=Priority.NORMAL,
                        tools_override=agent_tools,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    return (
                        result.get("final_text", ""),
                        result.get("token_usage", 0),
                        result.get("execution_ms", 0),
                    )
            except Exception as pe:
                logger.error("[BaseAgent] Prism agent routing failed for %s, falling back to local: %s", agent_name, pe)

        # Fallback to local agent loop
        from app.agents.agent_loop import run_agent_loop, run_split_agent_loop
        from app.agents.agent_budget import AgentBudget

        # Role-differentiated budget: risk=5, verifier=5, meta_audit=10, default=3
        budget_turns = get_agent_budget_turns(agent_name, enable_tools)
        budget = AgentBudget(max_turns=budget_turns)

        # Use Brain-Action split loop when tools are enabled to save context
        # The split loop first selects which tools are needed (lightweight call),
        # then runs the real agent loop with only the selected subset.
        if enable_tools and agent_tools:
            result = await run_split_agent_loop(
                system_prompt=actual_system_prompt,
                user_prompt=full_prompt,
                ticker=ticker,
                agent_name=agent_name,
                cycle_id=cycle_id,
                bot_id=bot_id,
                budget=budget,
                priority=Priority.NORMAL,
                tools_override=agent_tools,
                require_json_schema=True,
            )
        else:
            result = await run_agent_loop(
                system_prompt=actual_system_prompt,
                user_prompt=full_prompt,
                ticker=ticker,
                agent_name=agent_name,
                cycle_id=cycle_id,
                bot_id=bot_id,
                budget=budget,
                priority=Priority.NORMAL,
                tools_override=agent_tools,
                require_json_schema=True,
            )
        return (
            result.get("final_text", ""),
            result.get("token_usage", 0),
            result.get("execution_ms", 0),
        )

    content, tokens, elapsed_ms = await _agent_llm_call()

    if not content or not str(content).strip():
        content = f"Agent failed: empty response from {agent_name}"

    # ── Verbose output logging ──
    print(f"\n  {'~' * 50}")
    print(f"  AGENT OUTPUT: {agent_name} ({ticker}) [{tokens} tokens, {elapsed_ms}ms]")
    print(f"  {'~' * 50}")
    safe_content = sanitize_ascii(content) if content else ""
    print(f"    {safe_content}")
    print(f"  {'~' * 50}")

    return {
        "agent": agent_name,
        "ticker": ticker,
        "cycle_id": cycle_id,
        "bot_id": bot_id,
        "response": content,
        "tokens_used": tokens,
        "execution_ms": elapsed_ms,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "dynamic_prompt_used": enable_dynamic_prompt
        and actual_system_prompt != system_prompt,
        "dynamic_prompt": actual_system_prompt
        if actual_system_prompt != system_prompt
        else None,
        "dynamic_rationale": dynamic_rationale if dynamic_rationale else None,
    }
