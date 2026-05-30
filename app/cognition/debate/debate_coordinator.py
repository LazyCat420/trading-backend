"""
Debate Coordinator — Adversarial Bull/Bear Debate System.

Spawns bull and bear agents in parallel across both DGX Sparks,
then runs a cross-examiner to flag claims where cited values
don't match the evidence packet.

Architecture:
  1. BullAgent (forced BUY bias) → DGX Spark 1  ─┐
  2. BearAgent (forced SELL bias) → DGX Spark 2  ─┤ asyncio.gather()
  3. CrossExaminer receives both outputs           │
     and flags unverifiable/contradictory claims   ┘
  4. Returns raw debate output for ClaimVerifier + DebateJudge

All LLM calls go through app.services.vllm_client (Rule 2).
"""

import asyncio
import logging
from datetime import datetime, timezone

from app.services.vllm_client import llm, Priority
from app.config.config_cognition import LLM_TEMPERATURES, cognition_settings
from app.config.context_budget import get_context_budget
from app.cognition.contracts.evidence import EvidencePacket
from app.tools.registry import registry
from app.cognition.contracts.debate import DebateResult
from app.cognition.reflection_utils import generate_critique_prompt

from app.cognition.debate.debate_judge import judge_debate
from app.cognition.contracts.retrieval import SourceDocRef

logger = logging.getLogger(__name__)


def format_source_ref_for_prompt(s) -> str:
    """Format SourceDocRef or dict into a detailed string containing metadata if available."""
    source_type = getattr(s, "source_type", None) or (s.get("source_type") if isinstance(s, dict) else "unknown")
    summary = getattr(s, "summary", None) or (s.get("summary") if isinstance(s, dict) else "")
    metadata = getattr(s, "metadata", None) or (s.get("metadata") if isinstance(s, dict) else None)
    
    meta_parts = [f"Source: {source_type}"]
    if metadata and isinstance(metadata, dict):
        for k, v in metadata.items():
            if v is not None:
                meta_parts.append(f"{k}: {v}")
    meta_str = ", ".join(meta_parts)
    return f"[{meta_str}] {summary}"


# ── Context Budget Cap for Debate Prompts ────────────────────────────
def _cap_debate_text(text: str, max_chars: int, label: str = "debate") -> str:
    """Truncate text to max_chars with a tail marker.

    Used to prevent opponent quotes and user prompts from exceeding
    the model's effective context window.
    """
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    marker = f"\n... [{label}: truncated from {len(text):,} to {max_chars:,} chars]"
    logger.info(
        "[DEBATE] %s truncated: %d -> %d chars",
        label,
        len(text),
        max_chars,
    )
    return truncated + marker


# ── Analyst Personas ─────────────────────────────────────────

PERSONAS = {
    "Fundamental": "Focus purely on valuation multiples, earnings trends, balance sheet health, ratios, and margins.",
    "Technical": "Focus purely on price action, moving averages, relative strength (RSI), volume patterns, and momentum indicators.",
    "Macro_Sentiment": "Focus purely on the broader macroeconomic regime, interest rates, industry catalysts, and social/news sentiment.",
}

# ── Evidence Partitioning — prevent cross-persona fact anchoring ──
PERSONA_EVIDENCE_FILTER: dict[str, list[str]] = {
    "Fundamental": [
        "pe_ratio", "earnings", "revenue", "margins", "debt", "fcf",
        "book_value", "dividend", "eps", "roe", "roa", "p_e", "p_b",
        "operating", "net_income", "balance", "cash_flow", "valuation",
        "fundamental", "financial", "ratio",
    ],
    "Technical": [
        "rsi", "sma", "ema", "volume", "macd", "bollinger", "atr",
        "moving_average", "momentum", "price", "close", "open", "high",
        "low", "support", "resistance", "trend", "technical", "indicator",
    ],
    "Macro_Sentiment": [
        "fed_rate", "sector_flow", "news_sentiment", "reddit_score",
        "interest_rate", "inflation", "gdp", "unemployment", "sentiment",
        "macro", "catalyst", "industry", "social", "news", "youtube",
        "congress", "institutional", "insider",
    ],
}

# ── Per-Persona Temperature Diversity ────────────────────────
PERSONA_TEMPERATURES: dict[str, dict[str, float]] = {
    "Fundamental": {"bull": 0.3, "bear": 0.3},   # Low — be precise with numbers
    "Technical":   {"bull": 0.5, "bear": 0.5},   # Mid — pattern interpretation varies
    "Macro_Sentiment": {"bull": 0.7, "bear": 0.7},  # Higher — narrative/sentiment is fuzzy
}


def filter_packet_for_persona(
    packet: EvidencePacket, persona_name: str,
) -> EvidencePacket:
    """Return a shallow copy of the evidence packet filtered to this persona's focus area.

    Each persona only sees facts whose fact_type matches its allowed keywords.
    This prevents cross-persona fact anchoring — Technical can't cite P/E because
    it never saw the data.
    """
    allowed_keys = PERSONA_EVIDENCE_FILTER.get(persona_name)
    if not allowed_keys:
        return packet  # Unknown persona — pass full packet

    filtered_facts = [
        f for f in packet.structured_facts
        if any(k in f.fact_type.lower() for k in allowed_keys)
    ]

    # If filtering removed ALL facts, fall back to full packet so the
    # persona isn't left completely blind (edge case: misclassified fact_types).
    if not filtered_facts and packet.structured_facts:
        logger.warning(
            "[DEBATE] Evidence filter for %s matched 0/%d facts — using full packet",
            persona_name,
            len(packet.structured_facts),
        )
        return packet

    return packet.model_copy(update={"structured_facts": filtered_facts})



def build_system_prompt(
    bias: str,
    persona_instructions: str,
    position_context: dict | None = None,
) -> str:
    """Build system prompt for a biased analyst agent.

    When position_context indicates the bot holds a position, the
    debate reframes:
    - Bull (held): argue HOLD — the position is still good
    - Bear (held): argue SELL — the bot should exit now
    """
    held = position_context.get("held", False) if position_context else False

    if held and bias == "bull":
        # Bull becomes BUY (add to position) or HOLD advocate for existing positions
        action_word = "BUY"
        framing = (
            "The bot ALREADY holds this position. Your job is to "
            "argue that it should BUY (add to position) or HOLD (keep current size) based on the evidence. "
            "Argue the original buy thesis is still intact/valid, momentum or news "
            "supports adding/holding, and exiting now would be "
            "premature."
        )
    elif held and bias == "bear":
        # Bear becomes SELL advocate for existing positions
        action_word = "SELL"
        pnl = position_context.get("unrealized_pnl_pct", 0)
        days = position_context.get("holding_days", 0)
        entry = position_context.get("avg_entry", 0)
        framing = (
            f"The bot ALREADY holds this position "
            f"(entry=${entry}, P&L={pnl:+.1f}%, held {days}d). "
            f"Your job is to argue the bot should EXIT NOW. "
            f"Consider: deteriorating fundamentals, opportunity cost "
            f"of trapped capital, approaching stop-loss, or better "
            f"alternatives."
        )
    else:
        # Default: standard BUY vs SELL framing for new positions
        action_word = "BUY" if bias == "bull" else "SELL"
        framing = ""

    bias_label = bias.capitalize()
    base = f"""You are a {bias_label} Analyst. Your job is to construct the STRONGEST possible {action_word} case for this stock based ONLY on the provided evidence.

YOUR ANALYTICAL FOCUS: {persona_instructions}
Even if your focus area lacks strong data, do your best to formulate arguments based on what is available.

{framing}

TICKER ANCHOR: You are analyzing ONLY the ticker provided in the "Entity" field below.
All tool calls MUST use EXACTLY that ticker. Do NOT query data for any other ticker.
If you call get_market_data, get_technical_indicators, or any tool — the ticker parameter MUST match the entity.

CRITICAL RULES:
- Every claim MUST end with an inline citation: [source_table:value]
  Example: "RSI at 37.8 suggests oversold conditions [technical_data:RSI=37.8]"
- Do NOT invent data. Only cite values that appear in the Structured Facts or retrieved by your tools.
- Weigh evidence dynamically based on the source's metadata credibility (e.g. prioritize official news/SEC filings over individual Reddit posts/scores).
- If evidence is genuinely weak for a {action_word} case, acknowledge it — but still build the best {action_word} case you can.
- Be specific with numbers, dates, and metrics.

PRECISION QUERY TOOLS (use when needed):
- If you need to verify a specific metric (P/E, revenue, debt-to-equity, FCF), call query_financial_metrics.
- If you need to check a technical indicator (RSI, MACD, SMA levels), call query_technical_indicator.
- If you want to verify a claim from a news article or search for insider/congressional activity, call search_database_facts.
- These tools return ground-truth values from the database. Use them to cross-verify any claims you cite.
- You can request multiple metrics in a single query_financial_metrics call for efficiency.

Output exactly this JSON:
{{
  "action": "{action_word}",
  "claims": [
    "claim 1 with [source:value] citation",
    "claim 2 with [source:value] citation",
    "claim 3 with [source:value] citation"
  ],
  "confidence": 0-100,
  "key_argument": "single strongest argument for your case"
}}"""
    return base


# ── Cross-Examiner System Prompt ─────────────────────────────────────
CROSS_EXAM_SYSTEM_PROMPT = """You are a hostile cross-examiner and impartial Jury in a financial analysis hearing.

You have received arguments from a Bull Analyst (BUY case) and a Bear Analyst (SELL case).
Your job is to challenge BOTH sides by verifying their claims against the actual data.

1. For each claim, check if the cited [source:value] data point actually appears in the structured facts provided.
2. You must be intelligent: if a claim says "SMA_20=378.2" and the facts say "sma20: 378.24", this is VERIFIED. Do not fail it for minor formatting or rounding differences.
3. Flag any claim where the cited value seems hallucinated or blatantly contradicts the facts as UNVERIFIED.
4. Identify contradictions between bull and bear claims.

Output exactly this JSON:
{
  "summary": "1-2 sentence assessment of evidence quality on both sides",
  "verified_bull_claims": ["claim text 1", "claim text 2"],
  "unverified_bull_claims": ["claim text 3"],
  "verified_bear_claims": ["claim text 1"],
  "unverified_bear_claims": ["claim text 2"]
}"""


def _build_evidence_header(packet: EvidencePacket) -> str:
    facts = {f.fact_type: f.value for f in packet.structured_facts}
    lines = ["## EVIDENCE FILE (pre-verified, cite directly):"]
    for k, v in facts.items():
        lines.append(f"  {k}: {v}")
    if getattr(packet, "tool_cache", None):
        lines.append("## PRE-FETCHED TOOL DATA:")
        for tool_name, result in packet.tool_cache.items():
            lines.append(f"  [{tool_name}]: {result[:500]}")
    return "\n".join(lines)


USER_TEMPLATE = """## Entity: {entity_id}

{position_block}

{portfolio_dashboard}

{company_narrative}

## Structured Facts:
{structured_facts}

## Unstructured Context (Reddit/YouTube/News):
{unstructured_context}

## Available Claims from Evidence:
{claims_text}

## Missing Data:
{missing_fields}

## Specialist Agent Insights:
{agent_insights}

Construct your case based ONLY on the data above. Cite specific values with [source:value] format."""


CROSS_EXAM_USER_TEMPLATE = """## BULL ANALYST CLAIMS:
{bull_claims}

## BEAR ANALYST CLAIMS:
{bear_claims}

## IN-DEBATE TOOL RESEARCH (Ground Truth from Agent Tools):
{tool_research}

## UNSTRUCTURED CONTEXT (News, Reddit, YouTube):
{unstructured_context}

## ACTUAL STRUCTURED FACTS (ground truth):
{structured_facts}

Cross-examine both sets of claims against the actual data, context, AND the in-debate tool research above.
NOTE: Be highly tolerant of minor decimal rounding differences (e.g. 31.54 vs 31.539) and shorthand notations (e.g. $81.3B vs 81300000000.0). Do not flag these as unverified if the values represent the same underlying data point."""


async def _run_biased_agent(
    bias: str,
    system_prompt: str,
    entity_id: str,
    packet: EvidencePacket,
    cycle_id: str,
    bot_id: str,
    model_override: str | None = None,
    endpoint_override: str | None = None,
    agent_insights: dict[str, str] | None = None,
    override_user_prompt: str | None = None,
    position_block: str = "",
    debate_cache: dict[str, str] | None = None,
    portfolio_dashboard: str = "",
) -> tuple[str, int, list[str]]:
    """Run a single biased analyst agent with tool usage capability."""
    if override_user_prompt:
        if portfolio_dashboard:
            user_prompt = f"{portfolio_dashboard}\n\n{override_user_prompt}"
        else:
            user_prompt = override_user_prompt
    else:
        claims_text = (
            "\n".join(
                [
                    f"- [{c.provenance.source_table}] {c.subject_entity_id} "
                    f"{c.predicate} {c.object_value} (conf: {c.confidence:.2f})"
                    for c in packet.claims[:20]
                ]
            )
            or "No explicit claims available."
        )

        missing = ", ".join(packet.missing_fields) if packet.missing_fields else "None"

        insight_str = "None provided."
        if agent_insights:
            insight_str = "\n\n".join(
                f"=== {k.upper()} ===\n{v}" for k, v in agent_insights.items()
            )

        unstructured_context = "None available."
        if packet.source_summaries:
            unstructured_context = "\n".join(
                [format_source_ref_for_prompt(s) for s in packet.source_summaries[:15]]
            )

        company_narrative_str = "## Company Narrative & Evolving Themes:\nNo persistent narrative available."
        if getattr(packet, "company_story", None):
            narrative_lines = [
                "## Company Narrative & Evolving Themes:",
                f"[Story Summary]\n{packet.company_story}",
            ]
            if getattr(packet, "key_themes", None):
                narrative_lines.append("\n[Active & Historical Themes]")
                for theme in packet.key_themes:
                    theme_name = theme.get("theme", "Unknown")
                    cat = theme.get("category", "unknown")
                    status = theme.get("status", "unknown")
                    impact = theme.get("impact", "unknown")
                    rel_label = theme.get("market_relevance_label", "unknown")
                    sev_summary = theme.get("qualitative_severity_summary", "")
                    summary = theme.get("summary", "")
                    
                    narrative_lines.append(
                        f"- {theme_name} (Category: {cat}, Status: {status}, Impact: {impact}, Relevance: {rel_label})\n"
                        f"  Theme summary: {summary}\n"
                        f"  Contextual Relevance/Severity: {sev_summary}"
                    )
            
            if getattr(packet, "pillar_profiles", None) and "pillars" in packet.pillar_profiles:
                narrative_lines.append("\n## Structured Quantitative & Qualitative Pillar Profiles:")
                for pk, p_data in packet.pillar_profiles["pillars"].items():
                    p_name = pk.upper()
                    base_s = p_data.get("base_score", 5.0)
                    adj_s = p_data.get("adjusted_score", base_s)
                    label = p_data.get("profile_label", "Unknown")
                    drivers = p_data.get("active_drivers", [])
                    vetoes = p_data.get("veto_flags", [])
                    rat = p_data.get("adjustment_rationale", "")
                    
                    narrative_lines.append(f"\n### {p_name} PILLAR (Base Score: {base_s:.1f} | Adjusted: {adj_s:.1f})")
                    narrative_lines.append(f"  Profile Label: {label}")
                    if drivers:
                        narrative_lines.append(f"  Active Drivers: {', '.join(drivers)}")
                    if vetoes:
                        narrative_lines.append(f"  VETO FLAGS: {', '.join(vetoes)}")
                    if rat:
                        narrative_lines.append(f"  Adjustment Rationale: {rat}")

            company_narrative_str = "\n".join(narrative_lines)

        user_prompt = USER_TEMPLATE.format(
            entity_id=entity_id,
            position_block=position_block,
            portfolio_dashboard=portfolio_dashboard,
            company_narrative=company_narrative_str,
            structured_facts=_build_evidence_header(packet),
            unstructured_context=unstructured_context,
            claims_text=claims_text,
            missing_fields=missing,
            agent_insights=insight_str,
        )

    # ── Budget-aware truncation of assembled debate prompt ──
    budget = get_context_budget()
    # Debate user prompts get the data_context budget allocation
    max_prompt_chars = budget.data_context_chars
    user_prompt = _cap_debate_text(user_prompt, max_prompt_chars, label=f"{bias}_user_prompt")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if debate_cache:
        prior_research = "\n".join(f"### {k}\n{v}" for k, v in debate_cache.items())
        messages.append(
            {
                "role": "user",
                "content": f"## Prior Research From This Debate:\n{prior_research}\n\nContinue your analysis.",
            }
        )

    agent_name = f"{bias}_agent"
    total_tokens = 0
    final_response = ""
    tool_history = []

    from app.utils.text_utils import parse_json_response
    # Resolve persona whitelisted tools
    whitelist_key = None
    if "fundamental" in agent_name.lower():
        whitelist_key = "fundamental"
    elif "technical" in agent_name.lower():
        whitelist_key = "technical"
    elif "sentiment" in agent_name.lower() or "macro" in agent_name.lower():
        whitelist_key = "sentiment"
    
    from app.agents.tool_whitelists import get_agent_tools
    allowed_tools = None
    if whitelist_key:
        allowed_tools = get_agent_tools(whitelist_key)
    
    if allowed_tools is None:
        allowed_tools = registry.schemas

    # Brain-Action Split: select only needed tools when pool is large
    if allowed_tools and len(allowed_tools) > 5:
        try:
            from app.agents.tool_selector import select_tools_for_task
            task_desc = f"{system_prompt[:500]}\n\nTask: {user_prompt[:1500]}"
            allowed_tools = await select_tools_for_task(
                task_description=task_desc,
                available_tool_schemas=allowed_tools,
                agent_name=f"{agent_name}_selector",
                ticker=entity_id,
                cycle_id=cycle_id,
                priority=Priority.NORMAL,
                max_tools=5,
            )
            logger.info(
                "[Debate] Tool selection: %d tools selected for %s → %s",
                len(allowed_tools),
                agent_name,
                [t["function"]["name"] for t in allowed_tools],
            )
        except Exception as sel_err:
            logger.warning(
                "[Debate] Tool selection failed for %s, using full pool: %s",
                agent_name, sel_err,
            )

    # ── Try Prism /agent routing first (with strict 30s timeout) ──
    from app.config import settings
    if settings.PRISM_ENABLED and settings.PRISM_AGENT_ROUTING:
        try:
            prism_healthy = await asyncio.wait_for(
                llm.prism_client.check_health(), timeout=5.0,
            )
            if prism_healthy:
                from app.tools.prism_agent_harness import run_prism_agent
                logger.info("[Debate] Routing %s agentic loop to Prism /agent", agent_name)
                result = await asyncio.wait_for(
                    run_prism_agent(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        ticker=entity_id,
                        agent_name=agent_name,
                        cycle_id=cycle_id,
                        bot_id=bot_id,
                        priority=Priority.NORMAL,
                        tools_override=allowed_tools,
                        temperature=LLM_TEMPERATURES.get(agent_name, 0.4),
                        max_tokens=4096,
                    ),
                    timeout=120.0,  # Hard cap: allow time for CoT reasoning
                )
                return (
                    result.get("final_text", "").strip(),
                    result.get("token_usage", 0),
                    [],  # tool_history is not populated for Prism-delegated runs
                )
        except asyncio.TimeoutError:
            logger.warning("[Debate] Prism routing TIMEOUT for %s — falling back to local vLLM", agent_name)
        except Exception as pe:
            logger.error("[Debate] Prism routing failed for %s, falling back to local: %s", agent_name, pe)

    try:
        max_tool_turns = cognition_settings.DEBATE_MAX_TOOL_TURNS
        for turn_idx in range(max_tool_turns):  # Configurable tool-calling turns
            result = await llm.chat_with_tools(
                messages=messages,
                tools=allowed_tools,
                temperature=LLM_TEMPERATURES.get(agent_name, 0.4),
                max_tokens=4096,
                priority=Priority.NORMAL,
                agent_name=agent_name,
                ticker=entity_id,
                cycle_id=cycle_id,
                bot_id=bot_id,
                model_override=model_override,
                endpoint_override=endpoint_override,
            )
            total_tokens += result.get("total_tokens", 0)
            ms = result.get("elapsed_ms", 0)
            final_response = result.get("text", "")
            tool_calls = result.get("tool_calls")

            asst_msg = {"role": "assistant", "content": final_response}
            if tool_calls:
                asst_msg["tool_calls"] = tool_calls
            messages.append(asst_msg)

            if not tool_calls:
                # ENFORCEMENT: If they didn't use tools on the very first turn, and output no claims, force them.
                if turn_idx == 0:
                    parsed = parse_json_response(final_response)
                    claims = parsed.get("claims", [])
                    if not claims:
                        logger.warning(
                            "[DEBATE] %s returned no tools and no claims on turn 0 for %s. Forcing tool usage.",
                            bias.upper(),
                            entity_id,
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": f"You provided no claims and used no tools for {entity_id}. "
                                           "If you need data not in the Evidence File, use a tool. "
                                           "Otherwise, provide your claims JSON immediately.",
                            }
                        )
                        continue

                logger.info(
                    "[DEBATE] %s agent for %s: %d tokens, %dms",
                    bias.upper(),
                    entity_id,
                    total_tokens,
                    ms,
                )
                return final_response.strip(), total_tokens, tool_history

            # Early-warning on second-to-last turn
            if turn_idx == max_tool_turns - 2:
                messages.append(
                    {
                        "role": "user",
                        "content": f"WARNING: You have 1 tool turn remaining for {entity_id}. "
                                   "On your NEXT turn, you MUST output final JSON claims. "
                                   "If you need more data, use ONE tool now. Otherwise, output your JSON immediately.",
                    }
                )

            # Execute tool calls (with ticker-lock enforcement for debates)
            all_tools_empty = True
            for tc in tool_calls:
                combined_cache = {}
                if getattr(packet, "tool_cache", None):
                    combined_cache.update(packet.tool_cache)
                if debate_cache is not None:
                    combined_cache.update(debate_cache)

                tc_result = await registry.execute_tool_call(
                    tc,
                    agent_name=agent_name,
                    ticker=entity_id,
                    cycle_id=cycle_id,
                    tool_cache=combined_cache,
                    enforce_ticker=True,
                )
                messages.append(tc_result)

                # Track whether any tool returned useful data
                _tc_content = tc_result.get("content", "").strip()
                _tc_lower = _tc_content.lower()
                _is_empty = (
                    _tc_content in ("", "[]", "{}", "null", "None", "no data", "no results")
                    or any(k in _tc_lower for k in ["error", "exception", "traceback", "failed", "not found"])
                )
                if not _is_empty:
                    all_tools_empty = False

                # Record tool execution for the cross-examiner
                func_name = tc.get("function", {}).get("name", "unknown")
                args = tc.get("function", {}).get("arguments", "{}")
                tool_output = tc_result.get("content", "")
                tool_history.append(
                    f"### Tool Call: {func_name}({args})\n{tool_output[:5000]}"
                )
                if debate_cache is not None:
                    try:
                        import json
                        kwargs = json.loads(args)
                        normalized_args = json.dumps(kwargs, sort_keys=True, separators=(',', ':'))
                        cache_key = f"{func_name}:{normalized_args}"
                    except Exception:
                        cache_key = f"{func_name}:{args}"
                    debate_cache[cache_key] = tool_output[:3000]

            # Bail out of tool loop if ALL tools returned empty/error data.
            # This prevents cascading loops where the agent keeps retrying
            # the same failing tools without making progress.
            if all_tools_empty:
                logger.warning(
                    "[DEBATE] All tools returned empty/error for %s %s — breaking tool loop early",
                    bias, entity_id,
                )
                messages.append({
                    "role": "user",
                    "content": (
                        "The tools returned no useful data. Do NOT call more tools. "
                        "Use ONLY the evidence already provided in the Structured Facts to build your case. "
                        "Output your final JSON claims NOW."
                    ),
                })
                # Force a final text-only response
                try:
                    forced_result = await llm.chat_with_tools(
                        messages=messages,
                        tools=None,
                        temperature=LLM_TEMPERATURES.get(agent_name, 0.4),
                        max_tokens=2048,
                        priority=Priority.NORMAL,
                        agent_name=agent_name,
                        ticker=entity_id,
                        cycle_id=cycle_id,
                        bot_id=bot_id,
                        model_override=model_override,
                        endpoint_override=endpoint_override,
                    )
                    total_tokens += forced_result.get("total_tokens", 0)
                    final_response = forced_result.get("text", "")
                except Exception as bail_err:
                    logger.error("[DEBATE] Bail-out forced response failed for %s %s: %s", bias, entity_id, bail_err)
                break

        # ── Fix 2: JSON Extraction Guarantee ──────────────────────────────
        # If the agent used all tool turns but never output final JSON claims,
        # send one more prompt to force structured output.
        try:
            parsed_check = parse_json_response(final_response)
        except ValueError:
            parsed_check = {}
        if not parsed_check.get("claims"):
            logger.warning(
                "[DEBATE] %s agent exhausted tool turns without JSON claims for %s. Forcing final output.",
                bias.upper(),
                entity_id,
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"You have used all your tool turns for {entity_id}. You MUST now output your final verdict as JSON. "
                        "Based on the tool results you received, output EXACTLY this format:\n"
                        '{"action": "BUY or SELL", "claims": ["claim 1 [source:value]", "claim 2 [source:value]"], '
                        '"confidence": 0-100, "key_argument": "your strongest point"}'
                    ),
                }
            )
            try:
                forced_result = await llm.chat_with_tools(
                    messages=messages,
                    tools=None,  # No tools — force text output
                    temperature=LLM_TEMPERATURES.get(agent_name, 0.4),
                    max_tokens=2048,
                    priority=Priority.NORMAL,
                    agent_name=agent_name,
                    ticker=entity_id,
                    cycle_id=cycle_id,
                    bot_id=bot_id,
                    model_override=model_override,
                    endpoint_override=endpoint_override,
                )
                total_tokens += forced_result.get("total_tokens", 0)
                final_response = forced_result.get("text", "")
                logger.info(
                    "[DEBATE] %s forced JSON output for %s: %s",
                    bias.upper(),
                    entity_id,
                    final_response[:200],
                )
            except Exception as force_err:
                logger.error(
                    "[DEBATE] %s forced JSON call failed for %s: %s",
                    bias.upper(),
                    entity_id,
                    force_err,
                )

        return final_response.strip(), total_tokens, tool_history
    except Exception as e:
        logger.error("[DEBATE] %s agent failed for %s: %s", bias.upper(), entity_id, e)
        return (
            f'{{"action": "{bias.upper()}", "claims": [], "confidence": 0, "error": "{e}"}}',
            0,
            [],
        )


def _extract_claims_from_turns(
    turn_texts: list[str],
    side: str,
    persona_name: str,
) -> list[dict]:
    """Extract claims from multiple turn texts with fallback strategies.

    Merges claims from ALL turns (not just the latest non-empty one).
    Later turns' claims are marked survived_rebuttal=True. Deduplicates
    by claim text, preferring the version with survived_rebuttal=True.
    """
    import re
    from app.utils.text_utils import parse_json_response

    claims: list[dict] = []

    def get_turn_number(idx_from_start: int):
        if side == "bull":
            return 1 if idx_from_start == 0 else 3
        else:
            return 2 if idx_from_start == 0 else 4

    total_texts = len(turn_texts)

    # Strategy 1: Try JSON parsing on ALL turns — accumulate, don't return early
    for idx, text in enumerate(turn_texts):
        if not text or not text.strip():
            continue

        parsed = parse_json_response(text)
        turn_claims = parsed.get("claims", [])

        actual_turn = get_turn_number(idx)
        survived = actual_turn >= 3

        if isinstance(turn_claims, list) and len(turn_claims) > 0:
            for c in turn_claims:
                if isinstance(c, str):
                    claims.append(
                        {"claim": c, "turn": actual_turn, "survived_rebuttal": survived}
                    )
            logger.info(
                "[DEBATE] Extracted %d %s claims from %s turn %d (of %d)",
                len(turn_claims),
                side,
                persona_name,
                actual_turn,
                total_texts,
            )

    # If we got claims from JSON, deduplicate and return
    if claims:
        # Deduplicate by claim text — prefer the survived_rebuttal=True version
        seen: dict[str, dict] = {}
        for c in claims:
            key = c["claim"]
            if key not in seen or c["survived_rebuttal"]:
                seen[key] = c
        claims = list(seen.values())
        logger.info(
            "[DEBATE] Merged %d unique %s claims from %s across %d turns",
            len(claims),
            side,
            persona_name,
            total_texts,
        )
        return claims

    # Strategy 2: Regex fallback — extract claim-like sentences from ALL turns
    for idx, text in enumerate(turn_texts):
        if not text:
            continue
        actual_turn = get_turn_number(idx)
        survived = actual_turn >= 3

        citation_pattern = re.compile(
            r"[^.!?\n]*\[[\w_]+:[\w\s=.%$+\-]+\][^.!?\n]*[.!?]?",
            re.IGNORECASE,
        )
        matches = citation_pattern.findall(text)
        if matches:
            extracted = [m.strip().strip('"').strip("'").strip(",") for m in matches]
            extracted = [c for c in extracted if len(c) > 10]
            for c in extracted:
                claims.append(
                    {"claim": c, "turn": actual_turn, "survived_rebuttal": survived}
                )

    if claims:
        # Deduplicate regex claims
        seen_regex: dict[str, dict] = {}
        for c in claims:
            key = c["claim"]
            if key not in seen_regex or c["survived_rebuttal"]:
                seen_regex[key] = c
        claims = list(seen_regex.values())
        logger.warning(
            "[DEBATE] Used regex fallback to extract %d %s claims from %s",
            len(claims),
            side,
            persona_name,
        )
        return claims

    logger.error(
        "[DEBATE] EXTRACTION_FAILED: No %s claims could be extracted from %s across %d turns",
        side,
        persona_name,
        total_texts,
    )
    return claims


async def run_adversarial_debate(
    ticker: str,
    packet: EvidencePacket,
    cycle_id: str = "",
    bot_id: str = "",
    agent_insights: dict[str, str] | None = None,
    position_context: dict | None = None,
    portfolio_dashboard: str = "",
) -> DebateResult | None:
    """Run the full adversarial debate pipeline.

    Steps:
      1. Bull + Bear agents fire in parallel on different DGX Sparks
      2. Cross-examiner checks both outputs against ground truth
      3. ClaimVerifier (pure Python) validates cited values
      4. DebateJudge weighs verified claims and produces verdict

    When position_context indicates the bot holds a position,
    the debate reframes as HOLD-vs-SELL instead of BUY-vs-SELL.

    Returns DebateResult or None if debate is disabled/fails entirely.
    """
    from app.utils.text_utils import parse_json_response

    if not cognition_settings.DEBATE_ENABLED:
        logger.info("[DEBATE] Debate disabled via DEBATE_ENABLED=False")
        return None

    if ticker in ("USDC", "USDT", "DAI", "USD"):
        logger.info(
            "[DEBATE] Skipping debate for stablecoin/cash equivalent: %s", ticker
        )
        return None

    logger.info("[DEBATE] " + "═" * 50)
    held = position_context.get("held", False) if position_context else False
    debate_mode = "HOLD-vs-SELL" if held else "BUY-vs-SELL"
    logger.info("[DEBATE] Starting %s debate for %s", debate_mode, ticker)

    # Build position block text for user prompts
    _pos_block = ""
    if held:
        try:
            from app.tools.portfolio_tools import (
                format_position_context_for_prompt,
            )

            _pos_block = format_position_context_for_prompt(position_context)
        except Exception as pos_err:
            logger.warning("[DEBATE] Failed to format position context: %s", pos_err)
            _pos_block = ""

    # ── Step 1: Swarm Load Balancing ─────────────────────────────────
    # The vLLM client automatically handles routing to the least-busy nodes.
    bull_model = None
    bear_model = None
    bull_ep = None
    bear_ep = None
    logger.info("[DEBATE] Swarm load-balancing mode active for all personas.")

    # ── Confirmation Loop Guard ───────────────────────────────────────
    # Check if prior debates for this ticker keep producing the same verdict.
    # If so, bump temperatures to force more diverse reasoning paths.
    _temp_boost = 0.0
    try:
        from app.db.connection import get_db
        from datetime import timedelta

        with get_db() as _guard_db:
            _cutoff = datetime.now(timezone.utc) - timedelta(
                hours=cognition_settings.MAX_DEBATE_HISTORY_AGE_HOURS,
            )
            _prior = _guard_db.execute(
                "SELECT final_action FROM debate_history "
                "WHERE ticker = %s AND created_at > %s "
                "ORDER BY created_at DESC LIMIT 5",
                [ticker, _cutoff],
            ).fetchall()

            if len(_prior) >= cognition_settings.CONFIRMATION_LOOP_THRESHOLD:
                _actions = [r[0] for r in _prior]
                _dominant = _actions[0]
                _consecutive = sum(1 for a in _actions if a == _dominant)
                if _consecutive >= cognition_settings.CONFIRMATION_LOOP_THRESHOLD:
                    _temp_boost = 0.1
                    logger.warning(
                        "[DEBATE] Possible confirmation loop: %d consecutive %s verdicts for %s. "
                        "Boosting persona temperatures by +%.1f",
                        _consecutive,
                        _dominant,
                        ticker,
                        _temp_boost,
                    )
    except Exception as _guard_err:
        logger.debug("[DEBATE] Confirmation loop guard check failed (non-fatal): %s", _guard_err)

    max_retries = 3
    retry_count = 0
    bull_critique = ""
    bear_critique = ""

    while retry_count < max_retries:
        logger.info(f"[DEBATE] Attempt {retry_count + 1} / {max_retries}")
        # ── Step 2: N-Turn Sequence per Persona ──

        # To reduce latency, we run the personas concurrently, but within each persona it's sequential.
        # Pre-allocate per-persona caches — no cross-persona contamination
        persona_caches: dict[str, dict[str, str]] = {name: {} for name in PERSONAS}

        async def run_persona_debate(persona_name, persona_instr):
            _debate_cache = persona_caches[persona_name]
            b_tok = 0
            br_tok = 0

            # Per-turn timeout (seconds) — prevents one hung agent from blocking
            # the entire debate phase. Each agent turn gets 90s max.
            TURN_TIMEOUT = 90.0

            async def _timed_agent_call(agent_bias, *args, **kwargs):
                """Wrap _run_biased_agent with a per-turn timeout."""
                try:
                    return await asyncio.wait_for(
                        _run_biased_agent(agent_bias, *args, **kwargs),
                        timeout=TURN_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        "[DEBATE] TURN TIMEOUT: %s timed out after %.0fs for %s/%s",
                        agent_bias, TURN_TIMEOUT, persona_name, ticker,
                    )
                    return ("", 0, [])  # Empty fallback — debate continues

            # Filter evidence packet to this persona's focus area
            persona_packet = filter_packet_for_persona(packet, persona_name)
            logger.info(
                "[DEBATE] %s gets %d/%d facts (filtered)",
                persona_name,
                len(persona_packet.structured_facts),
                len(packet.structured_facts),
            )

            # Resolve persona-specific temperature
            _p_temps = PERSONA_TEMPERATURES.get(persona_name, {})

            # Enforce rule: Agents MUST use tools in debate!
            tool_enforcement = (
                "\n\nCRITICAL RULE: Only use tools if a SPECIFIC value is missing. "
                "Do NOT search endlessly. Once you have enough data, you MUST stop searching "
                "and IMMEDIATELY output your final JSON claims. If you have already run 1-2 tools, STOP and output the JSON!"
            )

            bull_sys = generate_critique_prompt(
                build_system_prompt(
                    "bull",
                    persona_instr + tool_enforcement,
                    position_context=position_context,
                ),
                bull_critique,
                retry_count + 1,
            )
            bear_sys = generate_critique_prompt(
                build_system_prompt(
                    "bear",
                    persona_instr + tool_enforcement,
                    position_context=position_context,
                ),
                bear_critique,
                retry_count + 1,
            )

            # Turn 1: Bull Opening
            bull_t1, tok1, th1 = await _timed_agent_call(
                f"bull_{persona_name.lower()}_t1",
                bull_sys,
                ticker,
                persona_packet,
                cycle_id,
                bot_id,
                model_override=bull_model,
                endpoint_override=bull_ep,
                agent_insights=agent_insights,
                position_block=_pos_block,
                debate_cache=_debate_cache,
                portfolio_dashboard=portfolio_dashboard,
            )
            b_tok += tok1

            if cognition_settings.FAST_DEBATE_MODE:
                _capped_bull_t1 = _cap_debate_text(bull_t1, 3000, "bull_t1_quote")
                bear_prompt = f"OPPONENT (BULL) OPENING STATEMENT:\n{_capped_bull_t1}\n\nFormulate your opening Bear argument AND a targeted rebuttal to the Bull. YOU MUST USE tools (quant or wiki) to verify or counteract their claims before answering!"
                bear_t2, tok2, th2 = await _timed_agent_call(
                    f"bear_{persona_name.lower()}_t2",
                    bear_sys,
                    ticker,
                    persona_packet,
                    cycle_id,
                    bot_id,
                    model_override=bear_model,
                    endpoint_override=bear_ep,
                    agent_insights=agent_insights,
                    override_user_prompt=bear_prompt,
                    debate_cache=_debate_cache,
                    portfolio_dashboard=portfolio_dashboard,
                )
                br_tok += tok2

                # Turn 3: Bull Rebuttal to Bear T2
                bear_research = "\n".join(th2) if th2 else "No tools used by Bear."
                _capped_bear_t2 = _cap_debate_text(bear_t2, 3000, "bear_t2_quote")
                _capped_bear_research = _cap_debate_text(bear_research, 2000, "bear_research")
                bull_prompt = f"OPPONENT (BEAR) REBUTTAL:\n{_capped_bear_t2}\n\nBEAR'S DATA SOURCES:\n{_capped_bear_research}\n\nFormulate a strict Bull counter-rebuttal. YOU MUST USE tools to find flaws in the Bear's math/logic before sending final JSON!"
                bull_t3, tok3, th3 = await _timed_agent_call(
                    f"bull_{persona_name.lower()}_t3",
                    bull_sys,
                    ticker,
                    persona_packet,
                    cycle_id,
                    bot_id,
                    model_override=bull_model,
                    endpoint_override=bull_ep,
                    agent_insights=agent_insights,
                    override_user_prompt=bull_prompt,
                    debate_cache=_debate_cache,
                    portfolio_dashboard=portfolio_dashboard,
                )
                b_tok += tok3

                # Turn 4: Bear Final Rebuttal (Fast Mode)
                bull_research = "\n".join(th3) if th3 else "No tools used by Bull."
                _capped_bull_t3 = _cap_debate_text(bull_t3, 3000, "bull_t3_quote")
                _capped_bull_research = _cap_debate_text(bull_research, 2000, "bull_research")
                bear_prompt_t4 = f"OPPONENT (BULL) COUNTER-REBUTTAL:\n{_capped_bull_t3}\n\nBULL'S DATA SOURCES:\n{_capped_bull_research}\n\nFormulate a 1-2 sentence final Bear conclusion. Ensure your JSON claims are robust and validated."
                bear_t4, tok4, th4 = await _timed_agent_call(
                    f"bear_{persona_name.lower()}_t4",
                    bear_sys,
                    ticker,
                    persona_packet,
                    cycle_id,
                    bot_id,
                    model_override=bear_model,
                    endpoint_override=bear_ep,
                    agent_insights=agent_insights,
                    override_user_prompt=bear_prompt_t4,
                    debate_cache=_debate_cache,
                    portfolio_dashboard=portfolio_dashboard,
                )
                br_tok += tok4

                bull_total_text = f"--- TURN 1 (OPENING) ---\n{bull_t1}\n--- TURN 3 (REBUTTAL) ---\n{bull_t3}"
                bear_total_text = f"--- TURN 2 (OPENING & REBUTTAL) ---\n{bear_t2}\n--- TURN 4 (FINAL REBUTTAL) ---\n{bear_t4}"
                all_bull_turns = [bull_t1, bull_t3]
                all_bear_turns = [bear_t2, bear_t4]
                all_th = th1 + th2 + th3 + th4
            else:
                # Turn 2: Bear Opening + Rebuttal to Bull T1
                _capped_bull_t1 = _cap_debate_text(bull_t1, 3000, "bull_t1_quote")
                bear_prompt = f"OPPONENT (BULL) OPENING STATEMENT:\n{_capped_bull_t1}\n\nFormulate your opening Bear argument AND a targeted rebuttal to the Bull. YOU MUST USE tools (quant or wiki) to verify or counteract their claims before answering!"
                bear_t2, tok2, th2 = await _timed_agent_call(
                    f"bear_{persona_name.lower()}_t2",
                    bear_sys,
                    ticker,
                    persona_packet,
                    cycle_id,
                    bot_id,
                    model_override=bear_model,
                    endpoint_override=bear_ep,
                    agent_insights=agent_insights,
                    override_user_prompt=bear_prompt,
                    debate_cache=_debate_cache,
                    portfolio_dashboard=portfolio_dashboard,
                )
                br_tok += tok2

                # Turn 3: Bull Rebuttal to Bear T2
                bear_research = "\n".join(th2) if th2 else "No tools used by Bear."
                _capped_bear_t2 = _cap_debate_text(bear_t2, 3000, "bear_t2_quote")
                _capped_bear_research = _cap_debate_text(bear_research, 2000, "bear_research")
                bull_prompt = f"OPPONENT (BEAR) REBUTTAL:\n{_capped_bear_t2}\n\nBEAR'S DATA SOURCES:\n{_capped_bear_research}\n\nFormulate a strict Bull counter-rebuttal. YOU MUST USE tools to find flaws in the Bear's math/logic before sending final JSON!"
                bull_t3, tok3, th3 = await _timed_agent_call(
                    f"bull_{persona_name.lower()}_t3",
                    bull_sys,
                    ticker,
                    persona_packet,
                    cycle_id,
                    bot_id,
                    model_override=bull_model,
                    endpoint_override=bull_ep,
                    agent_insights=agent_insights,
                    override_user_prompt=bull_prompt,
                    debate_cache=_debate_cache,
                    portfolio_dashboard=portfolio_dashboard,
                )
                b_tok += tok3

                # Turn 4: Bear Final Rebuttal
                bull_research = "\n".join(th3) if th3 else "No tools used by Bull."
                _capped_bull_t3 = _cap_debate_text(bull_t3, 3000, "bull_t3_quote")
                _capped_bull_research = _cap_debate_text(bull_research, 2000, "bull_research")
                bear_prompt = f"OPPONENT (BULL) COUNTER-REBUTTAL:\n{_capped_bull_t3}\n\nBULL'S DATA SOURCES:\n{_capped_bull_research}\n\nFormulate your final Bear conclusion. Ensure your JSON claims are robust and validated by any final tool calls."
                bear_t4, tok4, th4 = await _timed_agent_call(
                    f"bear_{persona_name.lower()}_t4",
                    bear_sys,
                    ticker,
                    persona_packet,
                    cycle_id,
                    bot_id,
                    model_override=bear_model,
                    endpoint_override=bear_ep,
                    agent_insights=agent_insights,
                    override_user_prompt=bear_prompt,
                    debate_cache=_debate_cache,
                    portfolio_dashboard=portfolio_dashboard,
                )
                br_tok += tok4

                bull_total_text = f"--- TURN 1 (OPENING) ---\n{bull_t1}\n--- TURN 3 (REBUTTAL) ---\n{bull_t3}"
                bear_total_text = f"--- TURN 2 (OPENING & REBUTTAL) ---\n{bear_t2}\n--- TURN 4 (FINAL REBUTTAL) ---\n{bear_t4}"

                all_th = th1 + th2 + th3 + th4
                all_bull_turns = [bull_t1, bull_t3]
                all_bear_turns = [bear_t2, bear_t4]

            return (
                bull_total_text,
                bear_total_text,
                b_tok,
                br_tok,
                all_bull_turns,
                all_bear_turns,
                all_th,
            )

        num_personas = len(PERSONAS)
        tasks = [run_persona_debate(name, instr) for name, instr in PERSONAS.items()]
        persona_results = await asyncio.gather(*tasks, return_exceptions=True)

        bull_claims = []
        bear_claims = []
        bull_tokens = 0
        bear_tokens = 0
        global_tool_research = []

        bull_formatted_history = []
        bear_formatted_history = []
        persona_outcomes = {}

        for i, res in enumerate(persona_results):
            if isinstance(res, BaseException):
                logger.error(
                    "[DEBATE] Persona %s failed: %s", list(PERSONAS.keys())[i], res
                )
                continue

            (
                bull_text,
                bear_text,
                b_tok,
                br_tok,
                all_bull_turns,
                all_bear_turns,
                tool_history,
            ) = res

            bull_tokens += b_tok
            bear_tokens += br_tok
            global_tool_research.extend(tool_history)

            bull_formatted_history.append(bull_text)
            bear_formatted_history.append(bear_text)

            # ── Fix 1+3: Extract claims from ALL turns with fallback ──────
            # Try each turn text individually; later turns take priority
            # but earlier turns provide fallback claims.
            p_name = list(PERSONAS.keys())[i]
            persona_bull_claims = _extract_claims_from_turns(
                all_bull_turns,
                "bull",
                p_name,
            )
            persona_bear_claims = _extract_claims_from_turns(
                all_bear_turns,
                "bear",
                p_name,
            )

            bull_claims.extend(persona_bull_claims)
            bear_claims.extend(persona_bear_claims)

            # Score persona winner by survived_rebuttal claims (higher signal)
            # Fall back to raw count if no claims survived rebuttal
            p_bull_survived = sum(1 for c in persona_bull_claims if c.get("survived_rebuttal"))
            p_bear_survived = sum(1 for c in persona_bear_claims if c.get("survived_rebuttal"))
            p_bull_count = len(persona_bull_claims)
            p_bear_count = len(persona_bear_claims)

            # Use survived count if available, otherwise fall back to raw count
            bull_score = p_bull_survived if (p_bull_survived + p_bear_survived) > 0 else p_bull_count
            bear_score = p_bear_survived if (p_bull_survived + p_bear_survived) > 0 else p_bear_count

            if bull_score > bear_score:
                p_winner = "bull"
            elif bear_score > bull_score:
                p_winner = "bear"
            else:
                p_winner = "split"

            persona_outcomes[p_name] = {
                "bull_claims_count": p_bull_count,
                "bear_claims_count": p_bear_count,
                "bull_survived": p_bull_survived,
                "bear_survived": p_bear_survived,
                "winner": p_winner,
            }

        # Deduplicate by claim string while preserving order
        bull_claims = list({c["claim"]: c for c in bull_claims}.values())
        bear_claims = list({c["claim"]: c for c in bear_claims}.values())

        logger.info(
            "[DEBATE] Aggregated %d Personas -> Bull: %d claims, Bear: %d claims",
            num_personas,
            len(bull_claims),
            len(bear_claims),
        )

        # ── Zero-Claim Debate Detection ──────────────────────────────────
        # If ALL personas produced 0 claims on both sides, the judge will
        # decide on unstructured context alone — defeating the adversarial
        # system.  Log a prominent WARNING so this is diagnosable.
        if not bull_claims and not bear_claims:
            logger.warning(
                "[DEBATE] ⚠️ EMPTY DEBATE: All %d personas produced 0 claims "
                "for %s. Judge will rule on unstructured context only. "
                "Per-persona outcomes: %s",
                num_personas,
                ticker,
                persona_outcomes,
            )

        seen_tools = set()
        deduped_research = []
        for entry in global_tool_research:
            tool_key = entry[:50]  # first 50 chars as dedup key
            if tool_key not in seen_tools:
                seen_tools.add(tool_key)
                deduped_research.append(entry)
        logger.info(
            "[DEBATE] Tool dedup: %d/%d unique calls",
            len(deduped_research),
            len(global_tool_research),
        )
        global_tool_research = deduped_research

        # ── Step 3: Per-Persona Blind Cross-examination ─────────────────
        # Each persona's claims are verified independently against its own
        # tool cache, preventing batch self-validation bias.

        bull_formatted = "\n---\n".join(bull_formatted_history)
        bear_formatted = "\n---\n".join(bear_formatted_history)

        transcript = f"### BULL ARGUMENTS\n{bull_formatted}\n\n### BEAR ARGUMENTS\n{bear_formatted}"

        unstructured_context = "None available."
        if packet.source_summaries:
            unstructured_context = "\n".join(
                [format_source_ref_for_prompt(s) for s in packet.source_summaries[:15]]
            )

        import json

        cross_tokens = 0
        cross_findings = ""

        # Build per-persona cross-exam tasks
        async def cross_examine_persona(p_name, p_bull_claims, p_bear_claims, p_cache):
            """Run cross-examination for a single persona's claims."""
            p_bull_strs = [c["claim"] for c in p_bull_claims]
            p_bear_strs = [c["claim"] for c in p_bear_claims]

            # Use persona-specific tool cache as research context
            p_tool_research = "\n\n".join(
                f"{k}: {v}" for k, v in p_cache.items()
            ) if p_cache else "No tools were executed by this persona."

            p_cross_user = CROSS_EXAM_USER_TEMPLATE.format(
                bull_claims=json.dumps(p_bull_strs, indent=2),
                bear_claims=json.dumps(p_bear_strs, indent=2),
                tool_research=p_tool_research[:15000],
                unstructured_context=unstructured_context[:5000],
                structured_facts=str(
                    filter_packet_for_persona(packet, p_name).structured_facts or {}
                )[:10000],
            )

            try:
                from app.utils.resilience import aresilient_call

                @aresilient_call(retries=2, backoff="exponential", base_delay=1.0, max_delay=10.0)
                async def _p_cross_exam():
                    return await asyncio.wait_for(
                        llm.chat(
                            system=CROSS_EXAM_SYSTEM_PROMPT,
                            user=p_cross_user,
                            temperature=LLM_TEMPERATURES.get("cross_examiner", 0.2),
                            max_tokens=2048,
                            priority=Priority.NORMAL,
                            agent_name=f"cross_examiner_{p_name.lower()}",
                            ticker=ticker,
                            cycle_id=cycle_id,
                            bot_id=bot_id,
                        ),
                        timeout=120.0,
                    )

                resp, tok, ms = await _p_cross_exam()
                parsed = parse_json_response(resp)
                logger.info(
                    "[DEBATE] Cross-exam %s: %d tokens, %dms",
                    p_name, tok or 0, ms,
                )
                return parsed, tok or 0
            except Exception as e:
                logger.error("[DEBATE] Cross-exam %s failed: %s", p_name, e)
                return {}, 0

        # Collect per-persona claims for cross-exam
        persona_claims_map: dict[str, dict] = {}
        for i, p_name in enumerate(PERSONAS.keys()):
            if isinstance(persona_results[i], Exception):
                continue
            persona_claims_map[p_name] = {
                "bull": [c for c in bull_claims if any(
                    c["claim"] == oc["claim"]
                    for oc in _extract_claims_from_turns(
                        persona_results[i][4], "bull", p_name
                    )
                )] if persona_results[i][4] else [],
                "bear": [c for c in bear_claims if any(
                    c["claim"] == oc["claim"]
                    for oc in _extract_claims_from_turns(
                        persona_results[i][5], "bear", p_name
                    )
                )] if persona_results[i][5] else [],
            }

        # Run all 3 cross-exams in parallel
        cross_tasks = []
        cross_persona_names = []
        for p_name in PERSONAS.keys():
            if p_name in persona_claims_map:
                pcm = persona_claims_map[p_name]
                cross_tasks.append(
                    cross_examine_persona(
                        p_name,
                        pcm["bull"],
                        pcm["bear"],
                        persona_caches.get(p_name, {}),
                    )
                )
                cross_persona_names.append(p_name)

        cross_results = await asyncio.gather(*cross_tasks, return_exceptions=True)

        # Merge per-persona cross-exam results
        merged_cross_parsed: dict = {
            "verified_bull_claims": [],
            "unverified_bull_claims": [],
            "verified_bear_claims": [],
            "unverified_bear_claims": [],
            "summary": [],
        }

        for idx, cr in enumerate(cross_results):
            if isinstance(cr, Exception):
                logger.error("[DEBATE] Cross-exam task %s failed: %s", cross_persona_names[idx], cr)
                continue
            parsed, tok = cr
            cross_tokens += tok

            for key in ("verified_bull_claims", "unverified_bull_claims",
                        "verified_bear_claims", "unverified_bear_claims"):
                merged_cross_parsed[key].extend(parsed.get(key, []))

            summary = parsed.get("summary", "")
            if summary:
                merged_cross_parsed["summary"].append(
                    f"[{cross_persona_names[idx]}] {summary}"
                )

        cross_findings = " | ".join(merged_cross_parsed["summary"]) if merged_cross_parsed["summary"] else ""
        cross_parsed = merged_cross_parsed

        logger.info(
            "[DEBATE] Per-persona cross-exam complete: %d personas, %d tokens",
            len(cross_persona_names),
            cross_tokens,
        )

        # --- AUDIT LOGGING (per-persona cross-exam details) ---
        # Written into the unified debate trace file at the end (A6)
        # to avoid creating redundant per-persona files.

        # ── Step 4: Claim verification (LLM Jury) ────────────────────────
        verified_bull = cross_parsed.get("verified_bull_claims", [])
        unverified_bull = cross_parsed.get("unverified_bull_claims", [])
        if not verified_bull and not unverified_bull:
            unverified_bull = [c["claim"] for c in bull_claims]

        verified_bear = cross_parsed.get("verified_bear_claims", [])
        unverified_bear = cross_parsed.get("unverified_bear_claims", [])
        if not verified_bear and not unverified_bear:
            unverified_bear = [c["claim"] for c in bear_claims]

        all_unverified = unverified_bull + unverified_bear

        # Map back to dicts to pass to judge
        verified_bull_meta = []
        for v in verified_bull:
            match = next(
                (
                    c
                    for c in bull_claims
                    if c["claim"] == v or v in c["claim"] or c["claim"] in v
                ),
                None,
            )
            verified_bull_meta.append(
                match if match else {"claim": v, "turn": 1, "survived_rebuttal": False}
            )

        verified_bear_meta = []
        for v in verified_bear:
            match = next(
                (
                    c
                    for c in bear_claims
                    if c["claim"] == v or v in c["claim"] or c["claim"] in v
                ),
                None,
            )
            verified_bear_meta.append(
                match if match else {"claim": v, "turn": 2, "survived_rebuttal": False}
            )

        unverified_meta = []
        for v in all_unverified:
            match = next(
                (
                    c
                    for c in bull_claims + bear_claims
                    if c["claim"] == v or v in c["claim"] or c["claim"] in v
                ),
                None,
            )
            unverified_meta.append(
                match if match else {"claim": v, "turn": 0, "survived_rebuttal": False}
            )

        # Determine integrity status
        reject_threshold = cognition_settings.CLAIM_REJECT_THRESHOLD
        integrity = "HIGH"
        if len(all_unverified) > reject_threshold:
            integrity = "LOW_INTEGRITY"
            logger.warning(
                "[DEBATE] LOW_INTEGRITY: %d unverified claims exceed threshold %d",
                len(all_unverified),
                reject_threshold,
            )

        logger.info(
            "[DEBATE] Verified: bull=%d/%d, bear=%d/%d, integrity=%s",
            len(verified_bull),
            len(bull_claims),
            len(verified_bear),
            len(bear_claims),
            integrity,
        )

        if integrity != "LOW_INTEGRITY":
            break  # Successful debate!

        logger.warning(
            "[DEBATE] LOW INTEGRITY DETECTED. Generating Critique and Retrying..."
        )

        # Generate Critique
        critic_sys = "You are a Debate Coach. The analysts failed cross-examination. Write a targeted critique explaining what they hallucinated and instructing them to fix it."
        critic_user = f"BULL CLAIMS REJECTED: {unverified_bull}\nBEAR CLAIMS REJECTED: {unverified_bear}\n\nCROSS EXAM FINDINGS: {cross_findings}\n\nWrite a strict, short critique for the Bull and Bear telling them what data they hallucinated and that they must use tools."

        try:
            from app.services.prism_agent_caller import call_prism_agent
            critic_res, _, _ = await call_prism_agent(
                agent_id="CUSTOM_DEBATE_CRITIC_AGENT",
                user_message=critic_user,
                fallback_system_prompt=critic_sys,
                fallback_agent_name="debate_critic",
                temperature=0.1,
                max_tokens=2048,
                priority=Priority.NORMAL,
                ticker=ticker,
                cycle_id=cycle_id,
                bot_id=bot_id,
            )
            bull_critique = critic_res
            bear_critique = critic_res
        except Exception as e:
            logger.error(f"[DEBATE] Critic failed: {e}")
            bull_critique = "You failed cross-examination. Stop hallucinating data and use your tools."
            bear_critique = "You failed cross-examination. Stop hallucinating data and use your tools."

        retry_count += 1

    if integrity == "LOW_INTEGRITY":
        logger.warning("[DEBATE] Max retries reached. Forcing ABSTAIN.")

    # ── Step 5: Judge weighs verified claims ─────────────────────────

    judge_result, judge_tokens = await judge_debate(
        ticker=ticker,
        verified_bull_claims=verified_bull_meta,
        verified_bear_claims=verified_bear_meta,
        cross_exam_findings=cross_findings,
        unverified_count=len(all_unverified),
        cycle_id=cycle_id,
        bot_id=bot_id,
        persona_outcomes=persona_outcomes,
        held=held,
        source_summaries=packet.source_summaries,
        position_context=position_context,
    )

    total_tokens = bull_tokens + bear_tokens + cross_tokens + judge_tokens

    from app.cognition.debate.action_gate import gate_action
    judge_action = gate_action(judge_result.get("action", "HOLD"), held)
    judge_confidence = judge_result.get("confidence", 0)

    # Ground-truth checklist: Force abstain if LOW_INTEGRITY
    if integrity == "LOW_INTEGRITY":
        abstain_action = gate_action("HOLD", held)
        logger.warning("[DEBATE] Forcing ABSTAIN (%s, 0%%) due to LOW_INTEGRITY", abstain_action)
        judge_action = abstain_action
        judge_confidence = 0
        judge_result["rationale"] = (
            f"[LOW INTEGRITY ABSTAIN] {len(all_unverified)} unverified claims exceeded threshold. Original rationale: {judge_result.get('rationale')}"
        )

    logger.info(
        "[DEBATE] VERDICT: %s @ %d%% | Winner: %s | Tokens: %d",
        judge_action,
        judge_confidence,
        judge_result.get("winning_side", "?"),
        total_tokens,
    )
    logger.info("[DEBATE] " + "═" * 50)

    # A5: Log debate history to database
    try:
        from app.db.connection import get_db
        import json
        import uuid as _uuid

        with get_db() as db:
            db.execute(
                """
                INSERT INTO debate_history
                (id, ticker, cycle_id, pro_argument, con_argument, winner, final_action, final_confidence, persona_outcomes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker, cycle_id) DO UPDATE SET 
                pro_argument = EXCLUDED.pro_argument, 
                con_argument = EXCLUDED.con_argument, 
                winner = EXCLUDED.winner, 
                final_action = EXCLUDED.final_action,
                final_confidence = EXCLUDED.final_confidence,
                persona_outcomes = EXCLUDED.persona_outcomes
                """,
                [
                    f"dh-{_uuid.uuid4().hex[:12]}",
                    ticker,
                    cycle_id or "manual",
                    json.dumps(verified_bull_meta),
                    json.dumps(verified_bear_meta),
                    judge_result.get("winning_side", "split"),
                    judge_action,
                    judge_confidence,
                    json.dumps(persona_outcomes),
                ],
            )
    except Exception as db_err:
        logger.error("[DEBATE] Failed to log debate history: %s", db_err)

    # A6: Write comprehensive debate audit JSONL
    try:
        import json as _json
        from pathlib import Path
        _audit_dir = Path("logs/audit")
        _audit_dir.mkdir(parents=True, exist_ok=True)
        run_time = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        cycle_suffix = f"_{cycle_id}" if cycle_id else ""
        log_filename = f"debate_audit_{ticker}{cycle_suffix}_{run_time}.jsonl"
        audit_entry = {
            "ticker": ticker,
            "cycle_id": cycle_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "debate_mode": debate_mode,
            "verdict": {
                "action": judge_action,
                "confidence": judge_confidence,
                "winner": judge_result.get("winning_side", "split"),
                "rationale": judge_result.get("rationale", ""),
                "key_factor": judge_result.get("key_deciding_factor", ""),
                "rejected_impact": judge_result.get("rejected_claim_impact", ""),
            },
            "integrity": integrity,
            "retry_count": retry_count,
            "tokens": {
                "bull": bull_tokens, "bear": bear_tokens,
                "cross_exam": cross_tokens, "judge": judge_tokens,
                "total": total_tokens,
            },
            "persona_outcomes": persona_outcomes,
            "claims": {
                "bull_total": len(bull_claims),
                "bear_total": len(bear_claims),
                "verified_bull": [c["claim"] for c in verified_bull_meta],
                "verified_bear": [c["claim"] for c in verified_bear_meta],
                "unverified": [c["claim"] for c in unverified_meta],
            },
            "cross_exam_summary": cross_findings,
        }
        with open(_audit_dir / log_filename, "w", encoding="utf-8") as f:
            f.write(_json.dumps(audit_entry, indent=2) + "\n")
    except Exception as audit_err:
        logger.error("[DEBATE] Failed to write debate audit: %s", audit_err)

    return DebateResult(
        bull_claims=bull_claims,
        bear_claims=bear_claims,
        verified_bull_claims=verified_bull_meta,
        verified_bear_claims=verified_bear_meta,
        unverified_claims=unverified_meta,
        cross_exam_findings=cross_findings,
        judge_action=judge_action,
        judge_confidence=judge_confidence,
        judge_rationale=judge_result.get("rationale", ""),
        winning_side=judge_result.get("winning_side", "split"),
        key_deciding_factor=judge_result.get("key_deciding_factor", ""),
        rejected_claim_impact=judge_result.get("rejected_claim_impact", ""),
        integrity_status=integrity,
        transcript=transcript,
        total_tokens=total_tokens,
        persona_outcomes=persona_outcomes,
        original_thesis_status=judge_result.get("original_thesis_status", "NOT_HELD"),
        original_thesis_explanation=judge_result.get("original_thesis_explanation", ""),
    )
