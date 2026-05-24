"""
Prism Agent Registry — Maps local trading-service agent names to Prism custom agent IDs.

When Prism is enabled, every LLM call routes through Prism's /agent endpoint
using the custom agent ID from this registry. If an agent doesn't exist in Prism
yet, the caller can auto-register it via POST /custom-agents.

This replaces the scattered if/elif chain that was in prism_client.py.
"""


# ── Central Agent ID Map ──
# Keys: local agent_name strings used in llm.chat() calls
# Values: Prism custom agent IDs (uppercase, CUSTOM_ prefix)
AGENT_ID_MAP: dict[str, str] = {
    # ── Canonical Prism Agent IDs (Self-Mapping) ──
    "CUSTOM_SYSTEM_JANITOR_AGENT": "CUSTOM_SYSTEM_JANITOR_AGENT",
    "CUSTOM_QUANT_RESEARCH_AGENT": "CUSTOM_QUANT_RESEARCH_AGENT",
    "CUSTOM_TECHNICAL_ANALYSIS_AGENT": "CUSTOM_TECHNICAL_ANALYSIS_AGENT",
    "CUSTOM_AGENT_ARCHITECT": "CUSTOM_AGENT_ARCHITECT",
    "CUSTOM_AGENT_BUDGET_MANAGER": "CUSTOM_AGENT_BUDGET_MANAGER",
    "CUSTOM_BULLISH_DEBATER": "CUSTOM_BULLISH_DEBATER",
    "CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT": "CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
    "CUSTOM_MARKET_ALPHA": "CUSTOM_MARKET_ALPHA",

    # ── System Janitor Agent Mappings ──
    "CUSTOM_DATA_JANITOR_AGENT": "CUSTOM_SYSTEM_JANITOR_AGENT",
    "CUSTOM_DATA_JANITOR_CRITIC_AGENT": "CUSTOM_SYSTEM_JANITOR_AGENT",
    "CUSTOM_DATA_CURATOR_AGENT": "CUSTOM_SYSTEM_JANITOR_AGENT",
    "CUSTOM_LIFECYCLE_SUMMARIZER_AGENT": "CUSTOM_SYSTEM_JANITOR_AGENT",
    "data_janitor": "CUSTOM_SYSTEM_JANITOR_AGENT",
    "data_janitor_critic": "CUSTOM_SYSTEM_JANITOR_AGENT",
    "data_curator": "CUSTOM_SYSTEM_JANITOR_AGENT",
    "database_curator": "CUSTOM_SYSTEM_JANITOR_AGENT",
    "lifecycle_summarizer": "CUSTOM_SYSTEM_JANITOR_AGENT",
    "purge_pass": "CUSTOM_SYSTEM_JANITOR_AGENT",
    "post_cycle_learner": "CUSTOM_SYSTEM_JANITOR_AGENT",
    "maintenance_agent": "CUSTOM_SYSTEM_JANITOR_AGENT",
    "janitor": "CUSTOM_SYSTEM_JANITOR_AGENT",

    # ── Quant Research Agent ──
    "quant_research": "CUSTOM_QUANT_RESEARCH_AGENT",
    "autoresearch": "CUSTOM_QUANT_RESEARCH_AGENT",
    "research_subagent": "CUSTOM_QUANT_RESEARCH_AGENT",
    "research_subagent_yield": "CUSTOM_QUANT_RESEARCH_AGENT",

    # ── Technical Analysis Agent ──
    "technical": "CUSTOM_TECHNICAL_ANALYSIS_AGENT",

    # ── Agent Architect ──
    "agent_architect": "CUSTOM_AGENT_ARCHITECT",

    # ── Agent Budget Manager ──
    "budget": "CUSTOM_AGENT_BUDGET_MANAGER",

    # ── Bullish Debater & Debate Agents ──
    "bull_agent": "CUSTOM_BULLISH_DEBATER",
    "bear_agent": "CUSTOM_BULLISH_DEBATER",
    "CUSTOM_BULL_AGENT": "CUSTOM_BULLISH_DEBATER",
    "CUSTOM_BEAR_AGENT": "CUSTOM_BULLISH_DEBATER",
    "debater": "CUSTOM_BULLISH_DEBATER",
    "debate_debater": "CUSTOM_BULLISH_DEBATER",
    "specialized_debater": "CUSTOM_BULLISH_DEBATER",
    "evolution_debater_proposed": "CUSTOM_BULLISH_DEBATER",
    "evolution_debater_critic": "CUSTOM_BULLISH_DEBATER",
    "evolution_debater_judge": "CUSTOM_BULLISH_DEBATER",
    "debate_meta": "CUSTOM_BULLISH_DEBATER",
    "debate_cross_examiner": "CUSTOM_BULLISH_DEBATER",
    "debate_synthesizer": "CUSTOM_BULLISH_DEBATER",
    "thesis_agent": "CUSTOM_BULLISH_DEBATER",
    "debate_judge": "CUSTOM_BULLISH_DEBATER",
    "debate_critic": "CUSTOM_BULLISH_DEBATER",
    "debate_coordinator": "CUSTOM_BULLISH_DEBATER",

    # ── Trading Cycle Analysis Agent (default for other specialists) ──
    "sentiment": "CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
    "fundamental": "CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
    "risk": "CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
    "fund_flow": "CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
    "comparative": "CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
    "retriever": "CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
    "verifier": "CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
    "synthesizer": "CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
    "pre_trade": "CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
    "meta_audit": "CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
    "decision_engine": "CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
    "decision_glance": "CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
    "trading_phase": "CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
    "trading_cycle": "CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
    "morning_briefing_analyst": "CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
    "briefing": "CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
    "flash_briefing": "CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
    "consolidator": "CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
    "context_compressor": "CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
    "swarm_quant": "CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
    "swarm_macro": "CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
    "swarm_cio": "CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
    "swarm_consensus": "CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
    "swarm_exec_planner": "CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
    "swarm_evaluator": "CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
    "reflector": "CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
    "evolution_runner": "CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
    "test_prove": "CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
    "strategy_auditor": "CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
    "judge_agent": "CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
    "deepeval_client": "CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
    "memory_consolidation": "CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
    "trading_memory": "CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
    "tool_analyst": "CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
    "benchmark_agent": "CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
    "curation_pass": "CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
    "macro_scout": "CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
}

# Agent names that map to CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT (catch-all for standard analysis)
_STANDARD_ANALYSIS_AGENTS = frozenset({
    "sentiment", "fundamental", "risk", "fund_flow", "comparative",
    "retriever", "verifier", "synthesizer", "pre_trade", "meta_audit",
    "decision_engine", "trading_phase", "trading_cycle",
})


def resolve_agent_id(agent_name: str, default_agent: str = "CUSTOM_MARKET_ALPHA") -> str:
    """Resolve a local agent name to a Prism custom agent ID.

    Args:
        agent_name: The local agent name (e.g. "data_janitor", "morning_briefing_analyst").
        default_agent: Fallback agent ID if no mapping exists.

    Returns:
        The Prism custom agent ID string.
    """
    if not agent_name:
        return default_agent

    # Direct lookup first (fast path)
    agent_id = AGENT_ID_MAP.get(agent_name)
    if agent_id:
        return agent_id

    # Fuzzy matching for standard analysis agents
    name_lower = agent_name.lower()
    if any(x in name_lower for x in _STANDARD_ANALYSIS_AGENTS):
        return "CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT"

    # Fuzzy matching for known prefixes (backward compat with prism_client.py logic)
    if "quant_research" in name_lower:
        return "CUSTOM_QUANT_RESEARCH_AGENT"
    if "janitor" in name_lower or "maintenance" in name_lower:
        return "CUSTOM_SYSTEM_JANITOR_AGENT"
    if "technical" in name_lower:
        return "CUSTOM_TECHNICAL_ANALYSIS_AGENT"
    if "agent_architect" in name_lower or "architect" in name_lower:
        return "CUSTOM_AGENT_ARCHITECT"
    if "budget" in name_lower:
        return "CUSTOM_AGENT_BUDGET_MANAGER"
    if "debater" in name_lower:
        return "CUSTOM_BULLISH_DEBATER"

    return default_agent
