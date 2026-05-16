import logging
from app.services.vllm_client import llm, Priority
from app.config.config_cognition import LLM_TEMPERATURES
from app.cognition.contracts.evidence import EvidencePacket

logger = logging.getLogger(__name__)


async def _run_specialized_agent(
    agent_name: str,
    system_prompt: str,
    entity_id: str,
    packet: EvidencePacket,
    cycle_id: str,
    bot_id: str,
) -> tuple[str, int]:
    """Helper to run a specialized agent."""
    user_prompt = f"## Entity ID: {entity_id}\n\n## Structured Facts:\n{packet.structured_facts}\n\nAnalyze the data from your unique perspective."

    tokens_used = 0
    try:
        response, tokens, ms = await llm.chat(
            system=system_prompt,
            user=user_prompt,
            temperature=LLM_TEMPERATURES.get(agent_name, 0.3),
            max_tokens=256,
            priority=Priority.NORMAL,
            agent_name=agent_name,
            ticker=entity_id,
            cycle_id=cycle_id,
            bot_id=bot_id,
        )
        tokens_used = tokens or 0
        return response.strip(), tokens_used
    except Exception as e:
        logger.error(f"[{agent_name.upper()}] Failed: {e}")
        return f"Failed: {e}", 0


async def analyze_sentiment(
    entity_id: str, packet: EvidencePacket, cycle_id: str, bot_id: str
) -> tuple[str, int]:
    sys = "You are a Sentiment Agent. Analyze the social and news sentiment purely based on the provided facts. Be concise (max 3 sentences)."
    return await _run_specialized_agent(
        "sentiment_agent", sys, entity_id, packet, cycle_id, bot_id
    )


async def analyze_macro_risk(
    entity_id: str, packet: EvidencePacket, cycle_id: str, bot_id: str
) -> tuple[str, int]:
    sys = "You are a Macro Risk Agent. Analyze macroeconomic conditions and geopolitical risks based on the provided facts. Be concise (max 3 sentences)."
    return await _run_specialized_agent(
        "macro_risk_agent", sys, entity_id, packet, cycle_id, bot_id
    )


async def analyze_fundamentals(
    entity_id: str, packet: EvidencePacket, cycle_id: str, bot_id: str
) -> tuple[str, int]:
    sys = "You are a Fundamental Value Agent. Analyze the price multiples, balance sheet, and income statement based on the provided facts. Be concise (max 3 sentences)."
    return await _run_specialized_agent(
        "fundamental_agent", sys, entity_id, packet, cycle_id, bot_id
    )

async def analyze_deep_research(
    entity_id: str, packet: EvidencePacket, cycle_id: str, bot_id: str
) -> tuple[str, int]:
    sys = "You are a Deep Research Agent. The provided data is highly redundant. Your mission is to find unique, non-obvious catalysts and hidden risks that the consensus is missing. Be concise (max 4 sentences)."
    return await _run_specialized_agent(
        "deep_research_agent", sys, entity_id, packet, cycle_id, bot_id
    )

