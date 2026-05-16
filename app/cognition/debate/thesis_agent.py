"""
Thesis Agent — Generates a structured trading thesis from an EvidencePacket.

When an adversarial debate result is present in extra_context, this agent
acts as a synthesis writer (lower temperature, formats the judge's verdict)
rather than reasoning from scratch.
"""

import logging
from app.services.vllm_client import llm, Priority
from app.utils.text_utils import parse_json_response
from app.config.config_cognition import LLM_TEMPERATURES
from app.cognition.contracts.evidence import EvidencePacket
from app.cognition.contracts.debate import ThesisDraft

logger = logging.getLogger(__name__)


SYSTEM_PROMPT_TEMPLATE = """You are an expert quantitative and fundamental financial analyst.
Your job is to synthesize a structured trading thesis based ONLY on the provided evidence.

Do not invent facts. If evidence is missing, state that.
Be specific with numbers, dates, and names provided in the context.

DECISION FRAMEWORK — apply these rules strictly:
{constitution_rules}

IMPORTANT — EXISTING POSITIONS:
If a CURRENT POSITION STATUS block is provided, you MUST factor the bot's
entry price, P&L, and holding duration into your decision. A position with
an unrealized loss that is approaching its stop-loss should bias toward SELL.
A position with a modest gain but deteriorating fundamentals should also
consider SELL. Do NOT default to HOLD for held positions — actively evaluate
whether the capital is best deployed here or elsewhere.

GROUNDING REQUIREMENTS (critical for evaluation):
- Your "rationale" MUST quote at least 2 specific data points VERBATIM from the provided evidence
  with source labels (e.g., "RSI(14): 62.3 [technical_data]", "P/E: 22.48 [fundamental_data]").
- Do NOT cite any numeric values that are not explicitly present in the evidence below.
- If a MACRO STRATEGY MEMO is provided, your rationale MUST acknowledge the current market regime and how it influences this thesis.
- State one explicit invalidation condition (e.g., "Invalidated if RSI > 70" or "Invalidated if VIX spikes above 30").

Respond with exactly this JSON schema:
{{
  "action": "{allowed_actions}",
  "confidence": 0-100,
  "core_claims": ["claim 1", "claim 2", "claim 3"],
  "evidence_refs": ["ref 1", "ref 2"],
  "weaknesses": ["known weakness 1", "known weakness 2"],
  "rationale": "2-4 sentences: cite specific data from context, reference macro regime if available, and state an invalidation condition"
}}"""


# Used when adversarial debate results are present — synthesis mode
SYNTHESIS_SYSTEM_PROMPT = """You are a financial thesis writer synthesizing a structured trading thesis.

An adversarial debate has ALREADY been conducted between bull and bear analysts,
with claims verified against ground truth and judged by a neutral arbiter.
The DEBATE RESULT is provided in the context below.

Your job is to FORMAT the debate verdict into a clean, structured thesis.
Do NOT override the judge's decision. Instead:
1. Adopt the judge's action and confidence as your starting point
2. Cite the verified claims from the winning side
3. Acknowledge the strongest counter-arguments from the losing side as weaknesses
4. State an explicit invalidation condition

Respond with exactly this JSON schema:
{{
  "action": "{allowed_actions}",
  "confidence": 0-100,
  "core_claims": ["verified claim 1", "verified claim 2", "verified claim 3"],
  "evidence_refs": ["ref 1", "ref 2"],
  "weaknesses": ["counter-argument 1", "counter-argument 2"],
  "rationale": "2-4 sentences: synthesize the debate verdict, cite verified values, state invalidation condition"
}}"""


USER_TEMPLATE = """## Entity ID: {entity_id}
## Bias / Direction: {bias}

## Available Claims:
{claims_text}

## Structured Facts:
{structured_facts}

## Context / Missing Data:
{context_meta}

Construct a trading thesis with the requested bias. Make sure you acknowledge the missing data in your weaknesses.
"""


async def generate_thesis(
    entity_id: str,
    packet: EvidencePacket,
    bias: str = "neutral",
    cycle_id: str = "",
    bot_id: str = "",
    extra_context: str = "",
    watchlist: list[str] | None = None,
    held: bool = False,
) -> tuple[ThesisDraft, int]:
    """Generate a structured thesis from an EvidencePacket.

    Returns:
        (ThesisDraft, tokens_used) — the draft and total tokens consumed.
    """

    claims_text = "\n".join(
        [
            f"- [{c.provenance.source_table}] {c.subject_entity_id} {c.predicate} {c.object_value} (conf: {c.confidence:.2f})"
            for c in packet.claims[:20]
        ]
    )
    if not claims_text:
        claims_text = "No explicit claims available."

    meta = []
    if packet.contradictions:
        meta.append(
            "Contradictions: "
            + "; ".join([c.description for c in packet.contradictions])
        )
    if packet.missing_fields:
        meta.append("Missing Critical Data: " + "; ".join(packet.missing_fields))
    context_meta = (
        "\n".join(meta) if meta else "No known contradictions or missing data."
    )

    user_prompt = USER_TEMPLATE.format(
        entity_id=entity_id,
        bias=bias,
        claims_text=claims_text,
        structured_facts=str(packet.structured_facts or {}),
        context_meta=context_meta,
    )

    # Inject ontology/macro context if provided
    if extra_context:
        user_prompt = extra_context.strip() + "\n\n" + user_prompt

    # Inject watchlist peer context so the LLM knows what other tickers
    # are being analysed in the same cycle.
    if watchlist:
        peers = [t for t in watchlist if t.upper() != entity_id.upper()]
        if peers:
            user_prompt += f"\n\n## Peer Tickers in This Cycle:\n{', '.join(peers)}"

    tokens_used = 0

    # Determine mode: synthesis (debate present) or independent reasoning
    is_synthesis = extra_context and "ADVERSARIAL DEBATE RESULT" in extra_context
    from app.db.constitution import format_constitution_for_prompt
    from app.cognition.debate.action_gate import get_allowed_actions_str, gate_action

    allowed_actions = get_allowed_actions_str(held)

    active_prompt = (
        SYNTHESIS_SYSTEM_PROMPT.format(allowed_actions=allowed_actions)
        if is_synthesis
        else SYSTEM_PROMPT_TEMPLATE.format(
            constitution_rules=format_constitution_for_prompt(),
            allowed_actions=allowed_actions,
        )
    )
    active_temp_key = "thesis_synthesis" if is_synthesis else "thesis_generation"
    if is_synthesis:
        logger.info(
            "[THESIS] Synthesis mode — formatting debate verdict for %s", entity_id
        )

    try:
        response, tokens, ms = await llm.chat(
            system=active_prompt,
            user=user_prompt,
            temperature=LLM_TEMPERATURES.get(active_temp_key, 0.5),
            max_tokens=768,
            priority=Priority.NORMAL,
            agent_name="thesis_agent",
            ticker=entity_id,
            cycle_id=cycle_id,
            bot_id=bot_id,
        )
        tokens_used = tokens or 0
        data = parse_json_response(response)
    except Exception as e:
        logger.error("[THESIS] Failed to generate thesis: %s", e)
        # Attempt to salvage the response if it was just a parsing error
        raw_text = (
            response if "response" in locals() and isinstance(response, str) else str(e)
        )
        data = {
            "error": f"Failed to parse thesis. Reason: {e}. Raw: {raw_text[:250]}..."
        }

    # If parsing returned an empty dict but no exception was thrown
    if not data and "response" in locals() and isinstance(response, str):
        data = {
            "error": f"Failed to parse thesis. Invalid JSON format. Raw: {response[:250]}..."
        }

    action = gate_action(data.get("action", "HOLD"), held)

    draft = ThesisDraft(
        action=action,
        confidence=int(data.get("confidence", 0)),
        core_claims=data.get("core_claims", []),
        evidence_refs=data.get("evidence_refs", []),
        weaknesses=data.get("weaknesses", []),
        rationale=data.get("rationale", data.get("error", "Failed to parse thesis")),
        iteration=0,
    )
    return draft, tokens_used
