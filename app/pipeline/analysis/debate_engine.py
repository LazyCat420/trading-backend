"""
Debate Engine — Dynamic persona-based challenge system for Config D.

Architecture:
  1. Config C produces a trading decision (BUY/SELL/HOLD + rationale)
  2. Meta-Prompt: LLM generates a custom devil's advocate persona
     tailored to the specific trade context
  3. Debate: The generated persona challenges Config C's thesis
     with evidence-based counterarguments
  4. Synthesis: Compare thesis vs antithesis → final decision

This replaces the old Config D (same RLM with thinking=ON) with a
dynamic debate that produces more diverse, rigorous outcomes.

Usage:
    from app.pipeline.analysis.debate_engine import run_debate
    result = await run_debate(ticker, config_c_result, context, ...)
"""

import logging
import time

from app.services.vllm_client import llm, Priority
from app.services.prism_agent_caller import call_prism_agent
from app.utils.text_utils import parse_json_response, sanitize_ascii
from app.utils.pipeline_utils import elapsed_ms

logger = logging.getLogger(__name__)


# ─── Meta-Prompt: generates a dynamic debate persona ───────────────
META_SYSTEM_PROMPT = """You are an expert at creating specialized analyst personas for stock market debates.

Your job: Given a trader's analysis and market data, create a CUSTOM SYSTEM PROMPT for an analyst who will cross-examine this specific trade thesis.

The persona should:
- Have a specific analytical framework (e.g., value investor, macro analyst, risk manager, contrarian, quant)
- Be chosen to be the BEST fit to evaluate THIS particular analysis
- Stay grounded in stock market trading — no generic debating
- Actively seek the strongest definitive trade (BUY or SELL). If the thesis is HOLD, the persona should push for a decisive BUY or SELL based on the data.

Respond with ONLY a JSON object:
{
  "persona_name": "short title, e.g. 'Macro Risk Analyst'",
  "persona_rationale": "1 sentence on why this persona fits",
  "system_prompt": "the full system prompt for the debate persona (2-4 paragraphs)"
}"""

META_USER_TEMPLATE = """## Trader's Analysis for {ticker}

**Decision:** {action} @ {confidence}% confidence
**Config Used:** {config_used}
**Rationale:** {rationale}

## Agent Summaries
{agent_summaries}

## Key Market Data (abbreviated)
{context_summary}

---

Based on the trader's analysis above, create a debate persona who would be the BEST-FIT devil's advocate to challenge this specific trade thesis. The persona should push back on the weakest points of the analysis using evidence and data."""


# ─── Debate prompt: the generated persona uses this to challenge ───
DEBATE_USER_TEMPLATE = """## Original Trade Thesis for {ticker}

**Proposed Action:** {action} @ {confidence}% confidence

**Rationale:**
{rationale}

## Full Market Data
{context}

---

Using your analytical framework, challenge this trade thesis. Be specific:
1. Identify the 2-3 weakest assumptions in the analysis
2. Cite specific data points from the market data that contradict the thesis
3. Assess what risks are being underweighted
4. Give your counter-recommendation

Respond with JSON:
{{
  "counter_action": "{allowed_actions}",
  "counter_confidence": 0-100,
  "challenges": ["specific challenge 1", "specific challenge 2", "specific challenge 3"],
  "risk_factors": ["risk 1", "risk 2"],
  "counter_rationale": "2-3 sentence summary of your counter-argument"
}}"""


# ─── Synthesis: merge thesis + antithesis into final call ──────────
_SYNTHESIS_SYSTEM_BASE = """You are a senior portfolio manager making a final trading decision.

You have TWO analyses to consider:
1. The original analyst's thesis (Config C)
2. A devil's advocate's counter-thesis (Config D debate)

Your job: weigh both perspectives and make a DEFINITIVE final decision. You MUST pick a winner based on the strongest empirical evidence. Do NOT use HOLD as a safe compromise between two conflicting arguments. You must choose the side (BUY or SELL) with the most asymmetric upside or downside, unless the data is completely neutral.

Respond with ONLY JSON:
{{
  "action": "{allowed_actions}",
  "confidence": 0-100,
  "rationale": "2-3 sentences explaining your final decision, citing which arguments won and why. You must explicitly declare a winner.",
  "thesis_won": true/false,
  "key_risk": "the single biggest risk identified by the debate"
}}"""

SYNTHESIS_USER_TEMPLATE = """## Ticker: {ticker}

## THESIS (Original Analyst — Config C)
**Action:** {c_action} @ {c_confidence}%
**Rationale:** {c_rationale}

## ANTITHESIS (Devil's Advocate — {persona_name})
**Counter-Action:** {d_action} @ {d_confidence}%
**Challenges:** {challenges}
**Counter-Rationale:** {d_rationale}

---

Make your final call. Who has the stronger argument and why%s"""


def _truncate_context(context: str, max_chars: int = 2000) -> str:
    """Create abbreviated context summary for the meta-prompt."""
    if len(context) <= max_chars:
        return context
    # Take first 1500 + last 500 chars
    return context[:1500] + "\n...[truncated]...\n" + context[-500:]


async def run_debate(
    ticker: str,
    config_c_result: dict,
    context: str,
    agent_summaries_text: str = "",
    cycle_id: str = "",
    bot_id: str = "",
    held: bool = False,
) -> dict:
    """Run the dynamic debate pipeline against Config C's thesis.

    Steps:
        1. Meta-prompt: generate tailored debate persona
        2. Debate: persona challenges Config C with evidence
        3. Synthesis: merge thesis + antithesis into final decision

    Args:
        ticker: Stock ticker
        config_c_result: Dict from Config C (action, confidence, rationale)
        context: Full context blob from context_builder
        agent_summaries_text: Formatted agent analysis text
        cycle_id: For audit logging
        bot_id: For audit logging

    Returns:
        Dict with final action, confidence, rationale, debate metadata
    """
    start = time.monotonic()
    from app.cognition.debate.action_gate import get_allowed_actions_str, gate_action
    
    allowed_actions = get_allowed_actions_str(held)
    
    c_action = gate_action(config_c_result.get("action", "HOLD"), held)
    c_confidence = config_c_result.get("confidence", 0)
    c_rationale = config_c_result.get("rationale", "")

    logger.info("[PIPELINE] " + "=" * 60)
    logger.info("[PIPELINE] [DEBATE] Starting dynamic debate for %s", ticker)
    logger.info(
        "[PIPELINE] [DEBATE] Config C thesis: %s @ %d%%", c_action, c_confidence
    )
    logger.info("[PIPELINE] " + "=" * 60)

    # ── Step 1: Generate debate persona ──────────────────────────
    logger.info(
        f"[PIPELINE] \n  >>> [DEBATE] Step 1/3 — Generating devil's advocate persona for {ticker}..."
    )
    logger.info("[PIPELINE] [DEBATE] Step 1: Generating debate persona...")
    t1 = time.monotonic()

    meta_user = META_USER_TEMPLATE.format(
        ticker=ticker,
        action=c_action,
        confidence=c_confidence,
        config_used=config_c_result.get("config_used", "C"),
        rationale=c_rationale[:500],
        agent_summaries=agent_summaries_text[:1000] if agent_summaries_text else "N/A",
        context_summary=_truncate_context(context, 2000),
    )

    try:
        meta_response, meta_tokens, meta_ms = await call_prism_agent(
            agent_id="CUSTOM_DEBATE_META_AGENT",
            user_message=meta_user,
            fallback_system_prompt=META_SYSTEM_PROMPT,
            fallback_agent_name="debate_meta",
            temperature=0.7,
            max_tokens=512,
            priority=Priority.NORMAL,
            ticker=ticker,
            cycle_id=cycle_id,
            bot_id=bot_id,
        )
    except Exception as e:
        logger.error("[PIPELINE] [DEBATE] Meta-prompt failed: %s", e)
        return _fallback_result(config_c_result, f"Meta-prompt failed: {e}", held)

    meta_parsed = parse_json_response(meta_response)
    persona_name = meta_parsed.get("persona_name", "Devil's Advocate")
    persona_prompt = meta_parsed.get("system_prompt", "")
    persona_rationale = meta_parsed.get("persona_rationale", "")

    if not persona_prompt:
        # Fallback: use a generic but strong devil's advocate prompt
        persona_name = "Generic Risk Analyst"
        persona_prompt = (
            "You are a skeptical risk analyst. Your job is to find flaws "
            "in trading theses by identifying overlooked risks, questionable "
            "assumptions, and contradictory data points. Be specific and "
            "cite numbers from the market data provided."
        )

    ms1 = elapsed_ms(t1)
    logger.info(
        f"[PIPELINE]   >>> [DEBATE] Step 1/3 done — Persona: '{persona_name}' ({ms1}ms)"
    )
    logger.info(
        "[DEBATE] Persona: '%s' (%s) [%d tokens, %dms]",
        persona_name,
        persona_rationale[:80],
        meta_tokens,
        ms1,
    )
    logger.info("[DEBATE] RAW PERSONA PROMPT:\n%s", persona_prompt)

    # ── Step 2: Run debate with generated persona ────────────────
    logger.info(
        f"[PIPELINE]   >>> [DEBATE] Step 2/3 — {persona_name} challenging the thesis..."
    )
    logger.info("[PIPELINE] [DEBATE] Step 2: %s challenging thesis...", persona_name)
    t2 = time.monotonic()

    # Smart truncation: prioritize analytical sections over raw price data
    safe_context = sanitize_ascii(context)
    if len(safe_context) > 10000:
        # Keep sections most relevant to debate (agent summaries, peer comp, risk)
        # Drop raw price history which the debate persona doesn't need
        priority_sections = []
        other_sections = []
        for section in safe_context.split("\n## "):
            header_lower = section[:80].lower()
            if any(
                kw in header_lower
                for kw in [
                    "agent",
                    "peer",
                    "risk",
                    "sentiment",
                    "technical",
                    "fundamental",
                    "youtube",
                    "reddit",
                    "news",
                    "unstructured",
                    "macro",
                    "congress",
                    "institutional",
                ]
            ):
                priority_sections.append(section)
            else:
                other_sections.append(section)
        # Rebuild: priority sections first (full), then other sections (truncated)
        priority_text = "\n## ".join(priority_sections)
        remaining_budget = max(10000 - len(priority_text), 2000)
        other_text = "\n## ".join(other_sections)[:remaining_budget]
        safe_context = (
            priority_text + "\n## " + other_text if other_text else priority_text
        )

    debate_user = DEBATE_USER_TEMPLATE.format(
        ticker=ticker,
        action=c_action,
        confidence=c_confidence,
        rationale=c_rationale,
        context=safe_context,
        allowed_actions=allowed_actions,
    )

    try:
        debate_response, debate_tokens, debate_ms = await call_prism_agent(
            agent_id="CUSTOM_DEBATE_CHALLENGE_AGENT",
            user_message=debate_user,
            fallback_system_prompt=persona_prompt,
            fallback_agent_name="debate_challenge",
            temperature=0.4,
            max_tokens=768,
            priority=Priority.NORMAL,
            ticker=ticker,
            cycle_id=cycle_id,
            bot_id=bot_id,
        )
    except Exception as e:
        logger.error("[PIPELINE] [DEBATE] Debate call failed: %s", e)
        return _fallback_result(config_c_result, f"Debate call failed: {e}", held)

    debate_parsed = parse_json_response(debate_response)
    d_action = gate_action(debate_parsed.get("counter_action", "HOLD"), held)
    d_confidence = int(debate_parsed.get("counter_confidence", 0))
    d_rationale = debate_parsed.get("counter_rationale", debate_response[:300])
    challenges = debate_parsed.get("challenges", [])
    risk_factors = debate_parsed.get("risk_factors", [])

    ms2 = elapsed_ms(t2)
    logger.info(
        f"[PIPELINE]   >>> [DEBATE] Step 2/3 done — {persona_name}: {d_action} @ {d_confidence}% ({ms2}ms)"
    )
    logger.info("[DEBATE] RAW CHALLENGE RESPONSE:\n%s", debate_response)
    logger.info(
        "[DEBATE] %s says: %s @ %d%% [%d tokens, %dms]",
        persona_name,
        d_action,
        d_confidence,
        debate_tokens,
        ms2,
    )
    if challenges:
        for i, c in enumerate(challenges[:3], 1):
            logger.info("[PIPELINE] [DEBATE]   Challenge %d: %s", i, c[:100])
            
        # ── Integrity Gate: Verify Devil's Advocate Claims ────────────
        logger.info("[PIPELINE] [DEBATE] Running Integrity Gate on challenges...")
        try:
            from app.cognition.debate.debate_coordinator import (
                CROSS_EXAM_SYSTEM_PROMPT,
                CROSS_EXAM_USER_TEMPLATE,
            )
            import json

            cross_user = CROSS_EXAM_USER_TEMPLATE.format(
                bull_claims="[]",
                bear_claims=json.dumps(challenges[:3], indent=2),
                tool_research="N/A",
                unstructured_context="N/A",
                structured_facts=safe_context[:15000],
            )

            cross_response, _, _ = await call_prism_agent(
                agent_id="CUSTOM_DEBATE_CROSS_EXAM_AGENT",
                user_message=cross_user,
                fallback_system_prompt=CROSS_EXAM_SYSTEM_PROMPT,
                fallback_agent_name="debate_cross_exam",
                temperature=0.1,
                max_tokens=512,
                priority=Priority.NORMAL,
                ticker=ticker,
                cycle_id=cycle_id,
                bot_id=bot_id,
            )
            cross_parsed = parse_json_response(cross_response)
            unverified_challenges = cross_parsed.get("unverified_bear_claims", [])

            # Integrity check: if all challenges are unverified, we override d_action
            if len(unverified_challenges) > 0 and len(unverified_challenges) == len(challenges[:3]):
                logger.warning(
                    "[DEBATE] ALL challenges failed verification (LOW_INTEGRITY). Discarding counter-argument."
                )
                d_action = "PASS"
                d_confidence = 0
                d_rationale = "[LOW INTEGRITY] All cited challenges were hallucinated or contradicted by context. Ignore this counter-argument and default to the original thesis."
                challenges = [f"FAILED VERIFICATION: {c}" for c in challenges]
            elif unverified_challenges:
                logger.info("[DEBATE] %d challenge(s) failed verification.", len(unverified_challenges))

        except Exception as e:
            logger.warning("[DEBATE] Cross-examiner verification failed: %s", e)

    # ── Step 3: Synthesis — merge thesis + antithesis ────────────
    logger.info("[PIPELINE]   >>> [DEBATE] Step 3/3 — Synthesizing final decision...")
    logger.info("[PIPELINE] [DEBATE] Step 3: Synthesizing final decision...")
    t3 = time.monotonic()

    synthesis_user = SYNTHESIS_USER_TEMPLATE.format(
        ticker=ticker,
        c_action=c_action,
        c_confidence=c_confidence,
        c_rationale=c_rationale[:400],
        persona_name=persona_name,
        d_action=d_action,
        d_confidence=d_confidence,
        challenges="; ".join(challenges[:3]) if challenges else "None specific",
        d_rationale=d_rationale[:400],
    )

    try:
        synth_response, synth_tokens, synth_ms = await call_prism_agent(
            agent_id="CUSTOM_DEBATE_SYNTHESIS_AGENT",
            user_message=synthesis_user,
            fallback_system_prompt=_SYNTHESIS_SYSTEM_BASE.format(allowed_actions=allowed_actions),
            fallback_agent_name="debate_synthesis",
            temperature=0.2,
            max_tokens=512,
            priority=Priority.NORMAL,
            ticker=ticker,
            cycle_id=cycle_id,
            bot_id=bot_id,
        )
    except Exception as e:
        logger.error("[PIPELINE] [DEBATE] Synthesis failed: %s", e)
        return _fallback_result(config_c_result, f"Synthesis failed: {e}", held)

    try:
        from app.pipeline.analysis.hallucination_checker import audit_numeric_divergence

        audit_numeric_divergence(synth_response, ticker, cycle_id, "debate_engine.py")
    except Exception as e:
        logger.warning(f"[DEBATE] Audit divergence failed: {e}")

    synth_parsed = parse_json_response(synth_response)
    final_action = gate_action(synth_parsed.get("action", c_action), held)
    final_confidence = int(synth_parsed.get("confidence", c_confidence))
    final_rationale = synth_parsed.get("rationale", synth_response[:300])
    thesis_won = synth_parsed.get("thesis_won", True)
    key_risk = synth_parsed.get("key_risk", "")

    ms3 = elapsed_ms(t3)
    logger.info("[DEBATE] RAW SYNTHESIS RESPONSE:\n%s", synth_response)

    total_time = time.monotonic() - start
    total_tokens = meta_tokens + debate_tokens + synth_tokens

    logger.info("[PIPELINE] " + "=" * 60)
    logger.info(
        "[DEBATE] FINAL: %s @ %d%% | Thesis %s | Key risk: %s",
        final_action,
        final_confidence,
        "WON" if thesis_won else "LOST",
        key_risk[:60],
    )
    logger.info(
        "[DEBATE] Tokens: %d (meta:%d + debate:%d + synth:%d) | Time: %.1fs",
        total_tokens,
        meta_tokens,
        debate_tokens,
        synth_tokens,
        total_time,
    )
    logger.info("[PIPELINE] " + "=" * 60)

    # ── Persist debate outcome for warm-start memory ──
    try:
        from app.db.connection import get_db
        import uuid as _uuid

        with get_db() as _db:
            _db.execute(
                """INSERT INTO debate_history
                (id, cycle_id, ticker, thesis_action, thesis_confidence,
                 counter_action, counter_confidence, winner, final_action,
                 final_confidence, persona_name, key_risk)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker, cycle_id) DO UPDATE SET
                thesis_action = EXCLUDED.thesis_action,
                thesis_confidence = EXCLUDED.thesis_confidence,
                counter_action = EXCLUDED.counter_action,
                counter_confidence = EXCLUDED.counter_confidence,
                winner = EXCLUDED.winner,
                final_action = EXCLUDED.final_action,
                final_confidence = EXCLUDED.final_confidence,
                persona_name = EXCLUDED.persona_name,
                key_risk = EXCLUDED.key_risk""",
                (
                    f"dh-{_uuid.uuid4().hex[:12]}",
                    cycle_id,
                    ticker,
                    c_action,
                    c_confidence,
                    d_action,
                    d_confidence,
                    "thesis" if thesis_won else "antithesis",
                    final_action,
                    final_confidence,
                    persona_name,
                    key_risk[:300] if key_risk else "",
                ),
            )
        logger.info(
            "[DEBATE] debate_history persisted for %s (winner=%s)",
            ticker,
            "thesis" if thesis_won else "antithesis",
        )
    except Exception as dh_err:
        logger.debug("[DEBATE] debate_history write failed (non-fatal): %s", dh_err)

    return {
        "action": final_action,
        "confidence": final_confidence,
        "rationale": final_rationale,
        "tokens_used": total_tokens,
        "execution_time_s": round(total_time, 2),
        "method": "debate",
        # Debate metadata
        "debate": {
            "persona_name": persona_name,
            "persona_rationale": persona_rationale,
            "persona_system_prompt": persona_prompt,
            "thesis_action": c_action,
            "thesis_confidence": c_confidence,
            "counter_action": d_action,
            "counter_confidence": d_confidence,
            "counter_rationale": d_rationale,
            "challenges": challenges,
            "risk_factors": risk_factors,
            "thesis_won": thesis_won,
            "key_risk": key_risk,
            "meta_tokens": meta_tokens,
            "debate_tokens": debate_tokens,
            "synthesis_tokens": synth_tokens,
            "meta_ms": ms1,
            "debate_ms": ms2,
            "synthesis_ms": ms3,
        },
    }


def _fallback_result(config_c_result: dict, reason: str, held: bool = False) -> dict:
    """Return Config C result with reduced confidence on debate failure."""
    from app.cognition.debate.action_gate import gate_action
    return {
        "action": gate_action(config_c_result.get("action", "HOLD"), held),
        "confidence": max(config_c_result.get("confidence", 0) - 10, 0),
        "rationale": (
            f"Debate failed ({reason}). Using Config C with reduced confidence. "
            f"Original: {config_c_result.get('rationale', '')[:200]}"
        ),
        "tokens_used": 0,
        "execution_time_s": 0,
        "method": "debate_fallback",
        "debate": {"error": reason},
    }
