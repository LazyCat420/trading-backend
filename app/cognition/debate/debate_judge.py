"""
Debate Judge — Weighs verified bull vs bear claims and produces verdict.

Receives ONLY verified claims (unverifiable claims have been removed by
ClaimVerifier). Produces a weighted decision with confidence and sourced
rationale.

The "court judge" framing prevents the LLM from introducing new data —
it can only reason about evidence already in the record.

All LLM calls go through app.services.vllm_client (Rule 2).
"""

import logging

from app.services.vllm_client import llm, Priority
from app.config.config_cognition import LLM_TEMPERATURES
from app.utils.text_utils import parse_json_response

logger = logging.getLogger(__name__)


_JUDGE_SYSTEM_BASE = """You are a neutral judge presiding over a financial analysis hearing.

You have been presented with TWO cases:
1. BULL CASE: Arguments for BUYING the stock (verified claims only)
2. BEAR CASE: Arguments for SELLING/AVOIDING the stock (verified claims only)

IMPORTANT RULES:
- Claims that were REJECTED for lack of ground truth have been removed.
  You are seeing ONLY verified, evidence-backed arguments.
- You may NOT introduce new data points not cited by either side.
- If one side had more claims rejected, their case was weakened by
  poor data quality — factor this into your confidence.
- Weigh evidence dynamically based on the source's metadata credibility presented under "## SOURCE METADATA & UNSTRUCTURED CONTEXT" (e.g. prioritize official news/SEC filings over individual Reddit posts/scores).
- Your rationale MUST cite at least 3 specific verified values from above.
- Claims tagged with [SURVIVED REBUTTAL] are from late turns and have successfully withstood attack by the opposing side. These are HIGHER SIGNAL and should be weighted more heavily than opening statements.
{hold_rule}
Output exactly this JSON:
{{
  "action": "{allowed_actions}",
  "confidence": 0-100,
  "winning_side": "bull|bear|split",
  "key_deciding_factor": "the specific claim that tipped the balance",
  "rejected_claim_impact": "how rejected claims affected your confidence",
  "rationale": "2-4 sentences citing specific verified values from above",
  "original_thesis_status": "VALID|PARTIALLY_VALID|INVALIDATED|NOT_HELD",
  "original_thesis_explanation": "explanation of whether the original buy thesis remains valid, partially valid, or has been invalidated by new evidence, and why (or empty string/NOT_HELD if not held)"
}}"""


def _build_judge_system_prompt(held: bool) -> str:
    """Build position-aware judge system prompt."""
    from app.cognition.debate.action_gate import get_allowed_actions_str

    allowed = get_allowed_actions_str(held)
    if held:
        hold_rule = ""
    else:
        hold_rule = (
            "\n- You MUST NOT output HOLD. The bot does not own this stock. "
            "You must decide BUY or SELL based on the evidence.\n"
        )
    return _JUDGE_SYSTEM_BASE.format(
        allowed_actions=allowed,
        hold_rule=hold_rule,
    )


JUDGE_USER_TEMPLATE = """## Ticker: {ticker}

{position_block}

## SOURCE METADATA & UNSTRUCTURED CONTEXT:
{source_metadata}

## BULL CASE (verified claims only):
{bull_claims}

## BEAR CASE (verified claims only):
{bear_claims}

## CROSS-EXAMINATION FINDINGS:
{cross_exam_findings}

## DATA QUALITY NOTE:
{unverified_note}

---

Weigh the verified evidence from both sides and make your final decision.
Claims that were REJECTED for lack of ground truth have been removed.
You may NOT introduce new data points."""


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


async def judge_debate(
    ticker: str,
    verified_bull_claims: list[dict],
    verified_bear_claims: list[dict],
    cross_exam_findings: str = "",
    unverified_count: int = 0,
    cycle_id: str = "",
    bot_id: str = "",
    persona_outcomes: dict | None = None,
    held: bool = False,
    source_summaries: list = None,
    position_context: dict | None = None,
) -> tuple[dict, int]:
    """Judge the adversarial debate and produce a weighted verdict.

    Args:
        ticker: Stock ticker symbol.
        verified_bull_claims: Bull claims that passed ground truth check (with metadata).
        verified_bear_claims: Bear claims that passed ground truth check (with metadata).
        cross_exam_findings: Cross-examiner's summary of contradictions.
        unverified_count: Number of claims rejected by ClaimVerifier.
        cycle_id: For audit logging.
        bot_id: For audit logging.

    Returns:
        (judge_result_dict, tokens_used) — the verdict and token count.
    """
    def format_claims(claims):
        if not claims:
            return "No verified claims survived verification."
        lines = []
        for c in claims:
            # Handle both dicts (new format) and strings (fallback)
            if isinstance(c, dict):
                claim_text = c.get("claim", "")
                turn = c.get("turn", 1)
                survived = c.get("survived_rebuttal", False)
                meta = f"[Turn {turn}{', SURVIVED REBUTTAL' if survived else ''}]"
                lines.append(f"- {claim_text} {meta}")
            else:
                lines.append(f"- {c}")
        return "\n".join(lines)

    bull_text = format_claims(verified_bull_claims)
    bear_text = format_claims(verified_bear_claims)

    unverified_note = "All claims verified — high evidence quality."
    if unverified_count > 0:
        unverified_note = (
            f"{unverified_count} claim(s) were REJECTED because their cited "
            f"values could not be found in the evidence packet. "
            f"The remaining claims above are verified."
        )

    # ── Dissent Preservation: force judge to address minority persona ──
    minority_block = ""
    if persona_outcomes and len(persona_outcomes) >= 3:
        winners = [v.get("winner", "split") for v in persona_outcomes.values()]
        bull_votes = sum(1 for w in winners if w == "bull")
        bear_votes = sum(1 for w in winners if w == "bear")

        if bull_votes >= 2 and bear_votes >= 1:
            # Bear is minority
            minority_personas = [k for k, v in persona_outcomes.items() if v.get("winner") == "bear"]
            minority_claims = sum(v.get("bear_claims_count", 0) for k, v in persona_outcomes.items() if v.get("winner") == "bear")
            minority_block = (
                f"\n## MINORITY DISSENT\n"
                f"Persona(s) {', '.join(minority_personas)} produced {minority_claims} verified claims "
                f"arguing SELL/BEAR. You MUST address why this dissent is overridden or "
                f"incorporate it into a confidence discount.\n"
            )
        elif bear_votes >= 2 and bull_votes >= 1:
            # Bull is minority
            minority_personas = [k for k, v in persona_outcomes.items() if v.get("winner") == "bull"]
            minority_claims = sum(v.get("bull_claims_count", 0) for k, v in persona_outcomes.items() if v.get("winner") == "bull")
            minority_block = (
                f"\n## MINORITY DISSENT\n"
                f"Persona(s) {', '.join(minority_personas)} produced {minority_claims} verified claims "
                f"arguing BUY/BULL. You MUST address why this dissent is overridden or "
                f"incorporate it into a confidence discount.\n"
            )

    source_metadata = "None available."
    if source_summaries:
        source_metadata = "\n".join(
            [format_source_ref_for_prompt(s) for s in source_summaries[:15]]
        )

    position_block = ""
    if held and position_context:
        from app.tools.portfolio_tools import format_position_context_for_prompt
        position_block = format_position_context_for_prompt(position_context)

    user_prompt = JUDGE_USER_TEMPLATE.format(
        ticker=ticker,
        position_block=position_block,
        source_metadata=source_metadata,
        bull_claims=bull_text,
        bear_claims=bear_text,
        cross_exam_findings=cross_exam_findings or "No cross-examination issues found.",
        unverified_note=unverified_note,
    )

    if minority_block:
        user_prompt += minority_block
        logger.info("[JUDGE] Injected minority dissent block for %s", ticker)

    from app.cognition.debate.action_gate import gate_action

    system_prompt = _build_judge_system_prompt(held)

    tokens_used = 0
    try:
        response, tokens, ms = await llm.chat(
            system=system_prompt,
            user=user_prompt,
            temperature=LLM_TEMPERATURES.get("debate_judge", 0.2),
            max_tokens=512,
            priority=Priority.NORMAL,
            agent_name="debate_judge",
            ticker=ticker,
            cycle_id=cycle_id,
            bot_id=bot_id,
        )
        tokens_used = tokens or 0
        result = parse_json_response(response)
        logger.info(
            "[JUDGE] %s verdict: %s @ %d%% (winner: %s) [%d tokens, %dms]",
            ticker,
            result.get("action", "?"),
            result.get("confidence", 0),
            result.get("winning_side", "?"),
            tokens_used,
            ms,
        )
    except Exception as e:
        logger.error("[JUDGE] Failed for %s: %s", ticker, e)
        error_default = gate_action("HOLD", held)
        result = {
            "action": error_default,
            "confidence": 0,
            "winning_side": "split",
            "key_deciding_factor": f"Judge failed: {e}",
            "rejected_claim_impact": "Unable to assess",
            "rationale": f"Debate judge failed to produce verdict: {e}",
            "original_thesis_status": "NOT_HELD" if not held else "VALID",
            "original_thesis_explanation": f"Failed to run judge: {e}",
        }

    # Validate and gate action based on position state
    raw_action = result.get("action", "HOLD").upper()
    result["action"] = gate_action(raw_action, held)

    return result, tokens_used
