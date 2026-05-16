"""
Agent Capsule — Compressed context unit for inter-agent communication.

A capsule is a two-layer structured object:
  Layer 1: Always injected into context (~150 tokens) — summary, signal, flags, source refs
  Layer 2: DB-stored raw data accessible via get_cycle_context tool on demand

This replaces the raw response passthrough pattern where full agent outputs
(1,000–3,000 tokens each) were forwarded verbatim between agents.

Usage:
    from app.agents.capsule import AgentCapsule, format_capsule_stack

    capsule = AgentCapsule(
        agent_name="retriever", cycle_id="abc", ticker="NVDA",
        summary="RSI=72 (overbought), vol 3x avg, EPS surprise +8%.",
        signal="BULLISH", confidence=0.78,
        flags=["overbought_momentum"], source_refs=["ref:capsule:abc-123"],
        raw_id="abc-123", tokens_estimated=38,
    )
    prompt_text = format_capsule_stack([capsule])
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentCapsule:
    """Compressed representation of an agent's findings for a single cycle."""

    agent_name: str  # "planner", "retriever", "verifier", "synthesizer"
    cycle_id: str
    ticker: str
    summary: str  # ~1-3 sentence dense finding (Layer 1)
    signal: str  # BUY | SELL | HOLD | NEUTRAL | CONFLICTED | UNKNOWN
    confidence: float  # 0.0–1.0
    flags: list[str] = field(default_factory=list)  # ["divergence", "low_volume"]
    source_refs: list[str] = field(default_factory=list)  # ["ref:capsule:<uuid>"]
    raw_id: str = ""  # UUID pointing to cycle_context table row
    tokens_estimated: int = 0  # For budget tracking


# Sentinel for failed/timed-out agents
EMPTY_CAPSULE = AgentCapsule(
    agent_name="unknown",
    cycle_id="",
    ticker="",
    summary="Agent failed or timed out. No data available.",
    signal="UNKNOWN",
    confidence=0.0,
    flags=["agent_failure"],
    source_refs=[],
    raw_id="",
    tokens_estimated=12,
)


def _estimate_tokens(text: str) -> int:
    """Fast heuristic token estimation (~4 chars per token)."""
    return len(text) // 4


def format_capsule_for_prompt(capsule: AgentCapsule) -> str:
    """Render a single capsule as the Layer 1 text block for prompt injection.

    Example output:
        RETRIEVER [signal:BULLISH, confidence:0.78]:
        RSI=72 (overbought), vol 3x avg, EPS surprise +8%.
        ⚠ Flags: overbought_momentum
        → Expand: ref:capsule:abc-123
    """
    lines = []
    lines.append(
        f"{capsule.agent_name.upper()} "
        f"[signal:{capsule.signal}, confidence:{capsule.confidence:.2f}]:"
    )
    lines.append(f"  {capsule.summary}")

    if capsule.flags:
        flags_str = ", ".join(capsule.flags)
        lines.append(f"  ⚠ Flags: {flags_str}")

    if capsule.source_refs:
        refs_str = ", ".join(capsule.source_refs[:3])  # Cap at 3 refs
        lines.append(f"  → Expand: {refs_str}")

    return "\n".join(lines)


def format_capsule_stack(
    capsules: list[AgentCapsule],
    max_tokens: int = 600,
) -> str:
    """Join all capsule Layer 1 summaries with a hard token cap.

    Capsules are added in order until the budget is exhausted.
    If the full stack exceeds the budget, later capsules are truncated
    to their signal+confidence header only (no summary text).

    Args:
        capsules: Ordered list of capsules (typically planner → retriever → verifier → synthesizer)
        max_tokens: Hard token ceiling for the entire stack

    Returns:
        Formatted text block ready for prompt injection
    """
    if not capsules:
        return ""

    header = "## PRIOR AGENT FINDINGS (compressed)\n"
    header += (
        "Use get_cycle_context(ref_id) tool to expand any finding if you need full details.\n"
    )
    parts = [header]
    budget_remaining = max_tokens - _estimate_tokens(header)

    for capsule in capsules:
        if capsule is EMPTY_CAPSULE:
            continue

        # Failed agents get a compact failure header so downstream agents
        # know a prior step failed, rather than silently seeing a shorter stack.
        if capsule.signal == "UNKNOWN":
            failure_header = (
                f"{capsule.agent_name.upper()} [FAILED]: {capsule.summary[:100]}"
            )
            if budget_remaining > _estimate_tokens(failure_header):
                parts.append(failure_header)
                budget_remaining -= _estimate_tokens(failure_header)
            continue

        full_text = format_capsule_for_prompt(capsule)
        full_tokens = _estimate_tokens(full_text)

        if full_tokens <= budget_remaining:
            # Full capsule fits
            parts.append(full_text)
            budget_remaining -= full_tokens
        elif budget_remaining > 30:
            # Truncated: signal header only
            truncated = (
                f"{capsule.agent_name.upper()} "
                f"[signal:{capsule.signal}, confidence:{capsule.confidence:.2f}]: "
                f"(truncated — use get_cycle_context to expand)"
            )
            parts.append(truncated)
            budget_remaining -= _estimate_tokens(truncated)
        else:
            # No budget left — skip
            logger.debug(
                "[CapsuleStack] Budget exhausted, skipping %s capsule",
                capsule.agent_name,
            )
            break

    return "\n\n".join(parts)
