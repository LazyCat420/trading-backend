"""
Degraded Retry — Simplified retry strategy for DEGRADED failures.

When an LLM produces invalid/empty output, this module provides a
stripped-down retry that uses minimal context to get *some* answer.

Strategy:
    1. Strip context to price data only (section header parsing)
    2. If that fails, ultra-minimal prompt ("based on ticker name only")
    3. Result always tagged with degraded=True so debate engine weights lower
"""

import logging

logger = logging.getLogger(__name__)

# Section headers used by context_builder.py
_PRICE_SECTIONS = {"## PRICE HISTORY", "## TECHNICAL INDICATORS", "## KEY LEVELS"}


def strip_to_price_context(full_context: str) -> str:
    """Strip a full context blob down to just price-related sections.

    Parses section headers from context_builder.py output and keeps only
    PRICE HISTORY, TECHNICAL INDICATORS, and KEY LEVELS.
    """
    if not full_context:
        return ""

    lines = full_context.split("\n")
    result_lines = []
    in_price_section = False

    for line in lines:
        stripped = line.strip()
        # Check if this line starts a new section
        if stripped.startswith("## "):
            in_price_section = any(stripped.startswith(s) for s in _PRICE_SECTIONS)

        if in_price_section:
            result_lines.append(line)

    stripped_context = "\n".join(result_lines)
    if stripped_context:
        logger.info(
            "[DEGRADED] Stripped context from %d to %d chars (price-only)",
            len(full_context),
            len(stripped_context),
        )
        return stripped_context

    # If no price sections found, take first 2000 chars as best-effort
    logger.warning("[DEGRADED] No price sections found — using first 2000 chars")
    return full_context[:2000]


def build_degraded_prompt(ticker: str) -> str:
    """Build an ultra-minimal prompt for last-resort analysis.

    Used when even the stripped context retry fails. The LLM gets
    no data at all — just the ticker — and is expected to return HOLD.
    """
    return (
        f"Analyze {ticker} with extreme caution. You have NO current data available. "
        f"Given this limitation, provide a conservative assessment. "
        f"Default to HOLD unless you have strong pre-training knowledge about "
        f"this ticker's fundamentals. Confidence should be very low (20-30%)."
    )


def build_degraded_result(ticker: str, reason: str) -> dict:
    """Build a fallback result dict when all retries are exhausted.

    This is the absolute last resort — a HOLD with 0 confidence and
    clear degraded=True flag so downstream consumers know.
    """
    return {
        "action": "HOLD",
        "confidence": 0,
        "rationale": f"[DEGRADED] Analysis unavailable: {reason}. "
        f"Defaulting to HOLD for safety.",
        "degraded": True,
        "degraded_reason": reason,
        "ticker": ticker,
    }
