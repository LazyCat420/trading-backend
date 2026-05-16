"""
Progressive Summarizer — Inline compression for debate turns and multi-turn context.

Provides deterministic (no LLM) methods for extracting the most information-dense
portion of a text block while maintaining readability. Designed for:

  1. Debate turn compression — extract claims + cited values only
  2. Opponent quote compression — reduce 4-turn exponential growth
  3. Tool research block compression — keep data, strip commentary

All methods are synchronous and require NO LLM calls (pure regex/heuristic).

Usage:
    from app.agents.progressive_summarizer import summarize_opponent_turn
    compact = summarize_opponent_turn(long_bull_output, side="bull", max_chars=2000)
"""

import re
import logging

logger = logging.getLogger(__name__)


def summarize_opponent_turn(
    text: str,
    side: str = "bull",
    max_chars: int = 2000,
) -> str:
    """Extract the most information-dense summary of a debate turn.

    Strategy:
      1. Try to extract JSON claims block (most structured)
      2. Fall back to extracting lines with [source:value] citations
      3. Fall back to head truncation with marker

    Args:
        text: Full debate turn output.
        side: "bull" or "bear" (for labeling).
        max_chars: Maximum output length.

    Returns:
        Compressed version of the debate turn.
    """
    if len(text) <= max_chars:
        return text

    result_parts = []

    # Strategy 1: Extract JSON claims array
    claims_match = re.search(
        r'"claims"\s*:\s*\[(.*?)\]',
        text,
        re.DOTALL,
    )
    if claims_match:
        claims_raw = claims_match.group(1)
        # Extract individual claim strings
        claim_strings = re.findall(r'"([^"]+)"', claims_raw)
        if claim_strings:
            result_parts.append(f"[{side.upper()} CLAIMS]")
            for i, c in enumerate(claim_strings, 1):
                result_parts.append(f"  {i}. {c}")

    # Strategy 2: Extract key_argument
    key_arg_match = re.search(r'"key_argument"\s*:\s*"([^"]+)"', text)
    if key_arg_match:
        result_parts.append(f"[KEY ARGUMENT] {key_arg_match.group(1)}")

    # Strategy 3: Extract confidence
    conf_match = re.search(r'"confidence"\s*:\s*(\d+)', text)
    if conf_match:
        result_parts.append(f"[CONFIDENCE] {conf_match.group(1)}/100")

    # Strategy 4: Extract lines with citations [source:value]
    if not result_parts:
        citation_pattern = re.compile(
            r'[^\n]*\[[a-zA-Z_]+:[^\]]+\][^\n]*', re.IGNORECASE
        )
        cited_lines = citation_pattern.findall(text)
        if cited_lines:
            result_parts.append(f"[{side.upper()} CITED EVIDENCE]")
            for line in cited_lines[:10]:
                result_parts.append(f"  • {line.strip()[:200]}")

    # Strategy 5: Extract lines with critical keywords
    if not result_parts:
        keyword_pattern = re.compile(
            r'[^\n]*(?:BUY|SELL|HOLD|UPGRADE|DOWNGRADE|TARGET|BANKRUPTCY|MERGER)[^\n]*', re.IGNORECASE
        )
        keyword_lines = keyword_pattern.findall(text)
        if keyword_lines:
            result_parts.append(f"[{side.upper()} KEYWORDS]")
            for line in keyword_lines[-3:]: # Get last 3 (often conclusions)
                result_parts.append(f"  • {line.strip()[:200]}")

    # If we extracted structured content, use that
    if result_parts:
        result = "\n".join(result_parts)
        if len(result) <= max_chars:
            return result
        return result[:max_chars] + f"\n... [{side} summary truncated]"

    # Fallback: head truncation
    return text[:max_chars] + f"\n... [{side} output truncated from {len(text):,} chars]"


def compress_tool_research_block(
    tool_history: list[str],
    max_total_chars: int = 4000,
) -> str:
    """Compress a list of tool execution records into a dense summary.

    Keeps the tool name and key data from each call, discarding verbose
    formatting and repetitive headers.

    Args:
        tool_history: List of strings like "### Tool Call: func(args)\\noutput"
        max_total_chars: Maximum total output length.

    Returns:
        Compressed multi-tool summary string.
    """
    if not tool_history:
        return "No tools used."

    compressed = []
    per_tool_budget = max_total_chars // max(len(tool_history), 1)

    for entry in tool_history:
        # Extract tool name from "### Tool Call: func_name(...)"
        name_match = re.match(r"###\s*Tool Call:\s*(\w+)", entry)
        tool_name = name_match.group(1) if name_match else "unknown"

        # Get the output (everything after the first newline)
        parts = entry.split("\n", 1)
        output = parts[1] if len(parts) > 1 else entry

        # Extract numbers, percentages, and key metrics
        numbers = re.findall(
            r'(?:[\w_]+\s*[=:]\s*)?[-+]?\$?[\d,]+\.?\d*%?', output[:per_tool_budget]
        )

        if len(output) <= per_tool_budget:
            compressed.append(f"[{tool_name}] {output}")
        else:
            # Keep head of output + extracted numbers
            head = output[:int(per_tool_budget * 0.8)]
            compressed.append(
                f"[{tool_name}] {head}\n... [truncated, {len(output):,} total chars]"
            )

    result = "\n".join(compressed)
    if len(result) > max_total_chars:
        result = result[:max_total_chars] + "\n... [research block truncated]"
    return result
