"""
Context Compressor — Summarizes older tool interactions to prevent TokenLimitError.

Also provides ``summarize_tool_result()`` for inline compression of oversized
tool outputs *before* they enter the message history.
"""

import logging
from typing import List, Dict

from app.services.vllm_client import llm, Priority
from app.config.context_budget import get_context_budget, estimate_tokens

logger = logging.getLogger(__name__)


# Very basic tokenizer heuristic for fast estimation (vLLM will count real tokens later)
def _estimate_tokens(text: str) -> int:
    return estimate_tokens(text)


def summarize_tool_result(content: str, tool_name: str = "unknown", budget_tokens: int | None = None) -> str:
    """Deterministic head+tail summarization for oversized tool results.

    If the tool result exceeds the budget, keeps the first 70% and last 15%
    of the budget, with a truncation marker in the middle. No LLM call needed.

    Args:
        content: Raw tool result string.
        tool_name: Name of the tool (for the truncation marker).
        budget_tokens: Max tokens for this result. If None, uses context budget default.

    Returns:
        Original content if within budget, otherwise truncated version.
    """
    if budget_tokens is None:
        budget = get_context_budget()
        budget_tokens = budget.tool_result_budget

    current_tokens = _estimate_tokens(content)
    if current_tokens <= budget_tokens:
        return content

    # Convert token budget to chars (4 chars/token heuristic)
    budget_chars = budget_tokens * 4
    head_chars = int(budget_chars * 0.70)
    tail_chars = int(budget_chars * 0.15)

    head = content[:head_chars]
    tail = content[-tail_chars:] if tail_chars > 0 else ""
    truncated_chars = len(content) - head_chars - tail_chars

    marker = (
        f"\n\n... [{tool_name}: {truncated_chars:,} chars truncated. "
        f"Use get_cycle_context tool for full data.] ...\n\n"
    )

    result = head + marker + tail
    logger.info(
        "[Compressor] Tool result '%s' truncated: %d -> %d tokens",
        tool_name,
        current_tokens,
        _estimate_tokens(result),
    )
    return result


async def compress_history(
    messages: List[Dict], threshold: int | None = None, keep_recent: int = 3
) -> List[Dict]:
    """
    Compresses chat history if the total tokens exceed the threshold.
    Preserves the system prompt, the most recent `keep_recent` turns, and replaces
    the middle messages with a dense LLM-generated summary.

    Args:
        messages: The full message list.
        threshold: Token threshold to trigger compression.
                   If None, uses the model-aware compressor_threshold from context_budget.
        keep_recent: Number of recent user/assistant pairs to preserve.
    """
    if threshold is None:
        budget = get_context_budget()
        threshold = budget.compressor_threshold

    # Count total estimated tokens
    total_tokens = sum(_estimate_tokens(m.get("content", "")) for m in messages)

    if total_tokens < threshold or len(messages) <= keep_recent * 2 + 1:
        return messages

    logger.warning(
        "[Compressor] Context size %d exceeds threshold %d. Compressing...",
        total_tokens,
        threshold,
    )

    # Extract segments
    system_prompt = messages[0] if messages[0]["role"] == "system" else None

    # We want to keep the last `keep_recent` user/assistant pairs (which means ~ keep_recent * 2 messages)
    recent_tail = messages[-(keep_recent * 2) :]

    # The middle segment to compress
    start_idx = 1 if system_prompt else 0
    middle_segment = messages[start_idx : -(keep_recent * 2)]

    if not middle_segment:
        return messages

    # Build a text representation of the middle segment for the LLM to summarize
    middle_text = ""
    for idx, msg in enumerate(middle_segment):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        middle_text += f"Turn {idx} [{role}]:\n"
        if content:
            middle_text += f"{content[:1000]}... (truncated)\n"
        if tool_calls:
            middle_text += f"Tool Calls: {[tc.get('function', {}).get('name') for tc in tool_calls]}\n"
        middle_text += "-" * 40 + "\n"

    # Ask LLM to summarize
    summary_prompt = (
        "The following is a transcript of older interactions in an ongoing agent loop. "
        "Summarize the key findings, data points, and context gathered so far. "
        "Keep it highly dense and factual. Do not output conversational filler. "
        f"\n\n{middle_text}"
    )

    try:
        summary_response, _, _ = await llm.chat(
            system="You are a context compression engine. Summarize facts densely.",
            user=summary_prompt,
            temperature=0.1,
            max_tokens=1000,
            priority=Priority.NORMAL,
            agent_name="context_compressor",
        )

        compressed_msg = {
            "role": "assistant",
            "content": f"[COMPRESSED HISTORY SUMMARY]\n{summary_response}",
        }

        new_messages = []
        if system_prompt:
            new_messages.append(system_prompt)
        new_messages.append(compressed_msg)
        new_messages.extend(recent_tail)

        new_tokens = sum(_estimate_tokens(m.get("content", "")) for m in new_messages)
        logger.info(
            "[Compressor] Compression successful. Reduced from ~%d to ~%d tokens (threshold=%d).",
            total_tokens,
            new_tokens,
            threshold,
        )
        return new_messages

    except Exception as e:
        logger.error(f"[Compressor] Failed to compress history: {e}")
        # On failure, return original to avoid breaking the loop (it might crash on token limit later, but better than failing here)
        return messages


# ── Capsule Generation (Layer 1 compression) ────────────────────────

import re
import json
import uuid
from app.agents.capsule import AgentCapsule


# Signal keywords for heuristic extraction from plain text
_SIGNAL_PATTERNS = {
    "BUY": re.compile(r"\b(BUY|BULLISH|LONG|STRONG\s*BUY)\b", re.IGNORECASE),
    "SELL": re.compile(r"\b(SELL|BEARISH|SHORT|STRONG\s*SELL)\b", re.IGNORECASE),
    "HOLD": re.compile(r"\b(HOLD|NEUTRAL|WAIT|NO\s*ACTION)\b", re.IGNORECASE),
}

_CONFIDENCE_PATTERN = re.compile(r"(?:confidence|conf)[:\s]*(\d{1,3})(?:\s*%)?", re.IGNORECASE)


def _extract_from_json(parsed: dict) -> tuple[str, str, float, list[str]]:
    """Extract signal, summary, confidence, and flags from a parsed JSON response."""
    # Signal extraction (try multiple field names)
    signal = "UNKNOWN"
    for key in ("signal", "action", "recommendation", "direction"):
        val = parsed.get(key, "")
        if val and isinstance(val, str):
            val_upper = val.upper().strip()
            if val_upper in ("BUY", "SELL", "HOLD", "BULLISH", "BEARISH", "NEUTRAL", "CONFLICTED"):
                signal = val_upper
                # Normalize aliases
                if signal == "BULLISH":
                    signal = "BUY"
                elif signal == "BEARISH":
                    signal = "SELL"
                elif signal == "NEUTRAL":
                    signal = "HOLD"
                break

    # Confidence extraction
    confidence = 0.0
    for key in ("confidence", "confidence_score", "conf"):
        val = parsed.get(key)
        if val is not None:
            try:
                conf_val = float(val)
                # Normalize: if > 1.0 assume it's 0-100 scale
                confidence = conf_val / 100.0 if conf_val > 1.0 else conf_val
                break
            except (ValueError, TypeError):
                pass

    # Summary extraction
    summary = ""
    for key in ("rationale", "summary", "reasoning", "explanation", "analysis"):
        val = parsed.get(key, "")
        if val and isinstance(val, str) and len(val) > 10:
            # Take first 300 chars as summary
            summary = val[:300].strip()
            if len(val) > 300:
                summary += "..."
            break

    if not summary:
        # Fallback: join all string values up to 300 chars
        parts = []
        for k, v in parsed.items():
            if isinstance(v, str) and len(v) > 5 and k not in ("signal", "action", "confidence"):
                parts.append(f"{k}: {v[:100]}")
        summary = "; ".join(parts)[:300] or "No summary extracted."

    # Flags extraction
    flags: list[str] = []
    for key in ("flags", "warnings", "risks", "concerns"):
        val = parsed.get(key)
        if isinstance(val, list):
            flags.extend(str(f) for f in val[:5])
        elif isinstance(val, str) and val:
            flags.append(val)

    return signal, summary, confidence, flags


def _extract_from_text(text: str) -> tuple[str, str, float, list[str]]:
    """Fallback heuristic extraction from plain text responses."""
    # Signal detection
    signal = "UNKNOWN"
    for sig, pattern in _SIGNAL_PATTERNS.items():
        if pattern.search(text):
            signal = sig
            break

    # Confidence detection
    confidence = 0.0
    conf_match = _CONFIDENCE_PATTERN.search(text)
    if conf_match:
        try:
            conf_val = int(conf_match.group(1))
            confidence = conf_val / 100.0 if conf_val > 1 else float(conf_val)
        except (ValueError, IndexError):
            pass

    # Summary: first 300 chars of meaningful text
    # Strip markdown headers and excessive whitespace
    clean = re.sub(r"#+\s*", "", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    summary = clean[:300]
    if len(clean) > 300:
        summary += "..."

    flags: list[str] = []
    # Detect common warning patterns
    if re.search(r"divergen", text, re.IGNORECASE):
        flags.append("divergence")
    if re.search(r"low.{0,5}volume", text, re.IGNORECASE):
        flags.append("low_volume")
    if re.search(r"missing.{0,10}data", text, re.IGNORECASE):
        flags.append("missing_data")
    if re.search(r"conflicting", text, re.IGNORECASE):
        flags.append("conflicting_signals")

    return signal, summary, confidence, flags


async def generate_capsule(
    agent_result: dict,
    agent_name: str,
    cycle_id: str,
    ticker: str,
) -> AgentCapsule:
    """Generate an AgentCapsule from a raw agent result dict.

    Uses heuristic extraction (JSON parsing + regex) — no LLM calls.
    Agents already output structured JSON via require_json_schema=True,
    so we can pull signal/confidence/rationale fields directly.

    Args:
        agent_result: The dict returned by run_agent() (has 'response', 'tokens_used', etc.)
        agent_name: Name of the agent that produced this result
        cycle_id: Current trading cycle ID
        ticker: Ticker being analyzed

    Returns:
        AgentCapsule with compressed Layer 1 summary and DB reference for Layer 2
    """
    if not isinstance(agent_result, dict):
        logger.warning("[Capsule] Agent result is not a dict for %s, returning empty", agent_name)
        return AgentCapsule(
            agent_name=agent_name, cycle_id=cycle_id, ticker=ticker,
            summary=f"Agent {agent_name} failed: {str(agent_result)[:100]}",
            signal="UNKNOWN", confidence=0.0, flags=["agent_failure"],
            raw_id="", tokens_estimated=25,
        )

    response_text = agent_result.get("response", "")
    if not response_text or len(response_text) < 5:
        return AgentCapsule(
            agent_name=agent_name, cycle_id=cycle_id, ticker=ticker,
            summary=f"Agent {agent_name} produced empty response.",
            signal="UNKNOWN", confidence=0.0, flags=["empty_response"],
            raw_id="", tokens_estimated=12,
        )

    # Try JSON extraction first (preferred — agents output structured JSON)
    from app.utils.text_utils import parse_json_response

    parsed = parse_json_response(response_text)
    if parsed and isinstance(parsed, dict) and len(parsed) > 0:
        signal, summary, confidence, flags = _extract_from_json(parsed)
    else:
        # Fallback to regex heuristic
        signal, summary, confidence, flags = _extract_from_text(response_text)

    # Generate raw_id and store Layer 2 in DB
    raw_id = str(uuid.uuid4())
    source_refs = [f"ref:capsule:{raw_id}"]

    capsule = AgentCapsule(
        agent_name=agent_name,
        cycle_id=cycle_id,
        ticker=ticker,
        summary=summary,
        signal=signal,
        confidence=confidence,
        flags=flags,
        source_refs=source_refs,
        raw_id=raw_id,
        tokens_estimated=0,  # Placeholder — computed below
    )

    # Compute accurate token estimate from the full rendered prompt output
    from app.agents.capsule import format_capsule_for_prompt
    rendered = format_capsule_for_prompt(capsule)
    # Rebuild with accurate token count (frozen dataclass, so re-create)
    capsule = AgentCapsule(
        agent_name=capsule.agent_name,
        cycle_id=capsule.cycle_id,
        ticker=capsule.ticker,
        summary=capsule.summary,
        signal=capsule.signal,
        confidence=capsule.confidence,
        flags=list(capsule.flags),
        source_refs=list(capsule.source_refs),
        raw_id=capsule.raw_id,
        tokens_estimated=_estimate_tokens(rendered),
    )

    return capsule


async def write_capsule_to_db(capsule: AgentCapsule, raw_response: str) -> None:
    """Write the capsule's Layer 2 data (full raw response) to the cycle_context table.

    This is a fire-and-forget write — failures are logged but don't block the pipeline.

    Args:
        capsule: The generated capsule (contains raw_id, summary, signal, etc.)
        raw_response: The full raw agent response text to store
    """
    if not capsule.raw_id:
        return

    try:
        from app.db.connection import get_db

        flags_json = json.dumps(capsule.flags) if capsule.flags else "[]"

        with get_db() as db:
            db.execute(
                """INSERT INTO cycle_context
                   (id, cycle_id, agent_name, ticker, raw_response, summary, signal, confidence, flags, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                   ON CONFLICT (id) DO NOTHING""",
                [
                    capsule.raw_id,
                    capsule.cycle_id,
                    capsule.agent_name,
                    capsule.ticker,
                    raw_response[:50000],  # Cap raw storage at 50k chars
                    capsule.summary,
                    capsule.signal,
                    capsule.confidence,
                    flags_json,
                ],
            )
        logger.debug(
            "[Capsule] Wrote Layer 2 for %s/%s (raw_id=%s, %d chars)",
            capsule.agent_name, capsule.ticker, capsule.raw_id[:8], len(raw_response),
        )
    except Exception as e:
        # Never let DB writes block the pipeline
        logger.error("[Capsule] Failed to write Layer 2 to cycle_context: %s", e)
