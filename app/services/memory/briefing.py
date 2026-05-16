"""
Memory Briefing Service — Developer 4

Responsible for extracting selected canonical memories, constitution rules,
and sector skills, and compressing them into a concise `Memory Brief`
via the vLLM queue.
"""

import logging
from typing import Any

from app.services.vllm_client import llm, Priority
from app.utils.text_utils import parse_json_response

logger = logging.getLogger(__name__)


async def generate_memory_brief(
    ticker: str,
    constitution: str,
    canonical_memories: list[dict[str, Any]],
    context_summary: str,
    skills: str,
) -> dict[str, Any]:
    """
    Compress constitution, selected canonical memories, and skills into a compact
    per-ticker memory brief block.

    Args:
        ticker: The stock ticker (e.g., "AAPL").
        constitution: The user's core portfolio rules.
        canonical_memories: Curated lessons/patterns returned by the retriever.
        context_summary: Extract of current context to provide situation awareness.
        skills: Sector-specific trading instructions.

    Returns:
        dict: The Memory Brief Result contract
            {
              "brief_text": str,
              "source_memory_ids": list[str],
              "char_count": int,
            }
    """
    source_ids = []
    for m in canonical_memories:
        mid = m.get("memory_id") or m.get("id")
        if mid:
            source_ids.append(mid)

    # 1. Handle empty state gracefully
    if not canonical_memories and not constitution and not skills:
        fallback = f"## MEMORY BRIEF\nNo specific memories or rules for {ticker}."
        return {
            "brief_text": fallback,
            "source_memory_ids": source_ids,
            "char_count": len(fallback),
        }

    # 2. Prepare the input text
    memories_text = "\n".join(
        f"- [{m.get('type', 'general')}] {m.get('summary', '')}"
        for m in canonical_memories
    )

    prompt = f"""You are compressing trading memories for the ticker {ticker}.
Your output will be injected directly into the trading analyst's system prompt.
Your goal is to synthesize the inputs into ONE compact section titled '## MEMORY BRIEF'.

## Inputs:
Constitution (Core Rules):
{constitution}

Skills (Sector-specific):
{skills}

Canonical Memories (Curated Lessons):
{memories_text}

Current Context Summary:
{context_summary[:1500]}

## Instructions:
1. Synthesize these inputs into a dense, actionable bulleted list.
2. IMPORTANT: You must include core Constitution rules so the agent does not violate them.
3. Keep it strictly under 1000 characters.
4. Output your answer purely as a JSON object with a single string key "brief_text". DO NOT use markdown code blocks like ```json.

Example output:
{{"brief_text": "## MEMORY BRIEF\\n- Rule: Never short before earnings\\n- TSLA Pattern: High volatility near resistance... "}}
"""
    try:
        response, tokens_used, _ = await llm.chat(
            system="You are a strict JSON data summarizer.",
            user=prompt,
            temperature=0.0,
            max_tokens=600,
            agent_name="memory_briefer",
            priority=Priority.HIGH,
        )

        result = parse_json_response(response)
        brief_text = result.get("brief_text", "").strip()

        # Fallback if the LLM output is malformed
        if not brief_text:
            logger.warning(
                "[MEMORY BRIEF] LLM returned empty brief text for %s", ticker
            )
            brief_text = _build_fallback_brief(constitution, skills, memories_text)

        char_count = len(brief_text)
        logger.debug(
            "[MEMORY BRIEF] Generated %d char brief for %s using %d tokens",
            char_count,
            ticker,
            tokens_used,
        )

        return {
            "brief_text": brief_text,
            "source_memory_ids": source_ids,
            "char_count": char_count,
        }

    except Exception as e:
        logger.error("[MEMORY BRIEF] Failed to generate brief for %s: %s", ticker, e)
        fallback = _build_fallback_brief(constitution, skills, memories_text)
        return {
            "brief_text": fallback,
            "source_memory_ids": source_ids,
            "char_count": len(fallback),
        }


def _build_fallback_brief(constitution: str, skills: str, memories: str) -> str:
    """Provides a safe concatenated backup string if LLM generation crashes."""
    parts = ["## MEMORY BRIEF\n"]
    if constitution:
        parts.append(f"Constitution:\n{constitution}\n")
    if skills:
        parts.append(f"Skills:\n{skills}\n")
    if memories:
        parts.append(f"Canonical Memories:\n{memories}\n")

    return "\n".join(parts)[:1500]
