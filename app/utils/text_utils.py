"""
Shared text utilities -- used across agents, pipeline, and services.

Consolidates duplicated text processing functions that were previously
copy-pasted across 4+ files. Each function existed in 2-4 places with
identical logic.

Usage:
    from app.utils.text_utils import (
        strip_think_tags,
        parse_json_response,
        sanitize_ascii,
        truncate,
        fmt_usd,
    )
"""

import hashlib
import json
import re
import logging

logger = logging.getLogger(__name__)


def strip_think_tags(text: str, return_think_content: bool = False):
    """Remove <think>...</think> blocks from LLM responses.

    Qwen3 inserts <think> blocks for chain-of-thought reasoning.
    These must be stripped before parsing the actual response content.
    If return_think_content is True, returns (cleaned_text, think_block_content)
    """
    think_content = ""
    # Extract think content if requested
    if return_think_content:
        match = re.search(r"<think>(.*?)(?:</think>|$)", text, flags=re.DOTALL)
        if match:
            think_content = match.group(1).strip()

    if "</think>" in text:
        cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
    else:
        # If unclosed, just remove the <think> tag itself so we don't delete the JSON!
        cleaned = text.replace("<think>", "").strip()

    if return_think_content:
        return cleaned, think_content
    return cleaned


def hash_prompt(prompt: str) -> str:
    """SHA256 hash of a system prompt for dedup/tracking."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


def parse_json_response(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown fences and nesting.

    Tries (in order):
        1. Markdown JSON code block (```json ... ```)
        2. Balanced brace-counting for nested JSON
        3. Raw text as JSON

    Previously duplicated in:
        - base_agent.py
        - debate_engine.py

    Args:
        text: Raw LLM response text (may contain <think> blocks, markdown, etc.)

    Returns:
        Parsed dict, or {} if no valid JSON found.
    """
    cleaned = strip_think_tags(text)

    # Strip __THINK__ streaming markers that may have leaked into pipeline responses.
    # These come from vllm_client.py's streaming mode and should never appear in
    # non-streaming chat() responses, but if they do, they kill the JSON parser.
    if "__THINK__" in cleaned:
        import logging

        logging.getLogger(__name__).warning(
            "[TEXT_UTILS] __THINK__ marker found in response — stripping before JSON parse. "
            "This indicates a streaming marker leaked into the pipeline. "
            "Preview: %s",
            cleaned[:200],
        )
        # Remove lines starting with __THINK__ (they're status markers, not JSON)
        lines = cleaned.split("\n")
        cleaned = "\n".join(l for l in lines if not l.strip().startswith("__THINK__"))
        cleaned = cleaned.strip()

    if not cleaned:
        raise ValueError(
            "LLM response is empty after stripping <think> tags (model failed to output JSON)."
        )

    # Try markdown JSON block first (greedy to capture full JSON)
    json_block = re.search(r"```(?:json)?\s*(\{.+\})\s*```", cleaned, re.DOTALL)
    if json_block:
        try:
            return json.loads(json_block.group(1))
        except json.JSONDecodeError:
            pass

    # Find balanced JSON objects using brace counting
    for start_idx in range(len(cleaned)):
        if cleaned[start_idx] != "{":
            continue
        depth = 0
        for end_idx in range(start_idx, len(cleaned)):
            if cleaned[end_idx] == "{":
                depth += 1
            elif cleaned[end_idx] == "}":
                depth -= 1
            if depth == 0:
                candidate = cleaned[start_idx : end_idx + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    break  # This opening brace didn't work, try next

    # Last resort: try the entire cleaned text
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        pass

    return {}


def sanitize_ascii(text: str) -> str:
    """Encode text as ASCII, replacing non-ASCII chars with '?'.

    Used for safe logging/printing on Windows (cp1252) and for
    sanitizing context before passing to RLM's LocalREPL which
    writes to temp files using system encoding.

    Previously duplicated as inline expressions in:
        - context_builder.py (_sanitize_text)
        - base_agent.py (inline .encode/.decode)
        - decision_engine.py (inline .encode/.decode)
        - debate_engine.py (inline .encode/.decode)
        - rlm_wrapper.py (inline .encode/.decode)
    """
    if not text:
        return ""
    # Strip invisible Unicode chars (zero-width, BOM, soft hyphens)
    text = re.sub(r"[\u200b\u200c\u200d\u200e\u200f\ufeff\u00ad\u2028\u2029]", "", text)
    return text.encode("ascii", errors="replace").decode("ascii")


def extract_reasoning_text(raw_response: str) -> str:
    """Extract meaningful reasoning/rationale text from a raw LLM response.

    Strips code blocks (```repl...```), tool call syntax, and JSON scaffolding
    to isolate the natural-language reasoning that should overlap with context.
    Also extracts the 'rationale' field from FINAL(...) JSON if present.

    Used for ROUGE-L grounding evaluation so we compare only the bot's
    reasoning against the source context — not code syntax or JSON keys.
    """
    if not raw_response:
        return ""

    text = strip_think_tags(raw_response)

    # 1. Extract rationale from FINAL({...}) if present
    rationale = ""
    final_match = re.search(r"FINAL\s*\(\s*(\{.*?\})\s*\)", text, re.DOTALL)
    if final_match:
        try:
            decision = json.loads(final_match.group(1))
            rationale = decision.get("rationale", "")
        except (json.JSONDecodeError, AttributeError):
            pass

    # 2. Strip code blocks (```repl ... ```, ```python ... ```, etc.)
    stripped = re.sub(r"```[\w]*\s*.*?```", " ", text, flags=re.DOTALL)

    # 3. Strip FINAL(...) call itself (already extracted rationale above)
    stripped = re.sub(r"FINAL\s*\(.*?\)", " ", stripped, flags=re.DOTALL)

    # 4. Strip tool output noise (lines that look like dict/list literals)
    stripped = re.sub(r"^\s*[\{\[].*?[\}\]]\s*$", " ", stripped, flags=re.MULTILINE)

    # 5. Strip REPL/tool function call patterns like get_technicals("AAPL")
    # Only strip patterns that look like tool calls (lowercase_with_underscores)
    # Preserves legitimate parentheticals like "earnings (Q4)" or "growth (YoY)"
    stripped = re.sub(r"\b[a-z_]+\([^)]*\)", " ", stripped)

    # 6. Collapse whitespace
    stripped = re.sub(r"\s+", " ", stripped).strip()

    # Combine natural-language reasoning with rationale
    parts = [p for p in [stripped, rationale] if p]
    return " ".join(parts)


def normalize_for_rouge(text: str) -> str:
    """Normalize text for ROUGE comparison.

    Strips markdown headers, table formatting, special chars, and
    collapses whitespace for fair token-level overlap measurement.
    """
    if not text:
        return ""
    # Strip markdown headers (## Header)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    # Strip markdown bold/italic
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    # Strip table separators (|---|---|)
    text = re.sub(r"\|[-:]+\|", " ", text)
    # Strip pipe chars from tables
    text = text.replace("|", " ")
    # Strip bullet markers
    text = re.sub(r"^\s*[-*•]\s*", "", text, flags=re.MULTILINE)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_citation_overlap(reasoning: str, context: str) -> float:
    """Compute how many numeric data points from the reasoning appear in context.

    Extracts numbers (decimals, percentages, dollar amounts) from the bot's
    reasoning and checks what fraction also appear in the source context.
    This directly measures whether the bot is citing real data.

    Returns a float 0.0–1.0 (fraction of cited numbers found in context).
    """
    if not reasoning or not context:
        return 0.0

    # Extract numeric tokens from reasoning: 37.8, 22.1%, $5.2B, 15%, etc.
    # Match patterns: digits with optional decimal, optional % or $ prefix,
    # optional magnitude suffix (B/M/K/T for billions/millions/etc.)
    number_pattern = re.compile(
        r"(?<!\w)(\$?\d+(?:\.\d+)?[BMKT]?%?)(?!\w)", re.IGNORECASE
    )
    cited_numbers = set(number_pattern.findall(reasoning))

    if not cited_numbers:
        return 0.0

    # Check how many appear in context (exact match)
    found = sum(1 for n in cited_numbers if n in context)
    return round(found / len(cited_numbers), 3)


def truncate(text: str, max_len: int = 500) -> str:
    """Truncate text to max_len, appending '...' if truncated.

    Previously duplicated in:
        - context_builder.py (_truncate)
        - debate_engine.py (_truncate_context with different split logic)

    Args:
        text: Text to truncate
        max_len: Maximum character length

    Returns:
        Original text if within limit, otherwise truncated with '...'
    """
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def fmt_usd(val) -> str:
    """Format USD value as human-readable: $6.2B, $551M, $42K.

    Previously duplicated in:
        - context_builder.py (_fmt_usd)
        - context_assembler.py (_fmt_usd)

    Args:
        val: Numeric value (or None)

    Returns:
        Formatted string like "$6.2B", "$551M", "$42K", or "N/A"
    """
    if val is None:
        return "N/A"
    v = float(val)
    if abs(v) >= 1e12:
        return f"${v / 1e12:.1f}T"
    if abs(v) >= 1e9:
        return f"${v / 1e9:.1f}B"
    if abs(v) >= 1e6:
        return f"${v / 1e6:.1f}M"
    if abs(v) >= 1e3:
        return f"${v / 1e3:.0f}K"
    return f"${v:.0f}"


# ── Scrape artifact detection ──────────────────────────────────────
# Previously duplicated in:
#   - context_builder.py (_SCRAPE_ARTIFACT_PATTERNS + _is_scrape_artifact)
#   - context_builder.py (identical copy)
#   - cognition/evidence/normalizer.py (identical copy)

SCRAPE_ARTIFACT_PATTERNS = [
    "access is temporarily restricted",
    "verification required",
    "enable javascript",
    "please complete the captcha",
    "checking your browser",
    "just a moment",
    "ray id:",
    "cloudflare",
    "are you a robot",
    "unusual traffic",
]


def is_scrape_artifact(summary: str) -> bool:
    """Return True if the summary looks like a scrape artifact (captcha, block page).

    Previously duplicated in:
        - context_builder.py (_is_scrape_artifact)
        - context_builder.py (_is_scrape_artifact)
        - cognition/evidence/normalizer.py (is_scrape_artifact)
    """
    if not summary:
        return False
    lower = summary.lower()
    for pattern in SCRAPE_ARTIFACT_PATTERNS:
        if pattern in lower:
            return True
    # Very short summaries with no real content
    if len(summary.strip()) < 15:
        return True
    return False


# ── DB section formatter ───────────────────────────────────────────
# Previously duplicated in:
#   - context_builder.py (_section)
#   - context_builder.py (_section)


def format_db_section(
    title: str, rows: list, columns: list[str], max_rows: int = 20
) -> str:
    """Format DB rows into a readable text section for LLM context.

    Previously duplicated as _section() in:
        - context_builder.py
        - context_builder.py
    """
    if not rows:
        return f"\n## {title}\nNo data available.\n"
    lines = [f"\n## {title}"]
    lines.append(f"({len(rows)} records, showing up to {max_rows})")
    for row in rows[:max_rows]:
        parts = []
        for i, col in enumerate(columns):
            val = row[i] if i < len(row) else ""
            if val is not None and val != "":
                # Round floats to 2dp to remove precision noise
                if isinstance(val, float):
                    val = round(val, 2)
                parts.append(f"{col}: {val}")
        lines.append("  " + " | ".join(parts))
    return "\n".join(lines) + "\n"

def parse_trading_decision(response: str) -> dict:
    """Extract a trading decision JSON from the RLM response string.

    Handles FINAL() wrapper and falls back to shared parse_json_response.
    Returns a dict with at minimum 'action' key, plus optional '_parse_meta'
    containing quality signals about the parse.
    """
    cleaned = strip_think_tags(response)

    parse_meta = {"source": "none", "warnings": []}

    # Strip markdown code fences wrapping FINAL (common LLM mistake)
    cleaned = re.sub(r"```(?:python|repl)?\s*\n?(FINAL\s*\()", r"\1", cleaned)
    cleaned = re.sub(r"(FINAL\s*\(\s*\{[^}]*\}\s*\))\s*\n?```", r"\1", cleaned)

    # Find ALL FINAL() matches
    final_matches = list(
        re.finditer(r"FINAL\s*\(\s*(\{.*?\})\s*\)", cleaned, re.DOTALL)
    )

    if final_matches:
        if len(final_matches) > 1:
            parse_meta["warnings"].append(f"multiple_finals:{len(final_matches)}")
            logger.warning(
                f"Multiple FINAL() calls found ({len(final_matches)}), using last one."
            )

        # Use the last FINAL — it's typically the refined answer
        match = final_matches[-1]
        try:
            decision = json.loads(match.group(1))
            parse_meta["source"] = "FINAL"

            # Check for trailing content after FINAL
            trailing = cleaned[match.end() :].strip()
            if trailing and len(trailing) > 50:
                parse_meta["warnings"].append("trailing_content")

            decision["_parse_meta"] = parse_meta
            return decision
        except json.JSONDecodeError:
            parse_meta["warnings"].append("json_decode_error")

    # Fall back to shared JSON parser
    result = parse_json_response(cleaned)
    if result and "action" in result:
        parse_meta["source"] = "fallback_json"
        result["_parse_meta"] = parse_meta
        return result

    return {}

