"""
Trading Constitution — Loader and prompt builder.

Reads active rules from the `trading_constitution` table and formats
them into a structured text block that gets injected into the thesis
agent's system prompt each cycle.

This is the key mechanism for self-improvement: rules are DB-stored
and versioned, not hardcoded. The benchmarking agent (Phase 3) can
propose amendments to these rules based on trading outcomes.
"""

import json
import logging

logger = logging.getLogger(__name__)

# Safety bounds — the ONLY remaining hardcoded guardrails.
# These prevent catastrophic parameter values regardless of what
# the benchmarking agent proposes.
SAFETY_BOUNDS = {
    "max_positions": (4, 20),
    "max_sector_pct": (15, 60),
    "rsi_threshold": (50, 90),
    "rsi_max": (40, 80),
    "pe_multiplier": (1.0, 3.0),
    "max_holding_days": (3, 60),
    "min_pct": (1, 15),
    "max_pct": (5, 30),
    "min_confidence": (40, 90),
}


def load_constitution() -> list[dict]:
    """Load all active Constitution rules from the DB.

    Returns a list of dicts with keys:
        id, rule_category, rule_text, rule_params, version
    """
    try:
        from app.db.connection import get_db

        with get_db() as db:
            rows = db.execute(
                "SELECT id, rule_category, rule_text, rule_params, "
                "version FROM trading_constitution "
                "WHERE is_active = TRUE "
                "ORDER BY rule_category, id"
            ).fetchall()

            rules = []
            for r in rows:
                params = r[3]
                if isinstance(params, str):
                    try:
                        params = json.loads(params)
                    except json.JSONDecodeError:
                        params = {}
                rules.append(
                    {
                        "id": r[0],
                        "rule_category": r[1],
                        "rule_text": r[2],
                        "rule_params": params or {},
                        "version": r[4],
                    }
                )
            return rules
    except Exception as e:
        logger.warning("[CONSTITUTION] Failed to load rules: %s", e)
        return []


def get_constitution_param(
    category: str,
    param_name: str,
    default: float | int = 0,
) -> float | int:
    """Get a specific parameter value from the Constitution.

    This is the replacement for hardcoded constants like
    MAX_CONCURRENT_POSITIONS. Instead of:
        MAX_CONCURRENT_POSITIONS = 8
    You now call:
        get_constitution_param("position_limits", "max_positions", 8)

    Falls back to the default if the rule doesn't exist or
    the parameter is missing.
    """
    rules = load_constitution()
    for rule in rules:
        if rule["rule_category"] == category:
            val = rule["rule_params"].get(param_name)
            if val is not None:
                return val
    return default


def format_constitution_for_prompt() -> str:
    """Build a text block of all active Constitution rules.

    This gets injected into the thesis agent's extra_context so
    the LLM knows what rules to follow. The rules are readable
    and the LLM can reason about them.
    """
    rules = load_constitution()
    if not rules:
        return ""

    categories: dict[str, list[str]] = {}
    for rule in rules:
        cat = rule["rule_category"]
        if cat not in categories:
            categories[cat] = []
        params_str = ""
        if rule["rule_params"]:
            params_str = (
                " ("
                + ", ".join(f"{k}={v}" for k, v in rule["rule_params"].items())
                + ")"
            )
        categories[cat].append(
            f"- [v{rule['version']}] {rule['rule_text']}{params_str}"
        )

    lines = ["# TRADING CONSTITUTION (Adaptive Rules)"]
    lines.append(
        "These are the bot's current operating rules. "
        "Follow them unless evidence strongly justifies "
        "deviation. Note any rule tensions in your rationale."
    )
    lines.append("")

    for cat, cat_rules in categories.items():
        lines.append(f"## {cat.upper().replace('_', ' ')}")
        lines.extend(cat_rules)
        lines.append("")

    return "\n".join(lines)


def validate_amendment(
    param_name: str, proposed_value: float | int
) -> tuple[bool, str]:
    """Check if a proposed amendment is within safety bounds.

    Returns (valid, reason).
    """
    bounds = SAFETY_BOUNDS.get(param_name)
    if not bounds:
        return True, "No safety bounds defined for this parameter."

    lo, hi = bounds
    if proposed_value < lo or proposed_value > hi:
        return False, (
            f"Value {proposed_value} outside safety bounds "
            f"[{lo}, {hi}] for {param_name}."
        )
    return True, "Within safety bounds."
