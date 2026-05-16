import json
import logging
from app.db.connection import get_db

logger = logging.getLogger(__name__)


def get_active_constitution_rules() -> list[dict]:
    """Fetch all active trading constitution rules."""
    try:
        with get_db() as db:
            rows = db.execute(
                "SELECT id, rule_category, rule_text, rule_params "
                "FROM trading_constitution WHERE is_active = TRUE"
            ).fetchall()
            rules = []
            for row in rows:
                params = row[3]
                if isinstance(params, str):
                    try:
                        params = json.loads(params)
                    except Exception:
                        params = {}

                rules.append(
                    {"id": row[0], "category": row[1], "text": row[2], "params": params}
                )
            return rules
    except Exception as e:
        logger.error("[CONSTITUTION] Failed to fetch rules: %s", e)
        return []


def format_constitution_for_prompt() -> str:
    """Format the active constitution rules into a string for the LLM prompt."""
    rules = get_active_constitution_rules()
    if not rules:
        return "- Default to holding unless overwhelming evidence suggests otherwise."

    formatted_rules = []

    # Group by category
    categories = {}
    for r in rules:
        cat = r["category"].upper()
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r["text"])

    for cat, texts in categories.items():
        formatted_rules.append(f"[{cat}]")
        for text in texts:
            formatted_rules.append(f"  * {text}")

    return "\n".join(formatted_rules)
