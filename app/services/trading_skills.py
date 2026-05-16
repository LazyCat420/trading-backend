"""
Trading Skills -- Sector-specific analysis instructions loaded from markdown files.

Simplified port of Hermes's skill_manager_tool.py. We just need:
1. A data/memory/skills/ directory with sector-specific SKILL.md files
2. A loader that reads the right skill based on ticker's sector
3. Injection into the system prompt alongside memory

Skills are manually created. Each SKILL.md has YAML frontmatter with a tickers list
and a markdown body with analysis instructions.

Usage:
    from app.services.trading_skills import load_skill_for_ticker
    skill = load_skill_for_ticker("NVDA")
    if skill:
        full_prompt = memory_block + skill + TRADING_SYSTEM_PROMPT
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
SKILLS_DIR = _PROJECT_ROOT / "data" / "memory" / "skills"

# Cache loaded skills to avoid repeated disk reads within a cycle
_skill_cache: dict[str, str | None] = {}
_ticker_to_skill: dict[str, str | None] = {}


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from a SKILL.md file.

    Returns (frontmatter_dict, body_text).
    """
    if not content.startswith("---"):
        return {}, content

    try:
        end_idx = content.index("---", 3)
    except ValueError:
        return {}, content

    frontmatter_raw = content[3:end_idx].strip()
    body = content[end_idx + 3 :].strip()

    # Simple YAML parsing without importing yaml (avoid dependency)
    fm: dict = {}
    for line in frontmatter_raw.split("\n"):
        line = line.strip()
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip()
        val = val.strip()

        # Handle list values: [XOM, CVX, COP]
        if val.startswith("[") and val.endswith("]"):
            items = [item.strip().strip("'\"") for item in val[1:-1].split(",")]
            fm[key] = items
        else:
            fm[key] = val.strip("'\"")

    return fm, body


def _load_all_skills() -> None:
    """Scan the skills directory and build ticker -> skill mapping."""
    global _skill_cache, _ticker_to_skill

    if not SKILLS_DIR.exists():
        logger.debug("[SKILLS] No skills directory at %s", SKILLS_DIR)
        return

    for skill_md in SKILLS_DIR.rglob("SKILL.md"):
        try:
            content = skill_md.read_text(encoding="utf-8")
            fm, body = _parse_frontmatter(content)

            skill_name = fm.get("name", skill_md.parent.name)
            tickers = fm.get("tickers", [])

            if not body:
                continue

            # Render as a prompt block
            skill_block = (
                f"{'=' * 40}\n"
                f"SECTOR SKILL: {skill_name}\n"
                f"{'=' * 40}\n"
                f"{body}\n"
                f"{'=' * 40}"
            )

            _skill_cache[skill_name] = skill_block

            # Map each ticker to this skill
            if isinstance(tickers, list):
                for t in tickers:
                    _ticker_to_skill[t.upper()] = skill_name
            else:
                _ticker_to_skill[str(tickers).upper()] = skill_name

        except Exception as e:
            logger.warning("[SKILLS] Failed to load %s: %s", skill_md, e)

    if _skill_cache:
        logger.info(
            "[SKILLS] Loaded %d skills covering %d tickers",
            len(_skill_cache),
            len(_ticker_to_skill),
        )


def load_skill_for_ticker(ticker: str) -> str | None:
    """Find and return the skill content for a ticker, or None.

    Lazy-loads all skills on first call, then uses cached mapping.
    """
    # Lazy load on first call
    if not _skill_cache and SKILLS_DIR.exists():
        _load_all_skills()

    ticker_upper = ticker.upper()
    skill_name = _ticker_to_skill.get(ticker_upper)

    if skill_name is None:
        # Try sector-based fallback from ticker_metadata
        try:
            from app.db.connection import get_db

            with get_db() as db:
                row = db.execute(
                    "SELECT sector FROM ticker_metadata WHERE ticker = %s",
                    [ticker_upper],
                ).fetchone()
            if row and row[0]:
                sector = row[0].lower().replace(" ", "-")
                # Check if we have a skill matching this sector name
                if sector in _skill_cache:
                    return _skill_cache[sector]
        except Exception:
            pass
        return None

    return _skill_cache.get(skill_name)


def reload_skills() -> None:
    """Force reload all skills from disk. Call if skills are modified."""
    global _skill_cache, _ticker_to_skill
    _skill_cache = {}
    _ticker_to_skill = {}
    _load_all_skills()
