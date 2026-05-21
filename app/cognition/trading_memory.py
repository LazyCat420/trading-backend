"""
Trading Memory -- Hermes-style bounded persistent memory for cross-cycle learning.

Ported from NousResearch/hermes-agent tools/memory_tool.py (MIT license).
Adapted: two trading-specific stores, Windows-safe atomic writes, higher char limits.

FILE OWNERSHIP (Does not use SQL):
- data/memory/MARKET_MEMORY.md
- data/memory/PORTFOLIO.md

Usage:
    from app.cognition.trading_memory import trading_memory
    trading_memory.load_from_disk()
    snapshot = trading_memory.get_frozen_snapshot()
"""

import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
MEMORY_DIR = _PROJECT_ROOT / "data" / "memory"
ENTRY_DELIMITER = "\u00a7"

_STORES = {
    "market": {"filename": "MARKET_MEMORY.md", "char_limit": 3000},
    "portfolio": {"filename": "PORTFOLIO.md", "char_limit": 1500},
}


class TradingMemory:
    """Bounded curated memory with file persistence.

    Frozen snapshot pattern: memory is read once at cycle start and
    rendered into a prompt block that never changes mid-cycle.
    This preserves vLLM prefix caching across all tickers.
    """

    def __init__(self, market_char_limit: int = 3000, portfolio_char_limit: int = 1500):
        self._entries: dict[str, list[str]] = {"market": [], "portfolio": []}
        self._limits: dict[str, int] = {
            "market": market_char_limit,
            "portfolio": portfolio_char_limit,
        }
        self._snapshot: dict[str, str] = {"market": "", "portfolio": ""}
        self._loaded = False

    def _store_path(self, target: str) -> Path:
        return MEMORY_DIR / str(_STORES[target]["filename"])

    def _read_entries(self, target: str) -> list[str]:
        path = self._store_path(target)
        if not path.exists():
            return []
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return []
        entries = [e.strip() for e in raw.split(ENTRY_DELIMITER)]
        return [e for e in entries if e]

    def _write_entries(self, target: str) -> None:
        path = self._store_path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = f" {ENTRY_DELIMITER}\n".join(self._entries[target])
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp", prefix=".mem_")
        try:
            os.write(fd, content.encode("utf-8"))
            os.close(fd)
            os.replace(tmp, str(path))
        except Exception as e:
            try:
                os.close(fd)
            except Exception:
                pass
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            logger.error("[MEMORY] Failed to persist %s: %s", target, e)

    def load_from_disk(self) -> None:
        """Load entries from disk and capture frozen snapshot.

        Call ONCE at cycle start. The snapshot returned by
        get_frozen_snapshot() will not change even if add/replace/remove
        are called mid-cycle.
        """
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        for target in self._entries:
            raw = self._read_entries(target)
            seen: set[str] = set()
            deduped: list[str] = []
            for entry in raw:
                if entry not in seen:
                    seen.add(entry)
                    deduped.append(entry)
            self._entries[target] = deduped

        self._snapshot = {t: self._render_block(t) for t in self._entries}
        self._loaded = True
        logger.info(
            "[MEMORY] Loaded: market=%d entries (%d/%d chars), "
            "portfolio=%d entries (%d/%d chars)",
            len(self._entries["market"]),
            self._char_count("market"),
            self._limits["market"],
            len(self._entries["portfolio"]),
            self._char_count("portfolio"),
            self._limits["portfolio"],
        )

    def get_frozen_snapshot(self) -> str:
        """Return the frozen prompt block captured at load time."""
        if not self._loaded:
            return ""
        parts = [
            self._snapshot[t] for t in ("market", "portfolio") if self._snapshot[t]
        ]
        return "\n".join(parts)

    def _char_count(self, target: str) -> int:
        if not self._entries[target]:
            return 0
        return len(f" {ENTRY_DELIMITER}\n".join(self._entries[target]))

    def add(self, target: str, content: str) -> dict:
        """Add an entry if within char limit. Persist immediately."""
        if target not in self._entries:
            return {"success": False, "message": f"Unknown store: {target}"}
        content = content.strip()
        if not content:
            return {"success": False, "message": "Empty content"}
        if content in self._entries[target]:
            return {"success": False, "message": "Duplicate entry"}

        new_size = self._char_count(target)
        if new_size > 0:
            new_size += len(f" {ENTRY_DELIMITER}\n") + len(content)
        else:
            new_size = len(content)

        limit = self._limits[target]
        if new_size > limit:
            return {
                "success": False,
                "message": f"Memory full: {new_size}/{limit} chars (entry is {len(content)} chars)",
                "usage_pct": round(self._char_count(target) / limit * 100, 1),
            }

        self._entries[target].append(content)
        self._write_entries(target)
        usage = round(self._char_count(target) / limit * 100, 1)
        logger.info("[MEMORY] Added to %s (%s%% full): %s", target, usage, content[:80])
        return {"success": True, "message": "Added", "usage_pct": usage}

    def replace(self, target: str, old_text: str, new_content: str) -> dict:
        """Find entry by substring, replace with new content."""
        if target not in self._entries:
            return {"success": False, "message": f"Unknown store: {target}"}
        old_text, new_content = old_text.strip(), new_content.strip()
        matches = [i for i, e in enumerate(self._entries[target]) if old_text in e]
        if not matches:
            return {"success": False, "message": "No entry matching substring"}
        if len(matches) > 1:
            return {
                "success": False,
                "message": f"Ambiguous: {len(matches)} entries match",
            }
        idx = matches[0]
        old_entry = self._entries[target][idx]
        self._entries[target][idx] = new_content
        self._write_entries(target)
        logger.info(
            "[MEMORY] Replaced in %s: '%s' -> '%s'",
            target,
            old_entry[:60],
            new_content[:60],
        )
        return {"success": True, "message": "Replaced"}

    def remove(self, target: str, old_text: str) -> dict:
        """Remove entry identified by substring."""
        if target not in self._entries:
            return {"success": False, "message": f"Unknown store: {target}"}
        old_text = old_text.strip()
        matches = [i for i, e in enumerate(self._entries[target]) if old_text in e]
        if not matches:
            return {"success": False, "message": "No entry matching substring"}
        if len(matches) > 1:
            return {
                "success": False,
                "message": f"Ambiguous: {len(matches)} entries match",
            }
        removed = self._entries[target].pop(matches[0])
        self._write_entries(target)
        logger.info("[MEMORY] Removed from %s: '%s'", target, removed[:60])
        return {"success": True, "message": "Removed", "removed": removed}

    def get_entries(self, target: str) -> list[str]:
        return list(self._entries.get(target, []))

    def _render_block(self, target: str) -> str:
        entries = self._entries[target]
        if not entries:
            return ""
        limit = self._limits[target]
        used = self._char_count(target)
        pct = round(used / limit * 100) if limit else 0
        title = target.upper().replace("_", " ")
        lines = [
            "=" * 40,
            f"{title} MEMORY [{pct}% -- {used:,}/{limit:,} chars]",
            "=" * 40,
        ]
        for entry in entries:
            lines.append(f"{entry} {ENTRY_DELIMITER}")
        lines.append("=" * 40)
        return "\n".join(lines)

    async def consolidate(self, target: str = "market") -> dict:
        """Merge related entries when memory is near capacity (>80%)."""
        limit = self._limits[target]
        used = self._char_count(target)
        if used < limit * 0.8:
            return {"skipped": True, "reason": "Under 80% capacity"}
        entries = self._entries[target]
        if len(entries) < 3:
            return {"skipped": True, "reason": "Too few entries to merge"}
        try:
            from app.services.vllm_client import llm, Priority

            joined = f" {ENTRY_DELIMITER}\n".join(entries)
            prompt = (
                f"These are {len(entries)} trading memory entries. "
                f"Consolidate related entries into fewer, denser entries. "
                f"Keep total under {limit} chars. Each entry max 120 chars. "
                f"Output consolidated entries separated by {ENTRY_DELIMITER}.\n\n"
                f"Current entries:\n{joined}"
            )
            response, tokens, _ = await llm.chat(
                system=(
                    "You consolidate trading notes. Merge related entries, "
                    "preserve specific numbers/dates. Output entries separated "
                    f"by {ENTRY_DELIMITER}. Max 120 chars per entry."
                ),
                user=prompt,
                temperature=0.0,
                max_tokens=500,
                agent_name="memory_consolidator",
                priority=Priority.LOW,
            )
            new_entries = [
                e.strip()
                for e in response.split(ENTRY_DELIMITER)
                if e.strip() and len(e.strip()) > 5
            ]
            if not new_entries:
                return {"success": False, "message": "Consolidation returned empty"}
            old_count = len(entries)
            self._entries[target] = new_entries
            self._write_entries(target)
            new_used = self._char_count(target)
            logger.info(
                "[MEMORY] Consolidated %s: %d -> %d entries, %d -> %d chars",
                target,
                old_count,
                len(new_entries),
                used,
                new_used,
            )
            return {
                "success": True,
                "old_entries": old_count,
                "new_entries": len(new_entries),
                "old_chars": used,
                "new_chars": new_used,
                "tokens_used": tokens,
            }
        except Exception as e:
            logger.warning("[MEMORY] Consolidation failed: %s", e)
            return {"success": False, "message": str(e)}


# Singleton -- import this everywhere
trading_memory = TradingMemory()
