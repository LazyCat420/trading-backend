import logging
from collections import deque, defaultdict
from typing import TypedDict
from threading import Lock

from app.config import settings
from app.services.memory.semantic_memory import semantic_memory_store
from app.services.memory.episodic_memory import episodic_memory_store
from app.services.memory.procedural_memory import procedural_memory_store
from app.services.memory.prospective_memory import prospective_memory_store

logger = logging.getLogger(__name__)


class MemorySlot(TypedDict):
    source: str
    content: str


class WorkingMemoryManager:
    """
    Assembles the multi-tier Prism memory system to be injected into the LLM context.
    - Resolves data from all memory stores
    - Combines them with live cycle messages
    - Includes scratchpad functionality
    """

    def __init__(self):
        self.max_slots = getattr(settings, "WORKING_MEMORY_MAX_SLOTS", 18)
        self._live_events: dict[str, deque[MemorySlot]] = defaultdict(
            lambda: deque(maxlen=self.max_slots)
        )
        self._scratchpads: dict[str, list[str]] = defaultdict(list)
        self._lock = Lock()

    def clear(self):
        with self._lock:
            self._live_events.clear()
            self._scratchpads.clear()
            logger.debug("[MEMORY] Cleared working memory and scratchpads")

    def add_event(self, content: str, source: str = "system", ticker: str = "global"):
        """Add a formatted memory event to the rolling deque for a ticker."""
        with self._lock:
            self._live_events[ticker].append({"content": content, "source": source})

    def add_scratchpad_note(self, agent_name: str, note: str, ticker: str):
        """Append to the active scratchpad for this cycle run."""
        with self._lock:
            self._scratchpads[ticker].append(f"[{agent_name}] {note}")

    def get_context(self, ticker: str = "global") -> str:
        """Assemble the active working memory for prompt injection."""

        reminders = prospective_memory_store.retrieve_pending(ticker)
        semantics = semantic_memory_store.retrieve(ticker, limit=6)
        episodes = episodic_memory_store.retrieve(ticker, limit=4)
        procedures = procedural_memory_store.retrieve(ticker, limit=3)

        lines = [
            "========================================",
            f"## Prism Working Memory Context [{ticker}]",
            "========================================",
        ]

        if reminders:
            lines.append("\n### Pending Reminders")
            for r in reminders:
                lines.append(
                    f"- 🔴 [{r['priority'].upper()}] {r['intention']} (Trigger: {r['trigger_condition']})"
                )
                try:
                    prospective_memory_store.mark_triggered(r["id"])
                except Exception as e:
                    logger.warning(f"Failed to mark prospective memory triggered: {e}")

        if semantics:
            lines.append(f"\n### Known Facts [{ticker}]")
            for s in semantics:
                lines.append(f"- [{s['type']}] {s['content']}")

        if episodes:
            lines.append("\n### Relevant Past Cycles")
            for e in episodes:
                date_str = e["timestamp"][:10] if e["timestamp"] else "Unknown"
                summary = e["summary"][:150]
                lines.append(
                    f"- [{date_str}] Outcome Score: {e['outcome_score']} — {summary}"
                )

        if procedures:
            lines.append("\n### Proven Patterns")
            for p in procedures:
                rate = round(p["success_rate"] * 100)
                lines.append(
                    f"- When {p['trigger_pattern']} → {p['procedure']} | Success: {rate}% ({p['total_uses']} uses)"
                )

        with self._lock:
            scratchpad = self._scratchpads.get(ticker, [])
            if scratchpad:
                lines.append("\n### Scratchpad (Active Notes)")
                for note in scratchpad:
                    lines.append(f"- {note}")

            live_msgs = self._live_events.get(ticker, [])
            live_globals = self._live_events.get("global", [])

            all_live = list(live_globals) + list(live_msgs)

            if all_live:
                lines.append(f"\n### Live Cycle Events ({len(all_live)} slots)")
                for i, slot in enumerate(all_live):
                    lines.append(
                        f"[{i + 1}] Source: {slot['source']}\n{slot['content']}\n---"
                    )

        lines.append("========================================\n")
        return "\n".join(lines)


# Singleton
working_memory = WorkingMemoryManager()
