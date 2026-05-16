"""
Obsidian LLMWiki Toolset.

Provides tools for creating, reading, and searching markdown-based
memory files under `memory/LLMWiki/` that acts as the shared mind map
for bots, allowing short/long-term persistence.
"""

import os
import re
import glob
import logging

from app.tools.registry import registry, PermissionLevel

logger = logging.getLogger(__name__)

WIKI_DIR = os.path.join(os.getcwd(), "memory", "LLMWiki")


def _safe_filename(topic: str) -> str:
    """Convert a topic string into a safe markdown filename."""
    # Remove invalid characters, replace spaces with underscores
    safe_topic = re.sub(r"[^a-zA-Z0-9_\- ]", "", topic)
    safe_topic = safe_topic.strip().replace(" ", "_")
    return f"{safe_topic}.md"


async def write_memory_note(topic: str, content: str) -> str:
    """Save or update a markdown note in the LLMWiki folder mapping a core concept or ticker."""
    os.makedirs(WIKI_DIR, exist_ok=True)
    filename = _safe_filename(topic)
    filepath = os.path.join(WIKI_DIR, filename)

    # Check if we're updating or creating
    action = "Updated" if os.path.exists(filepath) else "Created"

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# {topic}\n\n{content}")
        logger.info(f"[WIKI] {action} note: {filename}")
        return f"Success: {action} note '{topic}' at {filename}"
    except Exception as e:
        logger.error(f"[WIKI] Failed to write note '{topic}': {e}")
        return f"Error writing note '{topic}': {e}"


async def read_memory_note(topic: str) -> str:
    """Read a persistent concept, strategy, or ticker profile from the LLMWiki."""
    filename = _safe_filename(topic)
    filepath = os.path.join(WIKI_DIR, filename)

    if not os.path.exists(filepath):
        return f"Notice: No wiki note found for '{topic}'."

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Error reading note '{topic}': {e}"


async def search_wiki(query: str) -> str:
    """Lightweight search over the LLMWiki mind map."""
    # Guard: reject queries that look like LLM reasoning text, not search terms
    if len(query) > 80 or any(
        word in query.lower()
        for word in ["based on", "analysis", "therefore", "conclusion", "recommend"]
    ):
        return (
            f"Error: Query too long or appears to be reasoning text ({len(query)} chars). "
            "Please provide a short search term (1-5 words), e.g. 'NVDA strategy' or 'RSI threshold'."
        )

    search_results = []

    if not os.path.exists(WIKI_DIR):
        return "Wiki not initialized."

    for filepath in glob.glob(os.path.join(WIKI_DIR, "*.md")):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            if (
                query.lower() in content.lower()
                or query.lower() in os.path.basename(filepath).lower()
            ):
                # Extract a snippet
                idx = content.lower().find(query.lower())
                snippet_start = max(0, idx - 50)
                snippet_end = min(len(content), idx + 100)
                snippet = content[snippet_start:snippet_end].replace("\n", " ")
                search_results.append(
                    f"File: {os.path.basename(filepath)} | Snippet: ...{snippet}..."
                )
        except Exception:
            continue

    if not search_results:
        return f"No results found for query: '{query}'"
    return "\n".join(search_results)


# Register tools with the central ToolRegistry
registry.register(
    func=write_memory_note,
    name="write_memory_note",
    description="Save or update a markdown note in the shared LLMWiki folder (short/long-term memory). Useful for storing structural knowledge, agent rules, and mind maps.",
    parameters={
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "The concept, ticker, or structural name of the note (e.g. 'NVDA_strategy').",
            },
            "content": {
                "type": "string",
                "description": "The markdown-formatted content of the note.",
            },
        },
        "required": ["topic", "content"],
    },
    tier=0,
    source="wiki",
    permission=PermissionLevel.WRITE,
    tags=["memory", "wiki", "note", "persist"],
)

registry.register(
    func=read_memory_note,
    name="read_memory_note",
    description="Read a specific note from the LLMWiki folder into your context.",
    parameters={
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "The topic of the note to retrieve.",
            }
        },
        "required": ["topic"],
    },
    tier=0,
    source="wiki",
)

registry.register(
    func=search_wiki,
    name="search_wiki",
    description="Search all LLMWiki notes for a specific query string, returns matching filenames and snippets.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Text to search for across the mind map.",
            }
        },
        "required": ["query"],
    },
    tier=0,
    source="wiki",
)
