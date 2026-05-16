"""
Context Tools — Lazy-fetch expansion for AgentCapsule Layer 2 data.

Agents receive compressed capsule summaries (Layer 1, ~150 tokens) in their context.
When a capsule summary is insufficient, agents can call get_cycle_context()
to expand the full raw agent response from PostgreSQL (Layer 2).

This is the core mechanism that keeps token budgets flat while preserving
full data accessibility: summary travels in memory, raw data stays in DB.

Usage by LLM:
    get_cycle_context(ref_id="abc-123-def")
    → Returns the full raw response from the retriever agent for this cycle
"""

import json
import logging

from app.tools.registry import registry, PermissionLevel

logger = logging.getLogger(__name__)


@registry.register(
    name="get_cycle_context",
    description=(
        "Expand a capsule source reference to get the full raw agent findings from a prior step. "
        "Use this when the compressed capsule summary is insufficient for your analysis. "
        "The ref_id is found in the '→ Expand: ref:capsule:<id>' line of each capsule summary."
    ),
    parameters={
        "type": "object",
        "properties": {
            "ref_id": {
                "type": "string",
                "description": (
                    "The reference ID to expand. Extract from the capsule's source_refs. "
                    "Format: either the full 'ref:capsule:<uuid>' or just the UUID portion."
                ),
            }
        },
        "required": ["ref_id"],
    },
    tier=0,
    source="internal",
    permission=PermissionLevel.READ_ONLY,
    max_result_chars=10_000,
    tags=["context", "capsule", "expand", "lazy_fetch"],
)
async def get_cycle_context(ref_id: str) -> str:
    """Fetch the full raw response for a capsule reference from the cycle_context table.

    Strips the 'ref:capsule:' prefix if present, then queries by UUID.
    Returns the raw_response text, or a structured error if not found.
    """
    # Strip prefix if present
    clean_id = ref_id.strip()
    if clean_id.startswith("ref:capsule:"):
        clean_id = clean_id[len("ref:capsule:"):]

    if not clean_id:
        return json.dumps({"error": "Empty ref_id provided. Please provide a valid capsule reference."})

    try:
        from app.db.connection import get_db

        with get_db() as db:
            row = db.execute(
                "SELECT agent_name, ticker, raw_response, summary, signal, confidence "
                "FROM cycle_context WHERE id = %s",
                [clean_id],
            ).fetchone()

        if not row:
            return json.dumps({
                "error": f"No capsule found for ref_id '{clean_id}'. It may have expired or the ID is incorrect.",
                "hint": "Check the capsule summary for the correct ref:capsule:<id> value.",
            })

        agent_name, ticker, raw_response, summary, signal, confidence = row
        logger.info(
            "[ContextTool] Expanded capsule %s for %s/%s (%d chars)",
            clean_id[:8], agent_name, ticker, len(raw_response or ""),
        )

        return json.dumps({
            "agent_name": agent_name,
            "ticker": ticker,
            "signal": signal,
            "confidence": confidence,
            "summary": summary,
            "full_response": raw_response,
        })

    except Exception as e:
        logger.error("[ContextTool] Failed to expand capsule %s: %s", clean_id[:8], e)
        return json.dumps({"error": f"Database error expanding capsule: {str(e)}"})


@registry.register(
    name="get_cycle_context_all",
    description=(
        "Get ALL capsule summaries for the current cycle and ticker. "
        "Returns Layer 1 summaries from every agent that ran this cycle. "
        "Use this to re-ground yourself on what all agents found without "
        "needing individual ref_ids. Does NOT return full raw responses — "
        "use get_cycle_context(ref_id) for that."
    ),
    parameters={
        "type": "object",
        "properties": {
            "cycle_id": {
                "type": "string",
                "description": "The current cycle ID.",
            },
            "ticker": {
                "type": "string",
                "description": "The ticker symbol being analyzed.",
            },
        },
        "required": ["cycle_id", "ticker"],
    },
    tier=0,
    source="internal",
    permission=PermissionLevel.READ_ONLY,
    max_result_chars=5_000,
    tags=["context", "capsule", "cycle", "summary"],
)
async def get_cycle_context_all(cycle_id: str, ticker: str) -> str:
    """Return all capsule summaries for a given cycle_id + ticker.

    Returns Layer 1 data only (summaries, signals, confidence) — not raw responses.
    Useful for the synthesizer or RLM to re-ground on the full cycle picture.
    """
    if not cycle_id or not ticker:
        return json.dumps({"error": "Both cycle_id and ticker are required."})

    try:
        from app.db.connection import get_db

        with get_db() as db:
            rows = db.execute(
                "SELECT id, agent_name, summary, signal, confidence, flags "
                "FROM cycle_context "
                "WHERE cycle_id = %s AND ticker = %s "
                "ORDER BY created_at ASC",
                [cycle_id, ticker.upper()],
            ).fetchall()

        if not rows:
            return json.dumps({
                "cycle_id": cycle_id,
                "ticker": ticker,
                "agents": [],
                "note": "No capsules found for this cycle/ticker.",
            })

        agents = []
        for row in rows:
            ctx_id, agent_name, summary, signal, confidence, flags = row
            agents.append({
                "agent_name": agent_name,
                "signal": signal,
                "confidence": confidence,
                "summary": summary,
                "flags": flags,
                "ref_id": f"ref:capsule:{ctx_id}",
            })

        logger.info(
            "[ContextTool] Returned %d capsule summaries for cycle=%s ticker=%s",
            len(agents), cycle_id[:8], ticker,
        )

        return json.dumps({
            "cycle_id": cycle_id,
            "ticker": ticker,
            "agent_count": len(agents),
            "agents": agents,
        })

    except Exception as e:
        logger.error("[ContextTool] Failed to fetch cycle context: %s", e)
        return json.dumps({"error": f"Database error: {str(e)}"})
