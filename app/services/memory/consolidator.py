from app.utils.text_utils import parse_json_response
import logging
import uuid
import asyncio
from datetime import datetime, timezone

from app.db.memory_repo import (
    get_unpromoted_observations,
    get_active_canonical_memories,
    upsert_canonical_memories,
    deprecate_canonical_memories,
    mark_observations_promoted,
    log_consolidation_run,
)
from app.services.vllm_client import llm, Priority

logger = logging.getLogger(__name__)

NEW_EPISODIC_THRESHOLD = 5

CONSOLIDATION_SYSTEM_PROMPT = """
You are the Autodream Memory Consolidator, a background system optimizing a trading AI's knowledge base.
Your job is to read raw "episodic observations" and existing "canonical memories" for a given ticker, and merge them into a cleaner set of canonical rules.

RULES:
1. Combine redundant rules/observations.
2. Contradictions: If new episodic observations heavily contradict an existing canonical memory, you must DEPRECATE the old memory and replace it, or lower its confidence score.
3. Your output MUST be strictly valid JSON without markdown wrapping or backticks.

OUTPUT FORMAT:
{
  "new_or_updated_memories": [
    {
       "id": "UUID-OR-EXISTING-ID",
       "type": "market_pattern", // "market_pattern" | "ticker_quirk" | "failure_pattern" | "regime" | "execution_rule"
       "ticker": "...",
       "sector": "...",
       "summary": "...",
       "tags": ["..."],
       "confidence_score": 0.0 - 1.0,
       "evidence_count": integer
    }
  ],
  "deprecated_memory_ids": ["id-1", "id-2"]
}

Important notes on output:
- To UPDATE an existing canonical memory, emit it in `new_or_updated_memories` using its CURRENT `id`.
- To CREATE a new canonical memory, leave the `id` blank or generate a descriptive string.
- To DEPRECATE a memory completely, place its `id` in `deprecated_memory_ids`.
"""


async def should_consolidate(ticker: str) -> bool:
    unpromoted = get_unpromoted_observations(ticker)
    if len(unpromoted) >= NEW_EPISODIC_THRESHOLD:
        return True

    # Optional elapsed time threshold logic could go here
    return False


async def run_ticker_consolidation(ticker: str):
    logger.info(f"Starting consolidation for {ticker}...")
    observations = get_unpromoted_observations(ticker)

    if not observations:
        logger.info(f"No unpromoted observations for {ticker}.")
        return

    canonicals = get_active_canonical_memories(ticker)

    # Prompt synthesis
    user_prompt = f"TICKER: {ticker}\n\n"

    user_prompt += "=== EXISTING CANONICAL MEMORIES ===\n"
    if not canonicals:
        user_prompt += "(None)\n"
    else:
        for c in canonicals:
            user_prompt += f"ID: {c['id']} | Type: {c.get('type')} | Conf: {c.get('confidence_score')}\n"
            user_prompt += f"Summary: {c.get('summary')}\n\n"

    user_prompt += "=== NEW EPISODIC OBSERVATIONS ===\n"
    for o in observations:
        user_prompt += f"Obs [{o['created_at']}]: {o.get('observation_text')}\n"
        user_prompt += (
            f"Outcome: {o.get('outcome_label')} ({o.get('outcome_score')})\n\n"
        )

    # Execute LLM call
    try:
        response_text, _, _ = await llm.chat(
            system=CONSOLIDATION_SYSTEM_PROMPT,
            user=user_prompt,
            temperature=0.2,
            max_tokens=2048,
            priority=Priority.LOW,
            agent_name="memory_consolidator",
            ticker=ticker,
        )

        # Clean JSON blocks
        parsed_res = parse_json_response(response_text)

        updated_mems = parsed_res.get("new_or_updated_memories", [])
        deprecated_ids = parsed_res.get("deprecated_memory_ids", [])

        # Fill missing IDs and defaults
        for mem in updated_mems:
            if "id" not in mem or not mem["id"]:
                mem["id"] = str(uuid.uuid4())
            mem["ticker"] = ticker
            mem["status"] = "active"
            if "created_at" not in mem:
                mem["created_at"] = datetime.now(timezone.utc).isoformat()

        upsert_canonical_memories(updated_mems)
        deprecate_canonical_memories(deprecated_ids)

        obs_ids = [o["id"] for o in observations]
        mark_observations_promoted(obs_ids)

        log_consolidation_run(
            {
                "id": str(uuid.uuid4()),
                "ticker": ticker,
                "observations_consumed": len(obs_ids),
                "memories_created": len(updated_mems),
                "memories_deprecated": len(deprecated_ids),
            }
        )

        logger.info(
            f"Consolidation complete for {ticker}: {len(updated_mems)} upserted, {len(deprecated_ids)} deprecated."
        )

    except asyncio.TimeoutError:
        logger.error(f"Timeout during consolidation for {ticker}")
    except Exception as e:
        logger.error(f"Failed consolidation for {ticker}: {e}")


# _parse_json_response removed — use app.utils.text_utils.parse_json_response
