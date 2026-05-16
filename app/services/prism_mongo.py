"""
Prism MongoDB Logger — isolated component for writing request telemetry
directly to Prism's MongoDB `requests` collection. This allows offline
and non-proxied LLM calls to appear in the Retina dashboard analytics.
"""

import logging
from datetime import datetime, timezone

from app.config import settings

logger = logging.getLogger(__name__)

# Lazy import — motor is only needed when PRISM_ENABLED is True
_motor_client = None
_motor_db = None
_motor_init_failed = False  # Negative cache: stop retrying after first failure


def _get_mongo_db():
    """
    Lazy-initialise the motor async MongoDB client.
    Returns the database handle for Prism's MongoDB or None if unavailable.
    """
    global _motor_client, _motor_db, _motor_init_failed
    if _motor_db is not None:
        return _motor_db
    if _motor_init_failed:
        return None  # Don't retry after first failure (avoids log spam)
    try:
        from motor.motor_asyncio import AsyncIOMotorClient

        _motor_client = AsyncIOMotorClient(
            settings.PRISM_MONGO_URI,
            serverSelectionTimeoutMS=3000,
        )
        _motor_db = _motor_client[settings.PRISM_MONGO_DB]
        logger.info(
            "[PRISM] MongoDB client initialised: %s / %s",
            settings.PRISM_MONGO_URI.split("@")[-1],  # don't log credentials
            settings.PRISM_MONGO_DB,
        )
        return _motor_db
    except Exception as e:
        _motor_init_failed = True
        logger.warning("[PRISM] MongoDB client init failed (will not retry): %s", e)
        return None


async def log_request_to_mongo(
    *,
    request_id: str,
    conversation_id: str,
    model: str,
    agent_name: str,
    input_tokens: int,
    output_tokens: int,
    total_time_sec: float,
    messages: list[dict],
    response_text: str,
    project: str,
    username: str,
    success: bool = True,
    error_message: str | None = None,
    endpoint_name: str = "",
    ticker: str = "",
    cycle_id: str = "",
):
    """
    Write a request telemetry document directly to Prism's MongoDB `requests`
    collection. Matches the exact schema that Prism's internal
    ``RequestLogger.log()`` produces so the Retina dashboard aggregation
    pipelines pick it up identically to gateway-proxied calls.

    This is fire-and-forget — failures are logged and swallowed.
    """
    db = _get_mongo_db()
    if db is None:
        return

    # Calculate tok/s (guard against zero division)
    tokens_per_sec = None
    if total_time_sec > 0 and output_tokens > 0:
        tokens_per_sec = round(output_tokens / total_time_sec, 1)

    # Count input characters from messages
    input_chars = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            input_chars += len(content)

    doc = {
        "requestId": request_id,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        + "Z",
        "endpoint": "agent",
        "operation": "offline-sync",
        "project": project,
        "username": username,
        "clientIp": None,
        "agent": agent_name,
        "provider": "vllm",
        "model": model,
        "conversationId": conversation_id,
        "traceId": cycle_id or None,
        "agentSessionId": None,
        "toolsUsed": False,
        "toolDisplayNames": [],
        "toolApiNames": [],
        "success": success,
        "errorMessage": error_message,
        "inputTokens": input_tokens,
        "outputTokens": output_tokens,
        "estimatedCost": 0,  # self-hosted vLLM = $0
        "tokensPerSec": tokens_per_sec,
        "temperature": None,
        "maxTokens": None,
        "topP": None,
        "topK": None,
        "frequencyPenalty": None,
        "presencePenalty": None,
        "stopSequences": None,
        "messageCount": len(messages),
        "inputCharacters": input_chars,
        "outputCharacters": len(response_text),
        "timeToGeneration": None,
        "generationTime": None,
        "totalTime": round(total_time_sec, 3),
        "requestPayload": {
            "operation": "offline-sync",
            "agent": agent_name,
            "ticker": ticker or None,
            "endpoint": endpoint_name or None,
        },
        "responsePayload": {
            "textPreview": response_text[:200] if response_text else None,
        },
        "modalities": {"textIn": True, "textOut": True},
        "rateLimits": None,
    }

    try:
        await db.requests.insert_one(doc)
        logger.debug(
            "[PRISM] Request logged to MongoDB: %s | %s | %d+%d tok | %.1fs",
            agent_name,
            model,
            input_tokens,
            output_tokens,
            total_time_sec,
        )
    except Exception as e:
        logger.warning("[PRISM] MongoDB request log failed: %s", e)
