import logging
import uuid
import hashlib
from datetime import datetime, timezone
from app.db.connection import get_db

logger = logging.getLogger(__name__)


def log_rlm_audit_trail(
    cycle_id: str,
    bot_id: str,
    ticker: str,
    context: str,
    trading_system_prompt: str,
    active_model: str,
    response_text: str,
    tokens_used: int,
    execution_time: float,
    agent_step: str = "analysis",
    endpoint_name: str = "",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    queue_wait_ms: int = 0,
) -> None:
    """Log to PostgreSQL (with context dedup + per-box telemetry)."""
    try:
        with get_db() as db:
            # SHA256-hash context and system prompt for dedup storage
            ctx_hash = hashlib.sha256(context.encode("utf-8")).hexdigest()
            prompt_hash = hashlib.sha256(
                trading_system_prompt.encode("utf-8")
            ).hexdigest()

            # Insert blobs only if they don't already exist (dedup)
            for blob_hash, blob_content in [
                (ctx_hash, context),
                (prompt_hash, trading_system_prompt),
            ]:
                db.execute(
                    """
                    INSERT INTO context_blobs (context_hash, content, byte_size)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (context_hash) DO NOTHING
                """,
                    [blob_hash, blob_content, len(blob_content.encode("utf-8"))],
                )

            # Compute tokens per second
            exec_ms = int(execution_time * 1000)
            tok_per_sec = None
            if completion_tokens > 0 and exec_ms > 0:
                tok_per_sec = round(completion_tokens / (exec_ms / 1000), 1)

            # Store only hashes in the audit log row
            db.execute(
                """
                INSERT INTO llm_audit_logs (
                    id, cycle_id, bot_id, ticker, agent_step, model, system_prompt_hash,
                    context_hash, raw_response, tokens_used, execution_ms, created_at,
                    endpoint_name, prompt_tokens, completion_tokens,
                    queue_wait_ms, tokens_per_second
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
                [
                    str(uuid.uuid4()),
                    cycle_id,
                    bot_id,
                    ticker,
                    agent_step,
                    active_model,
                    prompt_hash,
                    ctx_hash,
                    response_text,
                    tokens_used,
                    exec_ms,
                    datetime.now(timezone.utc),
                    endpoint_name or None,
                    prompt_tokens,
                    completion_tokens,
                    queue_wait_ms,
                    tok_per_sec,
                ],
            )
            logger.debug(
                "[DB] Successfully wrote trace to llm_audit_logs for %s (ctx_hash=%s...)",
                ticker,
                ctx_hash[:12],
            )
    except Exception as db_e:
        logger.error("[RLM] [DB] Audit log un-writable for %s: %s", ticker, db_e)
