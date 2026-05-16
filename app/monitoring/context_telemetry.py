"""
Context Telemetry — Tracks prompt-level context utilization per agent/cycle.

Records how much of the effective context budget each agent consumed,
enabling operators to detect context bloat, validate compression effectiveness,
and tune budgets over time.

Data is logged to the ``context_telemetry`` table in PostgreSQL.

Usage:
    from app.monitoring.context_telemetry import log_context_usage
    log_context_usage(
        cycle_id="abc",
        agent_name="bull_fundamental_t1",
        model_id="google/gemma-4-26B-A4B-it",
        system_prompt_chars=3200,
        data_context_chars=12000,
        history_chars=4800,
        tool_result_chars=3600,
        total_prompt_chars=23600,
    )
"""

import logging

from app.db.connection import get_db
from app.config.context_budget import get_context_budget, CHARS_PER_TOKEN

logger = logging.getLogger(__name__)

# ── Schema Creation ────────────────────────────────────────────────────
_TABLE_CREATED = False


def _ensure_table():
    """Create the telemetry table if it doesn't exist."""
    global _TABLE_CREATED
    if _TABLE_CREATED:
        return

    try:
        with get_db() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS context_telemetry (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    cycle_id TEXT,
                    agent_name TEXT NOT NULL,
                    model_id TEXT,
                    system_prompt_tokens INT DEFAULT 0,
                    data_context_tokens INT DEFAULT 0,
                    history_tokens INT DEFAULT 0,
                    tool_result_tokens INT DEFAULT 0,
                    total_prompt_tokens INT DEFAULT 0,
                    effective_budget INT DEFAULT 0,
                    utilization_pct REAL DEFAULT 0,
                    was_compressed BOOLEAN DEFAULT FALSE,
                    compression_savings_tokens INT DEFAULT 0,
                    notes TEXT
                )
            """)

            # Index for cycle-level queries
            db.execute("""
                CREATE INDEX IF NOT EXISTS idx_ctx_telemetry_cycle
                ON context_telemetry(cycle_id)
            """)

            _TABLE_CREATED = True
            logger.info("[CTX_TELEMETRY] Table created/verified.")
    except Exception as e:
        logger.warning("[CTX_TELEMETRY] Table creation failed: %s", e)


def log_context_usage(
    cycle_id: str = "",
    agent_name: str = "unknown",
    model_id: str | None = None,
    system_prompt_chars: int = 0,
    data_context_chars: int = 0,
    history_chars: int = 0,
    tool_result_chars: int = 0,
    total_prompt_chars: int | None = None,
    was_compressed: bool = False,
    compression_savings_chars: int = 0,
    notes: str = "",
) -> None:
    """Log a context utilization event.

    All sizes are in *characters* — converted to estimated tokens internally
    using the standard 4-char-per-token heuristic.
    """
    _ensure_table()

    # Convert chars to estimated tokens
    sys_tok = system_prompt_chars // CHARS_PER_TOKEN
    data_tok = data_context_chars // CHARS_PER_TOKEN
    hist_tok = history_chars // CHARS_PER_TOKEN
    tool_tok = tool_result_chars // CHARS_PER_TOKEN

    if total_prompt_chars is not None:
        total_tok = total_prompt_chars // CHARS_PER_TOKEN
    else:
        total_tok = sys_tok + data_tok + hist_tok + tool_tok

    comp_savings = compression_savings_chars // CHARS_PER_TOKEN

    budget = get_context_budget(model_id)
    effective = budget.effective_context_tokens
    utilization = (total_tok / effective * 100) if effective > 0 else 0

    try:
        with get_db() as db:
            db.execute(
                """
                INSERT INTO context_telemetry
                    (cycle_id, agent_name, model_id,
                     system_prompt_tokens, data_context_tokens,
                     history_tokens, tool_result_tokens,
                     total_prompt_tokens, effective_budget,
                     utilization_pct, was_compressed,
                     compression_savings_tokens, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    cycle_id,
                    agent_name,
                    model_id or "unknown",
                    sys_tok,
                    data_tok,
                    hist_tok,
                    tool_tok,
                    total_tok,
                    effective,
                    round(utilization, 1),
                    was_compressed,
                    comp_savings,
                    notes,
                ],
            )

        # Log warning if utilization is getting dangerously high
        if utilization > 80:
            logger.warning(
                "[CTX_TELEMETRY] ⚠️ High utilization: %s used %.1f%% of %d effective tokens "
                "(total=%d, system=%d, data=%d, hist=%d, tool=%d)",
                agent_name,
                utilization,
                effective,
                total_tok,
                sys_tok,
                data_tok,
                hist_tok,
                tool_tok,
            )
        else:
            logger.debug(
                "[CTX_TELEMETRY] %s: %.1f%% utilization (%d/%d tokens)",
                agent_name,
                utilization,
                total_tok,
                effective,
            )

    except Exception as e:
        logger.warning("[CTX_TELEMETRY] Failed to log: %s", e)


def get_cycle_telemetry(cycle_id: str) -> list[dict]:
    """Retrieve all context telemetry entries for a given cycle."""
    _ensure_table()
    try:
        with get_db() as db:
            rows = db.execute(
                """
                SELECT agent_name, model_id, total_prompt_tokens,
                       effective_budget, utilization_pct,
                       was_compressed, compression_savings_tokens, notes
                FROM context_telemetry
                WHERE cycle_id = %s
                ORDER BY timestamp
                """,
                [cycle_id],
            ).fetchall()

            return [
                {
                    "agent_name": r[0],
                    "model_id": r[1],
                    "total_tokens": r[2],
                    "budget": r[3],
                    "utilization_pct": r[4],
                    "compressed": r[5],
                    "savings": r[6],
                    "notes": r[7],
                }
                for r in rows
            ]
    except Exception as e:
        logger.warning("[CTX_TELEMETRY] Failed to fetch: %s", e)
        return []
