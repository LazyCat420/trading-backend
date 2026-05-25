"""
Tool Logging Service — Tracks tool usage counts, latencies, success/failure status, and errors.
"""

import logging
from datetime import datetime, timezone
from app.db.connection import get_db

logger = logging.getLogger(__name__)


def log_tool_call(
    tool_name: str,
    agent_name: str = "",
    ticker: str = "",
    cycle_id: str = "",
    success: bool = True,
    execution_ms: int = 0,
    error_message: str | None = None,
    service_source: str = "trading-service"
):
    """
    Log a tool execution into the database.
    Fire-and-forget, suppresses all database connection issues to preserve tool reliability.
    """
    try:
        with get_db() as db:
            db.execute(
                """
                INSERT INTO tool_usage_stats 
                (tool_name, agent_name, ticker, cycle_id, success, execution_ms, error_message, service_source, called_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    tool_name,
                    agent_name or "",
                    ticker or "",
                    cycle_id or "",
                    success,
                    execution_ms,
                    error_message,
                    service_source,
                    datetime.now(timezone.utc)
                )
            )
    except Exception as e:
        logger.debug("[ToolLogger] Failed to log tool execution for '%s': %s", tool_name, e)
