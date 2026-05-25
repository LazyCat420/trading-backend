import json
import logging
import traceback
import uuid
from datetime import datetime, timezone
from app.utils.trace import get_trace_id

logger = logging.getLogger(__name__)

class DbLoggingHandler(logging.Handler):
    """
    Standard logging handler that writes log messages with level WARNING or higher
    directly into the PostgreSQL 'execution_errors' and 'cycle_audit_log' tables.
    Designed with zero-crash propagation — DB failures will not interrupt execution.
    """

    def __init__(self, level=logging.WARNING):
        super().__init__(level=level)

    def emit(self, record):
        try:
            # Avoid recursive loops if db client logs a warning/error
            if record.name.startswith("psycopg") or record.name.startswith("app.db"):
                return

            cycle_id = getattr(record, "cycle_id", get_trace_id())
            # If we don't have a cycle_id, use a fallback
            if not cycle_id:
                cycle_id = "system-log"

            phase = getattr(record, "phase", "unknown")
            ticker = getattr(record, "ticker", "system")
            error_type = getattr(record, "error_type", record.levelname)
            
            # Format message
            error_message = record.getMessage()
            
            # Format stack trace
            stack_trace = ""
            if record.exc_info:
                stack_trace = "".join(traceback.format_exception(*record.exc_info))
            else:
                stack_trace = record.stack_info or ""

            # Save to database
            self._write_to_db(cycle_id, phase, ticker, error_type, error_message, stack_trace, record.levelname)
        except Exception as e:
            # Suppress logging failures to prevent loop/hangs
            pass

    def _write_to_db(self, cycle_id: str, phase: str, ticker: str, error_type: str, error_message: str, stack_trace: str, levelname: str):
        try:
            from app.db.connection import get_db
            
            error_id = f"err_{uuid.uuid4().hex[:12]}"
            with get_db() as db:
                # 1. Insert into execution_errors
                db.execute(
                    """
                    INSERT INTO execution_errors (id, cycle_id, phase, ticker, error_type, error_message, stack_trace, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (error_id, cycle_id, phase, ticker, error_type, error_message[:1000], stack_trace[:4000], datetime.now(timezone.utc))
                )

                # 2. Duplicate to cycle_audit_log
                audit_id = f"aud_{uuid.uuid4().hex[:12]}"
                severity = "warning" if levelname == "WARNING" else "critical"
                db.execute(
                    """
                    INSERT INTO cycle_audit_log (id, cycle_id, timestamp, audit_type, event_type, phase, ticker, severity, message, data)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        audit_id,
                        cycle_id,
                        datetime.now(timezone.utc),
                        "log_event",
                        levelname.lower(),
                        phase,
                        ticker,
                        severity,
                        f"[{levelname}] {error_message[:500]}",
                        json.dumps({
                            "error_id": error_id,
                            "error_type": error_type,
                            "stack_trace_snippet": stack_trace[:500] if stack_trace else ""
                        })
                    )
                )
        except Exception:
            pass


def setup_db_logger():
    """Register the DbLoggingHandler globally."""
    root_logger = logging.getLogger()
    
    # Check if handler already registered
    for handler in root_logger.handlers:
        if isinstance(handler, DbLoggingHandler):
            return
            
    handler = DbLoggingHandler()
    root_logger.addHandler(handler)
    logger.info("[Logger] Unified DB error logger handler registered.")
