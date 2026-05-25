"""
Unified Logging, Auditing, and Self-Improvement Loops package.
"""

from app.services.logging.cycle_auditor import auditor
from app.services.logging.unified_logger import setup_db_logger, DbLoggingHandler
from app.services.logging.tool_logging import log_tool_call
from app.services.logging.autoresearch import (
    run_autoresearch,
    run_eval_worker,
    evaluate_trace,
    process_and_store_trace,
    evaluate_confidence_calibration,
    TraceRecord,
    classify_failure
)
from app.services.logging.pending_review import (
    get_pending_outliers,
    approve_outlier,
    reject_outlier,
    add_outlier_rule,
    get_pending_fixes,
    approve_fix,
    reject_fix
)

# Automatically set up the DB logger on import
setup_db_logger()
