import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from app.db.connection import get_db

with get_db() as db:
    cycle_row = db.execute("SELECT cycle_id, status FROM pipeline_state WHERE singleton_id = 'current'").fetchone()
    if cycle_row:
        cycle_id = cycle_row[0]
        
        # How many rows are in llm_audit_logs for this cycle?
        audit_rows = db.execute("SELECT count(*) FROM llm_audit_logs WHERE cycle_id = %s", [cycle_id]).fetchone()
        print("Total LLM audit logs (decisions + other logs):", audit_rows[0])
        
        # How many decisions are pending evaluation?
        # A decision is a log where context_hash is NOT NULL and is an actual trading decision.
        # Let's see how evaluate_pending_decisions finds them:
        pending_rows = db.execute(
            """
            SELECT count(*)
            FROM llm_audit_logs l
            WHERE l.cycle_id = %s
              AND l.context_hash IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM decision_evaluations e
                  WHERE e.decision_id = l.id
              )
            """,
            [cycle_id]
        ).fetchone()
        print("Pending decisions for evaluation:", pending_rows[0])
        
        # How many are already evaluated?
        eval_rows = db.execute(
            """
            SELECT count(*)
            FROM decision_evaluations
            WHERE cycle_id = %s
            """,
            [cycle_id]
        ).fetchone()
        print("Evaluated decisions:", eval_rows[0])
