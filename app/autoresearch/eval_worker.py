"""
Autoresearch Worker — Scheduled trace evaluation and playbook generation.

Periodically sweeps `agent_traces` that have not been evaluated, passes them 
through the `EvalEngine`, and aggregates statistics to update the `tool_playbook`.
"""

import logging
import uuid
from typing import List, Dict, Any  # noqa: F401 — kept for future use

from app.db.connection import get_db
from app.autoresearch.eval_engine import process_and_store_trace, TraceRecord, EvalStoreError

logger = logging.getLogger(__name__)

def process_pending_traces(limit: int = 50) -> int:
    """Find and evaluate pending traces."""
    processed_count = 0
    with get_db() as db:
        try:
            # Join against eval_scores treating eval_scores.run_id as agent_traces.id
            rows = db.execute(
                """
                SELECT t.id, t.run_id, t.agent_name, t.task_type, t.goal, 
                       t.planned_next_action, t.tool_name, t.tool_args, 
                       t.tool_result_summary, t.why_tool_was_called, 
                       t.tokens_before, t.tokens_after, t.latency_ms, 
                       t.did_tool_change_decision, t.loop_step, t.stop_reason
                FROM agent_traces t
                LEFT JOIN eval_scores e ON t.id = e.run_id
                WHERE e.id IS NULL
                ORDER BY t.created_at ASC
                LIMIT %s
                """,
                [limit],
            ).fetchall()

            columns = [
                "id", "cycle_id", "agent_name", "task_type", "goal", 
                "planned_next_action", "tool_name", "tool_args", 
                "tool_result_summary", "why_tool_was_called", 
                "tokens_before", "tokens_after", "latency_ms", 
                "did_tool_change_decision", "loop_step", "stop_reason"
            ]

            for row in rows:
                trace = dict(zip(columns, row))
                # Map trace 'id' to 'run_id' for EvalEngine backwards compatibility
                trace["run_id"] = trace["id"]
                
                # Fetch decision info to allow hold_bias check to work
                decision = db.execute(
                    """
                    SELECT action, confidence, pnl_pct 
                    FROM decision_outcomes 
                    WHERE cycle_id = %s
                    LIMIT 1
                    """,
                    [trace.get("cycle_id")]
                ).fetchone()
                
                if decision:
                    trace["decision_action"] = decision[0] or "HOLD"
                    trace["decision_confidence"] = decision[1] or 0
                    trace["pnl_pct"] = decision[2] or 0.0
                
                try:
                    record = TraceRecord(**trace)
                    process_and_store_trace(record)
                    processed_count += 1
                except ValueError as ve:
                    logger.warning("TraceRecord validation failed for run_id %s: %s", trace.get("run_id"), ve)
                except EvalStoreError as ee:
                    logger.warning("Failed to store trace %s: %s", trace.get("run_id"), ee)

            if processed_count > 0:
                logger.info(f"[EvalWorker] Processed {processed_count} pending agent traces.")
                
        except Exception as e:
            logger.error(f"[EvalWorker] Failed to process pending traces: {e}")
            
    return processed_count

def update_tool_playbook():
    """Aggregate trace eval scores and update the tool_playbook."""
    with get_db() as db:
        try:
            # Identify successful tool sequences for playbook
            rows = db.execute(
                """
                SELECT t.agent_name, t.tool_name, COUNT(*) as uses, AVG(e.final_score) as avg_score
                FROM agent_traces t
                JOIN eval_scores e ON t.id = e.run_id
                WHERE t.tool_name IS NOT NULL
                GROUP BY t.agent_name, t.tool_name
                HAVING COUNT(*) >= 5 AND AVG(e.final_score) >= 80.0
                """
            ).fetchall()

            for agent_name, tool_name, uses, avg_score in rows:
                playbook_id = str(uuid.uuid4())
                seq = f"Primary tool: {tool_name} (avg score: {avg_score:.1f} over {uses} uses)"
                
                # Insert tool_playbook
                db.execute(
                    """
                    INSERT INTO tool_playbook (id, task_type, market_context, agent_role, recommended_tool_sequence, required_preconditions)
                    VALUES (%s, 'general', 'any', %s, %s, 'None')
                    ON CONFLICT DO NOTHING
                    """,
                    [playbook_id, agent_name, seq]
                )
                
            logger.info("[EvalWorker] Updated tool playbook based on latest eval scores.")
        except Exception as e:
            logger.error(f"[EvalWorker] Failed to update tool playbook: {e}")

async def run_eval_worker(limit: int = 50):
    """Entry point for the scheduled task."""
    logger.info("[EvalWorker] Starting evaluation sweep...")
    count = process_pending_traces(limit)
    if count > 0:
        update_tool_playbook()
    logger.info("[EvalWorker] Evaluation sweep complete.")
