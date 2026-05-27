"""
AutoResearch Service — Consolidates post-cycle reflection, trace evaluations, 
and playbook aggregation.
"""

import json
import logging
import uuid
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, field_validator

from app.db.connection import get_db

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# TRACE EVALUATION & PLAYBOOK ENGINE (from eval_engine.py / eval_worker.py)
# ═══════════════════════════════════════════════════════════════════

class EvalStoreError(Exception):
    pass

class TraceRecord(BaseModel):
    id: str
    run_id: str
    cycle_id: Optional[str] = None
    agent_name: Optional[str] = None
    task_type: Optional[str] = None
    goal: Optional[str] = None
    planned_next_action: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[str] = None
    tool_result_summary: Optional[str] = None
    why_tool_was_called: Optional[str] = None
    tokens_before: int = 0
    tokens_after: int = 0
    latency_ms: int = 0
    did_tool_change_decision: Optional[bool] = None
    loop_step: Optional[int] = None
    stop_reason: Optional[str] = None
    decision_action: Optional[str] = None
    decision_confidence: Optional[float] = 0.0
    pnl_pct: Optional[float] = 0.0

    @field_validator("decision_confidence", "pnl_pct", mode="before")
    @classmethod
    def _coerce_none_to_zero(cls, v):
        """DB columns are nullable — coerce NULL → 0.0 so downstream math works."""
        return v if v is not None else 0.0


def evaluate_trace(trace: TraceRecord) -> Dict[str, Any]:
    """Score a single trace row based on the 5-part rubric."""
    completion_score = 40.0 if trace.stop_reason == "completed" else 0.0
    
    tool_correctness = 25.0
    if trace.tool_result_summary and "error" in str(trace.tool_result_summary).lower():
        tool_correctness -= 10.0
        
    tokens_used = trace.tokens_after - trace.tokens_before
    efficiency = 20.0
    if tokens_used > 5000:
        efficiency -= 10.0
        
    recovery = 10.0
    
    stop_quality = 5.0
    if trace.stop_reason == "budget_exhausted":
        stop_quality = 0.0

    final_score = max(0.0, completion_score + tool_correctness + efficiency + recovery + stop_quality)
    
    return {
        "completion_score": completion_score,
        "tool_correctness_score": tool_correctness,
        "efficiency_score": efficiency,
        "error_recovery_score": recovery,
        "stop_quality_score": stop_quality,
        "final_score": final_score
    }


def classify_failure(trace: TraceRecord, score: Dict[str, Any]) -> str | None:
    """Classifies runs with < 70 score into failure buckets."""
    if score["final_score"] >= 70.0:
        return None
        
    if score["completion_score"] == 0 and "budget_exhausted" in (trace.stop_reason or ""):
        return "over_research"
        
    action = str(trace.decision_action or "").upper()
    
    if action == "HOLD" and trace.decision_confidence >= 60 and abs(trace.pnl_pct) > 2.0:
        return "hold_bias"
        
    tool_summary = str(trace.tool_result_summary or "").lower()
    if "error" in tool_summary or "invalid" in tool_summary:
        return "bad_arguments"
        
    if trace.tokens_after - trace.tokens_before > 8000:
        return "loop_drift"
        
    return "wrong_tool_selected"


def process_and_store_trace(trace: TraceRecord):
    """Evaluate a trace and store the score and any failure bucket."""
    score = evaluate_trace(trace)
    bucket = classify_failure(trace, score)
    
    try:
        with get_db() as db:
            db.execute(
                """INSERT INTO eval_scores (id, run_id, completion_score, tool_correctness_score, 
                   efficiency_score, error_recovery_score, stop_quality_score, final_score)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                [
                    str(uuid.uuid4()), trace.run_id, score["completion_score"],
                    score["tool_correctness_score"], score["efficiency_score"],
                    score["error_recovery_score"], score["stop_quality_score"], score["final_score"]
                ]
            )
            
            if bucket:
                db.execute(
                    """INSERT INTO failure_buckets (id, run_id, bucket_type, description)
                       VALUES (%s, %s, %s, %s)""",
                    [str(uuid.uuid4()), trace.run_id, bucket, f"Auto-classified based on score {score['final_score']}"]
                )
    except Exception as e:
        logger.error("Failed to store eval results: %s", e)
        raise EvalStoreError(f"Failed to store eval results: {e}") from e


def evaluate_confidence_calibration(ticker: str | None = None, limit: int = 20) -> Dict[str, Any]:
    try:
        with get_db() as db:
            if ticker:
                rows = db.execute(
                    """
                    SELECT confidence, outcome, pnl_pct
                    FROM decision_outcomes
                    WHERE ticker = %s AND resolved_at IS NOT NULL
                      AND outcome IN ('WIN', 'LOSS')
                    ORDER BY resolved_at DESC LIMIT %s
                    """,
                    [ticker, limit],
                ).fetchall()
            else:
                rows = db.execute(
                    """
                    SELECT confidence, outcome, pnl_pct
                    FROM decision_outcomes
                    WHERE resolved_at IS NOT NULL
                      AND outcome IN ('WIN', 'LOSS')
                    ORDER BY resolved_at DESC LIMIT %s
                    """,
                    [limit],
                ).fetchall()

        if len(rows) < 3:
            return {
                "calibration_score": 50.0,
                "sample_count": len(rows),
                "status": "insufficient_data",
            }

        calibration_scores = []
        win_confs = []
        loss_confs = []

        for conf, outcome, pnl_pct in rows:
            normalized_conf = (conf or 50) / 100.0
            if outcome == "WIN":
                calibration_scores.append(normalized_conf)
                win_confs.append(conf or 50)
            elif outcome == "LOSS":
                calibration_scores.append(1.0 - normalized_conf)
                loss_confs.append(conf or 50)

        if not calibration_scores:
            cal_score = 50.0
        else:
            cal_score = (sum(calibration_scores) / len(calibration_scores)) * 100

        result = {
            "calibration_score": round(cal_score, 1),
            "sample_count": len(rows),
            "status": "ok",
            "avg_confidence_on_wins": round(sum(win_confs) / len(win_confs), 1) if win_confs else None,
            "avg_confidence_on_losses": round(sum(loss_confs) / len(loss_confs), 1) if loss_confs else None,
            "win_count": len(win_confs),
            "loss_count": len(loss_confs),
        }

        logger.info(
            "Confidence calibration: %.1f%% (%d samples, %d W / %d L)",
            cal_score, len(rows), len(win_confs), len(loss_confs),
        )
        return result

    except Exception as e:
        logger.error("Confidence calibration failed: %s", e)
        return {
            "calibration_score": 50.0,
            "sample_count": 0,
            "status": f"error: {e}",
        }


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


# ═══════════════════════════════════════════════════════════════════
# AUTORESEARCH REFLECTION & AUDITING (from pipeline/analysis/autoresearch.py)
# ═══════════════════════════════════════════════════════════════════

def _update_ar_state(report_id: str, **kwargs):
    updates = []
    params = []
    for k, v in kwargs.items():
        if k == "running":
            updates.append("status = %s")
            params.append("running" if v else "done")
        else:
            updates.append(f"{k} = %s")
            params.append(v)
    if not updates:
        return
    params.append(report_id)
    try:
        with get_db() as db:
            db.execute(
                f"UPDATE autoresearch_reports SET {', '.join(updates)} WHERE id = %s",
                params,
            )
    except Exception as e:
        logger.debug("Failed to update ar state: %s", e)


def get_autoresearch_status() -> dict:
    try:
        with get_db() as db:
            row = db.execute(
                "SELECT cycle_id, status, phase, error, created_at "
                "FROM autoresearch_reports ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            if row:
                return {
                    "running": row[1] == "running",
                    "cycle_id": row[0],
                    "phase": row[2] or "",
                    "error": row[3],
                    "started_at": row[4],
                }
    except Exception:
        pass
    return {
        "running": False,
        "cycle_id": None,
        "phase": None,
        "error": None,
        "started_at": None,
    }


async def run_autoresearch(cycle_id: str, cycle_summary: dict) -> dict:
    """Main entry point: run full autoresearch after a cycle."""
    report_id = f"ar-{uuid.uuid4().hex[:12]}"
    tickers = (
        cycle_summary.get("tickers_final")
        or cycle_summary.get("tickers_requested")
        or []
    )

    try:
        from app.utils.trace import set_trace_id
        set_trace_id(report_id)

        # Clean up stale reports stuck in 'running' from previous crashed cycles
        with get_db() as db:
            try:
                db.execute(
                    "UPDATE autoresearch_reports SET status = 'stale' "
                    "WHERE status = 'running' AND created_at < NOW() - INTERVAL '30 minutes'"
                )
                cleaned = db.execute(
                    "SELECT COUNT(*) FROM autoresearch_reports WHERE status = 'stale'"
                ).fetchone()
                if cleaned and cleaned[0] > 0:
                    logger.info(
                        "[AUTORESEARCH] Cleaned up stale 'running' reports (total stale: %d)",
                        cleaned[0],
                    )
            except Exception as cleanup_err:
                logger.debug("[AUTORESEARCH] Stale cleanup skipped: %s", cleanup_err)

            db.execute(
                "INSERT INTO autoresearch_reports (id, cycle_id, status, phase) VALUES (%s, %s, 'running', 'starting')",
                (report_id, cycle_id),
            )

        _update_ar_state(report_id, phase="data_quality")
        data_quality = _audit_data_quality(tickers)

        _update_ar_state(report_id, phase="decision_quality")
        decision_quality = _audit_decisions(cycle_id, cycle_summary)

        _update_ar_state(report_id, phase="llm_traces")
        llm_analysis = _audit_llm_traces(cycle_id)

        _update_ar_state(report_id, phase="performance")
        perf_metrics = _audit_performance(cycle_id, cycle_summary)

        _update_ar_state(report_id, phase="recovery")
        recovery = _audit_recovery()
        exec_errors = _audit_execution_errors(cycle_id)

        _update_ar_state(report_id, phase="reflection")
        audit_bundle = {
            "cycle_id": cycle_id,
            "tickers": tickers,
            "data_quality": data_quality,
            "decision_quality": decision_quality,
            "llm_analysis": llm_analysis,
            "performance": perf_metrics,
            "recovery": recovery,
            "execution_errors": exec_errors,
        }

        # Triage audit (evaluate triage distribution + attention health)
        _update_ar_state(report_id, phase="triage_audit")
        triage_audit = _audit_triage(cycle_id, cycle_summary, tickers)
        audit_bundle["triage_audit"] = triage_audit

        # Schedule health audit
        _update_ar_state(report_id, phase="schedule_audit")
        schedule_health = _audit_schedule_health()
        audit_bundle["schedule_health"] = schedule_health

        _update_ar_state(report_id, phase="reflection")
        reflection = await _reflect(audit_bundle)

        data_score = data_quality.get("avg_score", 0) * 100
        decision_score = decision_quality.get("score", 0) * 100
        llm_score = llm_analysis.get("score", 0) * 100
        overall = (data_score + decision_score + llm_score) / 3

        # Degenerate score detection
        degenerate_subs = []
        if data_score == 0.0:
            degenerate_subs.append("data")
        if decision_score == 0.0:
            degenerate_subs.append("decision")
        if llm_score == 0.0:
            degenerate_subs.append("llm")
        if degenerate_subs:
            logger.warning(
                "[AUTORESEARCH] DEGENERATE SCORES: %s at 0.0 — Flagging as anomaly.",
                ", ".join(degenerate_subs)
            )
            reflection["anomaly"] = True
            reflection["anomaly_detail"] = f"Degenerate sub-scores at 0.0: {', '.join(degenerate_subs)}"

        with get_db() as db:
            db.execute(
                """UPDATE autoresearch_reports SET
                    data_quality_score= %s, decision_quality_score= %s, llm_performance_score= %s,
                    overall_score= %s, data_gaps= %s, decision_issues= %s, llm_issues= %s,
                    performance_metrics= %s, reflection= %s, recovery_stats= %s, status='done'
                WHERE id=%s""",
                [
                    round(data_score, 1),
                    round(decision_score, 1),
                    round(llm_score, 1),
                    round(overall, 1),
                    json.dumps(data_quality.get("gaps", [])),
                    json.dumps(decision_quality.get("issues", [])),
                    json.dumps(llm_analysis.get("issues", [])),
                    json.dumps(perf_metrics),
                    json.dumps(reflection),
                    json.dumps(recovery),
                    report_id,
                ],
            )

        try:
            _store_lessons(reflection, cycle_id)

            if reflection.get("system_health") == "critical":
                from app.services.session_profile import profile_memory
                summary = reflection.get("summary", "Critical health detected by autoresearch.")
                profile_memory.add_agent_note(f"⚠️ AUTORESEARCH CRITICAL WARNING (Cycle {cycle_id[:8]}): {summary}")
        except Exception as ls_err:
            logger.warning("[AUTORESEARCH] Lesson store write failed: %s", ls_err)

        # Auto-resolve detected data gaps
        _update_ar_state(report_id, phase="gap_resolution")
        try:
            gap_result = await _resolve_data_gaps(data_quality.get("gaps", []), cycle_id)
            logger.info(
                "[AUTORESEARCH] Data gap resolution: resolved=%d, failed=%d, banned=%d",
                gap_result.get("resolved", 0), gap_result.get("failed", 0), gap_result.get("banned", 0)
            )
        except Exception as gap_err:
            logger.warning("[AUTORESEARCH] Data gap resolution failed: %s", gap_err)

        # Evolutionary Debate Council
        _EVO_TIMEOUT = 60
        try:
            from app.pipeline.analysis.evolution_router import router
            await asyncio.wait_for(router.run_router(cycle_id), timeout=_EVO_TIMEOUT)
        except Exception as e:
            logger.error("[AUTORESEARCH] Failed to trigger Evolution Router: %s", e)

        # Directives generation
        _update_ar_state(report_id, phase="directives")
        try:
            _generate_directives(reflection, cycle_id, triage_audit)
            _expire_old_directives()
        except Exception as dir_err:
            logger.warning("[AUTORESEARCH] Directive generation failed: %s", dir_err)

        # Benchmark Agent (Constitution review)
        _BENCH_TIMEOUT = 60
        try:
            from app.pipeline.analysis.benchmark_agent import run_benchmark_agent
            await asyncio.wait_for(run_benchmark_agent(cycle_id), timeout=_BENCH_TIMEOUT)
        except Exception as bench_err:
            logger.error("[AUTORESEARCH] Failed to trigger Benchmark Agent: %s", bench_err)

        # Record subsystem benchmarks
        try:
            from app.pipeline.subsystem_benchmarks import record_all
            record_all(cycle_id)
        except Exception as sb_err:
            logger.warning("[AUTORESEARCH] Subsystem benchmark recording failed: %s", sb_err)

        # Probation Rollbacks
        try:
            from app.cognition.evolution.rollback_monitor import check_probation_fixes
            rollback_summary = check_probation_fixes(cycle_id)
            if rollback_summary.get("rolled_back", 0) > 0:
                logger.warning("[AUTORESEARCH] Rolled back %d degrading fixes!", rollback_summary["rolled_back"])
        except Exception as rb_err:
            logger.warning("[AUTORESEARCH] Rollback monitor failed: %s", rb_err)

        # Meta-Agent Judge (prompt lifecycle management)
        _META_JUDGE_TIMEOUT = 120
        try:
            from app.agents.meta_agent_judge import run_meta_agent_judge
            _update_ar_state(report_id, phase="meta_judge")
            meta_result = await asyncio.wait_for(
                run_meta_agent_judge(cycle_id), timeout=_META_JUDGE_TIMEOUT
            )
            if meta_result.get("status") != "disabled":
                logger.info(
                    "[AUTORESEARCH] Meta-Agent Judge: benched=%d, promoted=%d, generated=%d",
                    len(meta_result.get("benched", [])),
                    len(meta_result.get("promoted", [])),
                    len(meta_result.get("generated", [])),
                )
        except Exception as mj_err:
            logger.warning("[AUTORESEARCH] Meta-Agent Judge failed: %s", mj_err)

        _update_ar_state(report_id, phase="done")
        return {"id": report_id, "overall_score": round(overall, 1), "status": "done"}

    except Exception as e:
        logger.error("[AUTORESEARCH] Failed: %s", e, exc_info=True)
        _update_ar_state(report_id, error=str(e), phase="error")
        try:
            with get_db() as db:
                db.execute("UPDATE autoresearch_reports SET status='error' WHERE id=%s", (report_id,))
        except Exception:
            pass
        return {"error": str(e)}
    finally:
        try:
            with get_db() as db:
                status = db.execute("SELECT status FROM autoresearch_reports WHERE id=%s", [report_id]).fetchone()
                if status and status[0] == 'running':
                    _update_ar_state(report_id, running=False)
        except:
            pass


async def run_partial_autoresearch(cycle_id: str, tickers: list[str]) -> dict:
    logger.info(f"[AUTORESEARCH] Running partial mid-cycle autoresearch for {len(tickers)} tickers.")
    data_quality = _audit_data_quality(tickers)
    return {"status": "partial_done", "data_quality": data_quality}


def _grade(score: float) -> str:
    if score >= 0.95: return "excellent"
    if score >= 0.80: return "good"
    if score >= 0.60: return "fair"
    if score >= 0.30: return "poor"
    return "critical"


def _safe_iso(val) -> str | None:
    if val is None: return None
    if hasattr(val, "isoformat"): return val.isoformat()
    return str(val)


def _audit_price_history(db, ticker: str) -> dict:
    try:
        stats = db.execute(
            """
            SELECT COUNT(*), MIN(date), MAX(date),
                   SUM(CASE WHEN close IS NULL THEN 1 ELSE 0 END),
                   SUM(CASE WHEN volume IS NULL OR volume = 0 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN open IS NULL OR high IS NULL OR low IS NULL THEN 1 ELSE 0 END)
            FROM price_history WHERE ticker = %s
            """,
            [ticker],
        ).fetchone()

        rows, min_d, max_d, null_close, zero_vol, null_ohlc = stats
        if rows == 0:
            return {"rows": 0, "quality": "critical", "quality_score": 0}

        gaps = db.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT date, LEAD(date) OVER (ORDER BY date) as next_date
                FROM price_history WHERE ticker = %s
            ) sub WHERE next_date::date - date::date > 4
            """,
            [ticker],
        ).fetchone()[0]

        latest = db.execute(
            """
            SELECT date, open, high, low, close, volume
            FROM price_history WHERE ticker = %s
            ORDER BY date DESC LIMIT 1
            """,
            [ticker],
        ).fetchone()

        null_pct = (null_close + zero_vol + null_ohlc) / (rows * 3) if rows else 1
        gap_penalty = min(gaps * 0.05, 0.3)
        score = max(0, 1.0 - null_pct - gap_penalty)

        return {
            "rows": rows,
            "date_range": [_safe_iso(min_d), _safe_iso(max_d)],
            "quality": _grade(score),
            "quality_score": round(score, 3),
            "null_close": null_close,
            "zero_volume_days": zero_vol,
            "null_ohlc": null_ohlc,
            "gaps_over_4_days": gaps,
            "latest": {
                "date": _safe_iso(latest[0]),
                "close": round(latest[4], 2) if latest[4] else None,
                "volume": latest[5],
            } if latest else None,
        }
    except Exception as e:
        logger.warning("audit price_history failed for %s: %s", ticker, e)
        return {"rows": 0, "quality": "error", "error": str(e)}


def _audit_technicals(db, ticker: str) -> dict:
    INDICATORS = [
        "rsi_14", "macd", "macd_signal", "macd_hist", "sma_20", "sma_50", "sma_200",
        "ema_12", "ema_26", "bb_upper", "bb_mid", "bb_lower", "atr_14", "adx_14",
        "stoch_k", "stoch_d", "obv", "vwap", "support", "resistance"
    ]
    try:
        stats = db.execute(
            "SELECT COUNT(*), MIN(date), MAX(date) FROM technicals WHERE ticker = %s",
            [ticker]
        ).fetchone()
        rows, min_d, max_d = stats

        if rows == 0:
            return {"rows": 0, "quality": "critical", "quality_score": 0, "indicators_computed": 0}

        indicator_health = {}
        total_nulls = 0
        indicators_ok = 0

        for col in INDICATORS:
            try:
                ind = db.execute(
                    f"""
                    SELECT COUNT({col}), SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END),
                           MIN({col}), MAX({col}), AVG({col})
                    FROM technicals WHERE ticker = %s
                    """,
                    [ticker],
                ).fetchone()

                non_null, nulls, min_v, max_v, avg_v = ind
                null_pct = nulls / rows if rows else 0
                total_nulls += nulls

                latest_val = db.execute(
                    f"SELECT {col} FROM technicals WHERE ticker = %s ORDER BY date DESC LIMIT 1",
                    [ticker],
                ).fetchone()

                status = "ok" if null_pct < 0.1 else "degraded" if null_pct < 0.5 else "poor"
                if non_null > 0:
                    indicators_ok += 1

                indicator_health[col] = {
                    "status": status,
                    "latest": round(latest_val[0], 4) if latest_val and latest_val[0] is not None else None,
                    "range": [round(min_v, 4) if min_v is not None else None, round(max_v, 4) if max_v is not None else None],
                    "nulls": nulls,
                    "null_pct": round(null_pct * 100, 1),
                }
            except Exception:
                indicator_health[col] = {"status": "error", "latest": None, "nulls": rows}

        total_cells = rows * len(INDICATORS) or 1
        score = max(0, 1.0 - (total_nulls / total_cells))

        return {
            "rows": rows,
            "date_range": [_safe_iso(min_d), _safe_iso(max_d)],
            "quality": _grade(score),
            "quality_score": round(score, 3),
            "indicators_computed": indicators_ok,
            "indicators_total": len(INDICATORS),
            "indicators_with_nulls": sum(1 for v in indicator_health.values() if v.get("nulls", 0) > 0),
            "indicator_health": indicator_health,
        }
    except Exception as e:
        logger.warning("audit technicals failed for %s: %s", ticker, e)
        return {"rows": 0, "quality": "error", "error": str(e)}


def _audit_fundamentals(db, ticker: str) -> dict:
    try:
        stats = db.execute(
            "SELECT COUNT(*), MIN(snapshot_date), MAX(snapshot_date) FROM fundamentals WHERE ticker = %s",
            [ticker]
        ).fetchone()
        rows, min_d, max_d = stats

        if rows == 0:
            return {"rows": 0, "quality": "critical", "quality_score": 0}

        key_fields = ["market_cap", "pe_ratio", "revenue", "profit_margin", "debt_to_equity"]
        latest = db.execute(
            "SELECT * FROM fundamentals WHERE ticker = %s ORDER BY snapshot_date DESC LIMIT 1",
            [ticker]
        ).fetchone()
        
        cols = [d[0] for d in db.execute("SELECT * FROM fundamentals LIMIT 0").description]
        data = dict(zip(cols, latest)) if latest else {}

        non_null_key = sum(1 for f in key_fields if data.get(f) is not None)
        score = non_null_key / len(key_fields) if key_fields else 0

        key_values = {}
        for f in key_fields:
            v = data.get(f)
            key_values[f] = round(v, 4) if isinstance(v, float) else v

        return {
            "rows": rows,
            "date_range": [_safe_iso(min_d), _safe_iso(max_d)],
            "quality": _grade(score),
            "quality_score": round(score, 3),
            "key_fields": key_values,
            "key_fields_present": f"{non_null_key}/{len(key_fields)}",
        }
    except Exception as e:
        logger.warning("audit fundamentals failed for %s: %s", ticker, e)
        return {"rows": 0, "quality": "error", "error": str(e)}


def _audit_news(db, ticker: str) -> dict:
    try:
        stats = db.execute(
            "SELECT COUNT(*), MIN(published_at), MAX(published_at), COUNT(DISTINCT source) FROM news_articles WHERE ticker = %s",
            [ticker]
        ).fetchone()
        rows, min_d, max_d, sources = stats

        source_list = []
        if rows > 0:
            src = db.execute(
                "SELECT source, COUNT(*) FROM news_articles WHERE ticker = %s GROUP BY source",
                [ticker]
            ).fetchall()
            source_list = [{"source": r[0], "count": r[1]} for r in src]

        score = min(1.0, rows / 5) if rows else 0
        return {
            "rows": rows,
            "date_range": [_safe_iso(min_d), _safe_iso(max_d)],
            "quality": _grade(score),
            "quality_score": round(score, 3),
            "source_count": sources,
            "sources": source_list,
        }
    except Exception as e:
        return {"rows": 0, "quality": "error", "error": str(e)}


def _audit_data_quality(tickers: list[str]) -> dict:
    if not tickers:
        return {"avg_score": 0, "gaps": [], "per_ticker": {}}
    from app.trading.watchlist import _snapshot_market_data, ban_ticker

    per_ticker, gaps, scores, purged_tickers = {}, [], [], []
    with get_db() as db:
        for ticker in tickers:
            try:
                cats = [
                    _audit_price_history(db, ticker),
                    _audit_technicals(db, ticker),
                    _audit_fundamentals(db, ticker),
                    _audit_news(db, ticker),
                ]
                cat_scores = [
                    c.get("quality_score", 0) for c in cats
                    if isinstance(c.get("quality_score"), (int, float)) and c.get("rows", 0) > 0
                ]
                avg = sum(cat_scores) / len(cat_scores) if cat_scores else 0
                scores.append(avg)
                per_ticker[ticker] = {"score": round(avg, 3)}
                missing = []
                for name, cat in zip(["price_history", "technicals", "fundamentals", "news"], cats):
                    if cat.get("rows", 0) == 0:
                        missing.append(name)
                if missing:
                    market_cap, price, volume = _snapshot_market_data(ticker)

                    is_junk = False
                    junk_reason = ""
                    if price is not None and price < 1.00:
                        is_junk = True
                        junk_reason = f"Penny stock (Price: ${price:.4f})"
                    elif market_cap is not None and market_cap > 0 and market_cap < 50_000_000:
                        is_junk = True
                        junk_reason = f"Micro-cap (Cap: ${market_cap:,.0f})"
                    elif price is not None and volume is not None and volume == 0:
                        is_junk = True
                        junk_reason = "Zero volume"

                    if is_junk:
                        ban_ticker(ticker, f"AutoResearch Context-Aware Pruning: {junk_reason}")
                        purged_tickers.append({"ticker": ticker, "reason": junk_reason})
                    else:
                        gaps.append({
                            "ticker": ticker,
                            "missing_sources": missing,
                            "recommendation": f"Re-collect {', '.join(missing)} for {ticker}",
                        })
            except Exception as e:
                scores.append(0)
                per_ticker[ticker] = {"score": 0, "error": str(e)}

    return {
        "avg_score": round(sum(scores) / len(scores), 3) if scores else 0,
        "gaps": gaps,
        "purged_tickers": purged_tickers,
        "per_ticker": per_ticker,
    }


def _audit_decisions(cycle_id: str, cycle_summary: dict) -> dict:
    buy = cycle_summary.get("buy_count", 0)
    sell = cycle_summary.get("sell_count", 0)
    hold = cycle_summary.get("hold_count", 0)
    total = buy + sell + hold
    issues = []
    outcome_stats = {}

    if total == 0:
        return {"score": 0, "issues": [{"issue": "No decisions produced", "severity": "critical"}]}

    try:
        with get_db() as db:
            rows = db.execute(
                "SELECT confidence FROM analysis_results WHERE cycle_id=%s AND confidence IS NOT NULL",
                [cycle_id],
            ).fetchall()
            if rows:
                confs = [r[0] for r in rows]
                if max(confs) - min(confs) < 10 and len(confs) >= 3:
                    issues.append({"issue": f"Uniform confidence ({min(confs)}-{max(confs)})", "severity": "info"})
                if sum(confs) / len(confs) < 40:
                    issues.append({"issue": f"Low avg confidence: {sum(confs) / len(confs):.0f}%", "severity": "warning"})
    except Exception:
        pass

    try:
        with get_db() as db:
            resolved = db.execute(
                """
                SELECT action, confidence, pnl_pct, outcome
                FROM decision_outcomes
                WHERE resolved_at IS NOT NULL AND outcome != 'CANCELED' AND resolved_at > CURRENT_TIMESTAMP - INTERVAL '30 days'
                ORDER BY resolved_at DESC LIMIT 100
                """,
            ).fetchall()

            if len(resolved) >= 3:
                wins = [r for r in resolved if r[3] == "WIN"]
                losses = [r for r in resolved if r[3] == "LOSS"]
                flats = [r for r in resolved if r[3] == "FLAT"]

                win_rate = len(wins) / len(resolved)
                avg_win_pnl = sum(r[2] for r in wins) / len(wins) if wins else 0
                avg_loss_pnl = sum(abs(r[2]) for r in losses) / len(losses) if losses else 0

                win_rate_score = min(1.0, win_rate)
                high_conf = [r for r in resolved if r[1] >= 70]
                low_conf = [r for r in resolved if r[1] < 50]

                if high_conf and low_conf:
                    high_win_rate = len([r for r in high_conf if r[3] == "WIN"]) / len(high_conf)
                    low_win_rate = len([r for r in low_conf if r[3] == "WIN"]) / len(low_conf)
                    calibration_gap = high_win_rate - low_win_rate
                    calibration_score = min(1.0, max(0.0, 0.5 + calibration_gap))
                elif high_conf:
                    calibration_score = len([r for r in high_conf if r[3] == "WIN"]) / len(high_conf)
                else:
                    calibration_score = 0.5

                if avg_loss_pnl > 0:
                    profit_factor = avg_win_pnl / avg_loss_pnl
                    risk_score = min(1.0, profit_factor / 2.0)
                elif avg_win_pnl > 0:
                    risk_score = 1.0
                else:
                    risk_score = 0.3

                score = (win_rate_score * 0.4 + calibration_score * 0.3 + risk_score * 0.3)

                outcome_stats = {
                    "total_resolved": len(resolved),
                    "wins": len(wins),
                    "losses": len(losses),
                    "flats": len(flats),
                    "win_rate": round(win_rate, 3),
                    "avg_win_pnl": round(avg_win_pnl, 2),
                    "avg_loss_pnl": round(avg_loss_pnl, 2),
                    "calibration_score": round(calibration_score, 3),
                    "risk_score": round(risk_score, 3),
                    "scoring_method": "outcome_based",
                }

                if win_rate < 0.40:
                    issues.append({"issue": f"Low win rate: {win_rate:.0%} ({len(wins)}/{len(resolved)})", "severity": "critical"})
                if avg_loss_pnl > 0 and avg_win_pnl < avg_loss_pnl:
                    issues.append({"issue": f"Avg loss ({avg_loss_pnl:.1f}%) > avg win ({avg_win_pnl:.1f}%)", "severity": "warning"})
                if calibration_score < 0.35:
                    issues.append({"issue": "Conviction miscalibrated", "severity": "warning"})

                try:
                    _backfill_cycle_summaries(db)
                except Exception as bf_err:
                    logger.debug("Summaries backfill failed (non-fatal): %s", bf_err)
            else:
                score = 0.5
                outcome_stats = {
                    "total_resolved": len(resolved),
                    "scoring_method": "cold_start",
                    "note": f"Need >= 3 resolved, have {len(resolved)}",
                }
                if buy + sell == 0 and total >= 3:
                    issues.append({"issue": "Zero BUY/SELL signals (cold start)", "severity": "info"})
                    score = 0.4
    except Exception as outcome_err:
        logger.warning("[AUTORESEARCH] Outcome-based scoring failed: %s", outcome_err)
        score = 0.5
        outcome_stats = {"scoring_method": "fallback_error", "error": str(outcome_err)}

    critical_issues = [i for i in issues if i.get("severity") == "critical"]
    if critical_issues:
        score *= max(0.5, 1.0 - len(critical_issues) * 0.2)

    return {
        "score": round(score, 3),
        "buy": buy,
        "sell": sell,
        "hold": hold,
        "issues": issues,
        "outcome_stats": outcome_stats,
    }


def _backfill_cycle_summaries(db) -> None:
    db.execute(
        """
        UPDATE cycle_summaries cs
        SET was_correct = CASE
                WHEN do.outcome = 'WIN' THEN TRUE
                WHEN do.outcome = 'LOSS' THEN FALSE
                ELSE NULL
            END,
            outcome_pnl = do.pnl_pct
        FROM decision_outcomes do
        WHERE cs.ticker = do.ticker AND cs.action = do.action
          AND do.resolved_at IS NOT NULL AND do.outcome != 'CANCELED' AND cs.was_correct IS NULL
          AND cs.cycle_date >= do.created_at - INTERVAL '1 day' AND cs.cycle_date <= do.created_at + INTERVAL '1 day'
        """
    )


def _audit_llm_traces(cycle_id: str) -> dict:
    issues = []
    try:
        from app.monitoring.llm_tracker import tracker
        stats = tracker.get_stats()
        total_calls = stats.get("total_calls", 0)
        failed = stats.get("failed_calls", 0)
        if total_calls == 0:
            return {"score": 0.5, "issues": []}
        fail_rate = failed / total_calls
        if fail_rate > 0.1:
            issues.append({"issue": f"LLM failure rate: {fail_rate:.0%}", "severity": "warning"})
        score = max(0, 1.0 - fail_rate * 2)
        return {
            "score": round(score, 3),
            "total_calls": total_calls,
            "failed_calls": failed,
            "fail_rate": round(fail_rate, 3),
            "issues": issues,
        }
    except Exception:
        return {"score": 0.5, "issues": []}


def _audit_performance(cycle_id: str, cycle_summary: dict) -> dict:
    return {
        "total_ms": cycle_summary.get("elapsed_ms", 0),
        "tickers_analyzed": cycle_summary.get("analysis_results_count", 0),
        "collector_ok": cycle_summary.get("collector_ok", 0),
        "collector_skipped": cycle_summary.get("collector_skipped", 0),
        "collector_error": cycle_summary.get("collector_error", 0),
        "trade_executed": cycle_summary.get("trade_executed", 0),
        "status": cycle_summary.get("status", "unknown"),
    }


def _audit_recovery() -> dict:
    try:
        from app.recovery.engine import recovery_engine
        return {
            **recovery_engine.get_stats(),
            "recent_events": recovery_engine.get_history(10),
        }
    except Exception:
        return {"total_failures": 0, "by_type": {}, "circuit_breakers_tripped": 0}


def _audit_execution_errors(cycle_id: str) -> list[dict]:
    try:
        with get_db() as db:
            rows = db.execute(
                "SELECT phase, error_type, error_message FROM execution_errors WHERE cycle_id = %s ORDER BY created_at DESC LIMIT 5",
                (cycle_id,),
            ).fetchall()
            return [{"phase": r[0], "error_type": r[1], "error_message": r[2]} for r in rows]
    except Exception as e:
        logger.debug("Failed to fetch execution errors: %s", e)
    return []


async def _reflect(audit_bundle: dict) -> dict:
    data_q = audit_bundle.get("data_quality", {})
    dec_q = audit_bundle.get("decision_quality", {})
    llm_a = audit_bundle.get("llm_analysis", {})
    perf = audit_bundle.get("performance", {})
    recovery = audit_bundle.get("recovery", {})
    sched = audit_bundle.get("schedule_health", {})
    exec_errs = audit_bundle.get("execution_errors", [])

    class DateTimeEncoder(json.JSONEncoder):
        def default(self, obj):
            if hasattr(obj, 'isoformat'): return obj.isoformat()
            return super().default(obj)

    def safe_dumps(obj):
        return json.dumps(obj, cls=DateTimeEncoder)

    sched_line = (
        f"Schedules: {sched.get('active_count', 0)} active, "
        f"avg interval {sched.get('avg_interval_hours', 'N/A')}h, "
        f"issues: {len(sched.get('issues', []))}"
    )

    prompt = (
        f"Review this trading cycle audit. Provide JSON with: summary, recommendations (list of 3), "
        f"urgent_data_gaps (ticker list), system_health (healthy/degraded/critical), "
        f"schedule_recommendation (optional string or null).\n\n"
        f"Data quality: {data_q.get('avg_score', 0):.0%}, gaps: {len(data_q.get('gaps', []))}\n"
        f"Decisions: {dec_q.get('buy', 0)} BUY, {dec_q.get('sell', 0)} SELL, {dec_q.get('hold', 0)} HOLD\n"
    )

    outcome_stats = dec_q.get("outcome_stats", {})
    if outcome_stats.get("scoring_method") == "outcome_based":
        prompt += (
            f"\n=== PREDICTION ACCURACY (last 30 days) ===\n"
            f"Resolved trades: {outcome_stats.get('total_resolved', 0)}\n"
            f"Win rate: {outcome_stats.get('win_rate', 0):.0%} "
            f"({outcome_stats.get('wins', 0)}W / {outcome_stats.get('losses', 0)}L / {outcome_stats.get('flats', 0)}F)\n"
            f"Avg win: +{outcome_stats.get('avg_win_pnl', 0):.1f}% | Avg loss: -{outcome_stats.get('avg_loss_pnl', 0):.1f}%\n"
            f"Conviction calibration: {outcome_stats.get('calibration_score', 0):.0%}\n"
            f"Risk score: {outcome_stats.get('risk_score', 0):.0%}\n"
            f"=== END PREDICTION ACCURACY ===\n\n"
        )
    else:
        prompt += f"Prediction accuracy: INSUFFICIENT DATA ({outcome_stats.get('note', 'cold start')})\n"

    prompt += (
        f"LLM calls: {llm_a.get('total_calls', 0)}, failures: {llm_a.get('failed_calls', 0)}\n"
        f"Duration: {perf.get('total_ms', 0) / 1000:.1f}s\n"
        f"Recovery failures: {recovery.get('total_failures', 0)}\n"
        f"{sched_line}\n"
        f"Data gaps: {safe_dumps(data_q.get('gaps', [])[:3])}\n"
        f"Issues: {safe_dumps(dec_q.get('issues', [])[:3])}\n"
        f"Schedule issues: {safe_dumps(sched.get('issues', [])[:3])}\n"
        f"System Execution Errors: {safe_dumps(exec_errs)}"
    )

    try:
        from app.services.vllm_client import llm, Priority
        response, tokens, elapsed = await llm.chat(
            system="You are a trading system auditor. Output valid JSON only.",
            user=prompt,
            temperature=0.1,
            max_tokens=500,
            agent_name="autoresearch_reflection",
            ticker="_system",
            priority=Priority.LOW
        )
        cleaned = response.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()
        result = json.loads(cleaned)
        result["tokens_used"] = tokens
        return result
    except Exception as e:
        logger.warning("[AUTORESEARCH] LLM reflection failed: %s", e)
        return _rule_based_reflection(audit_bundle)


def _rule_based_reflection(audit_bundle: dict) -> dict:
    data_q = audit_bundle.get("data_quality", {})
    dec_q = audit_bundle.get("decision_quality", {})
    recs = [g.get("recommendation", "") for g in data_q.get("gaps", [])[:2]]
    recs += [i.get("suggestion", "") for i in dec_q.get("issues", []) if i.get("suggestion")]
    health = "healthy" if data_q.get("avg_score", 1) >= 0.5 else "degraded" if data_q.get("avg_score", 1) >= 0.3 else "critical"
    return {
        "summary": f"Cycle completed with {len(data_q.get('gaps', []))} data gaps. Health: {health}.",
        "recommendations": [r for r in recs if r][:3],
        "urgent_data_gaps": [g["ticker"] for g in data_q.get("gaps", []) if g.get("missing_sources")][:5],
        "system_health": health,
        "fallback": True,
    }


def _store_lessons(reflection: dict, cycle_id: str):
    recs = reflection.get("recommendations", [])
    if not recs: return
    try:
        from app.cognition.lesson_store import add_lesson
        for rec in recs[:3]:
            if not rec or len(rec) < 10: continue
            add_lesson(
                text=rec[:120],
                metadata={
                    "session_id": f"autoresearch_{cycle_id[:8]}",
                    "round": 0, "score": 0, "status": "recommendation",
                    "source": "autoresearch", "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
    except Exception as e:
        logger.debug("Lesson store write failed: %s", e)


def _audit_triage(cycle_id: str, cycle_summary: dict, tickers: list[str]) -> dict:
    result = {
        "glance_count": 0, "standard_count": 0, "deep_count": 0,
        "neglect_count": 0, "avg_consecutive_skips": 0.0,
        "stale_tickers": [], "issues": []
    }
    try:
        triage = cycle_summary.get("triage", {})
        result["glance_count"] = triage.get("glance", 0)
        result["standard_count"] = triage.get("standard", 0)
        result["deep_count"] = triage.get("deep", 0)

        from app.pipeline.attention_tracker import get_attention_summary, get_neglect_flags
        attention = get_attention_summary(tickers)
        neglect = get_neglect_flags()
        result["neglect_count"] = len(neglect)

        skip_counts = [a.consecutive_skips for a in attention.values()]
        if skip_counts:
            result["avg_consecutive_skips"] = round(sum(skip_counts) / len(skip_counts), 1)

        cutoff_48h = datetime.now(timezone.utc) - timedelta(hours=48)
        for ticker, attn in attention.items():
            if attn.last_analyzed_at is None or attn.last_analyzed_at < cutoff_48h:
                result["stale_tickers"].append(ticker)

        if result["neglect_count"] > 0:
            result["issues"].append({
                "type": "neglect", "detail": f"{result['neglect_count']} tickers flagged as neglected",
                "tickers": [n["ticker"] for n in neglect[:5]]
            })

        if result["avg_consecutive_skips"] > 3:
            result["issues"].append({
                "type": "over_glancing",
                "detail": f"Average {result['avg_consecutive_skips']} consecutive Glance skips"
            })

        total = result["glance_count"] + result["standard_count"] + result["deep_count"]
        if total > 0 and result["glance_count"] / total > 0.7:
            result["issues"].append({
                "type": "too_many_glance",
                "detail": f"{result['glance_count']}/{total} tickers in Glance tier"
            })
    except Exception as e:
        logger.debug("Triage audit failed: %s", e)
        result["issues"].append({"type": "audit_error", "detail": str(e)})

    return result


def write_cycle_summary(cycle_id: str, analysis_results: list[dict]) -> None:
    if not analysis_results: return
    try:
        buy_count = sum(1 for r in analysis_results if r.get("action") == "BUY")
        sell_count = sum(1 for r in analysis_results if r.get("action") == "SELL")
        hold_count = sum(1 for r in analysis_results if r.get("action") == "HOLD")

        confidences = [r.get("confidence", 0) or 0 for r in analysis_results]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0

        top = max(analysis_results, key=lambda r: r.get("confidence", 0) or 0)
        top_confidence = top.get("confidence", 0) or 0
        top_ticker = top.get("ticker", "?") if top_confidence > 0 else None

        top_desc = f"Top pick: {top_ticker} @ {top_confidence}%." if top_ticker else "No high-confidence picks."
        lesson = f"{buy_count} BUY / {sell_count} SELL / {hold_count} HOLD. {top_desc}"

        with get_db() as db:
            db.execute(
                """INSERT INTO autoresearch_cycle_summaries
                (id, cycle_id, total_tickers, buy_count, sell_count, hold_count, avg_confidence, top_ticker, top_confidence, lesson_summary)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (cycle_id) DO UPDATE SET
                    total_tickers = EXCLUDED.total_tickers, buy_count = EXCLUDED.buy_count, sell_count = EXCLUDED.sell_count,
                    hold_count = EXCLUDED.hold_count, avg_confidence = EXCLUDED.avg_confidence, top_ticker = EXCLUDED.top_ticker,
                    top_confidence = EXCLUDED.top_confidence, lesson_summary = EXCLUDED.lesson_summary""",
                (
                    f"cs-{uuid.uuid4().hex[:12]}", cycle_id, len(analysis_results), buy_count, sell_count, hold_count,
                    round(avg_conf, 1), top_ticker, top_confidence, lesson[:500]
                )
            )
    except Exception as e:
        logger.warning("cycle_summaries write failed (non-fatal): %s", e)


def _generate_directives(reflection: dict, cycle_id: str, triage_audit: dict) -> None:
    directives_created = 0
    recs = reflection.get("recommendations", [])
    with get_db() as db:
        for rec in recs[:3]:
            if not rec or len(rec) < 15: continue
            severity = "info"
            rec_lower = rec.lower()
            if any(w in rec_lower for w in ["critical", "urgent", "immediate", "failing"]):
                severity = "critical"
            elif any(w in rec_lower for w in ["warn", "degrad", "poor", "missing"]):
                severity = "warning"

            directive_id = f"dir-{uuid.uuid4().hex[:12]}"
            db.execute(
                """INSERT INTO cycle_directives (id, cycle_id, directive_type, directive_text, severity, status, expires_after)
                VALUES (%s, %s, 'recommendation', %s, %s, 'active', 5) ON CONFLICT DO NOTHING""",
                [directive_id, cycle_id, rec[:300], severity]
            )
            directives_created += 1

        for issue in triage_audit.get("issues", [])[:3]:
            directive_id = f"dir-{uuid.uuid4().hex[:12]}"
            target_ticker = None
            tickers_list = issue.get("tickers", [])
            if tickers_list: target_ticker = tickers_list[0]
            severity = "warning" if issue["type"] in ("neglect", "over_glancing") else "info"
            db.execute(
                """INSERT INTO cycle_directives (id, cycle_id, directive_type, directive_text, target_ticker, severity, status, expires_after)
                VALUES (%s, %s, %s, %s, %s, %s, 'active', 3) ON CONFLICT DO NOTHING""",
                [directive_id, cycle_id, f"triage_{issue['type']}", issue["detail"][:300], target_ticker, severity]
            )
            directives_created += 1

        urgent_gaps = reflection.get("urgent_data_gaps", [])
        for ticker in urgent_gaps[:3]:
            directive_id = f"dir-{uuid.uuid4().hex[:12]}"
            db.execute(
                """INSERT INTO cycle_directives (id, cycle_id, directive_type, directive_text, target_ticker, severity, status, expires_after)
                VALUES (%s, %s, 'data_gap', %s, %s, 'warning', 'active', 3) ON CONFLICT DO NOTHING""",
                [directive_id, cycle_id, f"Critical data gap for {ticker}", ticker]
            )
            directives_created += 1

        sched_rec = reflection.get("schedule_recommendation")
        if sched_rec and isinstance(sched_rec, str) and len(sched_rec) >= 10:
            directive_id = f"dir-{uuid.uuid4().hex[:12]}"
            db.execute(
                """INSERT INTO cycle_directives (id, cycle_id, directive_type, directive_text, severity, status, expires_after)
                VALUES (%s, %s, 'schedule_recommendation', %s, 'info', 'active', 3) ON CONFLICT DO NOTHING""",
                [directive_id, cycle_id, sched_rec[:300]]
            )
            directives_created += 1


def _expire_old_directives() -> None:
    try:
        with get_db() as db:
            db.execute("UPDATE cycle_directives SET expires_after = expires_after - 1 WHERE status = 'active' AND expires_after > 0")
            db.execute("UPDATE cycle_directives SET status = 'expired', resolved_at = CURRENT_TIMESTAMP WHERE status = 'active' AND expires_after <= 0")
    except Exception as e:
        logger.debug("Directive expiry failed: %s", e)


def _audit_schedule_health() -> dict:
    result = {
        "active_count": 0, "total_count": 0, "avg_interval_hours": None,
        "has_premarket": False, "stuck_schedules": [], "issues": []
    }
    try:
        with get_db() as db:
            rows = db.execute(
                "SELECT id, name, schedule_type, cron_expression, interval_hours, is_active, last_run_at, next_run_at FROM cycle_schedules ORDER BY is_active DESC"
            ).fetchall()

        result["total_count"] = len(rows)
        active_rows = [r for r in rows if r[5]]
        result["active_count"] = len(active_rows)

        if result["active_count"] == 0:
            result["issues"].append({
                "type": "no_active_schedules", "severity": "critical",
                "detail": "Bot has NO active schedules."
            })
            return result

        intervals = [r[4] for r in active_rows if r[2] == "interval" and r[4] is not None and r[4] > 0]
        if intervals:
            result["avg_interval_hours"] = round(sum(intervals) / len(intervals), 1)

        for r in active_rows:
            if r[2] == "cron" and r[3]:
                cron_parts = r[3].split()
                if len(cron_parts) >= 2:
                    try:
                        hour = int(cron_parts[1])
                        if 7 <= hour <= 9: result["has_premarket"] = True
                    except ValueError: pass

        now = datetime.now(timezone.utc)
        for r in active_rows:
            if r[2] == "interval" and r[4] and r[6]:
                last_run = r[6]
                if hasattr(last_run, "timestamp"):
                    expected_gap = r[4] * 3600
                    actual_gap = (now - last_run).total_seconds()
                    if actual_gap > expected_gap * 2.5:
                        result["stuck_schedules"].append({
                            "id": r[0], "name": r[1], "expected_interval_h": r[4],
                            "actual_gap_h": round(actual_gap / 3600, 1)
                        })

        if result["stuck_schedules"]:
            result["issues"].append({
                "type": "stuck_schedules", "severity": "warning",
                "detail": f"{len(result['stuck_schedules'])} schedule(s) appear stuck",
                "schedules": [s["name"] for s in result["stuck_schedules"]]
            })

        if not result["has_premarket"] and result["active_count"] > 0:
            result["issues"].append({
                "type": "no_premarket", "severity": "info",
                "detail": "No pre-market (7-9 AM ET) schedule found."
            })
    except Exception as e:
        logger.debug("Schedule health audit failed: %s", e)
        result["issues"].append({"type": "audit_error", "detail": str(e)})

    return result


async def _resolve_data_gaps(gaps: list[dict], cycle_id: str) -> dict:
    if not gaps: return {"resolved": 0, "failed": 0, "banned": 0}
    resolved = 0
    failed = 0
    banned = 0

    COLLECTOR_MAP = {
        "news": ("app.collectors.news_collector", "collect_for_ticker"),
        "price_history": ("app.collectors.yfinance_collector", "collect_price_history"),
        "technicals": ("app.processors.technical_processor", "compute_technicals"),
        "fundamentals": ("app.collectors.yfinance_collector", "collect_fundamentals"),
    }

    for gap in gaps[:5]:
        ticker = gap.get("ticker", "")
        missing = gap.get("missing_sources", [])
        if not ticker or not missing: continue

        try:
            with get_db() as db:
                occurrence_row = db.execute(
                    "SELECT COUNT(*) FROM autoresearch_reports WHERE status = 'done' AND data_gaps LIKE %s",
                    [f'%"{ticker}"%']
                ).fetchone()

            if occurrence_row and occurrence_row[0] >= 5:
                from app.trading.watchlist import ban_ticker
                ban_ticker(ticker, f"AutoResearch: persistent data gap across {occurrence_row[0]} cycles")
                banned += 1
                continue
        except Exception as ban_err:
            logger.debug("Ban check failed for %s: %s", ticker, ban_err)

        import importlib
        for source in missing:
            collector_info = COLLECTOR_MAP.get(source)
            if not collector_info: continue
            module_path, func_name = collector_info
            try:
                mod = importlib.import_module(module_path)
                collect_fn = getattr(mod, func_name)

                if asyncio.iscoroutinefunction(collect_fn):
                    await asyncio.wait_for(collect_fn(ticker), timeout=30.0)
                else:
                    collect_fn(ticker)
                resolved += 1
            except Exception as coll_err:
                failed += 1
                logger.warning("Gap resolution failed: %s/%s — %s", ticker, source, coll_err)

    return {"resolved": resolved, "failed": failed, "banned": banned}
