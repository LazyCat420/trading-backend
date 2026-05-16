"""
AutoResearch Orchestrator — Post-cycle holistic audit + LLM reflection.

Runs automatically after every pipeline cycle completes. Collects audit
data across all dimensions (data quality, decisions, LLM traces,
performance, recovery) and feeds it to an LLM reflection pass.
"""

import json
import logging
import uuid
from datetime import datetime, timezone

from app.db.connection import get_db

logger = logging.getLogger(__name__)


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
        from app.db.connection import get_db

        with get_db() as db:
            db.execute(
                f"UPDATE autoresearch_reports SET {', '.join(updates)} WHERE id = %s",
                params,
            )
    except Exception as e:
        logger.debug("Failed to update ar state: %s", e)


def get_autoresearch_status() -> dict:
    try:
        from app.db.connection import get_db

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
                stale_count = db.execute(
                    "UPDATE autoresearch_reports SET status = 'stale' "
                    "WHERE status = 'running' AND created_at < NOW() - INTERVAL '30 minutes'"
                )
                # Log if we cleaned up any
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

        # Triage audit (new: evaluate triage distribution + attention health)
        _update_ar_state(report_id, phase="triage_audit")
        triage_audit = _audit_triage(cycle_id, cycle_summary, tickers)
        audit_bundle["triage_audit"] = triage_audit

        # Schedule health audit (can the bot wake itself up?)
        _update_ar_state(report_id, phase="schedule_audit")
        schedule_health = _audit_schedule_health()
        audit_bundle["schedule_health"] = schedule_health

        _update_ar_state(report_id, phase="reflection")
        reflection = await _reflect(audit_bundle)

        data_score = data_quality.get("avg_score", 0) * 100
        decision_score = decision_quality.get("score", 0) * 100
        llm_score = llm_analysis.get("score", 0) * 100
        overall = (data_score + decision_score + llm_score) / 3

        # ── Degenerate score detection (F-04 fix) ──
        degenerate_subs = []
        if data_score == 0.0:
            degenerate_subs.append("data")
        if decision_score == 0.0:
            degenerate_subs.append("decision")
        if llm_score == 0.0:
            degenerate_subs.append("llm")
        if degenerate_subs:
            logger.warning(
                "[AUTORESEARCH] DEGENERATE SCORES: %s at 0.0 — "
                "data=%.1f, decision=%.1f, llm=%.1f. Flagging as anomaly.",
                ", ".join(degenerate_subs),
                data_score,
                decision_score,
                llm_score,
            )
            reflection["anomaly"] = True
            reflection["anomaly_detail"] = (
                f"Degenerate sub-scores at 0.0: {', '.join(degenerate_subs)}"
            )

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

            # ── Global Critical Warning ──
            if reflection.get("system_health") == "critical":
                from app.services.session_profile import profile_memory

                summary = reflection.get(
                    "summary", "Critical health detected by autoresearch."
                )
                profile_memory.add_agent_note(
                    f"⚠️ AUTORESEARCH CRITICAL WARNING (Cycle {cycle_id[:8]}): {summary}"
                )
                logger.warning(
                    "[AUTORESEARCH] Pushed critical warning to global profile memory."
                )
        except Exception as ls_err:
            logger.warning("[AUTORESEARCH] Lesson store write failed: %s", ls_err)

        # ── Auto-resolve detected data gaps (F-05 fix) ──
        _update_ar_state(report_id, phase="gap_resolution")
        try:
            gap_result = await _resolve_data_gaps(
                data_quality.get("gaps", []), cycle_id
            )
            logger.info(
                "[AUTORESEARCH] Data gap resolution: resolved=%d, failed=%d, banned=%d",
                gap_result.get("resolved", 0),
                gap_result.get("failed", 0),
                gap_result.get("banned", 0),
            )
        except Exception as gap_err:
            logger.warning("[AUTORESEARCH] Data gap resolution failed: %s", gap_err)

        # Trigger Evolutionary Debate Council (BOUNDED — no more fire-and-forget)
        _EVO_TIMEOUT = 60  # seconds
        try:
            from app.pipeline.analysis.evolution_router import router

            await asyncio.wait_for(router.run_router(cycle_id), timeout=_EVO_TIMEOUT)
            logger.info("[AUTORESEARCH] Evolution Router completed.")
        except asyncio.TimeoutError:
            logger.warning("[AUTORESEARCH] Evolution Router timeout (%ds) — cancelling.", _EVO_TIMEOUT)
        except Exception as e:
            logger.error("[AUTORESEARCH] Failed to trigger Evolution Router: %s", e)

        logger.info("[AUTORESEARCH] Complete: score=%.1f", overall)

        # Generate cross-cycle directives from reflection recommendations
        _update_ar_state(report_id, phase="directives")
        try:
            _generate_directives(reflection, cycle_id, triage_audit)
            _expire_old_directives()
            logger.info("[AUTORESEARCH] Directive generation + expiry complete")
        except Exception as dir_err:
            logger.warning("[AUTORESEARCH] Directive generation failed: %s", dir_err)

        # Trigger Benchmark Agent for Constitution review (BOUNDED)
        _BENCH_TIMEOUT = 60  # seconds
        try:
            from app.pipeline.analysis.benchmark_agent import run_benchmark_agent

            await asyncio.wait_for(run_benchmark_agent(cycle_id), timeout=_BENCH_TIMEOUT)
            logger.info("[AUTORESEARCH] Benchmark Agent completed.")
        except asyncio.TimeoutError:
            logger.warning("[AUTORESEARCH] Benchmark Agent timeout (%ds) — cancelling.", _BENCH_TIMEOUT)
        except Exception as bench_err:
            logger.error(
                "[AUTORESEARCH] Failed to trigger Benchmark Agent: %s", bench_err
            )

        # ── Record subsystem benchmarks for trend analysis ──
        try:
            from app.pipeline.subsystem_benchmarks import record_all

            record_all(cycle_id)
        except Exception as sb_err:
            logger.warning(
                "[AUTORESEARCH] Subsystem benchmark recording failed: %s", sb_err
            )

        # ── Check probation fixes for rollback ──
        try:
            from app.cognition.evolution.rollback_monitor import check_probation_fixes

            rollback_summary = check_probation_fixes(cycle_id)
            if rollback_summary.get("rolled_back", 0) > 0:
                logger.warning(
                    "[AUTORESEARCH] Rolled back %d degrading fixes!",
                    rollback_summary["rolled_back"],
                )
        except Exception as rb_err:
            logger.warning("[AUTORESEARCH] Rollback monitor failed: %s", rb_err)

        _update_ar_state(report_id, phase="done")
        return {"id": report_id, "overall_score": round(overall, 1), "status": "done"}

    except BaseException as e:
        logger.error("[AUTORESEARCH] Failed: %s", e, exc_info=True)
        _update_ar_state(report_id, error=str(e), phase="error")
        try:
            with get_db() as db:
                db.execute(
                    "UPDATE autoresearch_reports SET status='error' WHERE id=%s",
                    (report_id,),
                )
        except Exception:
            pass
        return {"error": str(e)}
    finally:
        # Only set running=False (status=done) if we didn't error out
        try:
            with get_db() as db:
                status = db.execute("SELECT status FROM autoresearch_reports WHERE id=%s", [report_id]).fetchone()
                if status and status[0] == 'running':
                    _update_ar_state(report_id, running=False)
        except:
            pass


async def run_partial_autoresearch(cycle_id: str, tickers: list[str]) -> dict:
    """Mid-cycle autoresearch trigger for dynamic self-correction and data filling."""
    logger.info(
        f"[AUTORESEARCH] Running partial mid-cycle autoresearch for {len(tickers)} tickers."
    )

    data_quality = _audit_data_quality(tickers)

    # If there are gaps, we could trigger collectors here.
    if data_quality.get("gaps"):
        logger.warning(
            f"[AUTORESEARCH] Mid-cycle gaps found: {len(data_quality['gaps'])}"
        )

    return {"status": "partial_done", "data_quality": data_quality}


def _audit_data_quality(tickers: list[str]) -> dict:
    if not tickers:
        return {"avg_score": 0, "gaps": [], "per_ticker": {}}
    from app.routers.data_audit import (
        _audit_price_history,
        _audit_technicals,
        _audit_fundamentals,
        _audit_news,
    )
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
                    c.get("quality_score", 0)
                    for c in cats
                    if isinstance(c.get("quality_score"), (int, float))
                    and c.get("rows", 0) > 0
                ]
                avg = sum(cat_scores) / len(cat_scores) if cat_scores else 0
                scores.append(avg)
                per_ticker[ticker] = {"score": round(avg, 3)}
                missing = []
                for name, cat in zip(
                    ["price_history", "technicals", "fundamentals", "news"], cats
                ):
                    if cat.get("rows", 0) == 0:
                        missing.append(name)
                if missing:
                    # Context-Aware Pruning
                    market_cap, price, volume = _snapshot_market_data(ticker)

                    is_junk = False
                    junk_reason = ""
                    if price is not None and price < 1.00:
                        is_junk = True
                        junk_reason = f"Penny stock (Price: ${price:.4f})"
                    elif (
                        market_cap is not None
                        and market_cap > 0
                        and market_cap < 50_000_000
                    ):
                        is_junk = True
                        junk_reason = f"Micro-cap (Cap: ${market_cap:,.0f})"
                    elif price is not None and volume is not None and volume == 0:
                        is_junk = True
                        junk_reason = "Zero volume"

                    if is_junk:
                        ban_ticker(
                            ticker, f"AutoResearch Context-Aware Pruning: {junk_reason}"
                        )
                        purged_tickers.append({"ticker": ticker, "reason": junk_reason})
                        logger.warning(
                            "[AUTORESEARCH] Banned junk stock %s (%s) instead of treating as data gap",
                            ticker,
                            junk_reason,
                        )
                    else:
                        gaps.append(
                            {
                                "ticker": ticker,
                                "missing_sources": missing,
                                "recommendation": f"Re-collect {', '.join(missing)} for {ticker}",
                            }
                        )
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
    """Score decision quality using actual trade outcomes as ground truth.

    Scoring weights (when sufficient data exists):
      - Win rate:               40%  (did BUY/SELL decisions actually profit?)
      - Conviction calibration: 30%  (do high-confidence picks win more often?)
      - Risk management:        30%  (avg win > avg loss? Losses contained?)

    Falls back to a neutral 50% score during cold start (< 3 resolved trades).
    """
    buy = cycle_summary.get("buy_count", 0)
    sell = cycle_summary.get("sell_count", 0)
    hold = cycle_summary.get("hold_count", 0)
    total = buy + sell + hold
    issues = []
    outcome_stats = {}

    if total == 0:
        return {
            "score": 0,
            "issues": [{"issue": "No decisions produced", "severity": "critical"}],
        }

    # ── Confidence distribution analysis (unchanged) ──
    try:
        with get_db() as db:
            rows = db.execute(
                "SELECT confidence FROM analysis_results WHERE cycle_id=%s AND confidence IS NOT NULL",
                [cycle_id],
            ).fetchall()
            if rows:
                confs = [r[0] for r in rows]
                if max(confs) - min(confs) < 10 and len(confs) >= 3:
                    issues.append(
                        {
                            "issue": f"Uniform confidence ({min(confs)}-{max(confs)})",
                            "severity": "info",
                        }
                    )
                if sum(confs) / len(confs) < 40:
                    issues.append(
                        {
                            "issue": f"Low avg confidence: {sum(confs) / len(confs):.0f}%",
                            "severity": "warning",
                        }
                    )
    except Exception:
        pass

    # ── Ground-truth scoring from decision_outcomes ──
    try:
        with get_db() as db:
            # Fetch resolved outcomes from the last 30 days
            resolved = db.execute(
                """
                SELECT action, confidence, pnl_pct, outcome
                FROM decision_outcomes
                WHERE resolved_at IS NOT NULL
                  AND outcome != 'CANCELED'
                  AND resolved_at > CURRENT_TIMESTAMP - INTERVAL '30 days'
                ORDER BY resolved_at DESC
                LIMIT 100
                """,
            ).fetchall()

            if len(resolved) >= 3:
                # Enough data for outcome-based scoring
                wins = [r for r in resolved if r[3] == "WIN"]
                losses = [r for r in resolved if r[3] == "LOSS"]
                flats = [r for r in resolved if r[3] == "FLAT"]

                win_rate = len(wins) / len(resolved)
                avg_win_pnl = (
                    sum(r[2] for r in wins) / len(wins) if wins else 0
                )
                avg_loss_pnl = (
                    sum(abs(r[2]) for r in losses) / len(losses) if losses else 0
                )

                # ── Component 1: Win Rate (40%) ──
                # Score: win_rate directly (50% = 0.5, 60% = 0.6, etc.)
                win_rate_score = min(1.0, win_rate)

                # ── Component 2: Conviction Calibration (30%) ──
                # High-confidence picks (>=70) should win more than low-confidence (<50)
                high_conf = [r for r in resolved if r[1] >= 70]
                low_conf = [r for r in resolved if r[1] < 50]

                if high_conf and low_conf:
                    high_win_rate = len([r for r in high_conf if r[3] == "WIN"]) / len(high_conf)
                    low_win_rate = len([r for r in low_conf if r[3] == "WIN"]) / len(low_conf)
                    # Calibration is good if high-conf wins more than low-conf
                    calibration_gap = high_win_rate - low_win_rate
                    calibration_score = min(1.0, max(0.0, 0.5 + calibration_gap))
                elif high_conf:
                    # Only high-conf trades: score based on their win rate
                    calibration_score = len([r for r in high_conf if r[3] == "WIN"]) / len(high_conf)
                else:
                    # No high-conf trades at all: neutral
                    calibration_score = 0.5

                # ── Component 3: Risk Management (30%) ──
                # Good risk mgmt: avg_win > avg_loss (profit factor > 1)
                if avg_loss_pnl > 0:
                    profit_factor = avg_win_pnl / avg_loss_pnl
                    risk_score = min(1.0, profit_factor / 2.0)  # PF of 2.0 = perfect
                elif avg_win_pnl > 0:
                    risk_score = 1.0  # Wins with no losses = perfect risk mgmt
                else:
                    risk_score = 0.3  # No wins, no meaningful losses

                # ── Composite Score ──
                score = (
                    win_rate_score * 0.4
                    + calibration_score * 0.3
                    + risk_score * 0.3
                )

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

                # Add targeted issues based on outcomes
                if win_rate < 0.40:
                    issues.append(
                        {"issue": f"Low win rate: {win_rate:.0%} ({len(wins)}/{len(resolved)})", "severity": "critical"}
                    )
                if avg_loss_pnl > 0 and avg_win_pnl < avg_loss_pnl:
                    issues.append(
                        {"issue": f"Avg loss ({avg_loss_pnl:.1f}%) > avg win ({avg_win_pnl:.1f}%)", "severity": "warning"}
                    )
                if calibration_score < 0.35:
                    issues.append(
                        {"issue": "Conviction miscalibrated: high-confidence picks don't outperform", "severity": "warning"}
                    )

                logger.info(
                    "[AUTORESEARCH] Outcome-based scoring: win_rate=%.1f%% cal=%.2f risk=%.2f → score=%.3f (%d trades)",
                    win_rate * 100, calibration_score, risk_score, score, len(resolved),
                )

                # ── Backfill cycle_summaries with ground truth ──
                try:
                    _backfill_cycle_summaries(db)
                except Exception as bf_err:
                    logger.debug("[AUTORESEARCH] cycle_summaries backfill failed (non-fatal): %s", bf_err)

            else:
                # Cold start: not enough resolved trades for outcome scoring
                # Use neutral 50% baseline instead of the broken action-ratio formula
                score = 0.5
                outcome_stats = {
                    "total_resolved": len(resolved),
                    "scoring_method": "cold_start",
                    "note": f"Need >= 3 resolved trades, have {len(resolved)}",
                }
                if buy + sell == 0 and total >= 3:
                    issues.append({"issue": "Zero BUY/SELL signals (cold start)", "severity": "info"})
                    score = 0.4  # Mild penalty for total inaction during cold start

    except Exception as outcome_err:
        logger.warning("[AUTORESEARCH] Outcome-based scoring failed, using cold-start fallback: %s", outcome_err)
        score = 0.5
        outcome_stats = {"scoring_method": "fallback_error", "error": str(outcome_err)}

    # Apply issue penalty (max 30% reduction)
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
    """Backfill was_correct and outcome_pnl in cycle_summaries from decision_outcomes.

    Joins on ticker + close cycle_id match to connect predictions with outcomes.
    Only updates rows where was_correct is still NULL.
    """
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
        WHERE cs.ticker = do.ticker
          AND cs.action = do.action
          AND do.resolved_at IS NOT NULL
          AND do.outcome != 'CANCELED'
          AND cs.was_correct IS NULL
          AND cs.cycle_date >= do.created_at - INTERVAL '1 day'
          AND cs.cycle_date <= do.created_at + INTERVAL '1 day'
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
            issues.append(
                {"issue": f"LLM failure rate: {fail_rate:.0%}", "severity": "warning"}
            )
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
            return [
                {"phase": r[0], "error_type": r[1], "error_message": r[2]} for r in rows
            ]
    except Exception as e:
        logger.debug("[AUTORESEARCH] Failed to fetch execution_errors: %s", e)
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
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            return super().default(obj)

    def safe_dumps(obj):
        return json.dumps(obj, cls=DateTimeEncoder)

    # Build schedule status line for the reflection prompt
    sched_line = (
        f"Schedules: {sched.get('active_count', 0)} active, "
        f"avg interval {sched.get('avg_interval_hours', 'N/A')}h, "
        f"issues: {len(sched.get('issues', []))}"
    )

    prompt = (
        f"Review this trading cycle audit. Provide JSON with: summary, recommendations (list of 3), "
        f"urgent_data_gaps (ticker list), system_health (healthy/degraded/critical), "
        f"schedule_recommendation (optional string: suggest cadence changes like 'increase to 2h' or 'add pre-market run', "
        f"or null if current schedule is fine).\n\n"
        f"Data quality: {data_q.get('avg_score', 0):.0%}, gaps: {len(data_q.get('gaps', []))}\n"
        f"Decisions: {dec_q.get('buy', 0)} BUY, {dec_q.get('sell', 0)} SELL, {dec_q.get('hold', 0)} HOLD\n"
    )

    # ── Inject prediction accuracy scorecard (ground truth from decision_outcomes) ──
    outcome_stats = dec_q.get("outcome_stats", {})
    if outcome_stats.get("scoring_method") == "outcome_based":
        prompt += (
            f"\n=== PREDICTION ACCURACY (last 30 days) ===\n"
            f"Resolved trades: {outcome_stats.get('total_resolved', 0)}\n"
            f"Win rate: {outcome_stats.get('win_rate', 0):.0%} "
            f"({outcome_stats.get('wins', 0)}W / {outcome_stats.get('losses', 0)}L / {outcome_stats.get('flats', 0)}F)\n"
            f"Avg win: +{outcome_stats.get('avg_win_pnl', 0):.1f}%  |  Avg loss: -{outcome_stats.get('avg_loss_pnl', 0):.1f}%\n"
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
        f"System Execution Errors (CRITICAL): {safe_dumps(exec_errs)}"
    )
    try:
        from app.services.vllm_client import llm, Priority

        try:
            import asyncio
            response, tokens, elapsed = await asyncio.wait_for(
                llm.chat(
                    system="You are a trading system auditor. Output valid JSON only.",
                    user=prompt,
                    temperature=0.1,
                    max_tokens=500,
                    agent_name="autoresearch_reflection",
                    ticker="_system",
                    priority=Priority.LOW,
                ),
                timeout=60.0
            )
        except asyncio.TimeoutError:
            logger.warning("[AUTORESEARCH] LLM reflection timed out after 60s, falling back.")
            return _rule_based_reflection(audit_bundle)
            
        cleaned = response.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()
        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError:
            result = {
                "summary": cleaned[:500],
                "recommendations": [],
                "system_health": "unknown",
            }
        result["tokens_used"] = tokens
        return result
    except Exception as e:
        logger.warning("[AUTORESEARCH] LLM reflection failed: %s", e)
        return _rule_based_reflection(audit_bundle)


def _rule_based_reflection(audit_bundle: dict) -> dict:
    data_q = audit_bundle.get("data_quality", {})
    dec_q = audit_bundle.get("decision_quality", {})
    recs = [g.get("recommendation", "") for g in data_q.get("gaps", [])[:2]]
    recs += [
        i.get("suggestion", "") for i in dec_q.get("issues", []) if i.get("suggestion")
    ]
    health = (
        "healthy"
        if data_q.get("avg_score", 1) >= 0.5
        else "degraded"
        if data_q.get("avg_score", 1) >= 0.3
        else "critical"
    )
    return {
        "summary": f"Cycle completed with {len(data_q.get('gaps', []))} data gaps. Health: {health}.",
        "recommendations": [r for r in recs if r][:3],
        "urgent_data_gaps": [
            g["ticker"] for g in data_q.get("gaps", []) if g.get("missing_sources")
        ][:5],
        "system_health": health,
        "fallback": True,
    }


def _store_lessons(reflection: dict, cycle_id: str):
    recs = reflection.get("recommendations", [])
    if not recs:
        return
    try:
        from app.cognition.lesson_store import add_lesson

        for rec in recs[:3]:
            if not rec or len(rec) < 10:
                continue
            add_lesson(
                text=rec[:120],
                metadata={
                    "session_id": f"autoresearch_{cycle_id[:8]}",
                    "round": 0,
                    "score": 0,
                    "status": "recommendation",
                    "source": "autoresearch",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
    except Exception as e:
        logger.debug("[AUTORESEARCH] Lesson store write failed: %s", e)


# ═══════════════════════════════════════════════════════════════════
# TRIAGE AUDIT — Analyze attention tracker + triage distribution
# ═══════════════════════════════════════════════════════════════════


def _audit_triage(cycle_id: str, cycle_summary: dict, tickers: list[str]) -> dict:
    """Audit the triage distribution and attention tracker health."""
    result = {
        "glance_count": 0,
        "standard_count": 0,
        "deep_count": 0,
        "neglect_count": 0,
        "avg_consecutive_skips": 0.0,
        "stale_tickers": [],
        "issues": [],
    }
    try:
        # Get triage distribution from cycle summary
        triage = cycle_summary.get("triage", {})
        result["glance_count"] = triage.get("glance", 0)
        result["standard_count"] = triage.get("standard", 0)
        result["deep_count"] = triage.get("deep", 0)

        # Analyze attention tracker state
        from app.pipeline.attention_tracker import (
            get_attention_summary,
            get_neglect_flags,
        )

        attention = get_attention_summary(tickers)
        neglect = get_neglect_flags()
        result["neglect_count"] = len(neglect)

        # Calculate average consecutive skips
        skip_counts = [a.consecutive_skips for a in attention.values()]
        if skip_counts:
            result["avg_consecutive_skips"] = round(
                sum(skip_counts) / len(skip_counts), 1
            )

        # Identify stale tickers (not analyzed in 48+ hours)
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        cutoff_48h = now - timedelta(hours=48)
        for ticker, attn in attention.items():
            if attn.last_analyzed_at is None or attn.last_analyzed_at < cutoff_48h:
                result["stale_tickers"].append(ticker)

        # Generate issues
        if result["neglect_count"] > 0:
            result["issues"].append(
                {
                    "type": "neglect",
                    "detail": f"{result['neglect_count']} tickers flagged as neglected",
                    "tickers": [n["ticker"] for n in neglect[:5]],
                }
            )

        if result["avg_consecutive_skips"] > 3:
            result["issues"].append(
                {
                    "type": "over_glancing",
                    "detail": f"Average {result['avg_consecutive_skips']} consecutive Glance skips — "
                    "some tickers may not be getting enough attention",
                }
            )

        total = result["glance_count"] + result["standard_count"] + result["deep_count"]
        if total > 0 and result["glance_count"] / total > 0.7:
            result["issues"].append(
                {
                    "type": "too_many_glance",
                    "detail": f"{result['glance_count']}/{total} tickers in Glance tier — "
                    "system may be too conservative",
                }
            )

    except Exception as e:
        logger.debug("[AUTORESEARCH] Triage audit failed: %s", e)
        result["issues"].append({"type": "audit_error", "detail": str(e)})

    return result


# ═══════════════════════════════════════════════════════════════════
# CYCLE SUMMARY WRITER — Distill end-of-cycle stats for warm-start
# ═══════════════════════════════════════════════════════════════════


def write_cycle_summary(cycle_id: str, analysis_results: list[dict]) -> None:
    """Aggregate cycle analysis results into cycle_summaries for warm-start.

    Args:
        cycle_id: The cycle identifier.
        analysis_results: List of dicts with 'ticker', 'action', 'confidence' keys.
    """
    if not analysis_results:
        return

    try:
        buy_count = sum(1 for r in analysis_results if r.get("action") == "BUY")
        sell_count = sum(1 for r in analysis_results if r.get("action") == "SELL")
        hold_count = sum(1 for r in analysis_results if r.get("action") == "HOLD")

        confidences = [r.get("confidence", 0) or 0 for r in analysis_results]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0

        # Find top ticker by confidence (guard: only meaningful if confidence > 0)
        top = max(analysis_results, key=lambda r: r.get("confidence", 0) or 0)
        top_confidence = top.get("confidence", 0) or 0
        top_ticker = top.get("ticker", "?") if top_confidence > 0 else None

        # Generate a 1-line lesson summary
        top_desc = (
            f"Top pick: {top_ticker} @ {top_confidence}%."
            if top_ticker
            else "No high-confidence picks."
        )
        lesson = f"{buy_count} BUY / {sell_count} SELL / {hold_count} HOLD. {top_desc}"

        with get_db() as db:
            db.execute(
                """INSERT INTO autoresearch_cycle_summaries
                (id, cycle_id, total_tickers, buy_count, sell_count, hold_count,
                 avg_confidence, top_ticker, top_confidence, lesson_summary)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (cycle_id) DO UPDATE SET
                    total_tickers = EXCLUDED.total_tickers,
                    buy_count = EXCLUDED.buy_count,
                    sell_count = EXCLUDED.sell_count,
                    hold_count = EXCLUDED.hold_count,
                    avg_confidence = EXCLUDED.avg_confidence,
                    top_ticker = EXCLUDED.top_ticker,
                    top_confidence = EXCLUDED.top_confidence,
                    lesson_summary = EXCLUDED.lesson_summary
                """,
                (
                    f"cs-{uuid.uuid4().hex[:12]}",
                    cycle_id,
                    len(analysis_results),
                    buy_count,
                    sell_count,
                    hold_count,
                    round(avg_conf, 1),
                    top_ticker,
                    top_confidence,
                    lesson[:500],
                ),
            )
        logger.info(
            "[AUTORESEARCH] cycle_summaries written for %s: %d tickers, avg_conf=%.0f%%",
            cycle_id,
            len(analysis_results),
            avg_conf,
        )
    except Exception as e:
        logger.warning("[AUTORESEARCH] cycle_summaries write failed (non-fatal): %s", e)


# ═══════════════════════════════════════════════════════════════════
# DIRECTIVE GENERATION — Create actionable cycle_directives
# ═══════════════════════════════════════════════════════════════════


def _generate_directives(reflection: dict, cycle_id: str, triage_audit: dict) -> None:
    """Generate persistent directives from reflection recommendations.

    These directives are loaded at the start of the NEXT cycle and
    injected into the LLM context window as behavioral corrections.
    """
    directives_created = 0

    # 1. Convert reflection recommendations to directives
    recs = reflection.get("recommendations", [])
    with get_db() as db:
        for rec in recs[:3]:
            if not rec or len(rec) < 15:
                continue

            # Determine severity from the recommendation text
            severity = "info"
            rec_lower = rec.lower()
            if any(
                w in rec_lower for w in ["critical", "urgent", "immediate", "failing"]
            ):
                severity = "critical"
            elif any(w in rec_lower for w in ["warn", "degrad", "poor", "missing"]):
                severity = "warning"

            directive_id = f"dir-{uuid.uuid4().hex[:12]}"
            db.execute(
                """
                INSERT INTO cycle_directives
                (id, cycle_id, directive_type, directive_text, severity, status, expires_after)
                VALUES (%s, %s, 'recommendation', %s, %s, 'active', 5)
                ON CONFLICT (id) DO NOTHING
                """,
                [directive_id, cycle_id, rec[:300], severity],
            )
            directives_created += 1

        # 2. Convert triage issues to directives
        for issue in triage_audit.get("issues", [])[:3]:
            directive_id = f"dir-{uuid.uuid4().hex[:12]}"
            target_ticker = None
            tickers_list = issue.get("tickers", [])
            if tickers_list:
                target_ticker = tickers_list[0]  # Primary target

            severity = (
                "warning" if issue["type"] in ("neglect", "over_glancing") else "info"
            )
            db.execute(
                """
                INSERT INTO cycle_directives
                (id, cycle_id, directive_type, directive_text, target_ticker, severity, status, expires_after)
                VALUES (%s, %s, %s, %s, %s, %s, 'active', 3)
                ON CONFLICT (id) DO NOTHING
                """,
                [
                    directive_id,
                    cycle_id,
                    f"triage_{issue['type']}",
                    issue["detail"][:300],
                    target_ticker,
                    severity,
                ],
            )
            directives_created += 1

        # 3. Convert data gap findings to directives
        urgent_gaps = reflection.get("urgent_data_gaps", [])
        for ticker in urgent_gaps[:3]:
            directive_id = f"dir-{uuid.uuid4().hex[:12]}"
            db.execute(
                """
                INSERT INTO cycle_directives
                (id, cycle_id, directive_type, directive_text, target_ticker, severity, status, expires_after)
                VALUES (%s, %s, 'data_gap', %s, %s, 'warning', 'active', 3)
                ON CONFLICT (id) DO NOTHING
                """,
                [
                    directive_id,
                    cycle_id,
                    f"Critical data gap for {ticker} — force deep collection next cycle",
                    ticker,
                ],
            )
            directives_created += 1

        # 4. Convert schedule recommendations to directives
        sched_rec = reflection.get("schedule_recommendation")
        if sched_rec and isinstance(sched_rec, str) and len(sched_rec) >= 10:
            directive_id = f"dir-{uuid.uuid4().hex[:12]}"
            db.execute(
                """
                INSERT INTO cycle_directives
                (id, cycle_id, directive_type, directive_text, severity, status, expires_after)
                VALUES (%s, %s, 'schedule_recommendation', %s, 'info', 'active', 3)
                ON CONFLICT (id) DO NOTHING
                """,
                [directive_id, cycle_id, sched_rec[:300]],
            )
            directives_created += 1
            logger.info(
                "[AUTORESEARCH] Generated schedule recommendation directive: %s",
                sched_rec[:80],
            )

    if directives_created:
        logger.info(
            "[AUTORESEARCH] Generated %d directives for next cycle",
            directives_created,
        )


def _expire_old_directives() -> None:
    """Expire old directives that have exceeded their cycle lifespan.

    Each directive has an `expires_after` field (number of cycles).
    We decrement it each cycle and mark as 'expired' when it hits 0.
    """
    try:
        with get_db() as db:
            # Decrement remaining cycles
            db.execute(
                """
                UPDATE cycle_directives
                SET expires_after = expires_after - 1
                WHERE status = 'active' AND expires_after > 0
                """
            )

            # Expire directives that have run out
            expired = db.execute(
                """
                UPDATE cycle_directives
                SET status = 'expired', resolved_at = CURRENT_TIMESTAMP
                WHERE status = 'active' AND expires_after <= 0
                RETURNING id
                """
            ).fetchall()

            if expired:
                logger.info("[AUTORESEARCH] Expired %d old directives", len(expired))
    except Exception as e:
        logger.debug("[AUTORESEARCH] Directive expiry failed: %s", e)


# ═══════════════════════════════════════════════════════════════════
# SCHEDULE HEALTH AUDIT — Can the bot wake itself up?
# ═══════════════════════════════════════════════════════════════════


def _audit_schedule_health() -> dict:
    """Audit the bot's scheduling state for autonomy health.

    Checks:
    - Whether any active schedules exist (critical if not)
    - Average interval between runs
    - Whether any schedule is stuck (last_run_at >> expected interval)
    - Whether there's a pre-market schedule (market readiness)
    """
    result = {
        "active_count": 0,
        "total_count": 0,
        "avg_interval_hours": None,
        "has_premarket": False,
        "stuck_schedules": [],
        "issues": [],
    }

    try:
        with get_db() as db:
            rows = db.execute(
                """
                SELECT id, name, schedule_type, cron_expression, interval_hours,
                       is_active, last_run_at, next_run_at
                FROM cycle_schedules
                ORDER BY is_active DESC
                """
            ).fetchall()

        result["total_count"] = len(rows)
        active_rows = [r for r in rows if r[5]]  # is_active = True
        result["active_count"] = len(active_rows)

        if result["active_count"] == 0:
            result["issues"].append(
                {
                    "type": "no_active_schedules",
                    "severity": "critical",
                    "detail": "Bot has NO active schedules — it cannot wake itself up. "
                    "The Schedule Guardian should create a default, but this "
                    "indicates the bot is not self-scheduling.",
                }
            )
            return result

        # Average interval for interval-type schedules
        intervals = [
            r[4]
            for r in active_rows
            if r[2] == "interval" and r[4] is not None and r[4] > 0
        ]
        if intervals:
            result["avg_interval_hours"] = round(sum(intervals) / len(intervals), 1)

        # Check for pre-market cron schedule (e.g., '0 8 * * 1-5' or similar)
        for r in active_rows:
            if r[2] == "cron" and r[3]:
                cron_parts = r[3].split()
                if len(cron_parts) >= 2:
                    try:
                        hour = int(cron_parts[1])
                        if 7 <= hour <= 9:
                            result["has_premarket"] = True
                    except ValueError:
                        pass

        # Check for stuck schedules (last_run_at way past expected interval)
        now = datetime.now(timezone.utc)
        for r in active_rows:
            if r[2] == "interval" and r[4] and r[6]:
                last_run = r[6]
                if hasattr(last_run, "timestamp"):
                    expected_gap_seconds = r[4] * 3600
                    actual_gap_seconds = (now - last_run).total_seconds()
                    if actual_gap_seconds > expected_gap_seconds * 2.5:
                        result["stuck_schedules"].append(
                            {
                                "id": r[0],
                                "name": r[1],
                                "expected_interval_h": r[4],
                                "actual_gap_h": round(actual_gap_seconds / 3600, 1),
                            }
                        )

        if result["stuck_schedules"]:
            result["issues"].append(
                {
                    "type": "stuck_schedules",
                    "severity": "warning",
                    "detail": (
                        f"{len(result['stuck_schedules'])} schedule(s) appear stuck "
                        f"(last run >> expected interval)"
                    ),
                    "schedules": [s["name"] for s in result["stuck_schedules"]],
                }
            )

        if not result["has_premarket"] and result["active_count"] > 0:
            result["issues"].append(
                {
                    "type": "no_premarket",
                    "severity": "info",
                    "detail": "No pre-market (7-9 AM ET) schedule found. "
                    "Consider adding one for early market positioning.",
                }
            )

    except Exception as e:
        logger.debug("[AUTORESEARCH] Schedule health audit failed: %s", e)
        result["issues"].append(
            {
                "type": "audit_error",
                "detail": str(e),
            }
        )

    return result


# ═══════════════════════════════════════════════════════════════════
# DATA GAP AUTO-RESOLUTION — Close the "detect but don't fix" loop
# ═══════════════════════════════════════════════════════════════════


async def _resolve_data_gaps(gaps: list[dict], cycle_id: str) -> dict:
    """Attempt to fill detected data gaps by triggering specific collectors.

    This closes the 'detect but don't fix' anti-pattern: when AutoResearch
    identifies missing data sources for a ticker, this function triggers
    the appropriate collector to backfill the data.

    Also implements auto-ban for persistently unfillable tickers (5+ cycles
    with the same gap).
    """
    if not gaps:
        return {"resolved": 0, "failed": 0, "banned": 0}

    resolved = 0
    failed = 0
    banned = 0

    # Map of source name → collector module/function
    COLLECTOR_MAP = {
        "news": ("app.collectors.news_collector", "collect_for_ticker"),
        "price_history": ("app.collectors.yfinance_collector", "collect_price_history"),
        "technicals": ("app.processors.technical_processor", "compute_technicals"),
        "fundamentals": ("app.collectors.yfinance_collector", "collect_fundamentals"),
    }

    for gap in gaps[:5]:  # Cap at 5 to avoid overloading
        ticker = gap.get("ticker", "")
        missing = gap.get("missing_sources", [])

        if not ticker or not missing:
            continue

        # ── Check for persistent gaps (auto-ban after 5 cycles) ──
        try:
            with get_db() as db:
                occurrence_row = db.execute(
                    "SELECT COUNT(*) FROM autoresearch_reports "
                    "WHERE status = 'done' AND data_gaps LIKE %s",
                    [f'%"{ticker}"%'],
                ).fetchone()

            if occurrence_row and occurrence_row[0] >= 5:
                from app.trading.watchlist import ban_ticker

                ban_ticker(
                    ticker,
                    f"AutoResearch: persistent data gap across "
                    f"{occurrence_row[0]} cycles — auto-banned",
                )
                banned += 1
                logger.warning(
                    "[AUTORESEARCH] Auto-banned %s: persistent gap in %d cycles",
                    ticker,
                    occurrence_row[0],
                )
                continue
        except Exception as ban_err:
            logger.debug("[AUTORESEARCH] Ban check failed for %s: %s", ticker, ban_err)

        # ── Trigger collectors for each missing source ──
        import asyncio as _resolve_asyncio
        import importlib

        for source in missing:
            collector_info = COLLECTOR_MAP.get(source)
            if not collector_info:
                logger.debug(
                    "[AUTORESEARCH] No collector mapped for source '%s'", source
                )
                continue

            module_path, func_name = collector_info
            try:
                mod = importlib.import_module(module_path)
                collect_fn = getattr(mod, func_name)

                # Run with a 30-second timeout to prevent blocking
                if _resolve_asyncio.iscoroutinefunction(collect_fn):
                    await _resolve_asyncio.wait_for(collect_fn(ticker), timeout=30.0)
                else:
                    collect_fn(ticker)

                resolved += 1
                logger.info("[AUTORESEARCH] Resolved gap: %s/%s", ticker, source)
            except _resolve_asyncio.TimeoutError:
                failed += 1
                logger.warning(
                    "[AUTORESEARCH] Gap resolution timed out: %s/%s",
                    ticker,
                    source,
                )
            except Exception as coll_err:
                failed += 1
                logger.warning(
                    "[AUTORESEARCH] Gap resolution failed: %s/%s — %s",
                    ticker,
                    source,
                    coll_err,
                )

    return {"resolved": resolved, "failed": failed, "banned": banned}
