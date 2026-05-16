import json
import logging
import uuid
import datetime
import time
from pathlib import Path
from typing import Any, Callable, Dict

from app.services.vllm_client import llm, Priority
from app.utils.text_utils import parse_json_response
from app.db.connection import get_db
from .judge_agent import evaluate_decision
from app.trading.portfolio_drawdown import compute_portfolio_drawdown

logger = logging.getLogger(__name__)

AUDIT_LOG_DIR = Path("logs/audit")
AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)


def _write_audit_log(audit_id: str, data: dict):
    """Append a JSON line to a per-audit log file for debugging."""
    path = AUDIT_LOG_DIR / f"audit_{audit_id}.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, default=str) + "\n")


def get_latest_benchmark_cycle_id(db) -> str | None:
    """Return the most recent benchmarked cycle id if one exists."""
    row = db.execute(
        "SELECT cycle_id FROM cycle_benchmarks ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    return row[0] if row and row[0] else None


def compute_agent_metrics(db, cycle_id: str | None = None) -> Dict[str, Any]:
    """Aggregate grounding metrics grouped by model and agent_step."""
    scope = "cycle" if cycle_id else "global"
    query = """
        SELECT l.model, l.agent_step, e.red_cards, e.evidence_gathering 
        FROM decision_evaluations e
        JOIN llm_audit_logs l ON e.decision_id = l.id
    """
    params = []
    if cycle_id:
        query += " WHERE e.cycle_id = ?"
        params.append(cycle_id)

    evals = db.execute(query, params).fetchall()

    if not evals and cycle_id:
        logger.debug(
            "0 evaluated decisions for cycle %s — falling back to global scope",
            cycle_id,
        )
        query = """
            SELECT l.model, l.agent_step, e.red_cards, e.evidence_gathering 
            FROM decision_evaluations e
            JOIN llm_audit_logs l ON e.decision_id = l.id
        """
        evals = db.execute(query).fetchall()
        scope = "global"

    global_metrics = {
        "total_deepeval_red_cards": 0,
        "faithfulness_red_cards": 0,
        "relevancy_red_cards": 0,
        "error_red_cards": 0,
        "grounding_scores": [],
        "raw_rouge_scores": [],
        "citation_scores": [],
    }

    model_benchmarks = {}

    for model, agent_step, rc_json, evidence_json in evals:
        model = model or "Unknown"
        agent_step = agent_step or "unknown"
        key = f"{model}::{agent_step}"

        if key not in model_benchmarks:
            model_benchmarks[key] = {
                "model": model,
                "agent_step": agent_step,
                "decisions": 0,
                "red_cards": 0,
                "faithfulness": 0,
                "relevancy": 0,
                "tool_errors": 0,
                "grounding_scores": [],
                "citation_scores": [],
            }

        b = model_benchmarks[key]
        b["decisions"] += 1

        if rc_json:
            try:
                rcs = json.loads(rc_json)
                if isinstance(rcs, list):
                    global_metrics["total_deepeval_red_cards"] += len(rcs)
                    b["red_cards"] += len(rcs)
                    for rc in rcs:
                        if "Faithfulness Failure" in rc:
                            global_metrics["faithfulness_red_cards"] += 1
                            b["faithfulness"] += 1
                        elif "Answer Relevancy Failure" in rc:
                            global_metrics["relevancy_red_cards"] += 1
                            b["relevancy"] += 1
                        elif "Error" in rc:
                            global_metrics["error_red_cards"] += 1
                            if "Parse/Tool" in rc or "Error" in rc:
                                b["tool_errors"] += 1
            except Exception:
                pass
        if evidence_json:
            try:
                evidence = json.loads(evidence_json)
                if isinstance(evidence, dict):
                    gs = evidence.get("grounding_score", evidence.get("hf_rougeL"))
                    if gs is not None:
                        global_metrics["grounding_scores"].append(float(gs))
                        b["grounding_scores"].append(float(gs))
                    rr = evidence.get("raw_rougeL")
                    if rr is not None:
                        global_metrics["raw_rouge_scores"].append(float(rr))
                    co = evidence.get("citation_overlap")
                    if co is not None:
                        global_metrics["citation_scores"].append(float(co))
                        b["citation_scores"].append(float(co))
            except Exception:
                pass

    avg_grounding = (
        round(
            sum(global_metrics["grounding_scores"])
            / len(global_metrics["grounding_scores"]),
            3,
        )
        if global_metrics["grounding_scores"]
        else 0.0
    )
    avg_raw_rouge = (
        round(
            sum(global_metrics["raw_rouge_scores"])
            / len(global_metrics["raw_rouge_scores"]),
            3,
        )
        if global_metrics["raw_rouge_scores"]
        else 0.0
    )
    avg_citation = (
        round(
            sum(global_metrics["citation_scores"])
            / len(global_metrics["citation_scores"]),
            3,
        )
        if global_metrics["citation_scores"]
        else 0.0
    )

    benchmarks_list = []
    for b in model_benchmarks.values():
        benchmarks_list.append(
            {
                "model": b["model"],
                "agent_step": b["agent_step"],
                "decisions": b["decisions"],
                "red_cards": b["red_cards"],
                "faithfulness_failures": b["faithfulness"],
                "relevancy_failures": b["relevancy"],
                "tool_errors": b["tool_errors"],
                "avg_grounding_score": round(
                    sum(b["grounding_scores"]) / len(b["grounding_scores"]), 3
                )
                if b["grounding_scores"]
                else 0.0,
                "avg_citation_overlap": round(
                    sum(b["citation_scores"]) / len(b["citation_scores"]), 3
                )
                if b["citation_scores"]
                else 0.0,
            }
        )

    return {
        "total_deepeval_red_cards": global_metrics["total_deepeval_red_cards"],
        "faithfulness_red_cards": global_metrics["faithfulness_red_cards"],
        "relevancy_red_cards": global_metrics["relevancy_red_cards"],
        "error_red_cards": global_metrics["error_red_cards"],
        "avg_hf_rougeL": avg_grounding,
        "avg_grounding_score": avg_grounding,
        "avg_raw_rougeL": avg_raw_rouge,
        "avg_citation_overlap": avg_citation,
        "total_decisions_evaluated": len(evals),
        "scope_cycle_id": cycle_id,
        "scope": scope,
        "model_benchmarks": benchmarks_list,
    }


async def evaluate_pending_decisions(
    db,
    cycle_id: str | None = None,
    limit: int = 100,
    timeout_sec: float = 0,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> int:
    """Backfill missing decision evaluations before strategy scoring.

    If *cycle_id* is provided the query is scoped to that cycle first.
    When the scoped query returns 0 pending rows (common when cycle_id
    doesn't match any llm_audit_logs), the function falls back to a
    **global** query across all cycles so that decisions are never
    silently skipped.

    *timeout_sec* caps total wall-clock time.  0 = no limit (background).
    *on_progress(current, total, decision_id)* is called after each decision.
    """
    pending = []
    if cycle_id:
        pending = db.execute(
            """
            SELECT l.id
            FROM llm_audit_logs l
            LEFT JOIN decision_evaluations e ON l.id = e.decision_id
            WHERE l.cycle_id = %s
              AND e.decision_id IS NULL
              AND (l.raw_response LIKE '%%BUY%%' OR l.raw_response LIKE '%%SELL%%' OR l.raw_response LIKE '%%HOLD%%')
            ORDER BY l.created_at DESC NULLS LAST
            LIMIT %s
            """,
            [cycle_id, limit],
        ).fetchall()

    # Global fallback: no pending found for this cycle (or no cycle given)
    if not pending:
        if cycle_id:
            logger.warning(
                "No pending decisions for cycle %s — falling back to global scope",
                cycle_id,
            )
        pending = db.execute(
            """
            SELECT l.id
            FROM llm_audit_logs l
            LEFT JOIN decision_evaluations e ON l.id = e.decision_id
            WHERE e.decision_id IS NULL
              AND (l.raw_response LIKE '%%BUY%%' OR l.raw_response LIKE '%%SELL%%' OR l.raw_response LIKE '%%HOLD%%')
            ORDER BY l.created_at DESC NULLS LAST
            LIMIT %s
            """,
            [limit],
        ).fetchall()

    success_count = 0
    t0 = time.monotonic()
    total = len(pending)
    if on_progress:
        on_progress(0, total, "")
    for i, (decision_id,) in enumerate(pending):
        elapsed = time.monotonic() - t0
        if timeout_sec > 0 and elapsed > timeout_sec:
            logger.warning(
                "Backfill time budget (%.0fs) exhausted after %d/%d decisions",
                timeout_sec,
                success_count,
                total,
            )
            break
        try:
            # Emit sub-step so the frontend shows what's happening
            if on_progress:
                on_progress(
                    0,
                    -2,
                    f"Decision {i + 1}/{total}: Running DeepEval + ROUGE grounding...",
                )
            if await evaluate_decision(decision_id):
                success_count += 1
                logger.info(
                    "Backfill [%d/%d] decision %s OK (%.1fs elapsed)",
                    i + 1,
                    total,
                    decision_id[:12],
                    time.monotonic() - t0,
                )
        except Exception as exc:
            logger.error(
                "Failed to auto-evaluate %s before strategy audit: %s", decision_id, exc
            )
        if on_progress:
            on_progress(i + 1, total, decision_id)
    return success_count


SYSTEM_PROMPT = """You are a Lead Strategy Auditor evaluating an AI quantitative trading bot.
Your job is to read the bot's historical metrics, along with its configuration details,
and score the bot across 5 categories out of 5 points each.

CRITICAL CONTEXT - PAPER TRADING ONLY:
This bot is STRICTLY a paper-trading testbed designed to evaluate LLM reasoning and code architecture.
DO NOT penalize the system for low win rates (e.g., 0%) or negative PnL. This is expected during testing.
Your scores should reflect the quality of the architecture, the tracking of metrics, and logic, NOT actual profitability.

THE 5 CATEGORIES:
1. Risk Management (Weight 30%) - Evaluate if Stop Losses and Position Sizing exist programmatically. Do not penalize losses.
2. Performance & Profitability (Weight 25%) - Evaluate the system's ability to TRACK and report Win Rate/PnL properly. A 0% win rate is FINE (score 4-5/5) as long as it correctly tracks performance in paper mode.
3. Robustness & Reliability (Weight 25%) - Evaluate Error handling, modularity, and framework design. If DeepEval Red Cards > 0, severely penalize this score (1-2/5) as the bot is hallucinating.
4. Strategy Logic & Edge (Weight 10%) - Evaluate the integration of LLMs/agents and logic clarity.
   Grounding metrics guide:
   - Grounding Score is a composite of ROUGE-L precision (textual overlap) and Citation Overlap (numeric data cited from context).
   - Score 1/5 only if Grounding Score < 0.10 AND Raw ROUGE-L < 0.10 AND Citation Overlap < 0.10 (no engagement at all).
   - Score 2/5 if Grounding Score 0.10-0.25 (minimal grounding, mostly disconnected from context).
   - Score 3/5 if Grounding Score 0.25-0.45 (adequate grounding, some context engagement).
   - Score 4/5 if Grounding Score 0.45-0.65 (strong grounding, clear context integration).
   - Score 5/5 if Grounding Score > 0.65 (excellent grounding, deep context engagement).
   Consider each sub-metric: a low ROUGE-L but high Citation Overlap means the bot cites numbers but not prose — still partially grounded.
5. Operational Readiness (Weight 10%) - Evaluate logging, kill switches, security.

Return EXACTLY the following JSON format:
{
    "risk_score": <float 1-5>,
    "risk_reasoning": "<string>",
    "performance_score": <float 1-5>,
    "performance_reasoning": "<string>",
    "robustness_score": <float 1-5>,
    "robustness_reasoning": "<string>",
    "logic_score": <float 1-5>,
    "logic_reasoning": "<string>",
    "operational_score": <float 1-5>,
    "operational_reasoning": "<string>"
}
"""

USER_TEMPLATE = """### Bot Performance Metrics (Database Aggregation):
Total PnL: {total_pnl}
Win Rate: {win_rate}
Total Trades: {total_trades}
Cash Balance: {cash_balance}
Max Drawdown (Estimated): {mdd}

### Agent Ground-Truth Metrics:
Total Decisions Evaluated: {total_decisions_evaluated}
DeepEval Red Cards (Failures): {total_deepeval_red_cards}
  - Faithfulness Failures (hallucination): {faithfulness_red_cards}
  - Answer Relevancy Failures (off-topic): {relevancy_red_cards}
  - Evaluation Errors (timeout/crash): {error_red_cards}
Avg Grounding Score (composite): {avg_grounding_score}
Avg Raw ROUGE-L Precision: {avg_raw_rougeL}
Avg Citation Overlap: {avg_citation_overlap}

### Additional Context (Code & Operations):
- Environment: PAPER TRADING ONLY (Testing LLM capabilities).
- Stop Losses: Enforced at 8% per lot (default).
- Position Sizing: Static based on cash allocation.
- Code Clarity: Python 3.12, modular architecture, uses VLLM agents.
- Security: API keys stored in .env.

Act as the Strategy Auditor and score this system across the 5 dimensions considering the PAPER TRADING context. Output JSON only.
"""


async def evaluate_strategy(
    cycle_id: str | None = None,
    refresh_pending: bool = False,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> Dict[str, Any]:
    """Run the comprehensive Bot-Level strategy evaluation."""
    with get_db() as db:
        try:
            # Aggregating real stats from the database.
            # we will approximate some fields like MDD as we don't have equity curve simulation yet.
            bot = db.execute(
                "SELECT sum(total_pnl), avg(win_rate), sum(total_trades), sum(cash_balance) FROM bots"
            ).fetchone()

            total_pnl = bot[0] if bot[0] is not None else 0.0
            win_rate = bot[1] if bot[1] is not None else 0.0
            total_trades = bot[2] if bot[2] is not None else 0
            cash_balance = bot[3] if bot[3] is not None else 100000.0

            # Compute real max drawdown from trade history
            try:
                mdd_value = compute_portfolio_drawdown(db)
                mdd = (
                    f"{mdd_value * 100:.1f}%"
                    if mdd_value is not None
                    else "Unknown (no trade history)"
                )
            except Exception as e:
                logger.warning("Failed to compute portfolio drawdown: %s", e)
                mdd = "Unknown (computation error)"

            latest_cycle_id = cycle_id
            if latest_cycle_id is None and refresh_pending:
                latest_cycle_id = get_latest_benchmark_cycle_id(db)

            warnings: list[str] = []
            if refresh_pending:
                backfilled = await evaluate_pending_decisions(
                    db,
                    latest_cycle_id,
                    on_progress=on_progress,
                )
                if backfilled:
                    logger.info(
                        "Backfilled %d decision evaluations before strategy audit",
                        backfilled,
                    )
                else:
                    warnings.append(
                        f"No pending decisions found to backfill (cycle_id={latest_cycle_id})"
                    )

            agent_metrics = compute_agent_metrics(db, latest_cycle_id)
            if agent_metrics["total_decisions_evaluated"] == 0:
                warnings.append(
                    "0 decisions evaluated — run the pipeline first to generate trade decisions, "
                    "then click 'Run Strategy Audit' again."
                )

            logger.info(
                "Strategy audit metrics: decisions=%d, red_cards=%d, grounding=%.3f, rouge=%.3f, citation=%.3f, scope=%s",
                agent_metrics["total_decisions_evaluated"],
                agent_metrics["total_deepeval_red_cards"],
                agent_metrics["avg_grounding_score"],
                agent_metrics["avg_raw_rougeL"],
                agent_metrics["avg_citation_overlap"],
                agent_metrics.get("scope", "unknown"),
            )

            user_prompt = USER_TEMPLATE.format(
                total_pnl=total_pnl,
                win_rate=win_rate,
                total_trades=total_trades,
                cash_balance=cash_balance,
                mdd=mdd,
                total_decisions_evaluated=agent_metrics["total_decisions_evaluated"],
                total_deepeval_red_cards=agent_metrics["total_deepeval_red_cards"],
                faithfulness_red_cards=agent_metrics.get("faithfulness_red_cards", 0),
                relevancy_red_cards=agent_metrics.get("relevancy_red_cards", 0),
                error_red_cards=agent_metrics.get("error_red_cards", 0),
                avg_grounding_score=agent_metrics["avg_grounding_score"],
                avg_raw_rougeL=agent_metrics["avg_raw_rougeL"],
                avg_citation_overlap=agent_metrics["avg_citation_overlap"],
            )

            try:
                if on_progress:
                    on_progress(0, -1, "scoring")
                t_llm = time.monotonic()
                eval_response, tokens, ms = await llm.chat(
                    system=SYSTEM_PROMPT,
                    user=user_prompt,
                    temperature=0.1,
                    max_tokens=512,
                    priority=Priority.HIGH,
                    agent_name="strategy_evaluator",
                )
                logger.info(
                    "LLM scoring call completed in %.1fs (%d tokens)",
                    time.monotonic() - t_llm,
                    tokens or 0,
                )
            except Exception as api_err:
                logger.error(
                    f"llm.chat failed for strategy evaluation: {api_err}", exc_info=True
                )
                raise api_err

            payload = parse_json_response(eval_response)
            payload["agent_metrics_scope_cycle_id"] = latest_cycle_id
            payload["agent_metrics_total_decisions"] = agent_metrics[
                "total_decisions_evaluated"
            ]

            risk_score = float(payload.get("risk_score", 1.0))
            perf_score = float(payload.get("performance_score", 1.0))
            rob_score = float(payload.get("robustness_score", 1.0))
            log_score = float(payload.get("logic_score", 1.0))
            op_score = float(payload.get("operational_score", 1.0))

            # Calculate Total Score out of 100
            # 1-5 scale. So 5 is max points.
            # Risk: max 30. perf 25. rob 25. log 10. op 10.
            score_out_of_100 = (
                (risk_score / 5.0) * 30
                + (perf_score / 5.0) * 25
                + (rob_score / 5.0) * 25
                + (log_score / 5.0) * 10
                + (op_score / 5.0) * 10
            )

            total_score = round(score_out_of_100, 2)
            eval_id = str(uuid.uuid4())

            # Save to database
            db.execute(
                """
                INSERT INTO strategy_evaluations (
                    id, cycle_id, total_score, risk_score, performance_score, robustness_score, 
                    logic_score, operational_score, full_analysis
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    eval_id,
                    latest_cycle_id,
                    total_score,
                    risk_score,
                    perf_score,
                    rob_score,
                    log_score,
                    op_score,
                    json.dumps(payload),
                ],
            )

            logger.info(f"Strategy Evaluated! Total Score: {total_score}")

            # ── Write structured audit log ──
            _write_audit_log(
                eval_id,
                {
                    "step": "metrics",
                    "timestamp": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(),
                    "cycle_id": latest_cycle_id,
                    "bot_stats": {
                        "total_pnl": total_pnl,
                        "win_rate": win_rate,
                        "total_trades": total_trades,
                        "cash_balance": cash_balance,
                    },
                    "agent_metrics": agent_metrics,
                    "warnings": warnings,
                },
            )
            _write_audit_log(
                eval_id,
                {
                    "step": "llm_scores",
                    "timestamp": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(),
                    "total_score": total_score,
                    "risk_score": risk_score,
                    "performance_score": perf_score,
                    "robustness_score": rob_score,
                    "logic_score": log_score,
                    "operational_score": op_score,
                    "risk_reasoning": payload.get("risk_reasoning"),
                    "performance_reasoning": payload.get("performance_reasoning"),
                    "robustness_reasoning": payload.get("robustness_reasoning"),
                    "logic_reasoning": payload.get("logic_reasoning"),
                    "operational_reasoning": payload.get("operational_reasoning"),
                },
            )
            # Log each individual decision evaluation for debugging
            evals_for_log = db.execute(
                "SELECT decision_id, ticker, judge_a_score, final_quality_score, "
                "red_cards, evidence_gathering FROM decision_evaluations "
                "ORDER BY timestamp DESC LIMIT 100"
            ).fetchall()
            for ev in evals_for_log:
                _write_audit_log(
                    eval_id,
                    {
                        "step": "decision_detail",
                        "decision_id": ev[0],
                        "ticker": ev[1],
                        "judge_a_score": ev[2],
                        "final_quality_score": ev[3],
                        "red_cards": json.loads(ev[4]) if ev[4] else [],
                        "evidence": json.loads(ev[5]) if ev[5] else {},
                    },
                )

            result: Dict[str, Any] = {
                "id": eval_id,
                "total_score": total_score,
                "components": payload,
                "agent_metrics": agent_metrics,
            }
            if warnings:
                result["warnings"] = warnings
            return result

        except Exception as e:
            logger.error(f"Failed Strategy Evaluation: {e}", exc_info=True)
            return {"error": str(e)}
