"""
Decision Engine -- Hybrid Agent + RLM pipeline.

Architecture:
  1. Run 5 specialist agents in parallel (fast vLLM calls, ~5-10s each)
  2. Feed their structured summaries into RLM as context + tools
  3. RLM Config C synthesizes agent summaries + raw data → BUY/SELL/HOLD
  4. If confidence < 70 or HOLD → escalate to Config D (thinking ON)
  5. C and D disagree → flag for human review

Usage:
    from app.pipeline.analysis.decision_engine import analyze_ticker
    result = await analyze_ticker("NVDA")
"""

import asyncio
import time
import logging
from typing import Callable
import json
import uuid
import dataclasses
from datetime import datetime, timezone

from app.services.rlm.rlm_wrapper import rlm_analyze
from app.db.connection import get_db
from app.pipeline.analysis.agent_execution import (
    run_specialist_agents as _run_agents,
    format_agent_summaries as _format_agent_summaries,
)
from app.pipeline.analysis.debate_engine import run_debate
from app.utils.pipeline_utils import noop as _noop, elapsed_ms
from app.utils.text_utils import sanitize_ascii

logger = logging.getLogger(__name__)


# Fix #9: Import from shared source of truth

# Fix #10: Confidence threshold for Config C → Config D escalation.
# Current value chosen heuristically. To calibrate:
#   1. Run battle tests and log confidence distributions from analysis_results table
#   2. Query: SELECT confidence, action FROM analysis_results WHERE agent_name LIKE 'hybrid_%%'
#   3. If model reliably outputs 65-68 for strong signals, lower this to 60
#   4. If model outputs 80+ for weak signals, raise this to 80
CONFIDENCE_THRESHOLD = 70


# _run_agents and _format_agent_summaries removed — imported from app.pipeline.analysis.agent_execution


def _execute_quarantine(ticker: str, reason: str, cycle_id: str, bot_id: str, triage_tier: str, held: bool, emit: Callable, source: str = "data_sufficiency_gate") -> dict:
    """Isolate a ticker that failed critical checks, delete it from watchlists, and return a synthetic HOLD."""
    logger.warning(f"[PIPELINE] [QUARANTINE] {ticker}: {reason}")
    emit(
        "analyzing",
        f"quarantine_{ticker}",
        f"{ticker}: Quarantined — {reason}",
        status="error",
    )
    
    # 1. Update quarantine tracking tables
    try:
        with get_db() as db:
            # Try to insert into rejected_symbols
            try:
                db.execute(
                    "INSERT INTO rejected_symbols (symbol, reason, source, created_at) "
                    "VALUES (%s, %s, %s, NOW())",
                    [ticker, reason, source],
                )
            except Exception as e:
                logger.debug(f"rejected_symbols insert failed for {ticker}: {e}")

            # Try to insert into the actual ticker_quarantine table
            try:
                db.execute(
                    "INSERT INTO ticker_quarantine (ticker, reason, details) "
                    "VALUES (%s, %s, %s) "
                    "ON CONFLICT (ticker) DO UPDATE SET reason = EXCLUDED.reason, details = EXCLUDED.details, quarantined_at = NOW()",
                    [ticker, source, reason]
                )
            except Exception as e:
                logger.debug(f"ticker_quarantine insert failed for {ticker}: {e}")

            from app.processors.ticker_extractor import (
                get_registry as _get_reg_yf,
                _save_rejected_to_db as _reject_db,
                FALSE_TICKERS as _FT,
            )
            _reg_yf = _get_reg_yf()
            _reg_yf.add_rejected(ticker)
            _FT.add(ticker)
            try:
                _reject_db(ticker)
            except Exception:
                pass

            db.execute("DELETE FROM watchlist WHERE ticker = %s", [ticker])
            
            try:
                db.execute("UPDATE discovered_tickers SET validation_status = 'quarantine' WHERE ticker = %s", [ticker])
            except Exception as update_err:
                logger.debug(f"Failed to update discovered_tickers for {ticker}: {update_err}")
    except Exception as full_err:
        logger.warning("[PIPELINE] Error executing quarantine DB commands for %s: %s", ticker, full_err)

    logger.warning("[PIPELINE] %s is missing critical data/integrity — auto-rejecting, quarantining, and dropping from cycle.", ticker)
    
    from app.cognition.debate.action_gate import gate_action
    quarantine_action = gate_action("HOLD", held)
    
    # 2. Return a synthetic HOLD so the pipeline records this failure
    from datetime import datetime, timezone
    synthetic_result = {
        "ticker": ticker,
        "action": quarantine_action,
        "confidence": 0,
        "rationale": f"Quarantined: {reason}",
        "config_used": "quarantine",
        "decision_model": "none",
        "decision_role": "none",
        "escalated": False,
        "agent_results": {},
        "human_review": False,
        "agent_tokens": 0,
        "rlm_tokens": 0,
        "total_tokens": 0,
        "total_time_s": 0.0,
        "data_sources": [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "triage_tier": triage_tier,
    }
    _log_decision(synthetic_result, cycle_id, bot_id)
    return synthetic_result


async def analyze_ticker(
    ticker: str,
    cycle_id: str = "",
    bot_id: str = "",
    force_config_d: bool = False,
    skip_agents: bool = False,
    emit: Callable | None = None,
    macro_memo: str = "",
    watchlist: list[str] | None = None,
    use_tools: bool = False,
    triage_tier: str = "standard",
) -> dict:
    """Run the hybrid agent + RLM analysis for a single ticker.

    Pipeline:
        1. Run 5 specialist agents in parallel (~10s)
        2. Build context blob from DB
        3. Inject macro memo + agent summaries + raw data into RLM Config C
        4. If needed, escalate to Config D
        5. Return final decision

    Args:
        ticker: Stock ticker symbol
        cycle_id: Trading cycle ID for audit logging
        bot_id: Bot ID for audit logging
        force_config_d: Always run Config D (for testing)
        skip_agents: Skip agent phase, only run RLM (for A/B testing)
        emit: event callback(phase, step, detail, status, data, elapsed_ms)
        macro_memo: Pre-computed macro strategy memo from the Macro Scout.
                    Prepended to context so the LLM has big-picture awareness.

    Returns dict with action, confidence, rationale, agent/RLM details
    """
    if emit is None:
        emit = _noop
    start = time.monotonic()
    ticker = ticker.upper()

    logger.info("[PIPELINE] =" * 70)
    logger.info(
        "[PIPELINE] [DECISION ENGINE] Analyzing %s (hybrid mode, tier=%s)",
        ticker,
        triage_tier,
    )
    logger.info("[PIPELINE] =" * 70)
    emit(
        "analyzing",
        f"start_{ticker}",
        f"{ticker}: Starting hybrid analysis (tier={triage_tier})",
        status="running",
    )

    from app.tools.portfolio_tools import get_position_context
    position_context = get_position_context(ticker, bot_id)
    held = position_context.get("held", False)

    # Force Config D for Deep-tier tickers
    if triage_tier == "deep":
        force_config_d = True
        logger.info(
            "[PIPELINE] [TRIAGE] %s is Deep tier — forcing Config D escalation", ticker
        )

    # ── Tool Analyst Bypass ──
    from app.config import settings

    if use_tools or getattr(settings, "USE_TOOL_CALLING", False):
        from app.tools.tool_analyst import analyze_with_tools

        logger.info(
            "[PIPELINE] [DECISION ENGINE] Using Tool-Calling Analyst for %s", ticker
        )
        return await analyze_with_tools(ticker, cycle_id, bot_id)

    # ── Check for pause ──
    from app.pipeline.orchestration.cycle_control import cycle_control

    await cycle_control.wait_if_paused()

    # ── Step 0: Ensure data completeness (fill gaps before agents run) ──
    t0 = time.monotonic()
    from app.pipeline.data.data_completeness import check_and_fill

    data_report = await check_and_fill(ticker, emit=emit)
    ms0 = elapsed_ms(t0)
    filled = data_report.get("filled", [])
    if filled:
        logger.info("[PIPELINE] [DATA] Filled gaps: %s", filled)
        emit(
            "analyzing",
            f"data_completeness_{ticker}",
            f"{ticker}: Filled {len(filled)} data gaps",
            data={"filled": filled},
            elapsed_ms=ms0,
        )
    else:
        emit(
            "analyzing",
            f"data_completeness_{ticker}",
            f"{ticker}: Data complete, no gaps",
            elapsed_ms=ms0,
        )

    # ── Data sufficiency gate: quarantine if critical data still missing ──
    from app.pipeline.data.data_completeness import check_data_sufficiency

    sufficiency = check_data_sufficiency(data_report)
    if not sufficiency["sufficient"]:
        gap_names = ", ".join(g["category"] for g in sufficiency["gaps"])
        reason = f"Critical data missing after collection: {gap_names}"
        return _execute_quarantine(ticker, reason, cycle_id, bot_id, triage_tier, held, emit, source="data_sufficiency_gate")

    # ── GLANCE TIER: Lightweight change-detection check ──
    if triage_tier == "glance":
        try:
            from app.services.vllm_client import llm, Priority

            logger.info(
                "[PIPELINE] [TRIAGE] %s is Glance tier — running change detection only",
                ticker,
            )
            emit(
                "analyzing",
                f"glance_{ticker}",
                f"{ticker}: Glance tier — lightweight change detection",
                status="running",
            )

            # Quick context: last analysis + latest news headlines
            _glance_ctx_parts = []
            try:
                with get_db() as db:
                    # Last analysis result
                    last_row = db.execute(
                        "SELECT result_json FROM analysis_results "
                        "WHERE ticker = %s ORDER BY created_at DESC LIMIT 1",
                        [ticker],
                    ).fetchone()
                    if last_row:
                        _glance_ctx_parts.append(f"LAST ANALYSIS: {last_row[0][:500]}")

                    # Latest news headlines (last 24h)
                    from datetime import timedelta

                    _cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
                    _news = db.execute(
                        "SELECT title, published_at FROM news_articles "
                        "WHERE ticker = %s AND published_at > %s "
                        "AND (quality_status IS NULL OR quality_status != 'discarded') "
                        "ORDER BY published_at DESC LIMIT 5",
                        [ticker, _cutoff],
                    ).fetchall()
                    if _news:
                        headlines = "\n".join(f"- {n[0]}" for n in _news)
                        _glance_ctx_parts.append(f"RECENT NEWS:\n{headlines}")
                    else:
                        _glance_ctx_parts.append("RECENT NEWS: None")
            except Exception:
                pass

            _glance_ctx = (
                "\n\n".join(_glance_ctx_parts)
                if _glance_ctx_parts
                else "No context available."
            )

            glance_resp, glance_tokens, glance_ms = await llm.chat(
                system=(
                    "You are a fast market change detector. "
                    "Given a stock's last analysis and recent news, determine if anything "
                    "has MATERIALLY changed that would warrant a full re-analysis. "
                    "Respond with EXACTLY one of:\n"
                    "  SKIP — No material change\n"
                    "  CHANGED — Material change detected (explain briefly)\n"
                ),
                user=f"Ticker: {ticker}\n\n{_glance_ctx}",
                temperature=0.1,
                max_tokens=100,
                priority=Priority.LOW,
                agent_name="glance_detector",
                ticker=ticker,
                cycle_id=cycle_id,
            )

            glance_resp_upper = glance_resp.strip().upper()

            if glance_resp_upper.startswith("SKIP"):
                # No material change — try to return cached thesis
                from app.pipeline.analysis.thesis_store import get_thesis, mark_unchanged

                thesis = get_thesis(ticker)

                if thesis is None:
                    # No prior thesis — can't safely skip, promote to full analysis
                    logger.info(
                        "[PIPELINE] [TRIAGE] %s: Glance SKIP but no prior thesis — "
                        "promoting to Standard (%d tokens)",
                        ticker,
                        glance_tokens,
                    )
                    emit(
                        "analyzing",
                        f"glance_promote_{ticker}",
                        f"{ticker}: No prior thesis — promoting to full analysis",
                        status="warning",
                    )
                    triage_tier = "standard"  # Fall through to full analysis
                else:
                    # Cache hit — return cached verdict
                    logger.info(
                        "[PIPELINE] [TRIAGE] %s: Glance SKIP → cached thesis %s @ %d%% (%d tokens)",
                        ticker,
                        thesis.verdict,
                        thesis.confidence,
                        glance_tokens,
                    )
                    emit(
                        "analyzing",
                        f"glance_skip_{ticker}",
                        f"{ticker}: No material change — cached {thesis.verdict} @ {thesis.confidence}% "
                        f"(Glance skip, {glance_tokens} tok)",
                        status="ok",
                    )

                    mark_unchanged(ticker)

                    # Record as skipped in attention tracker
                    try:
                        from app.pipeline.attention_tracker import record_skip

                        record_skip(ticker)
                    except Exception:
                        pass

                    return {
                        "ticker": ticker,
                        "action": thesis.verdict,
                        "confidence": thesis.confidence,
                        "rationale": (
                            f"Glance SKIP: No material change. Thesis unchanged since "
                            f"{thesis.updated_at.strftime('%Y-%m-%d %H:%M')} UTC. "
                            f"({glance_resp.strip()[:100]})"
                        ),
                        "triage_tier": "glance",
                        "glance_skipped": True,
                        "glance_tokens": glance_tokens,
                        "glance_ms": glance_ms,
                        "thesis_verdict": thesis.verdict,
                        "thesis_confidence": thesis.confidence,
                    }
            else:
                # Material change detected — auto-promote to Standard
                logger.info(
                    "[PIPELINE] [TRIAGE] %s: Glance check = CHANGED — promoting to Standard analysis",
                    ticker,
                )
                emit(
                    "analyzing",
                    f"glance_promote_{ticker}",
                    f"{ticker}: Material change detected — promoting to full analysis",
                    status="warning",
                    data={"change": glance_resp.strip()[:200]},
                )
                triage_tier = "standard"  # Continue with full analysis

        except Exception as glance_err:
            logger.warning(
                "[PIPELINE] [TRIAGE] Glance check failed for %s (falling back to full analysis): %s",
                ticker,
                glance_err,
            )
            triage_tier = "standard"  # Fallback to full analysis

    # ── Step 1: Run 5 specialist agents in parallel ──
    agent_results = {}
    agent_summaries_text = ""
    if not skip_agents:
        emit(
            "analyzing",
            f"agents_{ticker}",
            f"{ticker}: Running specialist agents in parallel...",
            status="running",
        )
        t1 = time.monotonic()
        from app.pipeline.analysis.agent_execution import PipelineAbortError
        try:
            agent_results = await _run_agents(
                ticker,
                cycle_id,
                bot_id,
            )
        except PipelineAbortError as abort_err:
            reason = f"Upstream agent critical failure: {str(abort_err)}"
            return _execute_quarantine(ticker, reason, cycle_id, bot_id, triage_tier, held, emit, source="agent_failure")
        ms1 = elapsed_ms(t1)
        agent_summaries_text = _format_agent_summaries(agent_results)

        # ── Checkpoint: agent results complete ──
        try:
            from app.db.checkpoints import checkpoint_manager

            checkpoint_manager.save(cycle_id, "agents_complete", ticker=ticker)
        except Exception:
            pass  # Checkpointing must never block the pipeline

        # Emit individual agent results
        for name, result in agent_results.items():
            resp = (
                result.get("response", "")[:200]
                if isinstance(result, dict)
                else str(result)[:200]
            )
            tokens = result.get("tokens_used", 0) if isinstance(result, dict) else 0
            emit(
                "analyzing",
                f"agent_{name}_{ticker}",
                f"{ticker} {name}: {resp}",
                data={"tokens": tokens, "agent": name},
                elapsed_ms=ms1 // max(len(agent_results), 1),
            )
    else:
        logger.info("[PIPELINE] [AGENTS] Skipped (skip_agents=True)")
        emit(
            "analyzing",
            f"agents_{ticker}",
            f"{ticker}: Agents skipped",
            status="skipped",
        )

    # ── Step 2: Fetch current thesis (drives delta-mode context) ──
    from app.pipeline.analysis.thesis_store import get_thesis as _get_thesis

    current_thesis = _get_thesis(ticker)
    _since = current_thesis.updated_at if current_thesis else None

    # ── Step 3: Build context from DB ──
    # If a thesis exists, only pull data NEWER than the last analysis.
    # The thesis carries forward all prior understanding — no need to re-read old articles.
    from app.pipeline.analysis.context_builder import build_context_blob

    context = await build_context_blob(ticker, watchlist=watchlist, since=_since)

    # ── Semantic Gap Analysis ──
    from app.agents.gap_analyzer import analyze_gaps, fill_gaps

    gaps = analyze_gaps(context, ticker)
    fillable_gaps = [g for g in gaps if g["has_collector"]]
    if fillable_gaps:
        emit(
            "analyzing",
            f"gap_analysis_{ticker}",
            f"{ticker}: Detected {len(fillable_gaps)} semantic gaps, collecting...",
            status="running",
        )
        t_gap = time.monotonic()
        fill_results = await fill_gaps(fillable_gaps, ticker)
        ms_gap = elapsed_ms(t_gap)

        filled_count = sum(1 for v in fill_results.values() if v)
        if filled_count > 0:
            emit(
                "analyzing",
                f"gap_filled_{ticker}",
                f"{ticker}: Filled {filled_count} semantic gaps. Rebuilding context...",
                data={"fill_results": fill_results},
                elapsed_ms=ms_gap,
            )
            # Rebuild context since new data was collected
            context = await build_context_blob(ticker, watchlist=watchlist, since=_since)
        else:
            emit(
                "analyzing",
                f"gap_failed_{ticker}",
                f"{ticker}: Tested {len(fillable_gaps)} gap collectors (0 new rows)",
                data={"fill_results": fill_results},
                elapsed_ms=ms_gap,
                status="warning",
            )

    # Prepend macro memo + agent summaries to context so RLM sees them first
    prefix_sections = []

    # ── Thesis-Aware Context (Delta Analysis) ──
    if current_thesis and not current_thesis.unchanged:
        thesis_block = (
            f"# CURRENT THESIS (as of {current_thesis.updated_at.strftime('%Y-%m-%d %H:%M')} UTC)\n"
            f"Verdict: **{current_thesis.verdict}** | Confidence: {current_thesis.confidence}%\n"
            f"Summary: {current_thesis.summary}\n\n"
            "# YOUR TASK — DELTA ANALYSIS\n"
            "Do NOT rewrite the thesis from scratch. Only answer:\n"
            "1. Does the new data CONFIRM, WEAKEN, or REVERSE the current thesis?\n"
            "2. What specifically changed since the last analysis?\n"
            "3. Output your final verdict with updated rationale.\n"
            "If REVERSED, explain clearly why the prior thesis no longer holds.\n"
        )
        prefix_sections.append(thesis_block)
        logger.info(
            "[THESIS] Injecting current thesis for %s: %s @ %d%%",
            ticker,
            current_thesis.verdict,
            current_thesis.confidence,
        )

    if macro_memo:
        prefix_sections.append(macro_memo)
        logger.info(
            "[CONTEXT] Injecting macro memo (%d chars) for %s", len(macro_memo), ticker
        )

    if agent_summaries_text:
        logger.info(
            "[PIPELINE] [AGENTS] RAW SPECIALIST AGENT SUMMARIES:\n%s",
            agent_summaries_text,
        )
        prefix_sections.append(
            "# SPECIALIST AGENT ANALYSES\n"
            "The following are pre-computed analyses from 5 specialist agents. "
            "Use these as a starting point, then verify with raw data tools.\n\n"
            f"{agent_summaries_text}"
        )

    # ── Inject Autoresearch Lessons ──
    try:
        from app.cognition.lesson_store import retrieve_lessons

        lessons = retrieve_lessons(ticker, k=2)
        if lessons:
            lesson_texts = "\n".join(f"- {l.get('lesson_text', '')}" for l in lessons)
            prefix_sections.append(
                "# AUTORESEARCH LESSONS\n"
                "The following are critical lessons and recommendations from past autoresearch cycles. "
                "You MUST adhere to these rules to avoid repeating past mistakes:\n\n"
                f"{lesson_texts}"
            )
            emit(
                "analyzing",
                f"autoresearch_{ticker}",
                f"{ticker}: Injected {len(lessons)} Autoresearch lessons",
            )
            logger.info(
                "[MEMORY] Injected %d Autoresearch lessons for %s", len(lessons), ticker
            )
    except Exception as ar_err:
        logger.warning("[PIPELINE] Failed to retrieve autoresearch lessons: %s", ar_err)

    # B4 & B5: Inject TRADING_CONSTRAINTS
    from app.constants import TRADING_CONSTRAINTS

    prefix_sections.append(TRADING_CONSTRAINTS)

    prefix_sections.append(
        "# 🚨 TRADER GROUND TRUTH INSTRUCTION\n"
        "Trader Notes provided in the context are absolute GROUND TRUTH. You MUST weight these notes higher than any other data source and use them to actively steer your investigation and final decision."
    )

    if prefix_sections:
        context = "\n\n".join(prefix_sections) + "\n\n# RAW MARKET DATA\n" + context

    logger.info(
        "[PIPELINE] [CONTEXT] %s chars (agents + raw data)", f"{len(context):,}"
    )

    # ── Verbose: log full context being sent to RLM ──
    logger.debug("#" * 60)
    logger.debug("RLM INPUT CONTEXT (%s chars)", f"{len(context):,}")
    logger.debug("#" * 60)
    safe_ctx = sanitize_ascii(context)
    logger.debug("%s", safe_ctx)
    logger.debug("#" * 60)

    # ── Check for pause ──
    await cycle_control.wait_if_paused()

    # ── Data Flow Audit: log what data sources are available for the decision ──
    from app.data.market_data_store import get_latest_snapshot

    snapshot = get_latest_snapshot(ticker)
    data_manifest = {}
    if snapshot:
        for k, v in dataclasses.asdict(snapshot).items():
            data_manifest[k] = {"present": v is not None}

    present_sources = [k for k, v in data_manifest.items() if v.get("present")]
    missing_sources = [k for k, v in data_manifest.items() if not v.get("present")]
    emit(
        "analyzing",
        f"data_audit_{ticker}",
        f"{ticker}: Data flow audit — {len(present_sources)} sources present, "
        f"{len(missing_sources)} missing: {', '.join(missing_sources) if missing_sources else 'none'}",
        data={
            "present": present_sources,
            "missing": missing_sources,
            "manifest": data_manifest,
        },
    )

    # ── Resolve trader model for final decisions ──
    from app.services.vllm_client import llm as _llm_for_routing

    trader_model = _llm_for_routing.get_trader_model()
    trader_url = _llm_for_routing.get_trader_url()
    # Determine if we have a dedicated trader endpoint
    has_dedicated_trader = any(
        ep.role == "trader" and ep.enabled
        for ep in _llm_for_routing._endpoints.values()
    )
    decision_role = "trader" if has_dedicated_trader else "analyst"
    decision_model_name = trader_model or "unknown"
    logger.info(
        "[PIPELINE] [DECISION ROUTING] %s — using %s model: %s",
        ticker,
        decision_role.upper(),
        decision_model_name,
    )
    if has_dedicated_trader:
        emit(
            "analyzing",
            f"trader_routing_{ticker}",
            f"{ticker}: Final decision routed to TRADER model ({decision_model_name})",
        )

    # ── Step 3: Run RLM Config C ──
    logger.info("[PIPELINE] [CONFIG C] Running RLM synthesis (thinking OFF)...")
    emit(
        "analyzing",
        f"rlm_config_c_{ticker}",
        f"{ticker}: Running RLM Config C (thinking OFF) on {decision_role}...",
        status="running",
    )
    c_start = time.monotonic()
    c_result = await rlm_analyze(
        ticker=ticker,
        context=context,
        max_iterations=4,
        enable_thinking=False,
        cycle_id=cycle_id,
        bot_id=bot_id,
        target_role=decision_role,
    )
    c_time = time.monotonic() - c_start
    c_ms = int(c_time * 1000)
    
    from app.cognition.debate.action_gate import gate_action
    c_action = gate_action(c_result.get("action", "HOLD"), held)
    c_confidence = c_result.get("confidence", 0)
    c_tokens = c_result.get("tokens_used", 0)

    # B5: Rationale validation
    if c_action == "BUY":
        rationale_lower = c_result.get("rationale", "").lower()
        if not ("fee" in rationale_lower or "slippage" in rationale_lower):
            logger.warning(
                "[PIPELINE] Constraint validation failed for %s. Re-prompting once.",
                ticker,
            )
            emit(
                "analyzing",
                f"validation_{ticker}",
                f"{ticker}: Rationale ignored constraints. Re-prompting.",
                status="warning",
            )
            c_result = await rlm_analyze(
                ticker=ticker,
                context=context
                + "\n\nCRITICAL FEEDBACK: You recommended BUY but completely ignored the TRADING_CONSTRAINTS (fees, slippage). Please re-evaluate your decision and explicitly account for these constraints in your rationale.",
                max_iterations=2,
                enable_thinking=False,
                cycle_id=cycle_id,
                bot_id=bot_id,
                target_role=decision_role,
            )
            c_time = time.monotonic() - c_start
            c_ms = int(c_time * 1000)
            c_action = gate_action(c_result.get("action", "HOLD"), held)
            c_confidence = c_result.get("confidence", 0)
            c_tokens += c_result.get("tokens_used", 0)

    # ── Checkpoint: Config C complete ──
    try:
        from app.db.checkpoints import checkpoint_manager

        checkpoint_manager.save(
            cycle_id,
            "config_c_complete",
            ticker=ticker,
            state={
                "action": c_action,
                "confidence": c_confidence,
            },
        )
    except Exception:
        pass

    emit(
        "analyzing",
        f"rlm_config_c_{ticker}",
        f"{ticker}: Config C → {c_action} @ {c_confidence}% "
        f"({c_tokens:,} tokens, {c_time:.1f}s)",
        data={"action": c_action, "confidence": c_confidence, "tokens": c_tokens},
        elapsed_ms=c_ms,
    )

    # ── Verbose: log full RLM Config C response ──
    logger.debug("#" * 60)
    logger.debug("RLM CONFIG C OUTPUT [%s tokens, %.1fs]", c_tokens, c_time)
    logger.debug("#" * 60)
    safe_rationale = sanitize_ascii(c_result.get("rationale", ""))
    logger.debug("Action: %s | Confidence: %s%%", c_action, c_confidence)
    logger.debug("Rationale: %s", safe_rationale)
    logger.debug("#" * 60)

    # Fix #1: Escalate on low confidence AND on HOLD (for held positions).
    # We now force the debate engine to make a definitive BUY or SELL decision,
    # so escalating on HOLD will actually resolve indecision rather than confirming it.
    needs_escalation = force_config_d or c_confidence < CONFIDENCE_THRESHOLD or c_action == "HOLD"
    # Auto-escalate when Config C reverses an existing thesis (high-risk decision)
    if (
        not needs_escalation
        and current_thesis
        and c_action != current_thesis.verdict
        and c_action not in ("HOLD", "PASS")
    ):
        logger.info(
            "[THESIS] %s: Verdict REVERSED %s → %s — forcing Config D debate",
            ticker,
            current_thesis.verdict,
            c_action,
        )
        force_config_d = True
        needs_escalation = True

    d_result = None
    if needs_escalation:
        reason = (
            "forced"
            if force_config_d
            else f"confidence {c_confidence} < {CONFIDENCE_THRESHOLD}"
        )
        logger.info("[PIPELINE] [ESCALATION] -> Config D DEBATE (%s)...", reason)
        emit(
            "analyzing",
            f"escalation_{ticker}",
            f"{ticker}: Escalating to Config D Debate — {reason}",
            status="running",
        )

        d_start = time.monotonic()
        try:
            d_result = await asyncio.wait_for(
                run_debate(
                    ticker=ticker,
                    config_c_result=c_result,
                    context=context,
                    agent_summaries_text=agent_summaries_text,
                    cycle_id=cycle_id,
                    bot_id=bot_id,
                    held=held,
                ),
                timeout=300.0,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "[PIPELINE] [DEBATE ENGINE] TIMEOUT after 300s — falling back to Config C"
            )
            d_result = {
                "action": c_action,
                "confidence": max(c_confidence - 10, 0),
                "rationale": (
                    f"Debate engine timed out. Using Config C with reduced "
                    f"confidence. Original: "
                    f"{c_result.get('rationale', '')[:200]}"
                ),
                "tokens_used": 0,
            }
            emit(
                "analyzing",
                f"recursive_rlm_timeout_{ticker}",
                f"{ticker}: Debate TIMEOUT (300s) — using Config C",
                status="error",
                elapsed_ms=300000,
            )

        d_time = time.monotonic() - d_start
        d_ms = int(d_time * 1000)
        from app.cognition.debate.action_gate import gate_action
        d_action = gate_action(d_result.get("action", "HOLD"), held)
        d_confidence = d_result.get("confidence", 0)
        d_tokens = d_result.get("tokens_used", 0)

        if d_tokens > 0:
            emit(
                "analyzing",
                f"recursive_rlm_{ticker}",
                f"{ticker}: Recursive RLM (depth=2) → {d_action} @ "
                f"{d_confidence}% ({d_tokens:,} tokens, {d_time:.1f}s)",
                data={
                    "action": d_action,
                    "confidence": d_confidence,
                    "tokens": d_tokens,
                },
                elapsed_ms=d_ms,
            )

        # The returned action IS the final decision
        final_action = d_action
        final_confidence = d_confidence
        final_rationale = d_result.get("rationale", "")
        config_used = "C+D_recursive"
        human_review = False

        logger.info(
            "[RECURSIVE RLM] Finished deep analysis for %s with action %s",
            ticker,
            final_action,
        )
    else:
        final_action = c_action
        final_confidence = c_confidence
        final_rationale = c_result.get("rationale", "")
        config_used = "C"
        human_review = False
        logger.info("[PIPELINE] [CONFIDENT] Using Config C directly")
        emit(
            "analyzing",
            f"no_escalation_{ticker}",
            f"{ticker}: Config C confident ({c_confidence}%), no escalation needed",
        )



    total_time = time.monotonic() - start
    agent_tokens = sum(r.get("tokens_used", 0) for r in agent_results.values())
    rlm_tokens = c_result.get("tokens_used", 0) + (
        d_result.get("tokens_used", 0) if d_result else 0
    )
    total_tokens = agent_tokens + rlm_tokens

    # ── Hallucination Safety Gate (V1 parity with V2 runner.py) ──
    try:
        from app.pipeline.analysis.hallucination_checker import check_hallucinations
        from app.data.market_data_store import get_latest_snapshot

        snapshot = get_latest_snapshot(ticker)
        if snapshot:
            snapshot_dict = dataclasses.asdict(snapshot)
            context_provenance = {
                k: {"value": v, "source": "MarketSnapshot"}
                for k, v in snapshot_dict.items()
                if v is not None
            }
        else:
            context_provenance = {}

        hall_result = check_hallucinations(
            llm_output={
                "action": final_action,
                "confidence": final_confidence,
                "rationale": final_rationale,
            },
            context_provenance=context_provenance,
            raw_context=context,
            ticker=ticker,
        )

        if hall_result["rejected"]:
            logger.warning(
                "[PIPELINE] [HALLUCINATION] %s: REJECTED — %s. Downgrading action.",
                ticker,
                hall_result["rejection_reason"],
            )
            final_rationale += (
                f"\n\n⚠️ HALLUCINATION GATE REJECTED: {hall_result['rejection_reason']}"
            )
            final_action = gate_action("HOLD", held)
            final_confidence = max(10, final_confidence // 2)
            emit(
                "analyzing",
                f"hallucination_{ticker}",
                f"⚠️ {ticker}: Hallucination gate REJECTED — "
                f"downgraded to HOLD @ {final_confidence}%",
                status="warning",
            )
        elif hall_result["hallucinations"]:
            logger.info(
                "[PIPELINE] [HALLUCINATION] %s: %d minor hallucinations (below threshold)",
                ticker,
                len(hall_result["hallucinations"]),
            )
    except Exception as hall_err:
        logger.warning(
            "[PIPELINE] Hallucination check failed for %s (non-fatal): %s",
            ticker,
            hall_err,
        )

    result = {
        "ticker": ticker,
        "action": final_action,
        "confidence": final_confidence,
        "rationale": final_rationale,
        "config_used": config_used,
        "decision_model": decision_model_name,
        "decision_role": decision_role,
        "escalated": d_result is not None,
        "agent_results": {
            k: {
                "response": v.get("response", "")[:300],
                "tokens": v.get("tokens_used", 0),
            }
            for k, v in agent_results.items()
        },
        "c_result": c_result,
        "d_result": d_result,
        "human_review": human_review,
        "agent_tokens": agent_tokens,
        "rlm_tokens": rlm_tokens,
        "total_tokens": total_tokens,
        "total_time_s": round(total_time, 2),
        "data_sources": present_sources,
        "missing_sources": missing_sources,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    _log_decision(result, cycle_id, bot_id)

    # ── Execute Post-cycle Hooks ──
    try:
        from app.pipeline.orchestration.post_cycle_hooks import run_post_cycle_hooks

        await run_post_cycle_hooks(
            ticker=ticker,
            result=result,
            escalated=d_result is not None,
            cycle_id=cycle_id,
            final_action=final_action,
            final_confidence=final_confidence,
        )
    except Exception as hooks_err:
        logger.warning(
            "[PIPELINE] [HOOKS] Failed for %s (non-fatal): %s", ticker, hooks_err
        )

    logger.info(
        "[FINAL] %s @ %s%% | Config: %s | Agents: %s tok | RLM: %s tok | Time: %.1fs",
        final_action,
        final_confidence,
        config_used,
        f"{agent_tokens:,}",
        f"{rlm_tokens:,}",
        total_time,
    )
    if human_review:
        logger.warning("[PIPELINE] [!] HUMAN REVIEW REQUIRED")
    logger.info("[PIPELINE] =" * 70)

    # Emit final decision
    emoji = "🟢" if final_action == "BUY" else "🔴" if final_action == "SELL" else "🟡"
    emit(
        "analyzing",
        f"decision_{ticker}",
        f"{emoji} {ticker}: {final_action} @ {final_confidence}% "
        f"| {config_used} | {total_tokens:,} tokens | {total_time:.1f}s"
        + (" ⚠️ HUMAN REVIEW" if human_review else ""),
        data={
            "action": final_action,
            "confidence": final_confidence,
            "config": config_used,
            "tokens": total_tokens,
            "human_review": human_review,
            "rationale": final_rationale[:300],
        },
        elapsed_ms=int(total_time * 1000),
    )

    # Store triage tier in result for audit trail
    result["triage_tier"] = triage_tier

    # Record analysis in attention tracker
    try:
        from app.pipeline.attention_tracker import record_analysis as _record_attn

        _record_attn(
            ticker,
            action=final_action,
            confidence=final_confidence,
            was_deep=(triage_tier == "deep"),
        )
    except Exception as attn_err:
        logger.warning(
            "[PIPELINE] Attention tracker failed for %s: %s", ticker, attn_err
        )

    return result


async def analyze_tickers(
    tickers: list[str],
    cycle_id: str = "",
    bot_id: str = "",
    emit: Callable | None = None,
    macro_memo: str = "",
) -> list[dict]:
    """Analyze all tickers in parallel — the priority queue handles concurrency.

    No more TICKER_CHUNK_SIZE chunking. All tickers are fired at once via
    asyncio.gather(). The vllm_client.py priority queue dispatcher keeps
    its JETSON and DGX concurrent slots filled at all times.

    Args:
        macro_memo: Pre-computed macro strategy memo injected into every
                    ticker's analysis context.
    """
    if emit is None:
        emit = _noop
    from app.config import settings

    # ── Frozen snapshot pattern: load memory ONCE for the entire cycle ──
    # All tickers see the same memory snapshot. Mid-cycle add() calls
    # from post_cycle_learn don't affect the snapshot until next cycle.
    try:
        from app.services.trading_memory import trading_memory

        trading_memory.load_from_disk()
    except Exception as mem_err:
        logger.warning("[PIPELINE] [MEMORY] Failed to load (non-fatal): %s", mem_err)

    # ── Living Graph: seed ontology for all tickers ──
    try:
        from app.cognition.ontology.ontology_builder import BrainGraph

        for t in tickers:
            try:
                BrainGraph.seed_from_ticker_metadata(t)
            except Exception:
                pass
    except Exception as graph_err:
        logger.debug("[PIPELINE] Graph seeding failed (non-fatal): %s", graph_err)

    timeout_seconds = settings.CYCLE_TIMEOUT_MINUTES * 60

    logger.info(
        "[TICKERS] Analyzing %d tickers in parallel (queue-managed concurrency)%s",
        len(tickers),
        f" with macro memo ({len(macro_memo):,} chars)" if macro_memo else "",
    )
    emit(
        "analyzing",
        "all_tickers",
        f"Analyzing {len(tickers)} tickers in parallel (priority queue)"
        + (" + macro memo" if macro_memo else ""),
        status="running",
    )

    try:
        raw_results = await asyncio.wait_for(
            asyncio.gather(
                *[
                    analyze_ticker(
                        t,
                        cycle_id=cycle_id,
                        bot_id=bot_id,
                        emit=emit,
                        macro_memo=macro_memo,
                        watchlist=tickers,
                    )
                    for t in tickers
                ],
                return_exceptions=True,
            ),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "[TICKERS] CYCLE TIMEOUT after %d min",
            settings.CYCLE_TIMEOUT_MINUTES,
        )
        emit(
            "analyzing",
            "cycle_timeout",
            f"Cycle timeout ({settings.CYCLE_TIMEOUT_MINUTES}min)",
            status="error",
        )
        return [{"ticker": t, "error": "cycle_timeout"} for t in tickers]

    results = []
    for t, r in zip(tickers, raw_results):
        if isinstance(r, asyncio.CancelledError):
            raise r
        if isinstance(r, Exception):
            logger.error("[PIPELINE] [TICKERS] %s failed: %s", t, r)
            results.append({"ticker": t, "error": str(r)})
            emit(
                "analyzing", f"error_{t}", f"{t}: Analysis FAILED — {r}", status="error"
            )
        else:
            results.append(r)
    return results


def _log_decision(result: dict, cycle_id: str, bot_id: str) -> None:
    """Log the decision to analysis_results table.

    Stores ALL fields the frontend needs so they survive DB round-trips:
    estimate, agent_results, c_result, d_result, total_time_s, total_tokens.
    """
    try:
        with get_db() as db:
            result_id = str(uuid.uuid4())

            # Compute estimate inline for BUY actions so it persists in DB
            estimate = None
            action = result.get("action", "HOLD")
            confidence = result.get("confidence", 0)
            ticker = result["ticker"]

            if action == "BUY" and confidence > 0:
                try:
                    from app.pipeline.trading_phase import estimate_trade
                    from app.trading.paper_trader import get_portfolio

                    pf = get_portfolio(bot_id or "default")
                    cash = pf.get("cash", 0)
                    price_row = db.execute(
                        "SELECT close FROM price_history WHERE ticker = %s ORDER BY date DESC LIMIT 1",
                        [ticker],
                    ).fetchone()
                    if price_row and price_row[0] > 0:
                        estimate = estimate_trade(confidence, cash, price_row[0])
                except Exception as est_err:
                    logger.warning(
                        "[PIPELINE] Estimate calc failed for %s: %s", ticker, est_err
                    )

            # Build the full result payload for the frontend
            result_payload = {
                "action": action,
                "confidence": confidence,
                "rationale": result.get("rationale", ""),
                "config_used": result.get("config_used", ""),
                "escalated": result.get("escalated", False),
                "human_review": result.get("human_review", False),
                "agent_tokens": result.get("agent_tokens", 0),
                "rlm_tokens": result.get("rlm_tokens", 0),
                "total_tokens": result.get("total_tokens", 0),
                "total_time_s": result.get("total_time_s"),
                "agent_results": result.get("agent_results", {}),
                "c_result": {
                    "action": result.get("c_result", {}).get("action"),
                    "confidence": result.get("c_result", {}).get("confidence"),
                }
                if result.get("c_result")
                else None,
                "d_result": {
                    "action": result.get("d_result", {}).get("action"),
                    "confidence": result.get("d_result", {}).get("confidence"),
                }
                if result.get("d_result")
                else None,
            }

            if estimate:
                result_payload["estimate"] = estimate

            # Determine if this run should save thesis state
            _is_thesis_run = result.get("triage_tier") in ("standard", "deep")
            _thesis_now = datetime.now(timezone.utc) if _is_thesis_run else None

            db.execute(
                """
                INSERT INTO analysis_results
                (id, cycle_id, bot_id, ticker, agent_name, result_json, confidence, created_at, triage_tier,
                 thesis_verdict, thesis_confidence, thesis_summary, thesis_updated_at, thesis_unchanged)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, FALSE)
                ON CONFLICT (id) DO NOTHING
            """,
                [
                    result_id,
                    cycle_id or "manual",
                    bot_id or "decision-engine",
                    ticker,
                    f"hybrid_{result.get('config_used', 'C')}",
                    json.dumps(result_payload),
                    confidence,
                    result.get("timestamp"),
                    result.get("triage_tier", "standard"),
                    # Thesis fields — only populated for standard/deep runs
                    action if _is_thesis_run else None,
                    confidence if _is_thesis_run else None,
                    result.get("rationale", "")[:1500] if _is_thesis_run else None,
                    _thesis_now,
                ],
            )

            # A4: Log to cycle_summaries
            db.execute(
                """
                INSERT INTO cycle_summaries
                (ticker, cycle_id, cycle_date, agent_name, action, confidence, confidence_tier, rationale_summary, was_correct, outcome_pnl)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    ticker,
                    cycle_id or "manual",
                    result.get("timestamp"),
                    f"hybrid_{result.get('config_used', 'C')}",
                    action,
                    confidence,
                    "high"
                    if confidence >= 70
                    else "medium"
                    if confidence >= 40
                    else "low",
                    result.get("rationale", "")[:500],
                    None,  # was_correct
                    None,  # outcome_pnl
                ],
            )
    except Exception as e:
        logger.error("[PIPELINE] [decision_engine] log failed: %s", e)
