"""
Cognition V2 — Pipeline Runner.

Executes the full V2 cognition sequence for a single ticker:
  1. Ontology enrichment (optional, non-blocking)
  2. Evidence Packet build from DB
  3. Sufficiency Gate check
  4. Dynamic retrieval loop (max 1 retry on critical gaps)
  5. Memory context injection (prior episodes + procedural rules)
  6. Thesis generation via LLM
  7. Episodic memory write-back
  8. V2 cycle log

Returns a dict matching V1's analyze_ticker() shape so downstream
phases (trading, post-cycle hooks, reports) work unchanged.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from app.cognition.orchestration.models import CognitionRunResult
from app.log_manager import log_manager

logger = logging.getLogger(__name__)


async def execute_v2_pipeline(
    ticker: str,
    *,
    cycle_id: str = "",
    bot_id: str = "",
    emit: Callable[..., Any] | None = None,
    macro_memo: str = "",
    watchlist: list[str] | None = None,
    db_semaphore: asyncio.Semaphore | None = None,
    is_highly_redundant: bool = False,
) -> dict[str, Any]:
    """Run the full V2 cognition pipeline for a single ticker.

    Returns a dict with the same keys as V1's analyze_ticker() so the
    trading phase, post-cycle hooks, and report generation work unchanged.
    """
    from app.utils.pipeline_utils import noop as _noop
    from app.utils.pipeline_utils import elapsed_ms

    if emit is None:
        emit = _noop

    start = time.monotonic()
    ticker = ticker.upper()
    if not cycle_id:
        cycle_id = f"v2-{uuid.uuid4().hex[:8]}"

    total_tokens = 0
    stages: list[str] = []
    stage_timings: dict[str, int] = {}  # stage_name → elapsed_ms

    emit(
        "analyzing",
        f"v2_start_{ticker}",
        f"{ticker}: V2 cognition pipeline starting",
        status="running",
    )

    # ── Step 0: Pause check ──────────────────────────────────────────
    from app.pipeline.orchestration.cycle_control import cycle_control

    await cycle_control.wait_if_paused()
    
    # ── Fetch Position State & Risk Dashboard Early ──────────────────
    position_context: dict = {}
    try:
        from app.tools.portfolio_tools import get_position_context
        position_context = get_position_context(ticker, bot_id)
    except Exception as pos_err:
        logger.debug("[V2] Position context query failed early for %s: %s", ticker, pos_err)
    held = position_context.get("held", False)

    portfolio_dashboard: str = ""
    try:
        from app.tools.portfolio_tools import get_portfolio_risk_dashboard
        portfolio_dashboard = get_portfolio_risk_dashboard(ticker, bot_id)
    except Exception as port_err:
        logger.debug("[V2] Portfolio risk dashboard query failed early for %s: %s", ticker, port_err)

    log_manager.log_v2_cycle(cycle_id, "v2_start", {
        "ticker": ticker, "held": held,
        "position_context": {k: v for k, v in position_context.items() if k != "raw"} if position_context else {},
    })

    # ── Step 0.5: Data completeness (shared with V1) ─────────────────
    t0 = time.monotonic()
    from app.pipeline.data.data_completeness import check_and_fill

    try:
        data_report = await asyncio.wait_for(check_and_fill(ticker, emit=emit), timeout=30.0)
    except asyncio.TimeoutError:
        logger.warning("[V2] Data completeness TIMEOUT for %s (30s) — proceeding without gap fill", ticker)
        data_report = {}
    ms0 = elapsed_ms(t0)
    filled = data_report.get("filled", [])
    if filled:
        logger.info("[V2] [DATA] Filled gaps for %s: %s", ticker, filled)
        emit(
            "analyzing",
            f"v2_data_{ticker}",
            f"{ticker}: Filled {len(filled)} data gaps",
            data={"filled": filled},
            elapsed_ms=ms0,
        )
    stages.append("data_completeness")
    stage_timings["data_completeness"] = ms0
    log_manager.log_v2_cycle(cycle_id, "v2_data_completeness", {
        "ticker": ticker, "filled": filled, "elapsed_ms": ms0,
    })

    # ── Step 1: Ontology (optional, non-blocking) ────────────────────
    ontology_ctx: dict[str, Any] = {}
    try:
        from app.cognition.ontology.ontology_builder import OntologyBuilder

        await cycle_control.wait_if_paused()
        t1 = time.monotonic()
        ontology_ctx = await asyncio.wait_for(
            OntologyBuilder().execute(ticker, {"cycle_id": cycle_id}),
            timeout=15.0,
        )
        ms1 = elapsed_ms(t1)
        stages.append("ontology")
        stage_timings["ontology"] = ms1
        emit(
            "analyzing",
            f"v2_ontology_{ticker}",
            f"{ticker}: Ontology built "
            f"({len(ontology_ctx.get('ontology_nodes', []))} nodes)",
            elapsed_ms=ms1,
        )
        log_manager.log_v2_cycle(cycle_id, "v2_ontology", {
            "ticker": ticker, "node_count": len(ontology_ctx.get("ontology_nodes", [])),
            "elapsed_ms": ms1,
        })
    except Exception as e:
        logger.warning("[V2] Ontology failed for %s (non-fatal): %s", ticker, e)
        emit(
            "analyzing",
            f"v2_ontology_{ticker}",
            f"{ticker}: Ontology skipped — {e}",
            status="warning",
        )

    # ── Step 2: Build Evidence Packet ────────────────────────────────
    from app.cognition.evidence.packet_builder import build_evidence_packet

    await cycle_control.wait_if_paused()
    t2 = time.monotonic()
    try:
        packet = await asyncio.wait_for(build_evidence_packet(ticker), timeout=30.0)
    except asyncio.TimeoutError:
        logger.error("[V2] Evidence packet build TIMEOUT for %s (30s)", ticker)
        emit("analyzing", f"v2_evidence_timeout_{ticker}", f"{ticker}: Evidence build TIMEOUT", status="error")
        raise
    ms2 = elapsed_ms(t2)
    stages.append("evidence_build")
    stage_timings["evidence_build"] = ms2
    # Diagnostic logging for evidence packet health
    _teaser = (
        packet.source_quality_summary.teaser_artifact_risk
        if packet.source_quality_summary
        else 0.0
    )
    _diversity = (
        packet.source_quality_summary.source_diversity
        if packet.source_quality_summary
        else 0
    )
    logger.info(
        "[V2] [EVIDENCE] %s: %d claims, %d structured facts, %d sources, "
        "missing=%s, teaser_risk=%.2f, diversity=%d",
        ticker,
        len(packet.claims),
        len(packet.structured_facts),
        len(packet.source_summaries),
        packet.missing_fields or "none",
        _teaser,
        _diversity,
    )
    emit(
        "analyzing",
        f"v2_evidence_{ticker}",
        f"{ticker}: Evidence packet — {len(packet.claims)} claims, "
        f"{len(packet.structured_facts)} facts, "
        f"{len(packet.missing_fields)} missing, "
        f"teaser_risk={_teaser:.0%}, diversity={_diversity}",
        elapsed_ms=ms2,
    )
    log_manager.log_v2_cycle(cycle_id, "v2_evidence_build", {
        "ticker": ticker, "claims": len(packet.claims),
        "structured_facts": len(packet.structured_facts),
        "sources": len(packet.source_summaries),
        "missing_fields": packet.missing_fields,
        "teaser_risk": _teaser, "diversity": _diversity, "elapsed_ms": ms2,
    })

    # ── Step 3: Sufficiency Gate ─────────────────────────────────────
    from app.cognition.verification.sufficiency_gate import check_data_sufficiency

    sufficiency = check_data_sufficiency(ticker, packet)
    stages.append("sufficiency_gate")
    log_manager.log_v2_cycle(cycle_id, "v2_sufficiency_gate", {
        "ticker": ticker, "status": sufficiency.status,
        "blockers": sufficiency.blockers if hasattr(sufficiency, "blockers") else [],
        "warnings": sufficiency.warnings if hasattr(sufficiency, "warnings") else [],
    })

    # ── Step 4: Dynamic retrieval loop (max 1 retry) ─────────────────
    retrieval_retries = 0
    if sufficiency.status == "critical_gap" and packet.missing_fields:
        emit(
            "analyzing",
            f"v2_retrieval_{ticker}",
            f"{ticker}: Critical gap detected — triggering dynamic retrieval "
            f"for {packet.missing_fields}",
            status="running",
        )
        try:
            from app.pipeline.analysis.dynamic_tool_router import resolve_missing_data

            await cycle_control.wait_if_paused()
            t_r = time.monotonic()
            fetched = await resolve_missing_data(ticker, packet.missing_fields)
            ms_r = elapsed_ms(t_r)

            if fetched:
                retrieval_retries = 1
                # Rebuild packet with new data
                packet = await build_evidence_packet(ticker)
                sufficiency = check_data_sufficiency(ticker, packet)
                emit(
                    "analyzing",
                    f"v2_retrieval_done_{ticker}",
                    f"{ticker}: Retrieval done, re-evaluated "
                    f"sufficiency → {sufficiency.status}",
                    elapsed_ms=ms_r,
                )
            stages.append("dynamic_retrieval")
        except Exception as e:
            logger.warning("[V2] Dynamic retrieval failed for %s: %s", ticker, e)
            emit(
                "analyzing",
                f"v2_retrieval_fail_{ticker}",
                f"{ticker}: Dynamic retrieval failed — {e}",
                status="warning",
            )

    # ── Step 4.5: If still critical, ABSTAIN ──────────────────────────
    if sufficiency.status == "critical_gap":
        blockers_text = "; ".join(sufficiency.blockers)
        logger.warning(
            "[V2] ABSTAIN for %s — critical gaps remain: %s", ticker, blockers_text
        )

        # Check if missing critical price data, which indicates a fake or delisted ticker
        if "price" in packet.missing_fields or "Missing critical price history data." in sufficiency.blockers:
            logger.warning("[V2] %s is missing critical price data — auto-rejecting and removing from watchlist.", ticker)
            try:
                from app.processors.ticker_extractor import (
                    get_registry as _get_reg_yf,
                    _save_rejected_to_db as _reject_db,
                    FALSE_TICKERS as _FT,
                )
                from app.db.connection import get_db

                _reg_yf = _get_reg_yf()
                _reg_yf.add_rejected(ticker)
                _FT.add(ticker)
                _reject_db(ticker)

                with get_db() as db:
                    db.execute("DELETE FROM watchlist WHERE ticker = %s", [ticker])
            except Exception as rej_err:
                logger.debug("[V2] auto-reject write failed for %s: %s", ticker, rej_err)
            
            emit(
                "analyzing",
                f"v2_reject_{ticker}",
                f"{ticker}: THROWN OUT — missing critical price data",
                status="error",
            )
            # Return None so Phase 4 drops it entirely and does not log a decision
            return None

        emit(
            "analyzing",
            f"v2_abstain_{ticker}",
            f"{ticker}: ABSTAIN — {blockers_text}",
            status="warning",
        )
        from app.cognition.debate.action_gate import gate_action
        abstain_result = _build_v1_compatible_result(
            ticker=ticker,
            action=gate_action("HOLD", held),
            confidence=0,
            rationale=f"V2 ABSTAIN: Insufficient evidence. {blockers_text}",
            cycle_id=cycle_id,
            total_tokens=0,
            elapsed=time.monotonic() - start,
            stages=stages,
            config_used="v2_abstain",
        )
        # Log abstain decisions so they appear in analysis_results / frontend
        try:
            from app.pipeline.analysis.decision_engine import _log_decision

            if db_semaphore:
                async with db_semaphore:
                    _log_decision(abstain_result, cycle_id, bot_id)
            else:
                _log_decision(abstain_result, cycle_id, bot_id)
            stages.append("db_log")
        except Exception as e:
            logger.warning("[V2] _log_decision (abstain) failed for %s: %s", ticker, e)
        try:
            from app.pipeline.orchestration.post_cycle_hooks import run_post_cycle_hooks

            await run_post_cycle_hooks(
                ticker=ticker,
                result=abstain_result,
                escalated=False,
                cycle_id=cycle_id,
                final_action=gate_action("HOLD", held),
                final_confidence=0,
            )
            stages.append("post_cycle_hooks")
        except Exception as e:
            logger.warning(
                "[V2] post_cycle_hooks (abstain) failed for %s: %s", ticker, e
            )
        return abstain_result

    # ── Step 5: Memory context injection ──────────────────────────────
    memory_context: dict[str, Any] = {}
    try:
        from app.cognition.memory.reader import read_memories, read_procedural

        await cycle_control.wait_if_paused()
        loop = asyncio.get_running_loop()
        prior_episodes, procedural_rules = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: (
                    read_memories(ticker, memory_types=["episodic"], limit=5),
                    read_procedural(tags=[ticker.lower(), "all"], limit=10),
                ),
            ),
            timeout=10.0,
        )

        memory_lines: list[str] = []
        for ep in prior_episodes:
            payload = ep.payload or {}
            memory_lines.append(
                f"- [{ep.created_at[:10]}] {payload.get('event_type', 'run')}: "
                f"{payload.get('action', '?')} (conf={ep.confidence:.0f})"
            )
        for rule in procedural_rules:
            memory_lines.append(
                f"- RULE: {rule.rule_text} (conf={rule.confidence:.2f})"
            )

        log_manager.log_v2_cycle(cycle_id, "v2_memory_read", {
            "ticker": ticker, "episodes": len(prior_episodes),
            "rules": len(procedural_rules),
        })
        memory_context = {
            "episode_count": len(prior_episodes),
            "rule_count": len(procedural_rules),
            "memory_brief": (
                "\n".join(memory_lines) if memory_lines else "No prior memory."
            ),
        }
        stages.append("memory_read")
    except asyncio.TimeoutError:
        logger.warning("[V2] Memory read timed out for %s", ticker)
        memory_context = {
            "episode_count": 0,
            "rule_count": 0,
            "memory_brief": "Memory unavailable (timeout).",
        }
    except Exception as e:
        logger.warning("[V2] Memory read failed for %s (non-fatal): %s", ticker, e)
        memory_context = {
            "episode_count": 0,
            "rule_count": 0,
            "memory_brief": "Memory unavailable.",
        }

    # ── Step 5.5: Meta-Orchestrator Agent Routing ─────────────────────
    from app.cognition.orchestration.meta_orchestrator import MetaOrchestrator

    emit(
        "analyzing",
        f"v2_orchestrator_{ticker}",
        f"{ticker}: MetaOrchestrator determining sub-agent routing...",
        status="running",
    )
    await cycle_control.wait_if_paused()
    t_orch = time.monotonic()
    try:
        agent_insights, orch_tokens = await asyncio.wait_for(
            MetaOrchestrator.orchestrate(
                ticker, packet, sufficiency, cycle_id, bot_id, is_highly_redundant
            ),
            timeout=60.0,
        )
    except asyncio.TimeoutError:
        logger.warning("[V2] MetaOrchestrator TIMEOUT for %s (60s) — proceeding without agent insights", ticker)
        agent_insights, orch_tokens = {}, 0
        emit("analyzing", f"v2_orchestrator_timeout_{ticker}", f"{ticker}: MetaOrchestrator TIMEOUT", status="warning")
    total_tokens += orch_tokens
    ms_orch = elapsed_ms(t_orch)
    stages.append("meta_orchestration")
    log_manager.log_v2_cycle(cycle_id, "v2_meta_orchestration", {
        "ticker": ticker, "agent_count": len(agent_insights) if agent_insights else 0,
        "agent_keys": list(agent_insights.keys()) if agent_insights else [],
        "tokens": orch_tokens, "elapsed_ms": ms_orch,
    })

    if agent_insights:
        emit(
            "analyzing",
            f"v2_orchestrator_done_{ticker}",
            f"{ticker}: MetaOrchestrator completed {len(agent_insights)} specialist agents",
            elapsed_ms=ms_orch,
        )

    # ── Step 5.6: Position Context + Adversarial Debate ─────────────────
    # We already queried position_context early, but we log it now
    try:
        if held:
            emit(
                "analyzing",
                f"v2_position_{ticker}",
                f"{ticker}: Bot HOLDS position — "
                f"entry=${position_context['avg_entry']}, "
                f"P&L={position_context['unrealized_pnl_pct']:+.1f}%, "
                f"held {position_context['holding_days']}d. "
                f"Debate will include sell-thesis framing.",
                status="ok",
            )
    except Exception as pos_err:
        logger.debug(
            "[V2] Position context query failed for %s: %s",
            ticker,
            pos_err,
        )

    # ── Step 5.65: Read team findings from TaskBoard ─────────────────
    # Pulls findings posted by specialist agents (via MetaOrchestrator)
    # and injects them into agent_insights so the debate sees team context.
    team_findings_summary = ""
    try:
        from app.agents.task_board import task_board

        findings = await task_board.get_findings(
            ticker=ticker,
            cycle_id=cycle_id,
        )
        if findings:
            finding_lines = []
            for f in findings[:10]:  # Cap at 10 to avoid context bloat
                src = f.get("source_agent", "?")
                cat = f.get("category", "fact")
                content = f.get("content", "")[:200]
                conf = f.get("confidence", 0)
                finding_lines.append(
                    f"- [{cat.upper()}] ({src}, conf={conf}): {content}"
                )
            team_findings_summary = "\n".join(finding_lines)
            if agent_insights is None:
                agent_insights = {}
            agent_insights["team_findings"] = (
                f"# TEAM FINDINGS FROM SPECIALIST AGENTS\n"
                f"{len(findings)} findings shared by team:\n"
                f"{team_findings_summary}"
            )
            logger.info(
                "[V2] [COLLAB] Injected %d team findings for %s into debate context",
                len(findings), ticker,
            )
            emit(
                "analyzing",
                f"v2_team_findings_{ticker}",
                f"{ticker}: {len(findings)} team findings injected into debate context",
                status="ok",
            )
    except Exception as tb_err:
        logger.debug("[V2] TaskBoard read failed for %s (non-fatal): %s", ticker, tb_err)

    debate_result = None
    try:
        from app.cognition.debate.debate_coordinator import (
            run_adversarial_debate,
        )

        emit(
            "analyzing",
            f"v2_debate_{ticker}",
            f"{ticker}: Starting adversarial debate "
            f"({'HOLD-vs-SELL' if position_context.get('held') else 'BUY-vs-SELL'})...",
            status="running",
        )
        await cycle_control.wait_if_paused()
        t_debate = time.monotonic()
        # Add a granular timeout (5 minutes) to ensure a hung debate agent doesn't consume the entire cycle timeout.
        # With FAST_DEBATE_MODE and reduced tool turns, debates should complete in 2-3 minutes.
        debate_result = await asyncio.wait_for(
            run_adversarial_debate(
                ticker=ticker,
                packet=packet,
                cycle_id=cycle_id,
                bot_id=bot_id,
                agent_insights=agent_insights,
                position_context=position_context,
                portfolio_dashboard=portfolio_dashboard,
            ),
            timeout=300.0,
        )
        ms_debate = elapsed_ms(t_debate)

        if debate_result:
            total_tokens += debate_result.total_tokens
            stages.append("adversarial_debate")
            stage_timings["adversarial_debate"] = ms_debate
            emoji_d = (
                "🟢"
                if debate_result.judge_action == "BUY"
                else "🔴"
                if debate_result.judge_action == "SELL"
                else "🟡"
            )
            emit(
                "analyzing",
                f"v2_debate_done_{ticker}",
                f"{emoji_d} {ticker}: Debate verdict — {debate_result.judge_action} @ "
                f"{debate_result.judge_confidence}% (winner: {debate_result.winning_side}, "
                f"integrity: {debate_result.integrity_status})",
                elapsed_ms=ms_debate,
            )
            log_manager.log_v2_cycle(cycle_id, "v2_debate_result", {
                "ticker": ticker,
                "action": debate_result.judge_action,
                "confidence": debate_result.judge_confidence,
                "winner": debate_result.winning_side,
                "integrity": debate_result.integrity_status,
                "bull_claims": len(debate_result.bull_claims),
                "bear_claims": len(debate_result.bear_claims),
                "verified_bull": len(debate_result.verified_bull_claims),
                "verified_bear": len(debate_result.verified_bear_claims),
                "unverified": len(debate_result.unverified_claims),
                "rationale": debate_result.judge_rationale[:500] if debate_result.judge_rationale else "",
                "key_factor": debate_result.key_deciding_factor or "",
                "persona_outcomes": debate_result.persona_outcomes or {},
                "total_debate_tokens": debate_result.total_tokens,
                "elapsed_ms": ms_debate,
            })
        else:
            emit(
                "analyzing",
                f"v2_debate_skip_{ticker}",
                f"{ticker}: Debate skipped (disabled or no analyst endpoints)",
                status="warning",
            )
    except Exception as e:
        logger.warning(
            "[V2] Adversarial debate failed for %s (non-fatal): %s", ticker, e
        )
        emit(
            "analyzing",
            f"v2_debate_fail_{ticker}",
            f"{ticker}: Debate failed — {e}",
            status="warning",
        )

    # ── Step 6: Thesis generation (LLM call) ──────────────────────────
    from app.cognition.debate.thesis_agent import generate_thesis

    # Build extra context from ontology subgraph + macro memo + agent insights
    # Token budget guard: ~4 chars/token, cap extra_context to avoid exceeding
    # the LLM context window.  Trim least-critical parts first (ontology, then memory).
    MAX_EXTRA_CONTEXT_CHARS = 6000  # ~1500 tokens budget for injected context

    extra_context_parts: list[str] = []
    budget_used = 0

    # Inject portfolio risk & correlation dashboard as top priority context
    if portfolio_dashboard:
        extra_context_parts.append(portfolio_dashboard)
        budget_used += len(portfolio_dashboard)

    # Inject position context as HIGH-PRIORITY context (before debate)
    if position_context.get("held"):
        from app.tools.portfolio_tools import (
            format_position_context_for_prompt,
        )

        pos_block = format_position_context_for_prompt(position_context)
        if pos_block:
            extra_context_parts.append(pos_block)
            budget_used += len(pos_block)

    # Inject Trading Constitution (adaptive rules from DB)
    try:
        from app.pipeline.trading_constitution import (
            format_constitution_for_prompt,
        )

        constitution_block = format_constitution_for_prompt()
        if constitution_block:
            remaining = MAX_EXTRA_CONTEXT_CHARS - budget_used
            if remaining > 200:
                trimmed = constitution_block[:remaining]
                extra_context_parts.append(trimmed)
                budget_used += len(trimmed)
    except Exception as const_err:
        logger.debug(
            "[V2] Constitution load failed (non-fatal): %s",
            const_err,
        )

    # Inject debate result as FIRST context (highest priority)
    if debate_result and debate_result.judge_rationale:
        debate_summary = (
            f"# ADVERSARIAL DEBATE RESULT\n"
            f"**Verdict:** {debate_result.judge_action} @ {debate_result.judge_confidence}%\n"
            f"**Winner:** {debate_result.winning_side}\n"
            f"**Key Factor:** {debate_result.key_deciding_factor}\n"
            f"**Rationale:** {debate_result.judge_rationale}\n"
            f"**Evidence Quality:** {debate_result.integrity_status} "
            f"({len(debate_result.unverified_claims)} claims rejected)\n"
            f"**Bull claims verified:** {len(debate_result.verified_bull_claims)}/{len(debate_result.bull_claims)}\n"
            f"**Bear claims verified:** {len(debate_result.verified_bear_claims)}/{len(debate_result.bear_claims)}"
        )
        extra_context_parts.append(debate_summary)
        budget_used += len(debate_summary)

    if macro_memo:
        part = f"# MACRO STRATEGY MEMO\n{macro_memo}"
        extra_context_parts.append(part)
        budget_used += len(part)

    ontology_text = ontology_ctx.get("ontology_context", "")
    if ontology_text:
        remaining = MAX_EXTRA_CONTEXT_CHARS - budget_used
        if remaining > 200:
            extra_context_parts.append(ontology_text[:remaining])
            budget_used += min(len(ontology_text), remaining)

    mem_brief_pre = memory_context.get("memory_brief", "")
    if mem_brief_pre and mem_brief_pre != "No prior memory.":
        remaining = MAX_EXTRA_CONTEXT_CHARS - budget_used
        if remaining > 100:
            trimmed = mem_brief_pre[: min(500, remaining)]
            extra_context_parts.append(f"# PRIOR MEMORY\n{trimmed}")
            budget_used += len(trimmed) + 16

    if agent_insights:
        remaining = MAX_EXTRA_CONTEXT_CHARS - budget_used
        if remaining > 200:
            insights_str = "\n".join(
                [
                    f"## {k.upper()} AGENT INSIGHT\n{v}"
                    for k, v in agent_insights.items()
                ]
            )
            part = f"# SPECIALIZED AGENT INSIGHTS\n{insights_str}"
            extra_context_parts.append(part[:remaining])
            budget_used += len(part[:remaining])

    # ── Inject Autoresearch Lessons ──
    try:
        from app.cognition.lesson_store import retrieve_lessons

        lessons = retrieve_lessons(ticker, k=2)
        if lessons:
            lesson_texts = "\n".join(f"- {l.get('lesson_text', '')}" for l in lessons)
            part = (
                "# AUTORESEARCH LESSONS\n"
                "The following are critical lessons and recommendations from past autoresearch cycles. "
                "You MUST adhere to these rules to avoid repeating past mistakes:\n\n"
                f"{lesson_texts}"
            )
            remaining = MAX_EXTRA_CONTEXT_CHARS - budget_used
            if remaining > 200:
                extra_context_parts.append(part[:remaining])
                budget_used += len(part[:remaining])
            emit(
                "analyzing",
                f"autoresearch_{ticker}",
                f"{ticker}: Injected {len(lessons)} Autoresearch lessons",
            )
            logger.info(
                "[V2] [MEMORY] Injected %d Autoresearch lessons for %s",
                len(lessons),
                ticker,
            )
    except Exception as ar_err:
        logger.warning(
            "[V2] [PIPELINE] Failed to retrieve autoresearch lessons: %s", ar_err
        )

    extra_context = "\n\n".join(extra_context_parts)

    emit(
        "analyzing",
        f"v2_thesis_{ticker}",
        f"{ticker}: Generating thesis via LLM...",
        status="running",
    )

    await cycle_control.wait_if_paused()
    t6 = time.monotonic()
    try:
        thesis, thesis_tokens = await asyncio.wait_for(
            generate_thesis(
                entity_id=ticker,
                packet=packet,
                bias="neutral",
                cycle_id=cycle_id,
                bot_id=bot_id,
                extra_context=extra_context,
                watchlist=watchlist or [],
                held=held,
            ),
            timeout=120.0,
        )
    except asyncio.TimeoutError:
        logger.error("[V2] Thesis generation TIMEOUT for %s (120s)", ticker)
        emit("analyzing", f"v2_thesis_timeout_{ticker}", f"{ticker}: Thesis LLM TIMEOUT", status="error")
        raise
    total_tokens += thesis_tokens
    ms6 = elapsed_ms(t6)
    stages.append("thesis_generation")
    stage_timings["thesis_generation"] = ms6
    log_manager.log_v2_cycle(cycle_id, "v2_thesis", {
        "ticker": ticker, "action": thesis.action,
        "confidence": thesis.confidence,
        "claims_count": len(thesis.core_claims),
        "weaknesses_count": len(thesis.weaknesses),
        "tokens": thesis_tokens, "elapsed_ms": ms6,
    })

    emit(
        "analyzing",
        f"v2_thesis_done_{ticker}",
        f"{ticker}: Thesis \u2192 {thesis.action} @ {thesis.confidence}% "
        f"(claims: {len(thesis.core_claims)}, weaknesses: {len(thesis.weaknesses)})",
        elapsed_ms=ms6,
    )

    final_action = thesis.action
    final_confidence = thesis.confidence
    final_rationale = thesis.rationale

    # \u2500\u2500 Step 6.5: Hallucination Checker (Hard Safety Gate) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    hallucination_result = None
    try:
        from app.pipeline.analysis.hallucination_checker import check_hallucinations

        # Build provenance dict from evidence packet's structured facts
        context_provenance = {}
        raw_context_parts = []
        for fact in packet.structured_facts:
            field_name = fact.field_name if hasattr(fact, "field_name") else str(fact)
            field_val = fact.value if hasattr(fact, "value") else None
            source = fact.source if hasattr(fact, "source") else "unknown"
            context_provenance[field_name] = {
                "value": field_val,
                "source": source,
            }
            raw_context_parts.append(f"{field_name}: {field_val}")

        raw_context = "\n".join(raw_context_parts)

        await cycle_control.wait_if_paused()
        hallucination_result = check_hallucinations(
            llm_output={
                "action": final_action,
                "confidence": final_confidence,
                "rationale": final_rationale,
            },
            context_provenance=context_provenance,
            raw_context=raw_context,
            ticker=ticker,
        )

        if hallucination_result["rejected"]:
            logger.warning(
                "[V2] [HALLUCINATION] %s: REJECTED — %s. Downgrading to HOLD.",
                ticker,
                hallucination_result["rejection_reason"],
            )
            final_rationale += (
                f"\n\n\u26a0\ufe0f HALLUCINATION GATE REJECTED: "
                f"{hallucination_result['rejection_reason']}"
            )
            from app.cognition.debate.action_gate import gate_action
            final_action = gate_action("HOLD", held)
            final_confidence = max(10, final_confidence // 2)
            emit(
                "analyzing",
                f"v2_hallucination_{ticker}",
                f"\u26a0\ufe0f {ticker}: Hallucination gate REJECTED — "
                f"downgraded to HOLD @ {final_confidence}%",
                status="warning",
            )
        elif hallucination_result["hallucinations"]:
            logger.info(
                "[V2] [HALLUCINATION] %s: %d minor hallucinations (below threshold)",
                ticker,
                len(hallucination_result["hallucinations"]),
            )
        stages.append("hallucination_check")
        log_manager.log_v2_cycle(cycle_id, "v2_hallucination_check", {
            "ticker": ticker,
            "rejected": hallucination_result.get("rejected", False) if hallucination_result else False,
            "hallucination_count": len(hallucination_result.get("hallucinations", [])) if hallucination_result else 0,
            "rejection_reason": hallucination_result.get("rejection_reason", "") if hallucination_result else "",
        })
    except Exception as hall_err:
        logger.warning(
            "[V2] Hallucination check failed for %s (non-fatal): %s", ticker, hall_err
        )

    # Inject sufficiency warnings into rationale
    if sufficiency.warnings:
        final_rationale += (
            f"\n\n\u26a0\ufe0f Data warnings: {'; '.join(sufficiency.warnings)}"
        )
        
    # Inject data timeframe into rationale
    if packet.freshness_summary and packet.freshness_summary.oldest_timestamp and packet.freshness_summary.newest_timestamp:
        oldest_str = packet.freshness_summary.oldest_timestamp.strftime("%Y-%m-%d %H:%M UTC")
        newest_str = packet.freshness_summary.newest_timestamp.strftime("%Y-%m-%d %H:%M UTC")
        final_rationale += f"\n\n\ud83d\udcc5 Data timeframe: {oldest_str} to {newest_str}"

    mem_brief = memory_context.get("memory_brief", "")
    if mem_brief and mem_brief != "No prior memory.":
        final_rationale += f"\n\n\ud83d\udcdd Memory context: {mem_brief[:300]}"

    # ── Step 7: Episodic memory write-back ────────────────────────────
    # DB writes are serialized via db_semaphore to avoid conflicts
    try:
        from app.cognition.memory.writer import write_episode

        await cycle_control.wait_if_paused()
        run_result = CognitionRunResult(
            entity_id=ticker,
            cycle_id=cycle_id,
            final_action=final_action,
            final_confidence=final_confidence,
            summary=final_rationale[:500],
            rationale=final_rationale,
            tags=[ticker.lower(), "v2_stage"],
            evidence_packet=packet,
            thesis=thesis,
            sufficiency=sufficiency,
            memory_context=memory_context,
            total_tokens=total_tokens,
            total_ms=elapsed_ms(start),
            stages_completed=stages,
            retrieval_retries=retrieval_retries,
        )
        if db_semaphore:
            async with db_semaphore:
                episode_id = write_episode(run_result)
        else:
            episode_id = write_episode(run_result)
        stages.append("memory_write")
        ep_short = episode_id[:8] if episode_id else "?"
        logger.info("[V2] Wrote episode %s for %s", ep_short, ticker)
    except Exception as e:
        logger.warning("[V2] Memory write failed for %s (non-fatal): %s", ticker, e)

    # ── Step 8: V2 cycle log ──────────────────────────────────────────
    try:
        log_manager.log_v2_cycle(
            cycle_id=cycle_id,
            step_name="v2_pipeline_complete",
            payload={
                "ticker": ticker,
                "action": final_action,
                "confidence": final_confidence,
                "stages": stages,
                "retrieval_retries": retrieval_retries,
                "claims_count": len(packet.claims),
                "missing_fields": packet.missing_fields,
                "sufficiency_status": sufficiency.status,
            },
        )
        stages.append("v2_log")
    except Exception as e:
        logger.warning("[V2] Log write failed: %s", e)

    # ── Build V1-compatible result ────────────────────────────────────
    elapsed = time.monotonic() - start

    # ── PIPELINE_COMPLETE summary log ─────────────────────────────────
    # Single log line with per-stage timing breakdown for diagnostics.
    _timing_str = ", ".join(f"{k}:{v}ms" for k, v in stage_timings.items())
    logger.info(
        "[V2] PIPELINE_COMPLETE ticker=%s elapsed=%dms tokens=%d action=%s confidence=%d stages=[%s]",
        ticker,
        int(elapsed * 1000),
        total_tokens,
        final_action,
        final_confidence,
        _timing_str,
    )

    # Emit final decision
    emoji = "🟢" if final_action == "BUY" else "🔴" if final_action == "SELL" else "🟡"
    emit(
        "analyzing",
        f"v2_decision_{ticker}",
        f"{emoji} {ticker}: {final_action} @ {final_confidence}% "
        f"| V2 cognition | {elapsed:.1f}s"
        f" | {total_tokens:,} tokens",
        data={
            "action": final_action,
            "confidence": final_confidence,
            "rationale": final_rationale[:300],
        },
        elapsed_ms=int(elapsed * 1000),
    )

    result = _build_v1_compatible_result(
        ticker=ticker,
        action=final_action,
        confidence=final_confidence,
        rationale=final_rationale,
        cycle_id=cycle_id,
        total_tokens=total_tokens,
        elapsed=elapsed,
        stages=stages,
        config_used="v2_cognition",
        thesis=thesis,
        sufficiency=sufficiency,
        memory_context=memory_context,
        debate_result=debate_result,
    )

    # ── Step 9: Log decision to analysis_results (V1 parity) ──────────
    # DB writes are serialized via db_semaphore to avoid conflicts
    try:
        from app.pipeline.analysis.decision_engine import _log_decision

        if db_semaphore:
            async with db_semaphore:
                _log_decision(result, cycle_id, bot_id)
        else:
            _log_decision(result, cycle_id, bot_id)
        stages.append("db_log")
    except Exception as e:
        logger.warning("[V2] _log_decision failed for %s (non-fatal): %s", ticker, e)

    # ── Step 10: Post-cycle hooks (V1 parity) ─────────────────────────
    try:
        from app.pipeline.orchestration.post_cycle_hooks import run_post_cycle_hooks

        await run_post_cycle_hooks(
            ticker=ticker,
            result=result,
            escalated=False,
            cycle_id=cycle_id,
            final_action=final_action,
            final_confidence=final_confidence,
        )
        stages.append("post_cycle_hooks")
    except Exception as hooks_err:
        logger.warning(
            "[V2] Post-cycle hooks failed for %s (non-fatal): %s", ticker, hooks_err
        )

    # ── Step 11: Record analysis in attention tracker (V1 parity) ──────
    try:
        from app.pipeline.attention_tracker import record_analysis as _record_attn

        _record_attn(
            ticker,
            action=final_action,
            confidence=final_confidence,
            was_deep=True,  # V2 always does full evidence-based analysis
        )
        stages.append("attention_record")
    except Exception as attn_err:
        logger.warning(
            "[V2] Attention tracker failed for %s (non-fatal): %s", ticker, attn_err
        )

    return result


def _build_v1_compatible_result(
    *,
    ticker: str,
    action: str,
    confidence: int | float,
    rationale: str,
    cycle_id: str,
    total_tokens: int,
    elapsed: float,
    stages: list[str],
    config_used: str,
    thesis: Any = None,
    sufficiency: Any = None,
    memory_context: dict[str, Any] | None = None,
    debate_result: Any = None,
) -> dict[str, Any]:
    """Build a result dict matching V1's analyze_ticker() output shape.

    This ensures trading_phase, post_cycle_hooks, report_service, and
    the frontend all work without modification.
    """
    v2_meta: dict[str, Any] = {
        "stages_completed": stages,
        "sufficiency_status": sufficiency.status if sufficiency else None,
        "thesis_action": thesis.action if thesis else None,
        "thesis_confidence": thesis.confidence if thesis else None,
        "thesis_weaknesses": thesis.weaknesses if thesis else [],
        "memory_episodes": (
            memory_context.get("episode_count", 0) if memory_context else 0
        ),
        "memory_rules": (memory_context.get("rule_count", 0) if memory_context else 0),
    }

    # Include debate metadata if available
    if debate_result:
        v2_meta["debate"] = {
            "judge_action": debate_result.judge_action,
            "judge_confidence": debate_result.judge_confidence,
            "winning_side": debate_result.winning_side,
            "integrity_status": debate_result.integrity_status,
            "bull_claims_verified": f"{len(debate_result.verified_bull_claims)}/{len(debate_result.bull_claims)}",
            "bear_claims_verified": f"{len(debate_result.verified_bear_claims)}/{len(debate_result.bear_claims)}",
            "unverified_claims": len(debate_result.unverified_claims),
            "key_deciding_factor": debate_result.key_deciding_factor,
            "transcript": debate_result.transcript,
            "total_tokens": debate_result.total_tokens,
            "original_thesis_status": getattr(debate_result, "original_thesis_status", "NOT_HELD"),
            "original_thesis_explanation": getattr(debate_result, "original_thesis_explanation", ""),
        }

    return {
        "ticker": ticker,
        "action": action,
        "confidence": int(confidence),
        "rationale": rationale,
        "config_used": config_used,
        "escalated": debate_result is not None,
        "agent_results": {},
        "c_result": {
            "action": action,
            "confidence": int(confidence),
            "rationale": rationale,
        },
        "d_result": {
            "action": debate_result.judge_action,
            "confidence": debate_result.judge_confidence,
            "original_thesis_status": getattr(debate_result, "original_thesis_status", "NOT_HELD"),
            "original_thesis_explanation": getattr(debate_result, "original_thesis_explanation", ""),
        }
        if debate_result
        else None,
        "human_review": False,
        "agent_tokens": 0,
        "rlm_tokens": total_tokens,
        "total_tokens": total_tokens,
        "total_time_s": round(elapsed, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        # V2-specific metadata (ignored by V1 consumers, useful for debugging)
        "v2_metadata": v2_meta,
    }


async def execute_v2_tickers(
    tickers: list[str],
    *,
    cycle_id: str = "",
    bot_id: str = "",
    emit: Callable[..., Any] | None = None,
    macro_memo: str = "",
) -> list[dict[str, Any]]:
    """Run V2 pipeline for multiple tickers. Throttled parallel via semaphore.

    Matches the signature of V1's analyze_tickers() so it can be a
    drop-in replacement.

    Concurrency model:
      - Ticker parallelism: completely parallel (unbounded) to maximize throughput
      - LLM dispatch: gated by per-endpoint PriorityQueues in vllm_client.py
      - DB writes: serialized by a separate semaphore (8 concurrent max)
        to avoid TransactionException on the shared connection.
    """
    import asyncio
    from app.utils.pipeline_utils import noop as _noop
    from app.config import settings

    if emit is None:
        emit = _noop

    # Load memory snapshot once for the cycle (same pattern as V1)
    try:
        from app.cognition.trading_memory import trading_memory

        trading_memory.load_from_disk()
    except Exception as mem_err:
        logger.warning("[V2] Memory load failed (non-fatal): %s", mem_err)

    timeout_seconds = settings.CYCLE_TIMEOUT_MINUTES * 60

    # LLM analysis concurrency is actively handled by vllm_client's PriorityQueues.
    # Therefore, no global pipeline throttling semaphore is needed here.

    # DB write serialization — prevents concurrent connection pooling issues.
    # Only protects _log_decision() and write_episode(), NOT the full pipeline.
    db_semaphore = asyncio.Semaphore(8)

    async def _run_ticker(t: str) -> dict[str, Any]:
        return await execute_v2_pipeline(
            t,
            cycle_id=cycle_id,
            bot_id=bot_id,
            emit=emit,
            macro_memo=macro_memo,
            watchlist=tickers,
            db_semaphore=db_semaphore,
        )

    logger.info(
        "[V2] Launching parallel analysis for %d tickers (vLLM queues handle dispatch limits)",
        len(tickers),
    )
    emit(
        "analyzing",
        "v2_all_tickers",
        f"V2: Launching parallel analysis for {len(tickers)} tickers",
        status="running",
    )

    try:
        raw_results = await asyncio.wait_for(
            asyncio.gather(
                *[_run_ticker(t) for t in tickers],
                return_exceptions=True,
            ),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "[V2] CYCLE TIMEOUT after %d min",
            settings.CYCLE_TIMEOUT_MINUTES,
        )
        emit(
            "analyzing",
            "v2_timeout",
            f"V2 cycle timeout ({settings.CYCLE_TIMEOUT_MINUTES}min)",
            status="error",
        )
        return [{"ticker": t, "error": "cycle_timeout"} for t in tickers]

    results = []
    for t, r in zip(tickers, raw_results):
        if isinstance(r, Exception):
            logger.error("[V2] %s failed: %s", t, r)
            import traceback

            try:
                from app.pipeline.orchestration.state_manager import PipelineStateDB

                PipelineStateDB.log_execution_error(
                    cycle_id or "unknown",
                    "cognition_runner",
                    t,
                    type(r).__name__,
                    str(r),
                    "".join(traceback.format_exception(type(r), r, r.__traceback__)),
                )
            except Exception:
                pass
            results.append({"ticker": t, "error": str(r)})
            emit("analyzing", f"v2_error_{t}", f"{t}: V2 FAILED — {r}", status="error")
        else:
            results.append(r)

    return results
