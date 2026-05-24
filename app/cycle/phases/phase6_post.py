import logging
import json
import asyncio
from typing import Callable, Any

from app.db.connection import get_db

logger = logging.getLogger(__name__)


async def run_phase6_post(
    ctx: Any,
    bot_id: str,
    results: list[dict],
    trade_result: dict,
    emit: Callable,
    state: dict,
    cycle_summary: dict | None = None,
) -> None:
    """
    Phase 6: Post-Trade Enrichment and Purge.
    """
    # 0. Commit Memory
    try:
        from app.services.memory.cycle_closer import cycle_closer
        mode = state.get("execution_mode", "production")
        # Ensure we pass the cycle's actual summary details if needed, for now just empty dict
        await cycle_closer.close_cycle(
            cycle_id=ctx.cycle_id,
            tickers=ctx.tickers,
            mode=mode,
            summary=cycle_summary or {"phase": "post_trade", "collected": len(ctx.tickers)},
            results=results
        )
    except Exception as mem_err:
        logger.warning("Failed to store cycle memory: %s", mem_err)

    # 0.5. Post-Mortem Retrospective Audits for Closed Trades
    if ctx.trade and trade_result is not None:
        executed_sells = [
            t for t in trade_result.get("executed", [])
            if t.get("action") == "SELL"
        ]
        if executed_sells:
            emit("purge", "post_mortem_start", f"Running Post-Mortem retrospects for {len(executed_sells)} closed positions...", status="running")
            async def _run_single_post_mortem(t):
                try:
                    from app.agents.post_mortem_auditor_agent import run_post_mortem
                    # get entry price from decision_outcomes
                    entry_price = 0.0
                    with get_db() as db:
                        row = db.execute(
                            "SELECT entry_price FROM decision_outcomes WHERE ticker = %s AND action = 'BUY' AND resolved_at IS NOT NULL ORDER BY resolved_at DESC LIMIT 1",
                            [t["ticker"]]
                        ).fetchone()
                        if row:
                            entry_price = row[0]
                    
                    if entry_price == 0.0:
                        # Fallback: estimate from exit price and pnl_pct
                        exit_p = t.get("price", 0)
                        pct = t.get("pnl_pct", 0)
                        entry_price = exit_p / (1 + pct / 100.0) if pct != -100 else exit_p
                    
                    await run_post_mortem(
                        ticker=t["ticker"],
                        entry_price=entry_price,
                        exit_price=t.get("price", 0),
                        pnl_pct=t.get("pnl_pct", 0),
                        cycle_id=ctx.cycle_id,
                        bot_id=bot_id,
                    )
                except Exception as pm_err:
                    logger.warning("Post-mortem retrospective failed for %s: %s", t["ticker"], pm_err)

            try:
                # Limit total runtime to 90s max to prevent pipeline stalling
                await asyncio.wait_for(
                    asyncio.gather(*[_run_single_post_mortem(t) for t in executed_sells]),
                    timeout=90.0
                )
                emit("purge", "post_mortem_done", f"Retrospective audits complete for {len(executed_sells)} closed positions", status="ok")
            except asyncio.TimeoutError:
                logger.warning("Post-mortem retrospective timed out.")
                emit("purge", "post_mortem_timeout", "Post-Mortem retrospective audits timed out", status="warning")
            except Exception as pm_g_err:
                logger.warning("Post-mortem audits gather failed: %s", pm_g_err)

    # 1. Post-Trade Enrichment
    try:
        from app.cycle.trading_phase import estimate_trade
        from app.trading.paper_trader import get_portfolio as _get_pf

        pf = _get_pf(bot_id)
        cash = pf.get("cash", 0)

        with get_db() as _db:
            ticker_list = [r.get("ticker", "") for r in results if r.get("ticker")]
            
            # Step 1: Bulk fetch existing rows
            existing_map = {}
            if ticker_list and ctx.cycle_id:
                existing_rows = _db.execute(
                    "SELECT ticker, result_json FROM analysis_results WHERE cycle_id = %s AND ticker = ANY(%s)",
                    [ctx.cycle_id, ticker_list],
                ).fetchall()
                for row in existing_rows:
                    try:
                        existing_map[row[0]] = json.loads(row[1])
                    except Exception:
                        pass
            
            # Step 2: Bulk fetch price history for BUY estimates
            buy_tickers = [r.get("ticker") for r in results if r.get("action") == "BUY" and r.get("confidence", 0) > 0 and "estimate" not in r]
            price_map = {}
            if buy_tickers:
                price_rows = _db.execute(
                    """
                    SELECT DISTINCT ON (ticker) ticker, close 
                    FROM price_history 
                    WHERE ticker = ANY(%s) 
                    ORDER BY ticker, date DESC
                    """,
                    [buy_tickers]
                ).fetchall()
                price_map = {row[0]: row[1] for row in price_rows}

            updates = []
            inserts = []

            for result in results:
                ticker = result.get("ticker", "")
                if not ticker:
                    continue
                action = result.get("action", "HOLD")
                confidence = result.get("confidence", 0)
                needs_db_update = False

                if ctx.trade and trade_result is not None:
                    for exec_trade in trade_result.get("executed", []):
                        if exec_trade.get("ticker") == ticker:
                            result["trade_executed"] = exec_trade
                            needs_db_update = True
                    for skip in trade_result.get("skipped", []):
                        if skip.get("ticker") == ticker:
                            result["trade_skipped"] = skip
                            needs_db_update = True

                if action == "BUY" and confidence > 0 and "estimate" not in result:
                    close_price = price_map.get(ticker)
                    if close_price and close_price > 0:
                        result["estimate"] = estimate_trade(confidence, cash, close_price)
                        needs_db_update = True

                if needs_db_update and ctx.cycle_id:
                    if ticker in existing_map:
                        stored = existing_map[ticker]
                        if result.get("trade_executed"):
                            stored["trade_executed"] = result["trade_executed"]
                        if result.get("trade_skipped"):
                            stored["trade_skipped"] = result["trade_skipped"]
                        if result.get("estimate"):
                            stored["estimate"] = result["estimate"]
                        updates.append((json.dumps(stored), ctx.cycle_id, ticker))
                    else:
                        import uuid
                        from datetime import datetime, timezone
                        result_id = str(uuid.uuid4())
                        inserts.append((
                            result_id,
                            ctx.cycle_id or "manual",
                            bot_id or "decision-engine",
                            ticker,
                            "timeout_fallback",
                            json.dumps(result),
                            confidence,
                            datetime.now(timezone.utc).isoformat(),
                            "standard"
                        ))

            if updates:
                _db.executemany(
                    "UPDATE analysis_results SET result_json = %s WHERE cycle_id = %s AND ticker = %s",
                    updates,
                )
            if inserts:
                _db.executemany(
                    """
                    INSERT INTO analysis_results
                    (id, cycle_id, bot_id, ticker, agent_name, result_json, confidence, created_at, triage_tier)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    inserts,
                )

        state["results"] = results
    except Exception as e:
        logger.warning("Results enrichment failed: %s", e)

    # 2. Purge Pass — bounded housekeeping (no more fire-and-forget zombies)
    # All background tasks are gathered with a strict timeout so they can't
    # leak and keep the event loop alive for hours after the cycle ends.
    _HOUSEKEEPING_TIMEOUT = 120  # seconds — hard cap on all post-cycle work

    if ctx.analyze:
        async def _bg_purge():
            try:
                from app.pipeline.analysis.purge_pass import run_purge_pass
                purge_result = await run_purge_pass(
                    watchlist=ctx.tickers,
                    cycle_results=results,
                    emit=emit,
                    cycle_id=ctx.cycle_id,
                )
                if purge_result:
                    emit(
                        "purge",
                        "summary",
                        f"Purged {len(purge_result)}: {', '.join(purge_result)}",
                        status="ok",
                    )
            except Exception as e:
                logger.error("Purge pass failed: %s", e)

        async def _bg_knowledge_purge():
            try:
                from app.cognition.ontology.knowledge_purge import purge_stale_knowledge
                kg_result = await purge_stale_knowledge()
                kg_ops = sum(v for k, v in kg_result.items() if isinstance(v, int))
                if kg_ops > 0:
                    emit(
                        "purge",
                        "knowledge_purge",
                        f"Knowledge purge: {kg_result}",
                        status="ok",
                    )
            except Exception as kg_err:
                logger.debug("Knowledge purge failed (non-fatal): %s", kg_err)

        async def _bg_janitor():
            try:
                from app.pipeline.analysis.agent_maintenance import run_janitor_tasks
                await run_janitor_tasks()
            except Exception as e:
                logger.error("Agent maintenance failed: %s", e)

        async def _bg_benchmarks():
            try:
                from app.pipeline.subsystem_benchmarks import record_all
                record_all(ctx.cycle_id)
            except Exception as bench_err:
                logger.error("Benchmark recording failed: %s", bench_err)

        # ── Housekeeping: gathered with strict timeout (no orphan tasks) ──
        emit("purge", "start", "Launching bounded housekeeping...", status="ok")
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    _bg_purge(),
                    _bg_knowledge_purge(),
                    _bg_janitor(),
                    _bg_benchmarks(),
                    return_exceptions=True,
                ),
                timeout=_HOUSEKEEPING_TIMEOUT,
            )
            logger.info("[CYCLE] All housekeeping tasks completed within %ds", _HOUSEKEEPING_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning(
                "[CYCLE] Housekeeping timeout (%ds) — cancelling remaining tasks",
                _HOUSEKEEPING_TIMEOUT,
            )
        except Exception as hk_err:
            logger.error("[CYCLE] Housekeeping gather failed: %s", hk_err)

        # ── Post-cycle agents: gathered with strict timeout (no orphan tasks) ──
        async def _run_post_cycle_agents():
            try:
                import traceback
                from app.agents.meta_audit_agent import run_meta_audit
                from app.agents.quant_research_agent import run_quant_research

                bot_id_resolved = bot_id
                try:
                    from app.services.bot_manager import get_active_bot_id
                    bot_id_resolved = get_active_bot_id()
                except Exception:
                    pass

                # Run both agents with individual error handling
                async def _safe_agent(coro, name):
                    try:
                        await coro
                    except asyncio.CancelledError:
                        logger.info("[%s] Cancelled (timeout or shutdown).", name)
                    except Exception as e:
                        logger.error("[%s] Failed: %s", name, e)
                        try:
                            from app.db.pipeline_state import PipelineStateDB
                            PipelineStateDB.log_execution_error(
                                cycle_id=ctx.cycle_id,
                                phase="post_trade",
                                ticker="system",
                                error_type=f"{name}_failure",
                                error_message=str(e)[:500],
                                stack_trace=traceback.format_exc()[:2000],
                            )
                        except Exception:
                            pass

                await asyncio.gather(
                    _safe_agent(
                        run_meta_audit(cycle_id=ctx.cycle_id, bot_id=bot_id_resolved),
                        "meta_audit",
                    ),
                    _safe_agent(
                        run_quant_research(cycle_id=ctx.cycle_id, bot_id=bot_id_resolved),
                        "quant_research",
                    ),
                    return_exceptions=True,
                )
            except Exception as agent_err:
                logger.warning("Post-cycle agents failed (non-fatal): %s", agent_err)

        try:
            emit(
                "evaluated",
                "post_cycle_agents",
                "Running post-cycle agents (bounded)...",
                status="ok",
            )
            await asyncio.wait_for(
                _run_post_cycle_agents(),
                timeout=_HOUSEKEEPING_TIMEOUT,
            )
            logger.info("[CYCLE] Post-cycle agents completed within %ds", _HOUSEKEEPING_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning(
                "[CYCLE] Post-cycle agents timeout (%ds) — cancelling",
                _HOUSEKEEPING_TIMEOUT,
            )
        except Exception as pca_err:
            logger.warning("Post-cycle agent launch failed (non-fatal): %s", pca_err)
