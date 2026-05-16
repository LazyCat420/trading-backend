"""
Trading Phase -- Routes decision engine outputs to paper trader.

Takes BUY/SELL/HOLD decisions from the hybrid analysis pipeline
and executes them EXCLUSIVELY through the paper trader. This project does
not support live trading execution by design; all flows are simulated.

Position sizing (Kelly-inspired):
  - Linear scale: 2% at confidence=70 → 10% at confidence=100
  - Sub-70 confidence: minimum 2% (shouldn't reach here due to escalation)

Portfolio Gate:
  - Checks sector concentration, position count cap, and correlation
    BEFORE executing any BUY decision. Prevents overexposure.
"""

import logging
import time
from app.trading.paper_trader import buy, sell, get_portfolio
from app.cycle.portfolio_gate import check_portfolio_gate

logger = logging.getLogger(__name__)


def get_size_pct(confidence: int) -> float:
    """Kelly-inspired: scale 2%→10% linearly for confidence 70→100."""
    MIN_SIZE, MAX_SIZE = 0.02, 0.10
    MIN_CONF, MAX_CONF = 70, 100
    if confidence <= MIN_CONF:
        return MIN_SIZE
    if confidence >= MAX_CONF:
        return MAX_SIZE
    return MIN_SIZE + (confidence - MIN_CONF) / (MAX_CONF - MIN_CONF) * (
        MAX_SIZE - MIN_SIZE
    )


# Keep old name for internal usage
_get_size_pct = get_size_pct


def estimate_trade(confidence: int, cash: float, current_price: float) -> dict:
    """Estimate shares/$ for a BUY signal without executing.

    Returns: {"size_pct": 7.3, "amount": 7300, "qty": 52, "price": 140.38}
    """
    size_pct = get_size_pct(confidence)
    amount = cash * size_pct
    qty = amount / current_price if current_price > 0 else 0
    return {
        "size_pct": round(size_pct * 100, 1),
        "amount": round(amount, 2),
        "qty": round(qty, 2),
        "price": round(current_price, 2),
    }


async def execute_decisions(
    decisions: list[dict],
    bot_id: str = "default",
    cycle_id: str = "",
) -> dict:
    """
    Execute a list of trading decisions from the decision engine.

    Args:
        decisions: List of dicts from decision_engine.analyze_ticker()
        bot_id: Bot to trade with
        cycle_id: Unique cycle identifier for log traceability

    Returns:
        Summary dict with executed trades and portfolio state
    """
    start = time.monotonic()
    cid = cycle_id or "no-id"

    # Resolve bot_id: if 'default' or empty, use the active bot
    if not bot_id or bot_id == "default":
        try:
            from app.services.bot_manager import get_active_bot_id

            bot_id = get_active_bot_id()
        except Exception:
            from app.config import settings as _cfg

            bot_id = _cfg.BOT_ID

    logger.info(
        "[PIPELINE] TRADING PHASE START | bot_id=%s | cycle=%s | %d decisions",
        bot_id,
        cid,
        len(decisions),
    )

    # Pre-trade portfolio
    portfolio = get_portfolio(bot_id)
    logger.info(
        "[PIPELINE]   Pre-trade portfolio: $%s cash | %d positions | held: %s",
        f"{portfolio.get('cash', 0):,.2f}",
        portfolio.get("position_count", 0),
        [p["ticker"] for p in portfolio.get("positions", [])],
    )

    executed = []
    skipped = []

    # Observability: Categorized counts
    counts = {
        "holds": 0,
        "human_review": 0,
        "buy_executed": 0,
        "sell_executed": 0,
        "buy_failed": 0,
        "sell_failed": 0,
        "blocked": 0,
        "passes": 0,
    }
    
    consecutive_failures = 0

    for d in decisions:
        from app.cycle.orchestration.cycle_control import cycle_control
        await cycle_control.wait_if_paused()

        ticker = d.get("ticker", "???")
        action = d.get("action", "HOLD")
        confidence = d.get("confidence", 0)
        human_review = d.get("human_review", False)
        
        integrity_status = d.get("v2_metadata", {}).get("debate", {}).get("integrity_status", "HIGH")
        if integrity_status == "LOW_INTEGRITY" and action != "HOLD":
            logger.warning("[PIPELINE]   [%s] OVERRIDING %s TO HOLD due to LOW_INTEGRITY debate", ticker, action)
            action = "HOLD"
            confidence = 0
            
        if integrity_status == "LOW_INTEGRITY":
            consecutive_failures += 1
            logger.warning("[PIPELINE] [%s] Quarantining ticker due to LOW_INTEGRITY", ticker)
            try:
                from app.db.connection import get_db
                with get_db() as db:
                    db.execute(
                        "INSERT INTO ticker_quarantine (ticker, reason, details) "
                        "VALUES (%s, 'low_integrity', 'Swarm consensus failed integrity checks') "
                        "ON CONFLICT (ticker) DO UPDATE SET reason = EXCLUDED.reason, details = EXCLUDED.details, quarantined_at = NOW()",
                        [ticker]
                    )
            except Exception as e:
                logger.error("[PIPELINE] Failed to quarantine %s: %s", ticker, e)
                
            if consecutive_failures >= 3:
                logger.error("[PIPELINE] 3 consecutive LOW_INTEGRITY failures! Aborting trading phase to protect pipeline.")
                skipped.append({"ticker": ticker, "reason": "HOLD"})
                counts["holds"] += 1
                break
        else:
            consecutive_failures = 0

        # Skip human review items
        if human_review:
            logger.debug(
                "[PIPELINE]   [%s] SKIPPED -- flagged for human review", ticker
            )
            skipped.append({"ticker": ticker, "reason": "human_review"})
            counts["human_review"] += 1
            continue

        if action == "BUY":
            # ── Portfolio Gate: enforce position limits before BUY ──
            gate = check_portfolio_gate(ticker, action, bot_id, confidence)
            if gate["blocked"]:
                logger.warning(
                    "[PIPELINE]   [%s] BUY BLOCKED by portfolio gate: %s",
                    ticker,
                    gate["reason"],
                )
                skipped.append({"ticker": ticker, "reason": f"GATE: {gate['reason']}"})
                counts["blocked"] += 1
                continue

            if gate["warnings"]:
                for w in gate["warnings"]:
                    logger.info("[PIPELINE]   [%s] Gate warning: %s", ticker, w)

            # ── Pre-Trade Agent: run calculator tool chain before buying ──
            pre_trade_decision = None
            try:
                import asyncio
                from app.agents.pre_trade_agent import run_pre_trade

                pre_trade_decision = await asyncio.wait_for(
                    run_pre_trade(
                        ticker=ticker,
                        confidence=confidence,
                        cycle_id=cycle_id,
                        bot_id=bot_id,
                    ),
                    timeout=60.0,  # Hard timeout for pre-trade agent
                )

                if pre_trade_decision and pre_trade_decision.get("decision") == "VETO":
                    logger.warning(
                        "[PIPELINE]   [%s] BUY VETOED by pre-trade agent: %s",
                        ticker,
                        pre_trade_decision.get("veto_reason", "no reason"),
                    )
                    skipped.append({
                        "ticker": ticker,
                        "reason": f"PRE_TRADE VETO: {pre_trade_decision.get('veto_reason', 'risk check failed')}",
                    })
                    counts["blocked"] += 1
                    continue

            except asyncio.TimeoutError:
                logger.warning("[PIPELINE]   [%s] Pre-trade agent timed out, falling back to Kelly sizing", ticker)
                pre_trade_decision = None
            except Exception as pt_err:
                logger.warning("[PIPELINE]   [%s] Pre-trade agent failed (%s), falling back to Kelly sizing", ticker, pt_err)
                pre_trade_decision = None

            # Use pre-trade agent's position size if available, otherwise fall back to Kelly
            if pre_trade_decision and pre_trade_decision.get("decision") == "APPROVE":
                # Convert pre-trade shares to a size_pct for the buy() function
                total_cost = pre_trade_decision.get("total_cost", 0)
                cash_available = portfolio.get("cash", 0)
                size_pct = (total_cost / cash_available) if cash_available > 0 else 0.02
                size_pct = max(0.02, min(size_pct, 0.15))  # Clamp to 2%-15%
                logger.info(
                    "[PIPELINE]   [%s] Pre-trade APPROVED: %d shares, %.1f%% of cash, R:R=%.2f",
                    ticker,
                    pre_trade_decision.get("shares", 0),
                    size_pct * 100,
                    pre_trade_decision.get("risk_reward_ratio", 0),
                )
            else:
                size_pct = _get_size_pct(confidence)

            logger.info(
                "[PIPELINE]   [%s] EXECUTING BUY @ %d%% conf, %.0f%% of cash ($%.2f)",
                ticker,
                confidence,
                size_pct * 100,
                portfolio.get("cash", 0),
            )
            import asyncio

            try:
                result = await asyncio.wait_for(
                    buy(bot_id, ticker, size_pct, cycle_id=cycle_id), timeout=20.0
                )
            except asyncio.TimeoutError:
                logger.warning("[PIPELINE]   [%s] BUY TIMED OUT", ticker)
                result = {"error": "Timeout execution"}
            except Exception as e:
                logger.warning("[PIPELINE]   [%s] BUY EXCEPTION: %s", ticker, e)
                result = {"error": str(e)}

            if "error" in result:
                logger.warning(
                    "[PIPELINE]   [%s] BUY FAILED (cash=$%.2f): %s",
                    ticker,
                    portfolio.get("cash", 0),
                    result["error"],
                )
                skipped.append({"ticker": ticker, "reason": result["error"]})
                counts["buy_failed"] += 1
            else:
                executed.append(result)
                counts["buy_executed"] += 1

                # Emit live event to trigger real-time UI refresh
                try:
                    from app.services.pipeline_service import PipelineService

                    PipelineService.emit(
                        "trading",
                        ticker,
                        f"Executed BUY: {result['qty']:.2f} shares @ ${result['price']:.2f}",
                        data={
                            "action": "BUY",
                            "ticker": ticker,
                            "qty": result["qty"],
                            "price": result["price"],
                            "amount": result.get("amount", 0),
                        },
                    )
                except Exception as emit_err:
                    logger.debug("[TRADE] Failed to emit BUY event: %s", emit_err)

                # Refresh portfolio cash for accurate logging on next iteration
                portfolio = get_portfolio(bot_id)
                # Record trade in attention tracker
                try:
                    from app.cycle.attention_tracker import record_trade

                    record_trade(ticker)
                except Exception:
                    pass

        elif action == "SELL":
            # Defensive guard: verify we actually hold this ticker before selling
            # (Prevents wasted sell attempts when the LLM recommends SELL for unowned tickers)
            held_tickers = [p["ticker"] for p in portfolio.get("positions", [])]
            if ticker not in held_tickers:
                logger.info(
                    "[PIPELINE]   [%s] SELL skipped — not in portfolio (held: %s)",
                    ticker,
                    held_tickers,
                )
                skipped.append(
                    {"ticker": ticker, "reason": "SELL skipped: no open position"}
                )
                counts["holds"] += 1
                continue

            logger.debug("[PIPELINE]   [%s] EXECUTING SELL", ticker)
            import asyncio

            try:
                result = await asyncio.wait_for(
                    sell(bot_id, ticker, cycle_id=cycle_id), timeout=20.0
                )
            except asyncio.TimeoutError:
                logger.warning("[PIPELINE]   [%s] SELL TIMED OUT", ticker)
                result = {"error": "Timeout execution"}
            except Exception as e:
                logger.warning("[PIPELINE]   [%s] SELL EXCEPTION: %s", ticker, e)
                result = {"error": str(e)}

            if "error" in result:
                logger.warning(
                    "[PIPELINE]   [%s] SELL FAILED: %s", ticker, result["error"]
                )
                skipped.append({"ticker": ticker, "reason": result["error"]})
                counts["sell_failed"] += 1
            else:
                executed.append(result)
                counts["sell_executed"] += 1

                # ── Outcome Resolution: close the feedback loop ──
                # resolve_outcome() already computes WIN/LOSS, triggers
                # strategy_tracker.evaluate_pnl(), reinforces ontology claims,
                # and writes to lesson_store for the autoresearch evolve loop.
                try:
                    from app.pipeline.analysis.outcome_tracker import resolve_outcome

                    exit_price = result.get("price", 0)
                    outcome = resolve_outcome(
                        ticker, exit_price,
                        realized_pnl=result.get("realized_pnl"),
                    )
                    if outcome:
                        logger.info(
                            "[PIPELINE]   [%s] Outcome resolved: %s (%.1f%%)",
                            ticker,
                            outcome.get("outcome", "?"),
                            outcome.get("pnl_pct", 0),
                        )
                except Exception as outcome_err:
                    logger.warning(
                        "[PIPELINE]   [%s] Outcome resolution failed (non-fatal): %s",
                        ticker, outcome_err,
                    )
                # ── End Outcome Resolution ──

                # Emit live event to trigger real-time UI refresh
                try:
                    from app.services.pipeline_service import PipelineService

                    PipelineService.emit(
                        "trading",
                        ticker,
                        f"Executed SELL: {result.get('qty', 0):.2f} shares @ ${result.get('price', 0):.2f} (P&L: ${result.get('realized_pnl', 0):+.2f})",
                        data={
                            "action": "SELL",
                            "ticker": ticker,
                            "qty": result.get("qty", 0),
                            "price": result.get("price", 0),
                            "pnl": result.get("realized_pnl", 0),
                        },
                    )
                except Exception as emit_err:
                    logger.debug("[TRADE] Failed to emit SELL event: %s", emit_err)

                # Record trade in attention tracker
                try:
                    from app.cycle.attention_tracker import record_trade

                    record_trade(ticker)
                except Exception:
                    pass

        elif action == "HOLD":
            logger.info(
                "[PIPELINE]   [%s] EXECUTING HOLD (Confidence: %d%%)",
                ticker,
                confidence,
            )
            skipped.append({"ticker": ticker, "reason": "HOLD"})
            counts["holds"] += 1
            continue

        elif action == "PASS":
            logger.info(
                "[PIPELINE]   [%s] EXECUTING PASS (Confidence: %d%%)",
                ticker,
                confidence,
            )
            skipped.append({"ticker": ticker, "reason": "PASS"})
            counts["passes"] += 1
            continue

        else:
            logger.debug("[PIPELINE]   [%s] %s -- no action", ticker, action)
            skipped.append({"ticker": ticker, "reason": action.lower()})
            if action in ("HOLD", "PASS"):
                counts.setdefault(f"{action.lower()}es", 0)
                counts[f"{action.lower()}es"] += 1
            else:
                counts["holds"] += 1

    # Post-trade portfolio
    portfolio_after = get_portfolio(bot_id)
    elapsed = time.monotonic() - start

    logger.info(
        "[PIPELINE] TRADING COMPLETE | %d executed | %d skipped | Duration: %.1fs",
        len(executed),
        len(skipped),
        elapsed,
    )
    logger.info(
        "  Portfolio: $%s cash | %d positions",
        f"{portfolio_after['cash']:,.2f}",
        portfolio_after["position_count"],
    )

    return {
        "bot_id": bot_id,
        "executed": executed,
        "skipped": skipped,
        "counts": counts,
        "portfolio": portfolio_after,
        "elapsed_s": round(elapsed, 2),
    }
