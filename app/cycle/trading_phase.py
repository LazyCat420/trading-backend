"""
Trading Phase -- Routes decision engine outputs to paper trader.

Takes BUY/SELL/HOLD decisions from the hybrid analysis pipeline
and executes them EXCLUSIVELY through the paper trader. This project does
not support live trading execution by design; all flows are simulated.

Position sizing authority:
  PRIMARY: Portfolio Allocator Agent (agentic, regime-aware, constitution-driven)
  FALLBACK: get_size_pct() (deterministic Kelly formula, used only if agent crashes)

The allocator agent calls the composite `assess_risk_environment` tool
to get regime, portfolio state, brain graph, and constitution rules in
one shot, then makes informed sizing decisions. If the agent fails,
get_size_pct() provides a safe minimum-viable sizing.

Portfolio Gate:
  - Checks sector concentration, position count cap, and correlation
    BEFORE executing any BUY decision. Prevents overexposure.
"""

import asyncio
import logging
import time
from app.trading.paper_trader import buy, sell, get_portfolio, get_portfolio_value, _get_current_price
from app.cycle.portfolio_gate import check_portfolio_gate
from app.services.bot_manager import resolve_bot_id
from app.cycle.orchestration.cycle_control import cycle_control
from app.agents.portfolio_allocator_agent import run_portfolio_allocator
from app.agents.trade_execution_agent import run_trade_execution
from app.db.connection import get_db
from app.cycle.attention_tracker import record_trade
from app.pipeline.analysis.outcome_tracker import resolve_outcome
from app.utils.async_utils import run_with_timeout

logger = logging.getLogger(__name__)


def get_size_pct(confidence: int) -> float:
    """FALLBACK Kelly-inspired sizing: scale 2%→10% linearly for confidence 70→100.

    This is the FALLBACK sizing formula, used ONLY when the Portfolio Allocator
    Agent fails to produce valid allocations (crash, timeout, parse error).
    The primary sizing authority is the agentic allocator which uses regime,
    constitution rules, and brain graph data.

    This formula is intentionally conservative — it does NOT factor in
    market regime, correlations, or adaptive constitution parameters.
    """
    MIN_SIZE, MAX_SIZE = 0.02, 0.10
    MIN_CONF, MAX_CONF = 70, 100
    if confidence <= MIN_CONF:
        return MIN_SIZE
    if confidence >= MAX_CONF:
        return MAX_SIZE
    return MIN_SIZE + (confidence - MIN_CONF) / (MAX_CONF - MIN_CONF) * (
        MAX_SIZE - MIN_SIZE
    )




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
    from app.services.pipeline_service import PipelineService

    start = time.monotonic()
    cid = cycle_id or "no-id"

    # Resolve bot_id: if 'default' or empty, use the active bot
    bot_id = resolve_bot_id(bot_id)

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

    await cycle_control.wait_if_paused()

    # ── Call Portfolio Sizing Agent to allocate capital across all proposed BUYs as a batch ──
    allocations_map = {}
    try:
        allocations_map = await run_portfolio_allocator(decisions, cid, bot_id)
    except Exception as pa_err:
        logger.warning("[PIPELINE] Portfolio allocator agent failed to run: %s. Falling back to default sizing.", pa_err)

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
        "sell_skipped": 0,
    }


    for d in decisions:
        await cycle_control.wait_if_paused()

        # Always refresh portfolio at the start of each iteration to avoid stale snapshots
        portfolio = get_portfolio(bot_id)
        held_tickers = [p["ticker"] for p in portfolio.get("positions", [])]

        ticker = d.get("ticker", "???")
        action = d.get("action", "HOLD")
        confidence = d.get("confidence", 0)
        human_review = d.get("human_review", False)

        # Skip human review items first
        if human_review:
            logger.debug(
                "[PIPELINE]   [%s] SKIPPED -- flagged for human review", ticker
            )
            skipped.append({"ticker": ticker, "reason": "human_review"})
            counts["human_review"] += 1
            continue

        # ── Check HOLD advisory early to support CONVERT_SELL path ──
        if action == "HOLD" and ticker in held_tickers:
            try:
                hold_advisory = await run_with_timeout(
                    run_trade_execution(
                        ticker=ticker,
                        action="HOLD",
                        confidence=confidence,
                        cycle_id=cycle_id,
                        bot_id=bot_id,
                        rationale=d.get("rationale", ""),
                    ),
                    timeout=180.0,
                    label=f"hold_advisory_{ticker}",
                )
                if hold_advisory and hold_advisory.get("decision") == "CONVERT_SELL":
                    logger.warning(
                        "[PIPELINE]   [%s] HOLD advisory suggests CONVERT_SELL — converting action to SELL",
                        ticker,
                    )
                    action = "SELL"
            except Exception as hold_adv_err:
                logger.debug("[PIPELINE]   [%s] HOLD advisory agent skipped: %s", ticker, hold_adv_err)

        # ── BUY Pipeline Trace: log every decision entering the pipeline ──
        logger.info(
            "[PIPELINE]   [%s] DECISION RECEIVED: action=%s confidence=%d%% human_review=%s",
            ticker, action, confidence, human_review,
        )
        
        # Wire estimate_trade() into pre-trade logging for BUY decisions
        price_val, _ = _get_current_price(ticker)
        if action == "BUY" and price_val is not None and price_val > 0:
            est = estimate_trade(confidence, portfolio.get("cash", 0), price_val)
            logger.info(
                "[PIPELINE]   [%s] PRE-TRADE ESTIMATE: size_pct=%.1f%% of cash, amount=$%.2f, qty=%.2f shares @ $%.2f",
                ticker,
                est["size_pct"],
                est["amount"],
                est["qty"],
                est["price"],
            )

        integrity_status = d.get("v2_metadata", {}).get("debate", {}).get("integrity_status", "HIGH")
        if integrity_status == "LOW_INTEGRITY" and action != "HOLD":
            # Advisory: reduce confidence instead of overriding action
            original_conf = confidence
            confidence = max(confidence - 30, 10)
            logger.warning(
                "[PIPELINE]   [%s] LOW_INTEGRITY debate — reducing confidence %d%% → %d%% (action %s preserved)",
                ticker, original_conf, confidence, action,
            )
            
        if integrity_status == "LOW_INTEGRITY":
            logger.warning("[PIPELINE] [%s] LOW_INTEGRITY flag noted (advisory only, not blocking)", ticker)
            try:
                with get_db() as db:
                    db.execute(
                        "INSERT INTO ticker_quarantine (ticker, reason, details) "
                        "VALUES (%s, 'low_integrity', 'Swarm consensus failed integrity checks') "
                        "ON CONFLICT (ticker) DO UPDATE SET reason = EXCLUDED.reason, details = EXCLUDED.details, quarantined_at = NOW()",
                        [ticker]
                    )
            except Exception as e:
                logger.error("[PIPELINE] Failed to quarantine %s: %s", ticker, e)

        if action == "BUY":
            alloc_decision = allocations_map.get(ticker)
            if alloc_decision and alloc_decision.get("decision") == "VETO":
                logger.warning(
                    "[PIPELINE]   [%s] BUY VETOED by Portfolio Sizing Agent: %s",
                    ticker,
                    alloc_decision.get("veto_reason", "no reason"),
                )
                skipped.append({"ticker": ticker, "reason": f"VETO (Portfolio Sizing): {alloc_decision.get('veto_reason', 'no reason')}"})
                counts["blocked"] += 1
                continue

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

            # ── Trade Execution Agent: unified sizing for BUY ──
            pre_trade_decision = await run_with_timeout(
                run_trade_execution(
                    ticker=ticker,
                    action="BUY",
                    confidence=confidence,
                    cycle_id=cycle_id,
                    bot_id=bot_id,
                    rationale=d.get("rationale", ""),
                ),
                timeout=300.0,
                label=f"trade_execution_agent_{ticker}",
            )

            if pre_trade_decision and pre_trade_decision.get("decision") == "VETO":
                logger.warning(
                    "[PIPELINE]   [%s] BUY VETOED by Trade Execution Agent: %s",
                    ticker,
                    pre_trade_decision.get("veto_reason", "no reason"),
                )
                skipped.append({"ticker": ticker, "reason": f"VETO (Trade Execution): {pre_trade_decision.get('veto_reason', 'no reason')}"})
                counts["blocked"] += 1
                continue

            # Sizing logic selection
            if alloc_decision and alloc_decision.get("adjusted_size_pct", 0) > 0:
                adjusted_pct = alloc_decision.get("adjusted_size_pct", 0)
                total_portfolio_val = get_portfolio_value(bot_id)
                
                total_cost = total_portfolio_val * (adjusted_pct / 100.0)
                cash_available = portfolio.get("cash", 0)
                size_pct = (total_cost / cash_available) if cash_available > 0 else 0.02
                size_pct = max(0.02, min(size_pct, 0.15))  # Clamp to 2%-15%
                logger.info(
                    "[PIPELINE]   [%s] Using size approved by Portfolio Sizing Agent: %.1f%% of portfolio ($%.2f), equivalent to %.1f%% of cash",
                    ticker,
                    adjusted_pct,
                    total_cost,
                    size_pct * 100.0,
                )
            elif pre_trade_decision and pre_trade_decision.get("decision") == "APPROVE":
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
                size_pct = get_size_pct(confidence)
                logger.warning(
                    "[PIPELINE]   [%s] FALLBACK SIZING: Neither allocator nor pre-trade agent provided sizing. "
                    "Using deterministic get_size_pct(%d) = %.1f%%. This means the agentic allocator failed.",
                    ticker, confidence, size_pct * 100,
                )

            logger.info(
                "[PIPELINE]   [%s] EXECUTING BUY @ %d%% conf, %.0f%% of cash ($%.2f)",
                ticker,
                confidence,
                size_pct * 100,
                portfolio.get("cash", 0),
            )

            result = await run_with_timeout(
                buy(bot_id, ticker, size_pct, cycle_id=cycle_id),
                timeout=20.0,
                label=f"buy_execution_{ticker}",
                fallback={"error": "Timeout execution"},
            )

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
                    record_trade(ticker)
                except Exception as rt_err:
                    logger.debug("[TRADE] record_trade failed for %s (non-fatal): %s", ticker, rt_err)

        elif action == "SELL":
            # Defensive guard: verify we actually hold this ticker before selling
            # (Prevents wasted sell attempts when the LLM recommends SELL for unowned tickers)
            if ticker not in held_tickers:
                logger.info(
                    "[PIPELINE]   [%s] SELL skipped — not in portfolio (held: %s)",
                    ticker,
                    held_tickers,
                )
                skipped.append(
                    {"ticker": ticker, "reason": "SELL skipped: no open position"}
                )
                counts["sell_skipped"] += 1
                continue

            logger.debug("[PIPELINE]   [%s] EXECUTING SELL", ticker)

            sell_advisory = await run_with_timeout(
                run_trade_execution(
                    ticker=ticker,
                    action="SELL",
                    confidence=confidence,
                    cycle_id=cycle_id,
                    bot_id=bot_id,
                    rationale=d.get("rationale", ""),
                ),
                timeout=180.0,
                label=f"sell_advisory_{ticker}",
            )
            if sell_advisory:
                logger.info(
                    "[PIPELINE]   [%s] SELL advisory: sell_pct=%s%%, reason=%s",
                    ticker,
                    sell_advisory.get("sell_pct", 100),
                    sell_advisory.get("exit_reason", "full_exit"),
                )

            qty_pct = 1.0
            if sell_advisory and "sell_pct" in sell_advisory:
                try:
                    pct = float(sell_advisory["sell_pct"])
                    qty_pct = min(max(pct / 100.0, 0.0), 1.0)
                except (ValueError, TypeError):
                    pass

            result = await run_with_timeout(
                sell(bot_id, ticker, cycle_id=cycle_id, qty_pct=qty_pct),
                timeout=20.0,
                label=f"sell_execution_{ticker}",
                fallback={"error": "Timeout execution"},
            )

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

                # Fix C.2: Refresh portfolio after SELL so subsequent decisions
                # have accurate held_tickers and cash balance
                portfolio = get_portfolio(bot_id)

                # Record trade in attention tracker
                try:
                    record_trade(ticker)
                except Exception as rt_err:
                    logger.debug("[TRADE] record_trade failed for %s (non-fatal): %s", ticker, rt_err)

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
        f"{portfolio_after.get('cash', 0):,.2f}",
        portfolio_after.get("position_count", 0),
    )

    return {
        "bot_id": bot_id,
        "executed": executed,
        "skipped": skipped,
        "counts": counts,
        "portfolio": portfolio_after,
        "elapsed_s": round(elapsed, 2),
    }
