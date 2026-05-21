"""
Outcome Tracker -- Records BUY/SELL decisions and resolves outcomes.

Phase 4 of the memory plan: track whether past decisions were correct
so we can validate memory entries and feed the trade journal.

Usage:
    from app.pipeline.analysis.outcome_tracker import record_decision, resolve_outcome

    # After a BUY/SELL decision:
    record_decision(cycle_id, ticker, action, confidence, entry_price, lesson)

    # When a trade closes (sell or stop-loss):
    resolve_outcome(ticker, exit_price)
"""

import logging
import uuid
from datetime import datetime, timezone

from app.db.connection import get_db

logger = logging.getLogger(__name__)


def record_decision(
    cycle_id: str,
    ticker: str,
    action: str,
    confidence: int,
    entry_price: float | None = None,
    lesson: str | None = None,
) -> str | None:
    """Record a notable BUY/SELL decision for later outcome tracking.

    Only records BUY/SELL decisions (not HOLD). Returns the decision_outcome
    ID if recorded, None if skipped.
    """
    if action not in ("BUY", "SELL"):
        return None

    # Get current price if not provided
    if entry_price is None:
        try:
            with get_db() as db:
                row = db.execute(
                    "SELECT close FROM price_history WHERE ticker = %s "
                    "ORDER BY date DESC LIMIT 1",
                    [ticker],
                ).fetchone()
                if row:
                    entry_price = row[0]
        except Exception:
            pass

    outcome_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    try:
        with get_db() as db:
            db.execute(
                """
                INSERT INTO decision_outcomes
                (id, cycle_id, ticker, action, confidence, entry_price,
                 lesson_stored, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
                [
                    outcome_id,
                    cycle_id,
                    ticker,
                    action,
                    confidence,
                    entry_price,
                    lesson,
                    now,
                ],
            )
            logger.info(
                "[OUTCOME] Recorded %s %s @ $%.2f (conf=%d, id=%s)",
                action,
                ticker,
                entry_price or 0,
                confidence,
                outcome_id[:8],
            )
            return outcome_id
    except Exception as e:
        logger.warning(
            "[PIPELINE] [OUTCOME] Failed to record %s %s: %s", action, ticker, e
        )
        return None


def resolve_outcome(
    ticker: str,
    exit_price: float,
    realized_pnl: float | None = None,
) -> dict | None:
    """Resolve an open decision outcome when a trade closes.

    Finds the most recent unresolved BUY for this ticker and fills
    in exit_price, pnl_pct, and outcome (WIN/LOSS/FLAT).
    """
    try:
        with get_db() as db:
            row = db.execute(
                """
                SELECT id, entry_price, action, confidence FROM decision_outcomes
                WHERE ticker = %s AND action = 'BUY' AND resolved_at IS NULL
                ORDER BY created_at DESC LIMIT 1
            """,
                [ticker],
            ).fetchone()

            if not row:
                logger.debug("[OUTCOME] No unresolved BUY for %s to resolve", ticker)
                return None

            outcome_id, entry_price = row[0], row[1]
            action = row[2]
            confidence = row[3]

            if entry_price and entry_price > 0:
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            else:
                pnl_pct = 0.0

            # Classify outcome
            if pnl_pct > 0.5:
                outcome = "WIN"
            elif pnl_pct < -0.5:
                outcome = "LOSS"
            else:
                outcome = "FLAT"

            now = datetime.now(timezone.utc)
            db.execute(
                """
                UPDATE decision_outcomes
                SET exit_price = %s, pnl_pct = %s, outcome = %s, resolved_at = %s
                WHERE id = %s
            """,
                [exit_price, round(pnl_pct, 2), outcome, now, outcome_id],
            )

            logger.info(
                "[OUTCOME] Resolved %s: %s (%.1f%% PnL, entry=$%.2f exit=$%.2f)",
                ticker,
                outcome,
                pnl_pct,
                entry_price or 0,
                exit_price,
            )

            # ── Strategy Performance Tracking (Phase 5) ──
            try:
                from app.trading.strategy_tracker import evaluate_pnl

                resolved_strategies = evaluate_pnl(ticker, exit_price)
                if resolved_strategies:
                    logger.info(
                        "[OUTCOME] Evaluated P&L for %d strategy entries for %s",
                        len(resolved_strategies),
                        ticker,
                    )
            except Exception as strat_err:
                logger.debug(
                    "[OUTCOME] Strategy evaluation failed for %s (non-fatal): %s",
                    ticker,
                    strat_err,
                )

            # ── Lens Performance Tracking ──
            try:
                from app.pipeline.analysis.lens_scorecard import LensScorecard, LensReaper
                best_action = "BUY" if pnl_pct > 0.5 else ("SELL" if pnl_pct < -0.5 else "HOLD")
                LensScorecard.grade_lens(ticker, best_action, pnl_pct)
                LensReaper.reap_underperformers()
            except Exception as lens_err:
                logger.debug(
                    "[OUTCOME] Lens grading failed for %s (non-fatal): %s",
                    ticker,
                    lens_err,
                )

            # ── Living Graph: reinforce/weaken Claims based on outcome ──
            try:
                from app.cognition.ontology.graph_mutations import (
                    reinforce_claim,
                    get_claims_for_ticker,
                )

                claim_ids = get_claims_for_ticker(ticker)
                for claim_id in claim_ids:
                    reinforce_claim(claim_id, outcome)

                if claim_ids:
                    logger.info(
                        "[OUTCOME] Updated %d claims for %s → %s",
                        len(claim_ids),
                        ticker,
                        outcome,
                    )
            except Exception as graph_err:
                logger.debug(
                    "[OUTCOME] Graph reinforcement failed for %s (non-fatal): %s",
                    ticker,
                    graph_err,
                )

            # ── Dual-write to lesson_store for evolve loop ──
            try:
                from app.cognition.lesson_store import (
                    add_lesson as _add_evolution_lesson,
                )

                lesson_text = (
                    f"[{outcome}] {ticker} {action} conf={confidence}: "
                    f"entry=${entry_price:.2f} exit=${exit_price:.2f} "
                    f"pnl={pnl_pct:.1f}%"
                )
                _add_evolution_lesson(
                    text=lesson_text,
                    metadata={
                        "session_id": f"live_{datetime.now(timezone.utc).strftime('%b%d').lower()}",
                        "round": 0,
                        "score": round(pnl_pct, 2),
                        "status": outcome,
                        "source": "live_trade",
                        "timestamp": now.isoformat(),
                    },
                )
                logger.info(
                    "[OUTCOME] Wrote live lesson to lesson_store: %s",
                    lesson_text[:80],
                )
            except Exception as lesson_err:
                logger.debug(
                    "[OUTCOME] lesson_store write failed for %s (non-fatal): %s",
                    ticker,
                    lesson_err,
                )

            return {
                "id": outcome_id,
                "ticker": ticker,
                "outcome": outcome,
                "pnl_pct": round(pnl_pct, 2),
                "entry_price": entry_price,
                "exit_price": exit_price,
            }

    except Exception as e:
        logger.warning("[PIPELINE] [OUTCOME] Failed to resolve %s: %s", ticker, e)
        return None


def get_past_outcomes(
    ticker: str | None = None,
    limit: int = 10,
) -> list[dict]:
    """Get resolved decision outcomes for the trade journal.

    If ticker is None, returns outcomes across all tickers.
    """
    try:
        with get_db() as db:
            if ticker:
                rows = db.execute(
                    """
                    SELECT ticker, action, confidence, entry_price, exit_price,
                           pnl_pct, outcome, lesson_stored, created_at, resolved_at
                    FROM decision_outcomes
                    WHERE ticker = %s AND resolved_at IS NOT NULL
                    ORDER BY resolved_at DESC LIMIT %s
                """,
                    [ticker, limit],
                ).fetchall()
            else:
                rows = db.execute(
                    """
                    SELECT ticker, action, confidence, entry_price, exit_price,
                           pnl_pct, outcome, lesson_stored, created_at, resolved_at
                    FROM decision_outcomes
                    WHERE resolved_at IS NOT NULL
                    ORDER BY resolved_at DESC LIMIT %s
                """,
                    [limit],
                ).fetchall()

            return [
                {
                    "ticker": r[0],
                    "action": r[1],
                    "confidence": r[2],
                    "entry_price": r[3],
                    "exit_price": r[4],
                    "pnl_pct": r[5],
                    "outcome": r[6],
                    "lesson": r[7],
                    "decided_at": str(r[8]),
                    "resolved_at": str(r[9]),
                }
                for r in rows
            ]
    except Exception as e:
        logger.warning("[PIPELINE] [OUTCOME] Failed to get past outcomes: %s", e)
        return []


def cancel_outcome(
    ticker: str,
    reason: str = "Canceled",
) -> bool:
    """Cancel an unresolved outcome (e.g., trade blocked by portfolio gate).

    Marks the most recent unresolved BUY for this ticker as CANCELED
    with 0% PnL so it doesn't hang as unresolved forever.

    Returns True if an outcome was canceled, False if none found.
    """
    try:
        with get_db() as db:
            row = db.execute(
                """
                SELECT id, lesson_stored FROM decision_outcomes
                WHERE ticker = %s AND action = 'BUY' AND resolved_at IS NULL
                ORDER BY created_at DESC LIMIT 1
            """,
                [ticker],
            ).fetchone()

            if not row:
                logger.debug("[OUTCOME] No unresolved BUY for %s to cancel", ticker)
                return False

            outcome_id = row[0]
            old_lesson = row[1] or ""
            new_lesson = f"{old_lesson} [CANCELED: {reason}]".strip()

            now = datetime.now(timezone.utc)
            db.execute(
                """
                UPDATE decision_outcomes
                SET exit_price = entry_price, pnl_pct = 0.0,
                    outcome = 'CANCELED', lesson_stored = %s,
                    resolved_at = %s
                WHERE id = %s
            """,
                [new_lesson, now, outcome_id],
            )

            logger.info(
                "[OUTCOME] Canceled %s: %s (id=%s)", ticker, reason, outcome_id[:8]
            )
            return True

    except Exception as e:
        logger.warning("[PIPELINE] [OUTCOME] Failed to cancel %s: %s", ticker, e)
        return False
