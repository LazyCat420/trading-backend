"""
Order Triggers — Price-based automated trade execution.

Supports:
  - stop_loss:     Sell when price drops below trigger_price
  - take_profit:   Sell when price rises above trigger_price
  - buy_limit:     Buy when price drops to or below trigger_price
  - sell_limit:    Sell when price rises to or above trigger_price
  - trailing_stop: Sell when price drops trailing_pct from highest recorded price

All triggers are evaluated every 1 minute by the background scheduler.
The bot can also set triggers via agent tools during analysis.
"""

import uuid
import logging
from datetime import datetime, timezone

from app.db.connection import get_db
from app.trading.paper_trader import _get_current_price

logger = logging.getLogger(__name__)


async def create_trigger(
    bot_id: str,
    ticker: str,
    trigger_type: str,
    trigger_price: float,
    action: str = "SELL",
    qty_pct: float = 1.0,
    trailing_pct: float | None = None,
    reason: str | None = None,
    created_by: str = "bot",
) -> dict:
    """Create a new price trigger.

    Args:
        bot_id: Bot to associate with
        ticker: Ticker symbol
        trigger_type: stop_loss | take_profit | buy_limit | sell_limit | trailing_stop
        trigger_price: Price at which to trigger
        action: BUY or SELL
        qty_pct: Fraction of position (0.0-1.0, default 1.0 = full)
        trailing_pct: For trailing_stop: percentage drop from peak to trigger
        reason: Human-readable reason for the trigger
        created_by: bot | user | pipeline

    Returns:
        dict with trigger details
    """
    valid_types = (
        "stop_loss",
        "take_profit",
        "buy_limit",
        "sell_limit",
        "trailing_stop",
    )
    if trigger_type not in valid_types:
        return {
            "error": f"Invalid trigger_type: {trigger_type}. Must be one of {valid_types}"
        }

    if trigger_price <= 0:
        return {"error": f"trigger_price must be positive, got {trigger_price}"}

    if trigger_type == "trailing_stop" and (not trailing_pct or trailing_pct <= 0):
        return {"error": "trailing_stop requires a positive trailing_pct"}

    trigger_id = f"trg-{uuid.uuid4().hex[:10]}"
    now = datetime.now(timezone.utc)

    # For trailing stops, initialize highest_price to current price
    highest_price = None
    if trigger_type == "trailing_stop":
        current_price, _ = _get_current_price(ticker)
        highest_price = current_price or trigger_price

    with get_db() as db:
        db.execute(
            """
            INSERT INTO price_triggers (
                id, bot_id, ticker, trigger_type, trigger_price, action,
                qty_pct, trailing_pct, highest_price, reason, active,
                created_at, created_by
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, TRUE, %s, %s)
            """,
            [
                trigger_id,
                bot_id,
                ticker,
                trigger_type,
                trigger_price,
                action,
                qty_pct,
                trailing_pct,
                highest_price,
                reason,
                now,
                created_by,
            ],
        )

    logger.info(
        "[TRIGGER] Created %s trigger for %s: %s @ $%.2f (id=%s, by=%s)",
        trigger_type,
        ticker,
        action,
        trigger_price,
        trigger_id[:12],
        created_by,
    )

    return {
        "id": trigger_id,
        "bot_id": bot_id,
        "ticker": ticker,
        "trigger_type": trigger_type,
        "trigger_price": trigger_price,
        "action": action,
        "qty_pct": qty_pct,
        "trailing_pct": trailing_pct,
        "reason": reason,
        "created_by": created_by,
    }


async def check_triggers(bot_id: str) -> list[dict]:
    """Evaluate all active triggers against current prices.

    Called every 1 minute by the background scheduler.
    Returns list of triggered/executed results.
    """
    with get_db() as db:
        triggers = db.execute(
            """
            SELECT id, ticker, trigger_type, trigger_price, action,
                   qty_pct, trailing_pct, highest_price, reason
            FROM price_triggers
            WHERE bot_id = %s AND active = TRUE
            """,
            [bot_id],
        ).fetchall()

    if not triggers:
        return []

    results = []
    now = datetime.now(timezone.utc)

    for row in triggers:
        (
            trigger_id,
            ticker,
            trigger_type,
            trigger_price,
            action,
            qty_pct,
            trailing_pct,
            highest_price,
            reason,
        ) = row

        current_price, _ = _get_current_price(ticker)
        if current_price is None:
            continue

        triggered = False

        if trigger_type == "stop_loss":
            triggered = current_price <= trigger_price

        elif trigger_type == "take_profit":
            triggered = current_price >= trigger_price

        elif trigger_type == "buy_limit":
            triggered = current_price <= trigger_price

        elif trigger_type == "sell_limit":
            triggered = current_price >= trigger_price

        elif trigger_type == "trailing_stop":
            # Update highest_price if current is higher
            if highest_price is None or current_price > highest_price:
                highest_price = current_price
                with get_db() as db:
                    db.execute(
                        "UPDATE price_triggers SET highest_price = %s WHERE id = %s",
                        [highest_price, trigger_id],
                    )

            # Check if price dropped trailing_pct from peak
            if trailing_pct and highest_price and highest_price > 0:
                trail_price = highest_price * (1 - trailing_pct)
                triggered = current_price <= trail_price

        if triggered:
            logger.warning(
                "[TRIGGER] FIRED %s for %s @ $%.2f (trigger=$%.2f, type=%s, reason=%s)",
                action,
                ticker,
                current_price,
                trigger_price,
                trigger_type,
                reason or "N/A",
            )

            # Trigger an Edge Case Agent Cycle instead of a blind trade
            trade_result = None
            try:
                from app.services.pipeline_service import PipelineService
                
                logger.info(
                    "[TRIGGER] Spawning edge-case cycle for %s instead of blind %s", 
                    ticker, action
                )
                
                # Start a rapid response cycle
                res = await PipelineService.start_cycle(
                    tickers=[ticker],
                    collect=True,
                    analyze=True,
                    trade=True,
                    trigger_type=f"edge_case_{trigger_type}",
                )
                
                # If we successfully started a cycle, mark the trigger as fired so it doesn't repeatedly spawn cycles
                with get_db() as db:
                    db.execute(
                        "UPDATE price_triggers SET active = FALSE, triggered_at = %s WHERE id = %s",
                        [now, trigger_id],
                    )
                
                trade_result = {
                    "status": "cycle_started",
                    "cycle_id": res.get("cycle_id"),
                    "trigger_id": trigger_id,
                    "trigger_type": trigger_type,
                    "action_requested": action
                }
                results.append(trade_result)
                
            except ValueError as ve:
                # A cycle is likely already running
                logger.warning(
                    "[TRIGGER] Could not spawn cycle for %s (already running?): %s. Will retry next minute.",
                    ticker, ve
                )
            except Exception as e:
                logger.error(
                    "[TRIGGER] Execution error spawning cycle for %s/%s: %s",
                    ticker,
                    trigger_type,
                    e,
                )

    if results:
        logger.info(
            "[TRIGGER] Fired %d trigger(s) for bot '%s'",
            len(results),
            bot_id,
        )

    return results


async def cancel_trigger(trigger_id: str) -> dict:
    """Deactivate a specific trigger."""
    with get_db() as db:
        result = db.execute(
            "UPDATE price_triggers SET active = FALSE WHERE id = %s RETURNING id, ticker, trigger_type",
            [trigger_id],
        ).fetchone()

    if not result:
        return {"error": f"Trigger {trigger_id} not found"}

    logger.info(
        "[TRIGGER] Cancelled trigger %s (%s/%s)", result[0], result[1], result[2]
    )
    return {
        "status": "cancelled",
        "id": result[0],
        "ticker": result[1],
        "trigger_type": result[2],
    }


def list_triggers(bot_id: str, active_only: bool = True) -> list[dict]:
    """List triggers for a bot."""
    where = "bot_id = %s" + (" AND active = TRUE" if active_only else "")
    with get_db() as db:
        rows = db.execute(
            f"""
            SELECT id, ticker, trigger_type, trigger_price, action,
                   qty_pct, trailing_pct, highest_price, reason,
                   active, triggered_at, created_at, created_by
            FROM price_triggers
            WHERE {where}
            ORDER BY created_at DESC
            """,
            [bot_id],
        ).fetchall()

    return [
        {
            "id": r[0],
            "ticker": r[1],
            "trigger_type": r[2],
            "trigger_price": r[3],
            "action": r[4],
            "qty_pct": r[5],
            "trailing_pct": r[6],
            "highest_price": r[7],
            "reason": r[8],
            "active": bool(r[9]),
            "triggered_at": r[10].isoformat() if r[10] else None,
            "created_at": r[11].isoformat() if r[11] else None,
            "created_by": r[12],
        }
        for r in rows
    ]
