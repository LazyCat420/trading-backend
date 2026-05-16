"""
Trigger Management Tools — Agent-callable tools for price triggers.

Allows the bot to set stop-loss, take-profit, buy/sell limit, and
trailing stop triggers during analysis cycles.
"""

import json
import logging

from app.tools.registry import registry, PermissionLevel
from app.services.bot_manager import get_active_bot_id

logger = logging.getLogger(__name__)


@registry.register(
    name="set_price_trigger",
    description=(
        "Set a price-based trigger for automated trade execution. "
        "Use this to protect positions with stop-losses, lock in profits, "
        "or set limit orders. Types: stop_loss (sell when price drops below), "
        "take_profit (sell when price rises above), buy_limit (buy when price drops to), "
        "sell_limit (sell when price rises to), trailing_stop (sell when price drops X% from peak). "
        "Parameters: ticker (str), trigger_type (str), trigger_price (float), "
        "action (BUY|SELL), qty_pct (float 0-1, default 1.0), "
        "trailing_pct (float, required for trailing_stop), reason (str)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Ticker symbol (e.g., 'AAPL')",
            },
            "trigger_type": {
                "type": "string",
                "enum": [
                    "stop_loss",
                    "take_profit",
                    "buy_limit",
                    "sell_limit",
                    "trailing_stop",
                ],
                "description": "Type of trigger",
            },
            "trigger_price": {
                "type": "number",
                "description": "Price at which to trigger the action",
            },
            "action": {
                "type": "string",
                "enum": ["BUY", "SELL"],
                "description": "Action to execute when triggered. Default: SELL",
            },
            "qty_pct": {
                "type": "number",
                "description": "Fraction of position (0.0-1.0). Default: 1.0 (full position)",
            },
            "trailing_pct": {
                "type": "number",
                "description": "For trailing_stop: percentage drop from peak to trigger (e.g., 0.05 for 5%)",
            },
            "reason": {
                "type": "string",
                "description": "Why this trigger is being set",
            },
        },
        "required": ["ticker", "trigger_type", "trigger_price"],
    },
    permission=PermissionLevel.WRITE,
    tier=2,
    source="trading",
    tags=["trigger", "stop_loss", "take_profit", "automation"],
)
async def set_price_trigger(
    ticker: str,
    trigger_type: str,
    trigger_price: float,
    action: str = "SELL",
    qty_pct: float = 1.0,
    trailing_pct: float | None = None,
    reason: str | None = None,
) -> str:
    """Create a price-based trigger."""
    try:
        from app.trading.order_triggers import create_trigger

        bot_id = get_active_bot_id()
        if not bot_id:
            return json.dumps({"error": "No active bot"})

        result = await create_trigger(
            bot_id=bot_id,
            ticker=ticker,
            trigger_type=trigger_type,
            trigger_price=trigger_price,
            action=action,
            qty_pct=qty_pct,
            trailing_pct=trailing_pct,
            reason=reason,
            created_by="bot",
        )

        return json.dumps(result)
    except Exception as e:
        logger.error("[TRIGGER-TOOL] set_price_trigger failed: %s", e)
        return json.dumps({"error": str(e)})


@registry.register(
    name="list_active_triggers",
    description=(
        "List all active price triggers for the current bot. "
        "Shows trigger type, price, ticker, and status."
    ),
    parameters={
        "type": "object",
        "properties": {},
    },
    permission=PermissionLevel.READ_ONLY,
    tier=0,
    source="trading",
    tags=["trigger", "status"],
)
async def list_active_triggers() -> str:
    """List active triggers for the current bot."""
    try:
        from app.trading.order_triggers import list_triggers

        bot_id = get_active_bot_id()
        if not bot_id:
            return json.dumps({"error": "No active bot", "triggers": []})

        triggers = list_triggers(bot_id, active_only=True)
        return json.dumps(
            {
                "total": len(triggers),
                "triggers": triggers,
            }
        )
    except Exception as e:
        logger.error("[TRIGGER-TOOL] list_active_triggers failed: %s", e)
        return json.dumps({"error": str(e), "triggers": []})


@registry.register(
    name="cancel_price_trigger",
    description=("Cancel (deactivate) a specific price trigger by its ID."),
    parameters={
        "type": "object",
        "properties": {
            "trigger_id": {
                "type": "string",
                "description": "ID of the trigger to cancel",
            },
        },
        "required": ["trigger_id"],
    },
    permission=PermissionLevel.WRITE,
    tier=2,
    source="trading",
    tags=["trigger", "cancel"],
)
async def cancel_price_trigger(trigger_id: str) -> str:
    """Cancel a specific trigger."""
    try:
        from app.trading.order_triggers import cancel_trigger

        result = await cancel_trigger(trigger_id)
        return json.dumps(result)
    except Exception as e:
        logger.error("[TRIGGER-TOOL] cancel_price_trigger failed: %s", e)
        return json.dumps({"error": str(e)})
