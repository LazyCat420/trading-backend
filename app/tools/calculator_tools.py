"""
Deterministic Financial Calculator Tools.

Provides purpose-built financial calculation tools so the LLM never
has to do mental math for position sizing, stop-losses, or risk metrics.
All calculations are pure Python — no LLM inference, no side effects.
"""

import json
import logging

from app.tools.registry import registry

logger = logging.getLogger(__name__)


@registry.register(
    name="calculate_position_size",
    description=(
        "Calculate the number of shares to buy based on risk management rules. "
        "Uses the fixed-risk method: risk_amount = cash * risk_percent, "
        "shares = risk_amount / (entry_price - stop_loss_price)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "cash_available": {
                "type": "number",
                "description": "Total cash available for trading (e.g., 100000)",
            },
            "risk_percent": {
                "type": "number",
                "description": "Max percentage of cash to risk on this trade (e.g., 0.02 for 2%)",
            },
            "entry_price": {
                "type": "number",
                "description": "Planned entry price per share (e.g., 150.50)",
            },
            "stop_loss_price": {
                "type": "number",
                "description": "Planned stop-loss price per share (e.g., 142.00)",
            },
        },
        "required": [
            "cash_available",
            "risk_percent",
            "entry_price",
            "stop_loss_price",
        ],
    },
    tier=0,
    source="computed",
)
async def calculate_position_size(
    cash_available: float,
    risk_percent: float,
    entry_price: float,
    stop_loss_price: float,
) -> str:
    """Deterministic position sizing based on risk management."""
    if entry_price <= 0:
        return json.dumps({"error": "entry_price must be positive"})
    if stop_loss_price <= 0:
        return json.dumps({"error": "stop_loss_price must be positive"})

    risk_amount = cash_available * risk_percent
    risk_per_share = abs(entry_price - stop_loss_price)

    if risk_per_share <= 0:
        return json.dumps({"error": "stop_loss_price must differ from entry_price"})

    shares = int(risk_amount / risk_per_share)
    total_cost = shares * entry_price
    pct_of_portfolio = (total_cost / cash_available * 100) if cash_available > 0 else 0

    result = {
        "shares": shares,
        "total_cost": round(total_cost, 2),
        "risk_amount": round(risk_amount, 2),
        "risk_per_share": round(risk_per_share, 2),
        "pct_of_portfolio": round(pct_of_portfolio, 1),
    }
    logger.info("[CALC] Position size: %s", result)
    return json.dumps(result)


@registry.register(
    name="calculate_stop_loss",
    description=(
        "Calculate a stop-loss price based on ATR (Average True Range) volatility. "
        "stop_loss = entry_price - (ATR * multiplier)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "entry_price": {
                "type": "number",
                "description": "Planned entry price per share",
            },
            "atr": {
                "type": "number",
                "description": "Current ATR (Average True Range) value",
            },
            "multiplier": {
                "type": "number",
                "description": "ATR multiplier (default 2.0, higher = wider stop)",
            },
        },
        "required": ["entry_price", "atr"],
    },
    tier=0,
    source="computed",
)
async def calculate_stop_loss(
    entry_price: float,
    atr: float,
    multiplier: float = 2.0,
) -> str:
    """Calculate stop-loss based on ATR volatility."""
    if entry_price <= 0:
        return json.dumps({"error": "entry_price must be positive"})
    if atr <= 0:
        return json.dumps({"error": "atr must be positive"})

    stop_loss = entry_price - (atr * multiplier)
    risk_pct = ((entry_price - stop_loss) / entry_price) * 100

    result = {
        "stop_loss": round(stop_loss, 2),
        "risk_percent": round(risk_pct, 2),
        "atr_distance": round(atr * multiplier, 2),
    }
    logger.info("[CALC] Stop-loss: %s", result)
    return json.dumps(result)


@registry.register(
    name="calculate_risk_reward",
    description=(
        "Calculate the risk-to-reward ratio for a trade setup. "
        "A ratio >= 2.0 is generally considered favorable."
    ),
    parameters={
        "type": "object",
        "properties": {
            "entry_price": {
                "type": "number",
                "description": "Planned entry price per share",
            },
            "target_price": {
                "type": "number",
                "description": "Price target for profit taking",
            },
            "stop_loss_price": {
                "type": "number",
                "description": "Stop-loss price for the trade",
            },
        },
        "required": ["entry_price", "target_price", "stop_loss_price"],
    },
    tier=0,
    source="computed",
)
async def calculate_risk_reward(
    entry_price: float,
    target_price: float,
    stop_loss_price: float,
) -> str:
    """Calculate risk-to-reward ratio for a trade setup."""
    risk = abs(entry_price - stop_loss_price)
    reward = abs(target_price - entry_price)
    ratio = round(reward / risk, 2) if risk > 0 else 0

    result = {
        "risk": round(risk, 2),
        "reward": round(reward, 2),
        "ratio": ratio,
        "favorable": ratio >= 2.0,
    }
    logger.info("[CALC] Risk/Reward: %s", result)
    return json.dumps(result)


@registry.register(
    name="calculate_portfolio_allocation",
    description=(
        "Calculate how much capital to allocate to the next trade based on "
        "portfolio diversification rules (max position size, remaining slots)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "total_portfolio_value": {
                "type": "number",
                "description": "Total portfolio value (cash + positions)",
            },
            "max_position_pct": {
                "type": "number",
                "description": "Max percentage per position (e.g., 0.10 for 10%)",
            },
            "current_positions": {
                "type": "integer",
                "description": "Number of positions currently held",
            },
            "max_positions": {
                "type": "integer",
                "description": "Maximum number of positions allowed (e.g., 10)",
            },
        },
        "required": ["total_portfolio_value"],
    },
    tier=0,
    source="computed",
)
async def calculate_portfolio_allocation(
    total_portfolio_value: float,
    max_position_pct: float = 0.10,
    current_positions: int = 0,
    max_positions: int = 10,
) -> str:
    """Calculate how much capital to allocate to the next trade."""
    if current_positions >= max_positions:
        return json.dumps(
            {"error": "Max positions reached", "allocation": 0, "remaining_slots": 0}
        )

    max_per_position = total_portfolio_value * max_position_pct
    remaining_slots = max_positions - current_positions
    # Don't allocate more than 1/remaining_slots of total
    balanced = total_portfolio_value / remaining_slots if remaining_slots > 0 else 0
    allocation = min(max_per_position, balanced)

    result = {
        "allocation": round(allocation, 2),
        "max_per_position": round(max_per_position, 2),
        "remaining_slots": remaining_slots,
        "pct_of_portfolio": round(
            (allocation / total_portfolio_value * 100)
            if total_portfolio_value > 0
            else 0,
            1,
        ),
    }
    logger.info("[CALC] Portfolio allocation: %s", result)
    return json.dumps(result)
