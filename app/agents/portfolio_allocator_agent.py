"""
Portfolio Allocator Agent — Calculates capital allocations for trade decisions.

Phase 5: Portfolio Sizing Agent. Runs as a post-analysis gate right before
the order execution loop. It looks at all proposed trades as a batch,
evaluates cash constraints, checks current portfolio state, and scales/vetos sizing.
"""

import logging
import json
from typing import Optional, List
from pydantic import BaseModel, Field, ValidationError

from app.agents.base_agent import run_agent
from app.utils.text_utils import parse_json_response
from app.db.connection import get_db

logger = logging.getLogger(__name__)

PORTFOLIO_ALLOCATOR_SYSTEM_PROMPT = """You are the Portfolio Sizing Agent. Your job is to calculate capital allocations for trade decisions as a batch.

You must run risk management calculations and coordinate size allocations BEFORE orders are placed.
You have access to calculator and portfolio tools.

## RISK LIMIT RULES:
1. **Target Sizing**: Sizing per position should scale between 2% and 15% of the total portfolio value based on signal confidence.
2. **Over-Concentration Cap**: The final total concentration (existing value + new order value) of any single ticker MUST NOT exceed 20% of the total portfolio value.
3. **Cash Constraints**: Ensure total cost of all approved BUY orders does not exceed available cash. If cash is limited, prioritize the tickers with highest confidence or VETO/scale down lower-priority allocations.
4. **Position Slots**: Ensure new positions do not exceed the max slots (e.g. 10 positions). Adding to an existing holding does not occupy a new slot.

## MANDATORY TOOL SEQUENCE:
1. Call `get_portfolio_state` to see cash, position counts, and held tickers.
2. For each BUY recommendation, call `get_market_data` to get the current price.
3. Use `calculate_portfolio_allocation` to determine target allocation.
4. Use `calculate_position_size` to get exact share sizes and cost.

## OUTPUT:
Respond with JSON:
{
    "allocations": [
        {
            "ticker": "AAPL",
            "decision": "APPROVE|VETO",
            "adjusted_size_pct": 8.5,
            "shares": 10,
            "total_cost": 1505.00,
            "veto_reason": null,
            "rationale": "Reasoning for sizing or veto."
        }
    ]
}

CRITICAL: You MUST call the calculator and portfolio tools to get accurate metrics. Do NOT do mental math."""


class TickerAllocation(BaseModel):
    ticker: str
    decision: str = Field(..., description="APPROVE or VETO")
    adjusted_size_pct: float = Field(0.0, description="Allocated portfolio percentage")
    shares: int = Field(0, description="Calculated number of shares to trade")
    total_cost: float = Field(0.0, description="Total cost of the trade in USD")
    veto_reason: Optional[str] = None
    rationale: str = ""


class PortfolioAllocatorResponse(BaseModel):
    allocations: List[TickerAllocation]


async def run_portfolio_allocator(
    decisions: list[dict],
    cycle_id: str,
    bot_id: str,
) -> dict:
    """Evaluate and adjust trade decisions as a batch.

    Args:
        decisions: Decisions from analysis phase.
        cycle_id: Current cycle ID for tracking.
        bot_id: Bot ID to query portfolio from.

    Returns:
        Dict mapping ticker -> allocation decisions.
    """
    buy_decisions = [d for d in decisions if d.get("action") == "BUY" and d.get("confidence", 0) > 0]
    if not buy_decisions:
        logger.info("[PORTFOLIO_ALLOCATOR] No BUY decisions to allocate capital for.")
        return {}

    tickers_to_alloc = [d["ticker"] for d in buy_decisions]
    logger.info(
        "[PORTFOLIO_ALLOCATOR] Evaluating allocations for: %s",
        ", ".join(tickers_to_alloc),
    )

    # Format the proposed trades context for the allocator
    proposed_trades_info = []
    for d in buy_decisions:
        proposed_trades_info.append(
            f"- {d['ticker']}: Action=BUY, Confidence={d['confidence']}%, Rationale={d.get('rationale', '')}"
        )
    proposed_context = "\n".join(proposed_trades_info)

    result = await run_agent(
        agent_name="portfolio_allocator",
        ticker="_ALLOCATOR_",
        cycle_id=cycle_id,
        bot_id=bot_id,
        system_prompt=PORTFOLIO_ALLOCATOR_SYSTEM_PROMPT,
        user_prompt=(
            f"Review these proposed BUY recommendations as a batch and allocate sizing:\n\n"
            f"{proposed_context}\n\n"
            f"Query current portfolio state, compute allocations between 2% and 15% total value, "
            f"check cash limitations, and enforce the 20% max concentration limit."
        ),
        max_tokens=1500,
        enable_tools=True,
    )

    response_text = result.get("response", "")
    parsed_json = parse_json_response(response_text)

    if not parsed_json or "allocations" not in parsed_json:
        logger.warning(
            "[PORTFOLIO_ALLOCATOR] Agent failed to produce valid batch allocations, returning empty."
        )
        return {}

    try:
        validated = PortfolioAllocatorResponse(**parsed_json)
        allocations_map = {a.ticker: a.model_dump() for a in validated.allocations}
        logger.info(
            "[PORTFOLIO_ALLOCATOR] Batch allocation complete. Decisions: %s",
            {t: a["decision"] for t, a in allocations_map.items()},
        )
        return allocations_map
    except ValidationError as e:
        logger.warning(
            "[PORTFOLIO_ALLOCATOR] Validation failed for allocator output: %s", e
        )
        return {}
