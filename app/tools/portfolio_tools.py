"""
Portfolio Context Tools — Give agents full position visibility.

Provides tools for agents to query the bot's portfolio state and
individual position P&L during debates and analysis. This is the
foundation for position-aware sell decisions.
"""

import json
import logging
from datetime import datetime, timezone

from app.tools.registry import registry
from app.config import settings

logger = logging.getLogger(__name__)


def get_position_context(ticker: str, bot_id: str = "") -> dict:
    """Build a position context block for a ticker the bot may hold.

    Returns a dict describing the bot's current position (if any) including
    entry price, unrealized P&L, holding duration, and stop-loss level.
    This context is injected into the debate system so agents know whether
    the bot already holds this ticker and can argue for/against exiting.

    Returns:
        {
            "held": bool,
            "qty": float,
            "avg_entry": float,
            "current_price": float | None,
            "unrealized_pnl": float,
            "unrealized_pnl_pct": float,
            "holding_days": int,
            "stop_loss_pct": float,
            "stop_price": float,
            "original_thesis": str | None,
            "original_thesis_date": str | None,
            "original_thesis_conf": int | None,
        }
    """
    from app.db.connection import get_db
    from app.trading.paper_trader import _get_current_price

    if not bot_id:
        bot_id = settings.BOT_ID

    with get_db() as db:
        ticker = ticker.upper()

        try:
            row = db.execute(
                "SELECT qty, avg_entry_price, stop_loss_pct, opened_at "
                "FROM positions WHERE bot_id = %s AND ticker = %s",
                [bot_id, ticker],
            ).fetchone()
        except Exception as e:
            logger.debug(
                "[POSITION_CTX] Failed to query position for %s: %s",
                ticker,
                e,
            )
            return {"held": False}

        # Query original buy thesis
        original_thesis = None
        original_thesis_date = None
        original_thesis_conf = None
        if row and row[0] and row[0] > 0:
            try:
                row_thesis = db.execute(
                    """
                    SELECT created_at, thesis_summary, confidence
                    FROM analysis_results
                    WHERE ticker = %s AND (thesis_verdict = 'BUY' OR (result_json::jsonb->>'action') = 'BUY')
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    [ticker],
                ).fetchone()
                if row_thesis:
                    original_thesis_date, original_thesis, original_thesis_conf = row_thesis
                    if original_thesis_date:
                        if hasattr(original_thesis_date, "strftime"):
                            original_thesis_date = original_thesis_date.strftime("%Y-%m-%d")
                        else:
                            original_thesis_date = str(original_thesis_date)[:10]
            except Exception as e:
                logger.debug(
                    "[POSITION_CTX] Failed to query original thesis for %s: %s",
                    ticker,
                    e,
                )

    if not row or not row[0] or row[0] <= 0:
        return {"held": False}

    qty, avg_entry, stop_pct, opened_at = row
    stop_pct = stop_pct or 0.08  # Fallback default

    # Get current price
    current_price, _ = _get_current_price(ticker)

    # Calculate unrealized P&L
    unrealized_pnl = 0.0
    unrealized_pnl_pct = 0.0
    if current_price and avg_entry and avg_entry > 0:
        unrealized_pnl = (current_price - avg_entry) * qty
        unrealized_pnl_pct = ((current_price - avg_entry) / avg_entry) * 100

    # Calculate holding duration
    holding_days = 0
    if opened_at:
        try:
            if isinstance(opened_at, str):
                opened_dt = datetime.fromisoformat(opened_at.replace("Z", "+00:00"))
            else:
                opened_dt = opened_at
            if hasattr(opened_dt, "tzinfo") and opened_dt.tzinfo is None:
                opened_dt = opened_dt.replace(tzinfo=timezone.utc)
            delta = datetime.now(timezone.utc) - opened_dt
            holding_days = delta.days
        except Exception:
            pass

    stop_price = avg_entry * (1 - stop_pct) if avg_entry else 0

    return {
        "held": True,
        "qty": round(qty, 4),
        "avg_entry": round(avg_entry, 2),
        "current_price": (round(current_price, 2) if current_price else None),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "unrealized_pnl_pct": round(unrealized_pnl_pct, 2),
        "holding_days": holding_days,
        "stop_loss_pct": round(stop_pct * 100, 1),
        "stop_price": round(stop_price, 2),
        "original_thesis": original_thesis,
        "original_thesis_date": original_thesis_date,
        "original_thesis_conf": original_thesis_conf,
    }


def format_position_context_for_prompt(ctx: dict) -> str:
    """Format position context into a human-readable block for LLM prompts.

    Returns an empty string if no position is held.
    """
    if not ctx.get("held"):
        return ""

    pnl_emoji = "🟢" if ctx["unrealized_pnl_pct"] >= 0 else "🔴"
    
    thesis_text = "[No recorded BUY thesis found in database]"
    if ctx.get("original_thesis"):
        thesis_text = (
            f"Recorded on {ctx['original_thesis_date']} at {ctx['original_thesis_conf']}% confidence:\n"
            f"  \"\"\"\n  {ctx['original_thesis']}\n  \"\"\""
        )

    return (
        f"# CURRENT POSITION STATUS\n"
        f"You are evaluating a ticker the bot ALREADY HOLDS.\n"
        f"- Shares: {ctx['qty']}\n"
        f"- Entry Price: ${ctx['avg_entry']}\n"
        f"- Current Price: ${ctx.get('current_price', 'N/A')}\n"
        f"- Unrealized P&L: {pnl_emoji} {ctx['unrealized_pnl_pct']:+.1f}% "
        f"(${ctx['unrealized_pnl']:+,.2f})\n"
        f"- Holding Duration: {ctx['holding_days']} days\n"
        f"- Stop-Loss: {ctx['stop_loss_pct']}% "
        f"(triggers at ${ctx['stop_price']})\n"
        f"- Original Buy Thesis: {thesis_text}\n\n"
        f"IMPORTANT: Because this is an EXISTING POSITION, you can choose to BUY (add to position), HOLD (keep current sizing), or SELL (liquidate position).\n"
        f"Consider:\n"
        f"1. Is the original thesis still valid, or has it been invalidated by recent news/data?\n"
        f"2. If you are suggesting to BUY more, why is it justified to increase risk exposure now? Compare opportunity cost.\n"
        f"3. Has the price approached stop-loss or profit-taking targets?"
    )


# ── Tool: get_portfolio_state ────────────────────────────────────────
@registry.register(
    name="get_portfolio_state",
    description=(
        "Get the bot's full portfolio state including cash balance, "
        "all open positions with P&L, sector breakdown, and position "
        "count. Use this to understand portfolio health and make "
        "informed BUY/SELL decisions."
    ),
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    tier=1,
    source="paper_trader",
    tags=["portfolio", "positions", "cash", "pnl"],
)
async def get_portfolio_state_tool() -> str:
    """Return the bot's full portfolio state as JSON."""
    try:
        from app.trading.paper_trader import get_portfolio

        portfolio = get_portfolio(settings.BOT_ID)

        # Enrich positions with current prices and P&L
        enriched = []
        for pos in portfolio.get("positions", []):
            ctx = get_position_context(pos["ticker"], settings.BOT_ID)
            enriched.append(
                {
                    "ticker": pos["ticker"],
                    "qty": pos["qty"],
                    "avg_entry": pos["avg_entry_price"],
                    "current_price": ctx.get("current_price"),
                    "pnl_pct": ctx.get("unrealized_pnl_pct", 0),
                    "pnl_usd": ctx.get("unrealized_pnl", 0),
                    "holding_days": ctx.get("holding_days", 0),
                    "stop_loss_pct": ctx.get("stop_loss_pct"),
                }
            )

        # Sector breakdown
        sectors: dict[str, int] = {}
        try:
            from app.db.connection import get_db

            with get_db() as db:
                for pos in enriched:
                    row = db.execute(
                        "SELECT sector FROM ticker_metadata WHERE ticker = %s",
                        [pos["ticker"]],
                    ).fetchone()
                    sector = row[0] if row else "Unknown"
                    sectors[sector] = sectors.get(sector, 0) + 1
        except Exception:
            pass

        return json.dumps(
            {
                "status": "success",
                "cash": portfolio["cash"],
                "total_pnl": portfolio["total_pnl"],
                "position_count": len(enriched),
                "positions": enriched,
                "sector_breakdown": sectors,
            }
        )
    except Exception as e:
        logger.exception("[PortfolioTools] get_portfolio_state failed")
        return json.dumps({"status": "error", "message": str(e)})


# ── Tool: get_position_pnl ───────────────────────────────────────────
@registry.register(
    name="get_position_pnl",
    description=(
        "Check the P&L and health of a specific held position. "
        "Returns entry price, current price, unrealized P&L, "
        "holding duration, and stop-loss proximity. Use this "
        "during debates to evaluate whether to hold or sell."
    ),
    parameters={
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "The stock ticker to check.",
            },
        },
        "required": ["ticker"],
    },
    tier=1,
    source="paper_trader",
    tags=["position", "pnl", "health", "stop-loss"],
)
async def get_position_pnl_tool(ticker: str) -> str:
    """Return P&L details for a specific position."""
    try:
        ctx = get_position_context(ticker, settings.BOT_ID)
        if not ctx.get("held"):
            return json.dumps(
                {
                    "status": "success",
                    "ticker": ticker.upper(),
                    "held": False,
                    "message": (f"No open position in {ticker.upper()}."),
                }
            )
        return json.dumps(
            {
                "status": "success",
                "ticker": ticker.upper(),
                **ctx,
            }
        )
    except Exception as e:
        logger.exception(
            "[PortfolioTools] get_position_pnl failed for %s",
            ticker,
        )
        return json.dumps(
            {
                "status": "error",
                "ticker": ticker,
                "message": str(e),
            }
        )


# ── Tool: get_performance_metrics ─────────────────────────────────────
@registry.register(
    name="get_performance_metrics",
    description=(
        "Get bot's historical trading performance metrics over the last "
        "N trades or N days. Includes win rate, average profit, average "
        "loss, and average holding duration. Useful for benchmarking "
        "and suggesting strategy improvements."
    ),
    parameters={
        "type": "object",
        "properties": {
            "days_back": {
                "type": "integer",
                "description": "Number of days back to look. Default 30.",
            },
        },
        "required": [],
    },
    tier=1,
    source="paper_trader",
    tags=["performance", "metrics", "win-rate", "benchmark"],
)
async def get_performance_metrics_tool(days_back: int = 30) -> str:
    """Return bot performance metrics as JSON."""
    try:
        from app.db.connection import get_db
        from app.config import settings

        with get_db() as db:
            bot_id = settings.BOT_ID

            # Query lot closures for realized trades
            rows = db.execute(
                "SELECT realized_pnl, holding_days "
                "FROM lot_closures "
                "WHERE bot_id = %s "
                "AND closed_at >= CURRENT_TIMESTAMP + CAST(%s || ' days' AS INTERVAL)",
                [bot_id, f"-{days_back}"],
            ).fetchall()

            if not rows:
                return json.dumps(
                    {
                        "status": "success",
                        "message": f"No closed trades in the last {days_back} days.",
                        "total_trades": 0,
                    }
                )

            total_trades = len(rows)
            winning_trades = [r for r in rows if r[0] > 0]
            losing_trades = [r for r in rows if r[0] <= 0]

            win_rate = (len(winning_trades) / total_trades) * 100
            avg_profit = (
                sum(r[0] for r in winning_trades) / len(winning_trades)
                if winning_trades
                else 0
            )
            avg_loss = (
                sum(r[0] for r in losing_trades) / len(losing_trades)
                if losing_trades
                else 0
            )
            avg_holding = sum(r[1] or 0 for r in rows) / total_trades

            # Query total unrealized P&L from open positions
            open_rows = db.execute(
                "SELECT qty, avg_entry_price, ticker FROM positions WHERE bot_id = %s",
                [bot_id],
            ).fetchall()

            open_positions_count = len(open_rows)

        return json.dumps(
            {
                "status": "success",
                "days_back": days_back,
                "total_closed_trades": total_trades,
                "win_rate_pct": round(win_rate, 1),
                "avg_profit_usd": round(avg_profit, 2),
                "avg_loss_usd": round(avg_loss, 2),
                "avg_holding_days": round(avg_holding, 1),
                "open_positions_count": open_positions_count,
            }
        )
    except Exception as e:
        logger.exception("[PortfolioTools] get_performance_metrics failed")
        return json.dumps({"status": "error", "message": str(e)})


# ── Tool: propose_constitution_amendment ──────────────────────────────
@registry.register(
    name="propose_constitution_amendment",
    description=(
        "Propose an amendment to a Trading Constitution parameter. "
        "This triggers the Evolution Router to debate and potentially "
        "approve the change. Use this when performance data suggests "
        "a rule parameter is causing poor outcomes (e.g. max_positions "
        "is too high, holding_days is too long)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "rule_id": {
                "type": "string",
                "description": "The ID of the rule to amend (e.g., 'position_limit_v1').",
            },
            "param_name": {
                "type": "string",
                "description": "The parameter to change (e.g., 'max_positions').",
            },
            "proposed_value": {
                "type": "number",
                "description": "The new numeric value proposed for the parameter.",
            },
            "rationale": {
                "type": "string",
                "description": "A strong justification based on performance metrics for why this change is needed.",
            },
        },
        "required": ["rule_id", "param_name", "proposed_value", "rationale"],
    },
    tier=2,
    source="benchmark_agent",
    tags=["constitution", "amendment", "evolution", "rules"],
)
async def propose_constitution_amendment_tool(
    rule_id: str, param_name: str, proposed_value: float, rationale: str
) -> str:
    """Propose an amendment to the Trading Constitution."""
    try:
        from app.pipeline.trading_constitution import validate_amendment
        from app.pipeline.analysis.evolution_router import request_evolution_debate
        from app.db.connection import get_db

        with get_db() as db:
            # 1. Validate bounds
            valid, msg = validate_amendment(param_name, proposed_value)
            if not valid:
                return json.dumps(
                    {"status": "rejected", "reason": f"Safety bounds violation: {msg}"}
                )

            # 2. Check if rule exists
            row = db.execute(
                "SELECT rule_category, rule_params FROM trading_constitution WHERE id = %s AND is_active = TRUE",
                [rule_id],
            ).fetchone()

            if not row:
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"Rule {rule_id} not found or is inactive.",
                    }
                )

        # 3. Create the amendment proposal and send to Evolution Router
        proposal_payload = {
            "rule_id": rule_id,
            "rule_category": row[0],
            "param_name": param_name,
            "proposed_value": proposed_value,
            "rationale": rationale,
            "current_params": row[1],
        }

        # Dispatch to Evolution Router council
        # We use 'constitution_amendment' as the issue category
        issue_desc = (
            f"Constitution Amendment Proposal: Change {param_name} to {proposed_value} "
            f"in rule {rule_id}.\nRationale: {rationale}"
        )

        debate_req = await request_evolution_debate(
            category="constitution_amendment",
            issue_description=issue_desc,
            error_trace=json.dumps(proposal_payload),
            severity="high",
        )

        return json.dumps(
            {
                "status": "success",
                "message": "Amendment proposal submitted to the Evolution Council for debate.",
                "evolution_request": debate_req,
            }
        )

    except Exception as e:
        logger.exception("[PortfolioTools] propose_amendment failed")
        return json.dumps({"status": "error", "message": str(e)})


def get_portfolio_risk_dashboard(ticker: str, bot_id: str = "") -> str:
    """Build a quantitative risk and capacity dashboard for a target ticker.
    
    Includes cash balance, total portfolio value, active sector exposures,
    Constitution sector limits, remaining sector capital capacity,
    ticker concentration limits, and dynamic correlations (hedges & overlaps)
    with currently held positions.
    """
    from app.db.connection import get_db
    from app.trading.paper_trader import get_portfolio

    if not bot_id:
        bot_id = settings.BOT_ID

    ticker = ticker.upper()

    try:
        portfolio = get_portfolio(bot_id)
        cash = portfolio.get("cash", 0.0)
        positions = portfolio.get("positions", [])
        position_count = len(positions)
    except Exception as e:
        logger.warning("[PortfolioTools] Failed to load portfolio state: %s", e)
        return ""

    # Fetch thresholds from Constitution
    sector_cap = 30.0
    try:
        from app.pipeline.trading_constitution import get_constitution_param
        sector_cap = float(get_constitution_param("sector", "max_sector_pct", 30.0))
    except Exception:
        pass

    # Enrich positions with current prices and calculate total portfolio value
    total_value = cash
    enriched_positions = []
    sector_exposures = {}
    existing_ticker_value = 0.0

    with get_db() as db:
        for pos in positions:
            pos_ticker = pos["ticker"]
            pos_qty = pos["qty"]
            
            # Fetch current price
            ctx = get_position_context(pos_ticker, bot_id)
            current_price = ctx.get("current_price") or pos.get("avg_entry_price", 0.0)
            pos_value = pos_qty * current_price
            total_value += pos_value

            if pos_ticker == ticker:
                existing_ticker_value = pos_value

            # Fetch sector
            try:
                row = db.execute(
                    "SELECT sector FROM ticker_metadata WHERE ticker = %s",
                    [pos_ticker],
                ).fetchone()
                sector = row[0] if row else "Unknown"
            except Exception:
                sector = "Unknown"

            enriched_positions.append({
                "ticker": pos_ticker,
                "value": pos_value,
                "sector": sector,
            })
            
            sector_exposures[sector] = sector_exposures.get(sector, 0.0) + pos_value

    # Identify target ticker sector
    target_sector = "Unknown"
    with get_db() as db:
        try:
            row = db.execute(
                "SELECT sector FROM ticker_metadata WHERE ticker = %s",
                [ticker],
            ).fetchone()
            if row:
                target_sector = row[0]
        except Exception:
            pass

    # Calculate sector exposure and capacities
    current_sector_value = sector_exposures.get(target_sector, 0.0)
    current_sector_pct = (current_sector_value / total_value * 100.0) if total_value > 0 else 0.0

    max_sector_value = total_value * (sector_cap / 100.0)
    remaining_sector_capacity = max(0.0, max_sector_value - current_sector_value)

    # Ticker concentration limit (max 25% of portfolio value per ticker)
    max_concentration_pct = getattr(settings, "MAX_CONCENTRATION_PCT", 0.25)
    max_concentration_limit = total_value * max_concentration_pct
    max_allowed_for_ticker = max(0.0, max_concentration_limit - existing_ticker_value)

    # Max allowed BUY amount is the constraint of cash, sector headroom, and concentration headroom
    max_allowed_buy_amount = min(cash, remaining_sector_capacity, max_allowed_for_ticker)

    # Correlations & Hedging
    spy_corr = "N/A"
    qqq_corr = "N/A"
    high_overlaps = []
    hedges = []

    with get_db() as db:
        def _local_corr(t_a, t_b):
            try:
                row = db.execute(
                    """
                    SELECT correlation FROM ticker_correlations
                    WHERE (ticker_a = %s AND ticker_b = %s)
                       OR (ticker_a = %s AND ticker_b = %s)
                    ORDER BY computed_at DESC LIMIT 1
                    """,
                    [t_a, t_b, t_b, t_a],
                ).fetchone()
                return row[0] if row else None
            except Exception:
                return None

        # Fetch SPY/QQQ correlations
        s_c = _local_corr(ticker, "SPY")
        if s_c is not None:
            spy_corr = f"{s_c:+.2f}"
        q_c = _local_corr(ticker, "QQQ")
        if q_c is not None:
            qqq_corr = f"{q_c:+.2f}"

        # Fetch correlation with other held tickers
        for ep in enriched_positions:
            pos_ticker = ep["ticker"]
            if pos_ticker == ticker:
                continue
            corr = _local_corr(ticker, pos_ticker)
            if corr is not None:
                if corr > 0.70:
                    high_overlaps.append(f"{pos_ticker} ({corr:+.2f})")
                elif corr < 0.0:
                    hedges.append(f"{pos_ticker} ({corr:+.2f})")

    # Format the block
    lines = [
        "# PORTFOLIO RISK & CAPACITY LIMITS",
        "You must analyze this ticker within the context of the active portfolio and risk constraints.",
        "",
        "## Portfolio Allocation Status",
        f"- Available Cash: ${cash:,.2f}",
        f"- Total Portfolio Value: ${total_value:,.2f}",
        f"- Active Positions: {position_count}",
        "",
        "## Trade Capital & Sizing Budget",
        f"- Ticker Sector: {target_sector}",
        f"- Current Sector Exposure: ${current_sector_value:,.2f} ({current_sector_pct:.1f}% of portfolio)",
        f"- Max Allowed Sector Exposure: ${max_sector_value:,.2f} ({sector_cap:.1f}% limit)",
        f"- Remaining Sector Capacity: ${remaining_sector_capacity:,.2f}",
        f"- Max Concentration Limit for {ticker}: ${max_concentration_limit:,.2f} ({max_concentration_pct * 100.0:.1f}% limit)",
    ]

    if enriched_positions and any(ep["ticker"] == ticker for ep in enriched_positions):
        lines.append(
            f"- **MAXIMUM ALLOWED BUY AMOUNT FOR THIS TRADE**: ${max_allowed_buy_amount:,.2f}\n"
            f"  *(Note: You ALREADY hold {ticker}. Evaluators must prioritize HOLD/SELL. If recommending BUY/add to position, the maximum allowed is ${max_allowed_buy_amount:,.2f}.)*"
        )
    else:
        lines.append(
            f"- **MAXIMUM ALLOWED BUY AMOUNT FOR THIS TRADE**: ${max_allowed_buy_amount:,.2f}\n"
            f"  *(Any recommendation to BUY must stay within this budget. Exceeding it will be blocked by the risk engine.)*"
        )

    lines.extend([
        "",
        "## Portfolio Correlation & Hedging Analysis",
        f"- Correlation with SPY: {spy_corr}",
        f"- Correlation with QQQ: {qqq_corr}",
    ])

    if high_overlaps:
        lines.append(f"- High Correlation Overlaps (Correlation > 0.70): {', '.join(high_overlaps)}")
    else:
        lines.append("- High Correlation Overlaps (Correlation > 0.70): None")

    if hedges:
        lines.append(f"- Negative Correlation Hedges (Correlation < 0.0): {', '.join(hedges)}")
    else:
        lines.append("- Negative Correlation Hedges (Correlation < 0.0): None")

    lines.extend([
        "",
        "## Strategic Directives",
        f"1. If recommending BUY, ensure your sizing rationale aligns with the ${max_allowed_buy_amount:,.2f} maximum cap.",
        "2. If there are high correlation overlaps, justify why adding this ticker is worth the redundant portfolio risk.",
        "3. If this ticker has negative correlation hedges, evaluate if it serves as a valuable hedge for our active exposure.",
    ])

    return "\n".join(lines)

