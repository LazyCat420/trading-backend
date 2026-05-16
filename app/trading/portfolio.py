"""
Portfolio snapshots & equity curve — reads from PostgreSQL trading tables.

Uses actual schema:
  - portfolio_snapshots(id, bot_id, snapshot_ts, cash_balance, total_value, ...)
  - positions(id, bot_id, ticker, qty, avg_entry_price, stop_loss_pct)
  - orders(id, bot_id, ticker, side, qty, price, signal, created_at, filled_at)
  - bots(bot_id, cash_balance, total_pnl, total_trades, ...)

Usage:
    from app.trading.portfolio import (
        get_current_state, take_snapshot, get_equity_curve,
        get_recent_trades, get_performance_summary,
    )
"""

import logging
import uuid
import math
from datetime import datetime, timezone

from app.db.connection import get_db
from app.config import settings
from app.utils.tz import utc_iso

logger = logging.getLogger(__name__)


def _get_default_bot_id() -> str:
    """Resolve the active bot_id dynamically."""
    try:
        from app.services.bot_manager import get_active_bot_id

        return get_active_bot_id()
    except Exception:
        return settings.BOT_ID


def _safe_float(val, fallback=None):
    if val is None:
        return fallback
    try:
        fval = float(val)
        if math.isinf(fval) or math.isnan(fval):
            return fallback
        return fval
    except (ValueError, TypeError):
        return fallback


def get_current_state(bot_id: str = "") -> dict:
    """Return current portfolio: cash, positions, total value."""
    bid = bot_id or _get_default_bot_id()
    logger.info("[TRACE][STATE] resolved bid=%s (input was '%s')", bid, bot_id)

    # Bot state
    with get_db() as db:
        bot = db.execute(
            "SELECT cash_balance, total_pnl, total_trades "
            "FROM bots WHERE bot_id = %s LIMIT 1",
            [bid],
        ).fetchone()

    cash = (
        _safe_float(bot[0], fallback=settings.STARTING_CASH)
        if bot
        else settings.STARTING_CASH
    )
    total_pnl = _safe_float(bot[1], fallback=0.0) if bot else 0.0

    # Latest snapshot timestamp
    with get_db() as db:
        snap = db.execute(
            "SELECT snapshot_ts "
            "FROM portfolio_snapshots "
            "WHERE bot_id = %s ORDER BY snapshot_ts DESC LIMIT 1",
            [bid],
        ).fetchone()

    updated_at = snap[0] if snap else None

    # Open positions and dynamic equity calculation
    positions = []
    equity = 0.0
    try:
        with get_db() as db:
            pos_rows = db.execute(
                "SELECT ticker, qty, avg_entry_price, stop_loss_pct "
                "FROM positions WHERE bot_id = %s AND qty > 0 "
                "ORDER BY ticker",
                [bid],
            ).fetchall()
            logger.info(
                "[TRACE][STATE] raw pos_rows count=%d for bot_id=%s", len(pos_rows), bid
            )
            for p in pos_rows:
                logger.info(
                    "[TRACE][STATE] raw row: ticker=%s qty=%s avg_entry=%s",
                    p[0],
                    p[1],
                    p[2],
                )
            for p in pos_rows:
                ticker = p[0]
                qty = _safe_float(p[1], fallback=0.0)
                avg_entry_price = _safe_float(p[2], fallback=0.0)
                stop_loss_pct = _safe_float(p[3], fallback=0.0)

                # Fetch live current price
                price_row = db.execute(
                    "SELECT close FROM price_history "
                    "WHERE ticker = %s ORDER BY date DESC LIMIT 1",
                    [ticker],
                ).fetchone()
                if not price_row:
                    price_row = db.execute(
                        "SELECT close FROM asset_prices "
                        "WHERE symbol = %s ORDER BY date DESC LIMIT 1",
                        [ticker],
                    ).fetchone()

                curr_price = (
                    _safe_float(price_row[0], fallback=avg_entry_price)
                    if price_row and price_row[0]
                    else avg_entry_price
                )

                # ── Price sanity check ──
                # Prevent phantom gains from ticker collisions (e.g. ETH ETF vs ETH crypto)
                # or stale/corrupt price data.  If current price deviates >10x from entry,
                # the data is almost certainly wrong — fall back to entry price.
                if avg_entry_price and avg_entry_price > 0 and curr_price > 0:
                    ratio = curr_price / avg_entry_price
                    if ratio > 10 or ratio < 0.1:
                        logger.warning(
                            "price sanity: %s curr=$%.2f vs entry=$%.2f (%.1fx) — "
                            "using entry price to prevent phantom P&L",
                            ticker,
                            curr_price,
                            avg_entry_price,
                            ratio,
                        )
                        curr_price = avg_entry_price

                equity += qty * curr_price

                # Fetch extra data for table consistency
                meta = db.execute(
                    "SELECT sector, market_cap_tier, market_cap FROM ticker_metadata WHERE ticker = %s",
                    [ticker],
                ).fetchone()
                fund = db.execute(
                    "SELECT pe_ratio, revenue_growth FROM fundamentals WHERE ticker = %s ORDER BY snapshot_date DESC LIMIT 1",
                    [ticker],
                ).fetchone()
                tech = db.execute(
                    "SELECT rsi_14 FROM technicals WHERE ticker = %s ORDER BY date DESC LIMIT 1",
                    [ticker],
                ).fetchone()

                positions.append(
                    {
                        "ticker": ticker,
                        "qty": qty,
                        "avg_entry_price": avg_entry_price,
                        "current_price": curr_price,
                        "stop_loss_pct": stop_loss_pct,
                        "sector": meta[0] if meta else None,
                        "market_cap_tier": meta[1] if meta else None,
                        "market_cap": _safe_float(meta[2]) if meta else None,
                        "pe_ratio": _safe_float(fund[0]) if fund else None,
                        "revenue_growth": _safe_float(fund[1]) if fund else None,
                        "rsi_14": _safe_float(tech[0]) if tech else None,
                    }
                )
    except Exception as e:
        logger.warning("positions query failed: %s", e)

    total_value = cash + equity

    logger.info(
        "[TRACE][STATE] positions returned=%d cash=%.2f total_value=%.2f for bot_id=%s",
        len(positions),
        cash,
        total_value,
        bid,
    )

    return {
        "bot_id": bid,
        "cash": cash,
        "total_value": total_value,
        "total_pnl": total_pnl,
        "positions": positions,
        "position_count": len(positions),
        "updated_at": utc_iso(updated_at),
    }


def get_recent_trades(bot_id: str = "", limit: int = 50) -> list[dict]:
    """Return recent orders (paper trades)."""
    bid = bot_id or _get_default_bot_id()
    results = []
    with get_db() as db:
        rows = db.execute(
            "SELECT ticker, side, qty, price, signal, "
            "created_at, filled_at, realized_pnl "
            "FROM orders WHERE bot_id = %s "
            "ORDER BY created_at DESC LIMIT %s",
            [bid, limit],
        ).fetchall()
        for r in rows:
            ticker = r[0]
            meta = db.execute(
                "SELECT sector, market_cap_tier, market_cap FROM ticker_metadata WHERE ticker = %s",
                [ticker],
            ).fetchone()
            fund = db.execute(
                "SELECT pe_ratio, revenue_growth FROM fundamentals WHERE ticker = %s ORDER BY snapshot_date DESC LIMIT 1",
                [ticker],
            ).fetchone()
            tech = db.execute(
                "SELECT rsi_14 FROM technicals WHERE ticker = %s ORDER BY date DESC LIMIT 1",
                [ticker],
            ).fetchone()

            results.append(
                {
                    "ticker": ticker,
                    "side": r[1],
                    "qty": _safe_float(r[2], fallback=0.0),
                    "price": _safe_float(r[3], fallback=0.0),
                    "signal": r[4],
                    "created_at": utc_iso(r[5]),
                    "filled_at": utc_iso(r[6]),
                    "realized_pnl": _safe_float(r[7], fallback=0.0),
                    "sector": meta[0] if meta else None,
                    "market_cap_tier": meta[1] if meta else None,
                    "market_cap": _safe_float(meta[2]) if meta else None,
                    "pe_ratio": _safe_float(fund[0]) if fund else None,
                    "revenue_growth": _safe_float(fund[1]) if fund else None,
                    "rsi_14": _safe_float(tech[0]) if tech else None,
                }
            )
    return results


def get_equity_curve(bot_id: str = "", days: int = 30) -> list[dict]:
    """Return equity curve data from portfolio_snapshots."""
    bid = bot_id or _get_default_bot_id()
    days = max(1, min(int(days), 365))
    with get_db() as db:
        rows = db.execute(
            "SELECT total_value, cash_balance, snapshot_ts "
            "FROM portfolio_snapshots WHERE bot_id = %s "
            f"AND snapshot_ts >= CURRENT_TIMESTAMP - INTERVAL '{days} days' "
            "ORDER BY snapshot_ts ASC",
            [bid],
        ).fetchall()
    return [
        {
            "total_value": _safe_float(r[0], fallback=0.0),
            "cash": _safe_float(r[1], fallback=0.0),
            "timestamp": utc_iso(r[2]),
        }
        for r in rows
    ]


def take_snapshot(bot_id: str = "") -> dict:
    """Take a point-in-time snapshot. Returns the state."""
    state = get_current_state(bot_id)
    bid = bot_id or _get_default_bot_id()
    now = datetime.now(timezone.utc)
    snap_id = str(uuid.uuid4())
    try:
        with get_db() as db:
            db.execute(
                "INSERT INTO portfolio_snapshots "
                "(id, bot_id, snapshot_ts, cash_balance, total_value) "
                "VALUES (%s, %s, %s, %s, %s)",
                [snap_id, bid, now, state["cash"], state["total_value"]],
            )
        logger.info(
            "snapshot: %s total_value=%.2f",
            bid,
            state["total_value"],
        )
    except Exception as e:
        logger.error("snapshot failed: %s", e)
    return state


def get_performance_summary(bot_id: str = "") -> dict:
    """Calculate performance metrics from bots table."""
    bid = bot_id or _get_default_bot_id()
    state = get_current_state(bid)
    try:
        from app.services.bot_manager import get_bot_starting_cash

        starting = get_bot_starting_cash(bid)
    except Exception:
        starting = settings.STARTING_CASH
    total_val = state["total_value"]
    pnl = total_val - starting
    pnl_pct = (pnl / starting) * 100 if starting else 0

    # From bots table
    with get_db() as db:
        bot = db.execute(
            "SELECT total_trades, total_pnl, win_rate FROM bots WHERE bot_id = %s",
            [bid],
        ).fetchone()

    total_trades = bot[0] if bot else 0
    realized_pnl = bot[1] if bot else 0.0
    win_rate = bot[2] if bot else 0.0

    return {
        "bot_id": bid,
        "starting_cash": starting,
        "current_value": total_val,
        "cash": state["cash"],
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl_pct, 2),
        "realized_pnl": round(realized_pnl, 2),
        "win_rate": round(win_rate, 2),
        "total_trades": total_trades,
        "open_positions": state["position_count"],
    }


# ── BROKER LEDGER QUERIES ──


def get_position_lots(bot_id: str = "", ticker: str | None = None) -> list[dict]:
    """Get open/partial lots, optionally filtered by ticker."""
    bid = bot_id or _get_default_bot_id()
    with get_db() as db:
        if ticker:
            rows = db.execute(
                "SELECT lot_id, ticker, opened_at, original_qty, remaining_qty, "
                "entry_price, status, cycle_id, is_legacy "
                "FROM position_lots "
                "WHERE bot_id = %s AND ticker = %s AND status IN ('open', 'partial') "
                "ORDER BY opened_at ASC",
                [bid, ticker.upper()],
            ).fetchall()
        else:
            rows = db.execute(
                "SELECT lot_id, ticker, opened_at, original_qty, remaining_qty, "
                "entry_price, status, cycle_id, is_legacy "
                "FROM position_lots "
                "WHERE bot_id = %s AND status IN ('open', 'partial') "
                "ORDER BY ticker, opened_at ASC",
                [bid],
            ).fetchall()
    return [
        {
            "lot_id": r[0],
            "ticker": r[1],
            "opened_at": utc_iso(r[2]),
            "original_qty": _safe_float(r[3], fallback=0.0),
            "remaining_qty": _safe_float(r[4], fallback=0.0),
            "entry_price": _safe_float(r[5], fallback=0.0),
            "status": r[6],
            "cycle_id": r[7],
            "is_legacy": bool(r[8]),
        }
        for r in rows
    ]


def get_lot_closures(
    bot_id: str = "", ticker: str | None = None, limit: int = 50
) -> list[dict]:
    """Get recent lot closures (realized trades with per-lot P&L)."""
    bid = bot_id or _get_default_bot_id()
    limit = max(1, min(int(limit), 500))
    params: list = [bid]
    where_extra = ""
    if ticker:
        where_extra = " AND ticker = %s"
        params.append(ticker.upper())
    params.append(limit)
    with get_db() as db:
        rows = db.execute(
            "SELECT closure_id, ticker, closed_qty, entry_price, exit_price, "
            "realized_pnl, closed_at, holding_days, lot_id "
            "FROM lot_closures "
            f"WHERE bot_id = %s{where_extra} "
            "ORDER BY closed_at DESC LIMIT %s",
            params,
        ).fetchall()
    return [
        {
            "closure_id": r[0],
            "ticker": r[1],
            "closed_qty": _safe_float(r[2], fallback=0.0),
            "entry_price": _safe_float(r[3], fallback=0.0),
            "exit_price": _safe_float(r[4], fallback=0.0),
            "realized_pnl": _safe_float(r[5], fallback=0.0),
            "closed_at": utc_iso(r[6]),
            "holding_days": r[7],
            "lot_id": r[8],
        }
        for r in rows
    ]


def get_trade_fills(bot_id: str = "", limit: int = 50) -> list[dict]:
    """Get recent trade fills from the broker ledger."""
    bid = bot_id or _get_default_bot_id()
    limit = max(1, min(int(limit), 500))
    with get_db() as db:
        rows = db.execute(
            "SELECT fill_id, order_id, ticker, side, fill_qty, fill_price, "
            "fill_value, fees, filled_at, cycle_id "
            "FROM trade_fills WHERE bot_id = %s "
            "ORDER BY filled_at DESC LIMIT %s",
            [bid, limit],
        ).fetchall()
    return [
        {
            "fill_id": r[0],
            "order_id": r[1],
            "ticker": r[2],
            "side": r[3],
            "fill_qty": _safe_float(r[4], fallback=0.0),
            "fill_price": _safe_float(r[5], fallback=0.0),
            "fill_value": _safe_float(r[6], fallback=0.0),
            "fees": _safe_float(r[7], fallback=0.0),
            "filled_at": utc_iso(r[8]),
            "cycle_id": r[9],
        }
        for r in rows
    ]


def get_lot_count_by_ticker(bot_id: str = "") -> dict[str, int]:
    """Return {ticker: open_lot_count} for all open positions."""
    bid = bot_id or _get_default_bot_id()
    with get_db() as db:
        rows = db.execute(
            "SELECT ticker, COUNT(*) FROM position_lots "
            "WHERE bot_id = %s AND status IN ('open', 'partial') "
            "GROUP BY ticker",
            [bid],
        ).fetchall()
    return {r[0]: r[1] for r in rows}
