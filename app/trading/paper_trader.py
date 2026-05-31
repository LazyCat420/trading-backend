"""
Paper Trader — simulated trade execution.

This is the ONLY supported execution mode for the application by design.
It reads/writes positions, orders, and cash balance to PostgreSQL.
No real money or live broker APIs are intended to be integrated here.
Pure paper trading is an architectural invariant to ensure safe simulations.

Features:
- Buy/sell with position tracking
- P&L calculation
- Per-position ATR-based stop-loss enforcement
- Price staleness checks
"""

import datetime
import uuid
import logging
from app.db.connection import get_db
from app.config import settings
from app.config.config_tickers import classify_asset as _classify_asset
from app.services.alert_service import record_fund_alert

logger = logging.getLogger(__name__)

# Stop-loss bounds by asset class
_STOP_BOUNDS = {
    "crypto": (0.12, 0.25, 0.18),  # min, max, default
    "commodity": (0.06, 0.15, 0.10),
    "stock": (0.04, 0.12, 0.08),
}


def _compute_stop_loss_pct(ticker: str, entry_price: float) -> float:
    """Compute a volatility-adjusted stop-loss % using ATR-14.

    Logic:
        stop_pct = (ATR_14 * 2) / entry_price
        Clamped to asset-class bounds so crypto gets wider stops
        and blue-chips get tighter stops.

    Falls back to class default if no ATR data.
    """
    asset_class = _classify_asset(ticker)
    min_stop, max_stop, default_stop = _STOP_BOUNDS[asset_class]

    try:
        with get_db() as db:
            row = db.execute(
                """
                SELECT atr_14 FROM technicals
                WHERE ticker = %s AND atr_14 IS NOT NULL
                ORDER BY date DESC LIMIT 1
            """,
                [ticker],
            ).fetchone()

        if row and row[0] and entry_price > 0:
            atr = row[0]
            raw_pct = (atr * 2.0) / entry_price  # 2x ATR = standard stop
            stop_pct = max(min_stop, min(max_stop, raw_pct))
            logger.debug(
                "[stop] %s: ATR=%.2f, raw=%.3f, clamped=%.3f (%s)",
                ticker,
                atr,
                raw_pct,
                stop_pct,
                asset_class,
            )
            return round(stop_pct, 4)
    except Exception as e:
        logger.warning("[stop] %s: ATR lookup failed (%s), using default", ticker, e)

    logger.debug(
        "[stop] %s: no ATR data, using %s default=%s", ticker, asset_class, default_stop
    )
    return default_stop


# Fix #3: Maximum age for price data before we refuse to trade
MAX_PRICE_AGE_HOURS = 96


def _ensure_bot(bot_id: str):
    """Create bot row if it doesn't exist."""
    with get_db() as db:
        existing = db.execute(
            "SELECT bot_id FROM bots WHERE bot_id = %s", [bot_id]
        ).fetchone()
        if not existing:
            db.execute(
                """
                INSERT INTO bots (bot_id, display_name, cash_balance, total_pnl, created_at)
                VALUES (%s, %s, %s, 0.0, %s)
            """,
                [
                    bot_id,
                    bot_id,
                    settings.STARTING_CASH,
                    datetime.datetime.now(datetime.UTC),
                ],
            )
            logger.info(
                "[paper] Created bot '%s' with $%s",
                bot_id,
                f"{settings.STARTING_CASH:,.0f}",
            )


def _get_current_price(ticker: str) -> tuple[float | None, float | None]:
    """Get latest price and its age in hours.

    Returns (price, age_hours) or (None, None) if no price data.
    Checks price_history first, then asset_prices (crypto/commodity).
    """
    with get_db() as db:
        # Try price_history first (stocks)
        price_row = db.execute(
            "SELECT close, date FROM price_history WHERE ticker = %s ORDER BY date DESC LIMIT 1",
            [ticker],
        ).fetchone()

        # Fallback: asset_prices (crypto/commodity)
        if not price_row:
            price_row = db.execute(
                "SELECT close, date FROM asset_prices WHERE symbol = %s ORDER BY date DESC LIMIT 1",
                [ticker],
            ).fetchone()

    if not price_row:
        return None, None

    price = price_row[0]
    price_date = price_row[1]

    # Calculate age — handle date, datetime, and string types from DB
    if price_date:
        if isinstance(price_date, str):
            try:
                price_date = datetime.datetime.fromisoformat(
                    price_date.replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                price_date = None
        elif isinstance(price_date, datetime.date) and not isinstance(
            price_date, datetime.datetime
        ):
            # DB returned date (no time) — convert to datetime at midnight
            price_date = datetime.datetime.combine(
                price_date, datetime.time(), tzinfo=datetime.UTC
            )
        if price_date:
            # Make aware if naive
            if hasattr(price_date, "tzinfo") and price_date.tzinfo is None:
                price_date = price_date.replace(tzinfo=datetime.UTC)
            age = datetime.datetime.now(datetime.UTC) - price_date
            age_hours = age.total_seconds() / 3600
            return price, age_hours

    return price, None


def get_portfolio(bot_id: str) -> dict:
    """Get current portfolio state: cash + positions."""
    _ensure_bot(bot_id)

    with get_db() as db:
        bot = db.execute(
            "SELECT cash_balance, total_pnl FROM bots WHERE bot_id = %s", [bot_id]
        ).fetchone()

        positions = db.execute(
            """
            SELECT ticker, qty, avg_entry_price, stop_loss_pct, opened_at
            FROM positions
            WHERE bot_id = %s
        """,
            [bot_id],
        ).fetchall()

    pos_list = []
    for p in positions:
        pos_list.append(
            {
                "ticker": p[0],
                "qty": p[1],
                "avg_entry_price": p[2],
                "stop_loss_pct": p[3],
                "opened_at": str(p[4]),
            }
        )

    return {
        "bot_id": bot_id,
        "cash": bot[0],
        "total_pnl": bot[1],
        "positions": pos_list,
        "position_count": len(pos_list),
    }


def get_portfolio_value(bot_id: str) -> float:
    """Compute the total portfolio value (cash + mark-to-market positions)."""
    portfolio = get_portfolio(bot_id)
    cash = portfolio.get("cash", 0.0)
    total_position_value = 0.0
    for p in portfolio.get("positions", []):
        price, _ = _get_current_price(p["ticker"])
        if price is None:
            price = p["avg_entry_price"]
        total_position_value += p["qty"] * price
    return cash + total_position_value


async def buy(
    bot_id: str,
    ticker: str,
    size_pct: float,
    current_price: float | None = None,
    cycle_id: str | None = None,
) -> dict:
    """
    Execute a paper BUY.
    size_pct: fraction of cash to use (0.01 = 1%, 0.10 = 10%)
    Enforces a maximum portfolio concentration cap per ticker.
    """
    logger.info(
        "[TRACE][BUY] START bot_id=%s ticker=%s size_pct=%s", bot_id, ticker, size_pct
    )
    _ensure_bot(bot_id)

    if cycle_id:
        with get_db() as db:
            existing = db.execute(
                "SELECT fill_id FROM trade_fills WHERE cycle_id = %s AND ticker = %s AND side = 'BUY'",
                [cycle_id, ticker]
            ).fetchone()
            if existing:
                logger.warning("[TRACE][BUY] ABORT — Duplicate BUY order in same cycle for %s", ticker)
                return {"error": f"Duplicate BUY order in cycle {cycle_id} for {ticker}"}

    with get_db() as db:
        cash = db.execute(
            "SELECT cash_balance FROM bots WHERE bot_id = %s", [bot_id]
        ).fetchone()[0]
    logger.info("[TRACE][BUY] cash=%.2f for bot_id=%s", cash, bot_id)

    # Fix #3: Get price with staleness check
    price_age_hours = None
    if current_price is None:
        current_price, price_age_hours = _get_current_price(ticker)
        logger.info(
            "[TRACE][BUY] price=%.4f age_hours=%s for %s",
            current_price or 0,
            price_age_hours,
            ticker,
        )
        if current_price is None:
            logger.warning("[TRACE][BUY] ABORT — no price data for %s", ticker)
            return {"error": f"No price data for {ticker}"}
        if price_age_hours is not None and price_age_hours > MAX_PRICE_AGE_HOURS:
            logger.warning(
                "[TRACE][BUY] ABORT — stale price for %s (%.0fh old)",
                ticker,
                price_age_hours,
            )
            return {
                "error": f"Price data for {ticker} is {price_age_hours:.0f}h old "
                f"(max {MAX_PRICE_AGE_HOURS}h). Refusing stale trade.",
                "price_age_hours": round(price_age_hours, 1),
            }

    # Calculate position size
    if current_price <= 0:
        return {"error": f"Invalid price for {ticker}: ${current_price}"}

    # ── Price sanity gate: reject obviously corrupt prices ──
    # Stablecoins should be ~$1.00 — reject if way off
    _STABLECOINS = {"USDC", "USDT", "DAI", "BUSD", "TUSD", "USDP", "GUSD", "FRAX"}
    if ticker.upper() in _STABLECOINS:
        if current_price < 0.90 or current_price > 1.10:
            logger.warning(
                "[paper] PRICE SANITY: %s at $%.6f is outside stablecoin range ($0.90-$1.10)",
                ticker,
                current_price,
            )
            return {
                "error": f"Price sanity failed: {ticker} at ${current_price:.6f} "
                f"(expected ~$1.00 for stablecoin)"
            }

    # General sanity: compare to recent historical price if available
    try:
        with get_db() as _sanity_db:
            hist_row = _sanity_db.execute(
                "SELECT AVG(close) FROM price_history "
                "WHERE ticker = %s AND date > CURRENT_DATE - INTERVAL '30 days'",
                [ticker],
            ).fetchone()
        if hist_row and hist_row[0] and hist_row[0] > 0:
            avg_30d = hist_row[0]
            ratio = current_price / avg_30d
            # Reject if price is <5% or >500% of 30-day average (likely bad data)
            if ratio < 0.05 or ratio > 5.0:
                logger.warning(
                    "[paper] PRICE SANITY: %s at $%.4f is %.1fx the 30d avg ($%.4f)",
                    ticker,
                    current_price,
                    ratio,
                    avg_30d,
                )
                return {
                    "error": f"Price sanity failed: {ticker} at ${current_price:.4f} "
                    f"is {ratio:.1f}x the 30-day avg (${avg_30d:.4f})"
                }
    except Exception as e:
        logger.warning("[paper] Historical price sanity check skipped for %s: %s", ticker, e)

    # Calculate portfolio total value to enforce concentration cap
    with get_db() as db:
        positions = db.execute(
            "SELECT ticker, qty FROM positions WHERE bot_id = %s", [bot_id]
        ).fetchall()
    portfolio_value = cash
    existing_ticker_value = 0.0

    for pticker, pqty in positions:
        pprice, _ = _get_current_price(pticker)
        val = pqty * (pprice or 0.0)
        portfolio_value += val
        if pticker == ticker:
            existing_ticker_value = val

    amount = cash * min(size_pct, 1.0)

    # Enforce concentration cap (e.g., max 25% of portfolio per ticker)
    max_concentration_pct = getattr(settings, "MAX_CONCENTRATION_PCT", 0.25)
    max_allowed_value = portfolio_value * max_concentration_pct

    logger.info(
        "[TRACE][BUY] amount=%.2f portfolio_value=%.2f existing_val=%.2f max_allowed=%.2f for %s",
        amount,
        portfolio_value,
        existing_ticker_value,
        max_allowed_value,
        ticker,
    )

    if existing_ticker_value + amount > max_allowed_value:
        amount = max(0.0, max_allowed_value - existing_ticker_value)
        if amount < 1.0:
            logger.warning(
                "[TRACE][BUY] ABORT — concentration cap (%.0f%%) reached for %s",
                max_concentration_pct * 100,
                ticker,
            )
            return {
                "error": f"Concentration cap ({max_concentration_pct * 100}%) reached for {ticker}"
            }
        logger.info(
            "[paper] Scaling down BUY for %s to enforce %.0f%% concentration cap",
            ticker,
            max_concentration_pct * 100,
        )

    qty = amount / current_price
    logger.info(
        "[TRACE][BUY] final qty=%.4f amount=%.2f price=%.4f for %s",
        qty,
        amount,
        current_price,
        ticker,
    )

    if amount < 1.0:
        logger.warning(
            "[TRACE][BUY] ABORT — insufficient cash ($%.2f) for %s", amount, ticker
        )
        return {"error": "Insufficient cash for this position"}

    now = datetime.datetime.now(datetime.UTC)
    pos_id = str(uuid.uuid4())
    order_id = str(uuid.uuid4())

    # Fix #5: Wrap in transaction
    with get_db() as db:
        try:
            with db.transaction():
                # Check for existing position
                existing = db.execute(
                    """
                    SELECT id, qty, avg_entry_price FROM positions
                    WHERE bot_id = %s AND ticker = %s
                """,
                    [bot_id, ticker],
                ).fetchone()

                # Compute volatility-adjusted stop-loss
                stop_pct = _compute_stop_loss_pct(ticker, current_price)

                if existing:
                    old_id, old_qty, old_price = existing
                    new_qty = old_qty + qty
                    new_avg = ((old_qty * old_price) + (qty * current_price)) / new_qty
                    # Recompute stop for new avg price
                    stop_pct = _compute_stop_loss_pct(ticker, new_avg)
                    db.execute(
                        """
                        UPDATE positions SET qty = %s, avg_entry_price = %s, stop_loss_pct = %s
                        WHERE id = %s
                    """,
                        [new_qty, new_avg, stop_pct, old_id],
                    )
                else:
                    db.execute(
                        """
                        INSERT INTO positions (id, bot_id, ticker, qty, avg_entry_price, stop_loss_pct, opened_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                        [pos_id, bot_id, ticker, qty, current_price, stop_pct, now],
                    )

                # Deduct cash and log trade count
                db.execute(
                    "UPDATE bots SET cash_balance = cash_balance - %s, total_trades = total_trades + 1 WHERE bot_id = %s",
                    [amount, bot_id],
                )

                # Log order
                db.execute(
                    """
                    INSERT INTO orders (id, bot_id, ticker, side, qty, price, signal, created_at, filled_at)
                    VALUES (%s, %s, %s, 'BUY', %s, %s, 'pipeline', %s, %s)
                """,
                    [order_id, bot_id, ticker, qty, current_price, now, now],
                )

                # ── BROKER LEDGER: create fill + lot ──
                fill_id = str(uuid.uuid4())
                lot_id = str(uuid.uuid4())
                db.execute(
                    """
                    INSERT INTO trade_fills
                      (fill_id, order_id, bot_id, ticker, side, fill_qty, fill_price, fill_value, filled_at, cycle_id)
                    VALUES (%s, %s, %s, %s, 'BUY', %s, %s, %s, %s, %s)
                """,
                    [
                        fill_id,
                        order_id,
                        bot_id,
                        ticker,
                        qty,
                        current_price,
                        amount,
                        now,
                        cycle_id,
                    ],
                )
                db.execute(
                    """
                    INSERT INTO position_lots
                      (lot_id, bot_id, ticker, fill_id, opened_at, original_qty, remaining_qty, entry_price, status, cycle_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'open', %s)
                """,
                    [
                        lot_id,
                        bot_id,
                        ticker,
                        fill_id,
                        now,
                        qty,
                        qty,
                        current_price,
                        cycle_id,
                    ],
                )
                logger.info(
                    "[TRACE][BUY] COMMITTED OK — order_id=%s fill_id=%s lot_id=%s",
                    order_id,
                    fill_id,
                    lot_id,
                )
        except Exception as e:
            logger.error(
                "[TRACE][BUY] ROLLBACK triggered for %s: %s",
                ticker,
                repr(e),
                exc_info=True,
            )
            return {"error": f"Transaction failed: {e}"}

    result = {
        "action": "BUY",
        "ticker": ticker,
        "qty": round(qty, 4),
        "price": current_price,
        "amount": round(amount, 2),
    }
    if price_age_hours is not None:
        result["price_age_hours"] = round(price_age_hours, 1)
    logger.info(
        "[paper] BUY %s: %.4f shares @ $%.2f = $%.2f (lot=%s)",
        ticker,
        qty,
        current_price,
        amount,
        lot_id[:8],
    )
    return result


async def sell(
    bot_id: str,
    ticker: str,
    current_price: float | None = None,
    cycle_id: str | None = None,
    qty_pct: float = 1.0,
) -> dict:
    """
    Execute a paper SELL.
    qty_pct: fraction of position to sell (default 1.0 = full close).
    """
    logger.info(
        "[TRACE][SELL] START bot_id=%s ticker=%s qty_pct=%s", bot_id, ticker, qty_pct
    )
    _ensure_bot(bot_id)

    if cycle_id:
        with get_db() as db:
            existing = db.execute(
                "SELECT fill_id FROM trade_fills WHERE cycle_id = %s AND ticker = %s AND side = 'SELL'",
                [cycle_id, ticker]
            ).fetchone()
            if existing:
                logger.warning("[TRACE][SELL] ABORT — Duplicate SELL order in same cycle for %s", ticker)
                return {"error": f"Duplicate SELL order in cycle {cycle_id} for {ticker}"}

    with get_db() as db:
        pos = db.execute(
            """
            SELECT id, qty, avg_entry_price FROM positions
            WHERE bot_id = %s AND ticker = %s
        """,
            [bot_id, ticker],
        ).fetchone()

    if not pos:
        logger.warning(
            "[TRACE][SELL] ABORT — no open position for %s (bot_id=%s)", ticker, bot_id
        )
        return {"error": f"No open position for {ticker}"}

    pos_id, total_qty, avg_entry_price = pos
    logger.info(
        "[TRACE][SELL] pos found: id=%s total_qty=%.4f avg_entry=%.2f",
        pos_id,
        total_qty,
        avg_entry_price,
    )

    qty_to_sell = total_qty * min(max(qty_pct, 0.0), 1.0)
    if qty_to_sell <= 0:
        return {"error": "Invalid sell quantity"}

    # Fix #3: Get price with staleness check
    price_age_hours = None
    if current_price is None:
        current_price, price_age_hours = _get_current_price(ticker)
        if current_price is None:
            return {"error": f"No price data for {ticker}"}
        if price_age_hours is not None and price_age_hours > MAX_PRICE_AGE_HOURS:
            return {
                "error": f"Price data for {ticker} is {price_age_hours:.0f}h old "
                f"(max {MAX_PRICE_AGE_HOURS}h). Refusing stale trade.",
                "price_age_hours": round(price_age_hours, 1),
            }

    logger.info(
        "[TRACE][SELL] qty_to_sell=%.4f price=%.4f for %s",
        qty_to_sell,
        current_price,
        ticker,
    )
    # Calculate proceeds
    proceeds = current_price * qty_to_sell
    now = datetime.datetime.now(datetime.UTC)
    order_id = str(uuid.uuid4())
    sell_fill_id = str(uuid.uuid4())

    # Fix #5: Wrap in transaction
    with get_db() as db:
        try:
            with db.transaction():
                # 1. FIFO lot matching: consume oldest lots first to calculate true P&L
                open_lots = db.execute(
                    """
                    SELECT lot_id, remaining_qty, entry_price, opened_at
                    FROM position_lots
                    WHERE bot_id = %s AND ticker = %s AND status IN ('open', 'partial')
                    ORDER BY opened_at ASC
                """,
                    [bot_id, ticker],
                ).fetchall()

                remaining_to_sell = qty_to_sell
                total_realized_pnl = 0.0
                remaining_lots_value = 0.0
                remaining_lots_qty = 0.0

                for lot_row in open_lots:
                    lot_id, lot_remaining, lot_entry, lot_opened = lot_row

                    if remaining_to_sell > 0:
                        closed_qty = min(remaining_to_sell, lot_remaining)
                        lot_pnl = (current_price - lot_entry) * closed_qty
                        total_realized_pnl += lot_pnl

                        new_remaining = lot_remaining - closed_qty
                        new_status = "closed" if new_remaining <= 0.0001 else "partial"

                        if new_remaining > 0.0001:
                            remaining_lots_qty += new_remaining
                            remaining_lots_value += new_remaining * lot_entry

                        # Calculate holding duration
                        holding_days = None
                        if lot_opened:
                            try:
                                if isinstance(lot_opened, str):
                                    lot_opened_dt = datetime.datetime.fromisoformat(
                                        lot_opened.replace("Z", "+00:00")
                                    )
                                elif isinstance(lot_opened, datetime.datetime):
                                    lot_opened_dt = lot_opened
                                else:
                                    lot_opened_dt = datetime.datetime.combine(
                                        lot_opened, datetime.time(), tzinfo=datetime.UTC
                                    )
                                if lot_opened_dt.tzinfo is None:
                                    lot_opened_dt = lot_opened_dt.replace(
                                        tzinfo=datetime.UTC
                                    )
                                holding_days = (now - lot_opened_dt).days
                            except Exception as parse_err:
                                logger.warning("[paper] Failed to calculate holding_days for %s: %s", ticker, parse_err)

                        # Create closure record
                        closure_id = str(uuid.uuid4())
                        db.execute(
                            """
                            INSERT INTO lot_closures
                              (closure_id, bot_id, ticker, sell_fill_id, lot_id,
                               closed_qty, entry_price, exit_price, realized_pnl, closed_at, holding_days)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                            [
                                closure_id,
                                bot_id,
                                ticker,
                                sell_fill_id,
                                lot_id,
                                closed_qty,
                                lot_entry,
                                current_price,
                                lot_pnl,
                                now,
                                holding_days,
                            ],
                        )

                        # Update lot status
                        db.execute(
                            """
                            UPDATE position_lots SET remaining_qty = %s, status = %s WHERE lot_id = %s
                        """,
                            [max(new_remaining, 0), new_status, lot_id],
                        )

                        remaining_to_sell -= closed_qty
                    else:
                        # Lot is not sold, track its value for the new average cost basis
                        remaining_lots_qty += lot_remaining
                        remaining_lots_value += lot_remaining * lot_entry

                if remaining_to_sell > 0.0001:
                    logger.warning(
                        "[paper] SELL %s: %.4f shares unmatched to lots (legacy position)",
                        ticker,
                        remaining_to_sell,
                    )
                    total_realized_pnl += (
                        current_price - avg_entry_price
                    ) * remaining_to_sell

                # 2. Update positions table and recalculate cost basis if partial
                if qty_to_sell >= total_qty - 0.0001:
                    db.execute("DELETE FROM positions WHERE id = %s", [pos_id])
                else:
                    new_pos_qty = total_qty - qty_to_sell
                    new_avg_price = (
                        (remaining_lots_value / remaining_lots_qty)
                        if remaining_lots_qty > 0.0001
                        else avg_entry_price
                    )
                    db.execute(
                        "UPDATE positions SET qty = %s, avg_entry_price = %s WHERE id = %s",
                        [new_pos_qty, new_avg_price, pos_id],
                    )

                # 3. Add cash back, update P&L, and log trade count
                db.execute(
                    "UPDATE bots SET cash_balance = cash_balance + %s, total_pnl = total_pnl + %s, "
                    "total_trades = total_trades + 1 WHERE bot_id = %s",
                    [proceeds, total_realized_pnl, bot_id],
                )

                # 4. Log order
                db.execute(
                    """
                    INSERT INTO orders (id, bot_id, ticker, side, qty, price, signal,
                                       created_at, filled_at, realized_pnl)
                    VALUES (%s, %s, %s, 'SELL', %s, %s, 'pipeline', %s, %s, %s)
                """,
                    [
                        order_id,
                        bot_id,
                        ticker,
                        qty_to_sell,
                        current_price,
                        now,
                        now,
                        total_realized_pnl,
                    ],
                )

                # 5. BROKER LEDGER: create sell fill
                db.execute(
                    """
                    INSERT INTO trade_fills
                      (fill_id, order_id, bot_id, ticker, side, fill_qty, fill_price, fill_value, filled_at, cycle_id)
                    VALUES (%s, %s, %s, %s, 'SELL', %s, %s, %s, %s, %s)
                """,
                    [
                        sell_fill_id,
                        order_id,
                        bot_id,
                        ticker,
                        qty_to_sell,
                        current_price,
                        proceeds,
                        now,
                        cycle_id,
                    ],
                )

                # Update bot win_rate dynamically
                db.execute(
                    """
                    UPDATE bots SET win_rate = (
                        SELECT coalesce(CAST(sum(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) AS DOUBLE PRECISION) / NULLIF(count(*), 0) * 100.0, 0.0)
                        FROM lot_closures WHERE bot_id = %s
                    ) WHERE bot_id = %s
                """,
                    [bot_id, bot_id],
                )

                logger.info(
                    "[TRACE][SELL] COMMITTED OK — order_id=%s sell_fill_id=%s",
                    order_id,
                    sell_fill_id,
                )
        except Exception as e:
            logger.error(
                "[TRACE][SELL] ROLLBACK triggered for %s: %s",
                ticker,
                repr(e),
                exc_info=True,
            )
            return {"error": f"Transaction failed: {e}"}

    cost_basis = avg_entry_price * qty_to_sell
    pnl_pct = (total_realized_pnl / cost_basis) * 100 if cost_basis != 0 else 0.0
    result = {
        "action": "SELL",
        "ticker": ticker,
        "qty": round(qty_to_sell, 4),
        "price": current_price,
        "proceeds": round(proceeds, 2),
        "realized_pnl": round(total_realized_pnl, 2),
        "pnl_pct": round(pnl_pct, 2),
    }
    if price_age_hours is not None:
        result["price_age_hours"] = round(price_age_hours, 1)
    logger.info(
        "[paper] SELL %s: %.4f shares @ $%.2f = $%.2f (P&L: $%+.2f / %+.1f%%)",
        ticker,
        qty_to_sell,
        current_price,
        proceeds,
        total_realized_pnl,
        pnl_pct,
    )
    return result


# Fix #13: Stop-loss enforcement — now per-position with ATR-based levels
async def check_stop_losses(
    bot_id: str, default_stop_pct: float = 0.08, cycle_id: str | None = None
) -> list[dict]:
    """Check all open positions against their per-position stop-loss levels.

    Each position stores its own stop_loss_pct (set at buy time from ATR).
    Falls back to default_stop_pct only if the stored value is NULL.

    Args:
        bot_id: The bot to check
        default_stop_pct: Fallback stop-loss for positions without a stored value
        cycle_id: Identifier to tag generated fills/orders with the specific cycle
    """
    _ensure_bot(bot_id)

    with get_db() as db:
        positions = db.execute(
            """
            SELECT id, ticker, qty, avg_entry_price, stop_loss_pct FROM positions
            WHERE bot_id = %s
        """,
            [bot_id],
        ).fetchall()

    triggered = []
    for pos in positions:
        pos_id, ticker, qty, entry_price, stop_pct = pos
        # Use stored stop or fall back to default
        effective_stop = stop_pct if stop_pct is not None else default_stop_pct

        current_price, age_hours = _get_current_price(ticker)

        if current_price is None:
            logger.warning("[stop-loss] %s: no price data, skipping", ticker)
            continue

        # Check if price has dropped below stop-loss level
        stop_price = entry_price * (1 - effective_stop)
        if current_price <= stop_price:
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            logger.warning(
                "[stop-loss] %s: TRIGGERED @ $%.2f (entry=$%.2f, stop=%.1f%% =$%.2f, loss=%.1f%%)",
                ticker,
                current_price,
                entry_price,
                effective_stop * 100,
                stop_price,
                pnl_pct,
            )

            # Fix: restored missing sell call
            result = await sell(
                bot_id, ticker, current_price=current_price, cycle_id=cycle_id
            )
            if "error" not in result:
                triggered.append(result)
                # Resolve outcome so the feedback loop captures stop-loss exits
                try:
                    from app.pipeline.analysis.outcome_tracker import resolve_outcome

                    resolve_outcome(
                        ticker, current_price,
                        realized_pnl=result.get("realized_pnl"),
                    )
                except Exception as outcome_err:
                    logger.error("[stop-loss] Failed to resolve outcome for %s: %s", ticker, outcome_err)
                
                try:
                    record_fund_alert(
                        alert_type="stop_loss",
                        entity_name=bot_id,
                        detail=f"Stop loss triggered for {ticker} at ${current_price:.2f} (entry=${entry_price:.2f}, loss={pnl_pct:.1f}%)",
                        severity="high",
                        ticker=ticker
                    )
                except Exception as e:
                    logger.error("[stop-loss] Alert error: %s", e)
            else:
                logger.error(
                    "[stop-loss] Sell failed for %s: %s", ticker, result.get("error")
                )

    if triggered:
        logger.info(
            "[stop-loss] Triggered %d stop-losses for bot '%s'", len(triggered), bot_id
        )
    return triggered


# Fix #13: Take-Profit Harvesting
async def check_take_profits(
    bot_id: str,
    reward_risk_ratio: float = 2.0,
    default_tp_pct: float = 0.20,
    cycle_id: str | None = None,
) -> list[dict]:
    """Check all open positions against a take-profit target (harvesting).

    Uses a dynamic target based on the Risk/Reward ratio and the position's
    stop-loss. If stop-loss is 8%, take-profit is 16% (at 2.0 R:R).
    """
    _ensure_bot(bot_id)

    with get_db() as db:
        positions = db.execute(
            """
            SELECT id, ticker, qty, avg_entry_price, stop_loss_pct FROM positions
            WHERE bot_id = %s
        """,
            [bot_id],
        ).fetchall()

    triggered = []
    for pos in positions:
        pos_id, ticker, qty, entry_price, stop_pct = pos
        current_price, age_hours = _get_current_price(ticker)

        if current_price is None:
            continue

        effective_stop = (
            stop_pct if stop_pct is not None else (default_tp_pct / reward_risk_ratio)
        )
        effective_tp = effective_stop * reward_risk_ratio

        target_price = entry_price * (1 + effective_tp)
        if current_price >= target_price:
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            logger.info(
                "[take-profit] %s: HARVEST TRIGGERED @ $%.2f (entry=$%.2f, target=%.1f%% =$%.2f, gain=%.1f%%)",
                ticker,
                current_price,
                entry_price,
                effective_tp * 100,
                target_price,
                pnl_pct,
            )

            # Harvest the full position
            result = await sell(
                bot_id, ticker, current_price=current_price, cycle_id=cycle_id
            )
            if "error" not in result:
                triggered.append(result)
                # Resolve outcome so the feedback loop captures take-profit exits
                try:
                    from app.pipeline.analysis.outcome_tracker import resolve_outcome

                    resolve_outcome(
                        ticker, current_price,
                        realized_pnl=result.get("realized_pnl"),
                    )
                except Exception as outcome_err:
                    logger.error("[take-profit] Failed to resolve outcome for %s: %s", ticker, outcome_err)

    if triggered:
        logger.info(
            "[take-profit] Harvested %d positions for bot '%s'", len(triggered), bot_id
        )
    return triggered
