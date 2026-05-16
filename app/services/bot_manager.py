"""Bot Profile Manager — centralized service for multi-profile trading.

All backend code should use these functions to resolve the active bot,
instead of reading settings.BOT_ID directly. This enables profile switching
without restarting the server.

Public API:
    get_active_bot_id()         → str
    get_bot_starting_cash(bid)  → float
    set_active_bot(bot_id)      → None
    is_cycle_running()          → bool
    create_bot_profile(...)     → dict
    delete_bot_profile(bot_id)  → dict
    reset_bot_profile(bot_id)   → dict
    list_bot_profiles()         → list[dict]
"""

import logging
import re
import uuid
from datetime import datetime, timezone

from app.config import settings
from app.db.connection import get_db

logger = logging.getLogger(__name__)

# ── In-memory cache of the active bot_id ──
# Initialized lazily from DB on first call, updated on set_active_bot().
_active_bot_id: str | None = None


def _slugify(name: str) -> str:
    """Convert display name to a URL-safe bot_id slug."""
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)  # remove special chars
    slug = re.sub(r"[\s]+", "-", slug)  # spaces → hyphens
    slug = re.sub(r"-+", "-", slug).strip("-")  # collapse hyphens
    return slug or f"bot-{uuid.uuid4().hex[:8]}"


def get_active_bot_id() -> str:
    """Return the currently active bot_id.

    Resolution order:
    1. In-memory cache (fastest)
    2. DB lookup (is_active = TRUE)
    3. Fallback to settings.BOT_ID
    """
    global _active_bot_id
    if _active_bot_id is not None:
        logger.debug(
            "[TRACE][BOT_MANAGER] get_active_bot_id() cache hit: %s", _active_bot_id
        )
        return _active_bot_id

    try:
        with get_db() as db:
            rows = db.execute(
                "SELECT bot_id FROM bots WHERE is_active = TRUE ORDER BY created_at ASC"
            ).fetchall()
            if rows:
                _active_bot_id = rows[0][0]
                # Enforce single-active invariant: if multiple are active, fix it
                if len(rows) > 1:
                    logger.warning(
                        "[BOT_MANAGER] INVARIANT VIOLATION: %d bots marked active! "
                        "Keeping '%s', deactivating others: %s",
                        len(rows),
                        _active_bot_id,
                        [r[0] for r in rows[1:]],
                    )
                    db.execute(
                        "UPDATE bots SET is_active = FALSE WHERE is_active = TRUE AND bot_id != %s",
                        [_active_bot_id],
                    )
                logger.info(
                    "[TRACE][BOT_MANAGER] get_active_bot_id() from DB: %s",
                    _active_bot_id,
                )
                return _active_bot_id
    except Exception as e:
        logger.warning("Failed to query active bot: %s", e)

    _active_bot_id = settings.BOT_ID
    logger.info(
        "[TRACE][BOT_MANAGER] get_active_bot_id() fallback to settings.BOT_ID: %s",
        _active_bot_id,
    )
    return _active_bot_id


def get_bot_starting_cash(bot_id: str = "") -> float:
    """Return the starting cash for a bot profile.

    Falls back to settings.STARTING_CASH if the bot doesn't have
    a starting_cash column or the bot doesn't exist.
    """
    bid = bot_id or get_active_bot_id()
    try:
        with get_db() as db:
            row = db.execute(
                "SELECT starting_cash FROM bots WHERE bot_id = %s", [bid]
            ).fetchone()
            if row and row[0] is not None:
                return float(row[0])
    except Exception as e:
        logger.debug("starting_cash lookup failed for %s: %s", bid, e)
    return settings.STARTING_CASH


def get_bot_description(bot_id: str = "") -> str:
    """Return the custom description/instructions for a bot profile.

    This is used to inject custom trading personas into the RLM system prompt.
    """
    bid = bot_id or get_active_bot_id()
    try:
        with get_db() as db:
            row = db.execute(
                "SELECT description FROM bots WHERE bot_id = %s", [bid]
            ).fetchone()
            if row and row[0]:
                return str(row[0]).strip()
    except Exception as e:
        logger.debug("description lookup failed for %s: %s", bid, e)
    return ""


def set_active_bot(bot_id: str) -> None:
    """Switch the active bot profile.

    Sets is_active=FALSE on all bots, then TRUE on the target.
    Raises ValueError if bot_id doesn't exist or a cycle is running.
    """
    global _active_bot_id

    if is_cycle_running():
        raise ValueError(
            "Cannot switch profiles while a pipeline cycle is running. "
            "Stop the cycle first."
        )

    with get_db() as db:
        # Verify bot exists
        exists = db.execute("SELECT 1 FROM bots WHERE bot_id = %s", [bot_id]).fetchone()
        if not exists:
            raise ValueError(f"Bot profile '{bot_id}' does not exist")

        # Deactivate all, activate target
        db.execute("UPDATE bots SET is_active = FALSE WHERE is_active = TRUE")
        db.execute("UPDATE bots SET is_active = TRUE WHERE bot_id = %s", [bot_id])
    _active_bot_id = bot_id
    logger.info("[BOT_MANAGER] Active bot switched to: %s", bot_id)

    # Reset pipeline state — the old bot's cycle state is irrelevant to the new bot.
    # Any interrupted/stopped checkpoint belongs to the previous bot and should not
    # be offered for resume under a different profile.
    from app.pipeline.orchestration.state_manager import PipelineStateDB

    PipelineStateDB.save_state(PipelineStateDB.default_state())
    logger.info("[BOT_MANAGER] Pipeline state reset to idle for new profile %s", bot_id)


def is_cycle_running() -> bool:
    """Check if a pipeline cycle is currently running."""
    try:
        from app.services.pipeline_service import PipelineService

        status = PipelineService.get_status()
        return status.get("status") in (
            "collecting",
            "analyzing",
            "trading",
            "starting",
        )
    except Exception:
        return False


def list_bot_profiles() -> list[dict]:
    """Return all bot profiles with summary stats."""
    with get_db() as db:
        rows = db.execute(
            "SELECT bot_id, display_name, model_name, status, "
            "cash_balance, starting_cash, total_pnl, win_rate, "
            "total_trades, is_active, created_at, last_run_at, description "
            "FROM bots ORDER BY is_active DESC, created_at ASC"
        ).fetchall()
    return [
        {
            "bot_id": r[0],
            "display_name": r[1] or r[0],
            "model_name": r[2],
            "status": r[3] or "idle",
            "cash_balance": float(r[4]) if r[4] else 0.0,
            "starting_cash": float(r[5]) if r[5] else settings.STARTING_CASH,
            "total_pnl": float(r[6]) if r[6] else 0.0,
            "win_rate": float(r[7]) if r[7] else 0.0,
            "total_trades": r[8] or 0,
            "is_active": bool(r[9]),
            "created_at": r[10].isoformat() if r[10] else None,
            "last_run_at": r[11].isoformat() if r[11] else None,
            "description": r[12] or "",
        }
        for r in rows
    ]


import time


def create_bot_profile(
    display_name: str,
    starting_cash: float = 100_000.0,
    description: str = "",
) -> dict:
    """Create a new bot profile.

    Returns the created profile dict.
    Raises ValueError if slug collides with existing bot_id.
    """
    bot_id = _slugify(display_name)
    start_time = time.perf_counter()

    # Handle slug collisions
    db_start = time.perf_counter()
    with get_db() as db:
        db_acquired = time.perf_counter()
        existing = db.execute(
            "SELECT 1 FROM bots WHERE bot_id = %s", [bot_id]
        ).fetchone()

        if existing:
            # Append short UUID suffix
            bot_id = f"{bot_id}-{uuid.uuid4().hex[:6]}"

        now = datetime.now(timezone.utc)
        insert_start = time.perf_counter()
        db.execute(
            """
            INSERT INTO bots (
                bot_id, display_name, model_name, status,
                cash_balance, starting_cash, total_pnl, win_rate,
                total_trades, is_active, created_at, description
            ) VALUES (%s, %s, %s, 'idle', %s, %s, 0.0, 0.0, 0, FALSE, %s, %s)
            """,
            [
                bot_id,
                display_name,
                settings.ACTIVE_MODEL,
                starting_cash,
                starting_cash,
                now,
                description,
            ],
        )
        insert_end = time.perf_counter()

    total_end = time.perf_counter()

    logger.info(
        "[BOT_MANAGER] Created profile: %s (%s) with $%.2f "
        "[Timing: get_db=%.3fs, insert=%.3fs, total=%.3fs]",
        display_name,
        bot_id,
        starting_cash,
        db_acquired - db_start,
        insert_end - insert_start,
        total_end - start_time,
    )
    return {
        "bot_id": bot_id,
        "display_name": display_name,
        "starting_cash": starting_cash,
        "description": description,
        "created_at": now.isoformat(),
    }


def update_bot_profile(
    bot_id: str,
    display_name: str | None = None,
    description: str | None = None,
    starting_cash: float | None = None,
) -> dict:
    """Update a bot profile's metadata.

    starting_cash can only be updated if the bot has 0 trades.
    """
    with get_db() as db:
        row = db.execute(
            "SELECT display_name, description, starting_cash, total_trades "
            "FROM bots WHERE bot_id = %s",
            [bot_id],
        ).fetchone()
        if not row:
            raise ValueError(f"Bot profile '{bot_id}' does not exist")

        current_name, current_desc, current_cash, trades = row

        if starting_cash is not None and trades and trades > 0:
            raise ValueError(
                f"Cannot change starting cash for '{bot_id}' — "
                f"it already has {trades} trades. Reset the profile first."
            )

        new_name = display_name if display_name is not None else current_name
        new_desc = description if description is not None else current_desc
        new_cash = starting_cash if starting_cash is not None else current_cash

        db.execute(
            "UPDATE bots SET display_name = %s, description = %s, starting_cash = %s "
            "WHERE bot_id = %s",
            [new_name, new_desc, new_cash, bot_id],
        )

        # If starting_cash changed and no trades, also update cash_balance
        if starting_cash is not None:
            db.execute(
                "UPDATE bots SET cash_balance = %s WHERE bot_id = %s",
                [new_cash, bot_id],
            )

    logger.info("[BOT_MANAGER] Updated profile: %s", bot_id)
    return {"bot_id": bot_id, "display_name": new_name, "updated": True}


def reset_bot_profile(bot_id: str) -> dict:
    """Reset a bot profile to its starting cash.

    Wipes: positions, orders, trade_fills, position_lots,
           lot_closures, portfolio_snapshots, decision_outcomes,
           analysis_results for this bot_id.
    Resets: cash_balance, total_pnl, total_trades, win_rate.
    """
    if is_cycle_running():
        raise ValueError("Cannot reset while a pipeline cycle is running")

    with get_db() as db:
        row = db.execute(
            "SELECT starting_cash FROM bots WHERE bot_id = %s", [bot_id]
        ).fetchone()
        if not row:
            raise ValueError(f"Bot profile '{bot_id}' does not exist")

        starting_cash = float(row[0]) if row[0] else settings.STARTING_CASH

        # Wipe trading data for this bot
        tables_to_clear = [
            "positions",
            "orders",
            "trade_fills",
            "position_lots",
            "lot_closures",
            "portfolio_snapshots",
        ]
        cleared = {}
        for table in tables_to_clear:
            try:
                result = db.execute(f"DELETE FROM {table} WHERE bot_id = %s", [bot_id])
                # psycopg doesn't return rowcount easily via our wrapper,
                # but the delete still works
                cleared[table] = "cleared"
            except Exception as e:
                cleared[table] = f"error: {e}"

        # Reset bot stats
        db.execute(
            """
            UPDATE bots SET
                cash_balance = %s,
                total_pnl = 0.0,
                total_trades = 0,
                win_rate = 0.0,
                status = 'idle'
            WHERE bot_id = %s
            """,
            [starting_cash, bot_id],
        )

    logger.info(
        "[BOT_MANAGER] Reset profile '%s' to $%.2f",
        bot_id,
        starting_cash,
    )
    return {
        "bot_id": bot_id,
        "starting_cash": starting_cash,
        "cleared_tables": cleared,
        "reset": True,
    }


def delete_bot_profile(bot_id: str) -> dict:
    """Delete a bot profile and ALL its trading data.

    Cannot delete the currently active profile.
    """
    if is_cycle_running():
        raise ValueError("Cannot delete while a pipeline cycle is running")

    with get_db() as db:
        row = db.execute(
            "SELECT is_active FROM bots WHERE bot_id = %s", [bot_id]
        ).fetchone()
        if not row:
            raise ValueError(f"Bot profile '{bot_id}' does not exist")
        if row[0]:
            raise ValueError(
                f"Cannot delete the active profile '{bot_id}'. "
                "Switch to a different profile first."
            )

        # Wipe all data for this bot
        tables_to_clear = [
            "positions",
            "orders",
            "trade_fills",
            "position_lots",
            "lot_closures",
            "portfolio_snapshots",
        ]
        for table in tables_to_clear:
            try:
                db.execute(f"DELETE FROM {table} WHERE bot_id = %s", [bot_id])
            except Exception as e:
                logger.warning("delete %s for %s: %s", table, bot_id, e)

        # Delete the bot row itself
        db.execute("DELETE FROM bots WHERE bot_id = %s", [bot_id])
    logger.info("[BOT_MANAGER] Deleted profile: %s", bot_id)

    return {"bot_id": bot_id, "deleted": True}
