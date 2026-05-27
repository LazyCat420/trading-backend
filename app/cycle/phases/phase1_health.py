import logging
from datetime import datetime, timezone
from typing import Callable

from app.config import settings
from app.db.connection import get_db
from app.services.vllm_client import llm
from app.trading.paper_trader import check_stop_losses, check_take_profits
from app.cycle.attention_tracker import (
    flag_neglected_tickers,
    get_attention_summary,
    increment_days_since_deep,
)
from app.pipeline.ticker_triage import classify_tickers
from app.cycle.context import CycleContext
from app.utils.emit import noop_emit

logger = logging.getLogger(__name__)


async def run_phase1_health(
    ctx: CycleContext,
    bot_id: str,
    emit: Callable = noop_emit,
    cycle_summary: dict = None,
    state: dict = None,
) -> None:
    """
    Phase 1: Health Checks, Stop Losses, Triage, and Directives.
    """
    emit(
        "started",
        "start",
        f"Cycle started: ID {ctx.cycle_id} | Bot {bot_id} | "
        f"{len(ctx.tickers)} tickers | Mode: sequential",
        status="running",
    )

    # 1. Bot Health Check
    health_status = await llm.health_all()
    jetson_ok = health_status.get("jetson", False)
    dgx_ok = any(
        health_status.get(k, False) for k in health_status if k.startswith("dgx")
    )

    cycle_summary["jetson_healthy_start"] = jetson_ok
    cycle_summary["dgx_healthy_start"] = dgx_ok

    if not jetson_ok and not dgx_ok:
        emit(
            "started",
            "bots_offline",
            "⚠️ All LLM endpoints are UNREACHABLE. Aborting cycle.",
            status="error",
        )
        cycle_summary["no_trade_reason"] = "all_bots_down"
        raise RuntimeError("All LLM endpoints unreachable.")
    elif not dgx_ok:
        emit(
            "started",
            "dgx_offline",
            "⚠️ All DGX Sparks are down. Degraded mode.",
            status="warning",
        )
    elif not jetson_ok:
        emit(
            "started",
            "jetson_offline",
            "⚠️ Jetson is down. Degraded mode.",
            status="warning",
        )

    # 2. Stop-Loss Check
    if ctx.trade:
        try:
            triggered_stops = await check_stop_losses(bot_id)
            triggered_tps = await check_take_profits(bot_id)
            triggered = triggered_stops + triggered_tps
            if triggered:
                cycle_summary["stop_loss_triggered"] = len(triggered)
                cycle_summary["stop_loss_tickers"] = [t["ticker"] for t in triggered]
                emit(
                    "trading",
                    "stop_losses",
                    f"Stop-loss triggered: {len(triggered)} positions closed",
                    status="ok",
                    data={"triggered": cycle_summary["stop_loss_tickers"]},
                )
            else:
                cycle_summary["stop_loss_triggered"] = 0
                emit(
                    "trading",
                    "stop_losses",
                    "Stop-loss check: all positions within bounds",
                )
        except Exception as e:
            logger.error("Stop-loss check failed: %s", e)
            cycle_summary["stop_loss_check_failed"] = True
            emit(
                "trading",
                "stop_loss_error",
                f"⚠️ CRITICAL: Stop-loss check failed: {e}",
                status="error",
            )

    # 3. Position-Watchlist Reconciliation
    try:
        with get_db() as _db:
            pos_rows = _db.execute(
                "SELECT DISTINCT ticker FROM positions WHERE qty > 0 AND bot_id = %s",
                [bot_id],
            ).fetchall()
            wl_rows = _db.execute(
                "SELECT ticker FROM watchlist WHERE status = 'active'"
            ).fetchall()
            pos_set = {r[0] for r in pos_rows}
            wl_set = {r[0] for r in wl_rows}
            orphaned = pos_set - wl_set

            if orphaned:
                now = datetime.now(timezone.utc)
                for orphan in orphaned:
                    try:
                        _db.execute(
                            "INSERT INTO watchlist (ticker, status, source, added_at) VALUES (%s, 'active', 'position_hold', %s) "
                            "ON CONFLICT (ticker) DO UPDATE SET status = 'active'",
                            [orphan, now],
                        )
                    except Exception as recon_err:
                        logger.error(
                            "Failed to reconcile orphaned position %s: %s",
                            orphan,
                            recon_err,
                        )

            existing_set = set(ctx.tickers)
            added = [t for t in orphaned if t not in existing_set]
            if added:
                ctx.tickers.extend(added)
                state["tickers"] = ctx.tickers

            if orphaned:
                emit(
                    "started",
                    "position_reconciliation",
                    f"Added {len(orphaned)} orphaned position tickers to watchlist",
                    status="warning",
                    data={"orphaned": sorted(orphaned)},
                )
    except Exception as e:
        logger.warning("Reconciliation failed: %s", e)

    # 4. Smart Triage
    if ctx.tickers and settings.TRIAGE_ENABLED:
        try:
            emit(
                "started", "triage", "Running smart ticker triage...", status="running"
            )
            increment_days_since_deep(ctx.tickers)

            neglected = flag_neglected_tickers(
                max_days=settings.TRIAGE_NEGLECT_MAX_DAYS
            )
            if neglected:
                emit(
                    "started",
                    "triage_neglect",
                    f"⚠️ {len(neglected)} neglected tickers",
                    status="warning",
                )

            attention_data = get_attention_summary(ctx.tickers)

            _user_added = set()
            try:
                with get_db() as _db:
                    _user_rows = _db.execute(
                        "SELECT ticker FROM watchlist WHERE status = 'active' AND source = 'user'"
                    ).fetchall()
                _user_added = {r[0] for r in _user_rows}
            except Exception:
                pass

            positions_list = state.get("position_tickers", [])
            triage_result = classify_tickers(
                ctx.tickers, attention_data, positions_list, _user_added
            )

            state["triage"] = {
                "glance": triage_result.glance,
                "standard": triage_result.standard,
                "deep": triage_result.deep,
            }
            emit(
                "started",
                "triage_result",
                f"Triage: {triage_result.summary()}",
                status="ok",
            )
            cycle_summary["triage"] = {k: len(v) for k, v in state["triage"].items()}
        except Exception as e:
            logger.warning("Triage failed: %s", e)

    # 5. Load Autoresearch Directives
    try:
        with get_db() as _db:
            dir_rows = _db.execute(
                "SELECT directive_type, directive_text, target_ticker, severity FROM cycle_directives "
                "WHERE status = 'active' ORDER BY severity DESC, created_at DESC LIMIT 5"
            ).fetchall()
        active_directives = [
            {"type": r[0], "text": r[1], "ticker": r[2], "severity": r[3]}
            for r in dir_rows
        ]
        if active_directives:
            emit(
                "started",
                "directives_loaded",
                f"Loaded {len(active_directives)} directives",
                status="ok",
            )
        state["active_directives"] = active_directives
    except Exception as e:
        logger.debug("Directive loading skipped: %s", e)
        state["active_directives"] = []
