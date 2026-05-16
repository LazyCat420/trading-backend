"""
Alpha Decay Purge — Mathematical pruning pass for toxic assets.

Runs right after data collection. If a ticker mathematically crosses into
"toxic" territory (massive debt, 52-week crash, penny stock rot), it is
immediately BANNED from the watchlist before the LLM wastes tokens on it.

IMPORTANT: This module must NEVER crash the pipeline. All errors are
swallowed and logged. Only truly catastrophic metrics trigger a ban,
and MULTIPLE concurrent failures are required to prevent false positives.
"""

import logging
import traceback
from typing import Callable

from app.config import settings
from app.db.connection import get_db
from app.utils.pipeline_utils import noop as _noop

logger = logging.getLogger(__name__)

# Minimum simultaneous metric failures required to ban a ticker.
# 1 = any single failure bans it (aggressive, causes false positives on
#     legitimate high-leverage companies like ORCL, PSX, HAL).
# 2 = needs two concurrent red flags (recommended — catches real decay).
MIN_FAILURES_TO_BAN = 2


def _is_exempt_sector(sector: str | None) -> bool:
    """Check if the sector is exempt from debt checks (e.g. Banks)."""
    if not sector:
        return False
    exempt = [s.lower() for s in settings.ALPHA_EXEMPT_DEBT_SECTORS]
    return sector.lower() in exempt


def run_alpha_decay_purge(
    watchlist: list[str], emit: Callable | None = None
) -> list[str]:
    """Mathematical scan for terminally decaying assets.

    Requires at least MIN_FAILURES_TO_BAN concurrent metric violations
    (e.g. penny stock AND massive debt) before banning. This prevents
    false positives on legitimate high-leverage companies.

    Returns the list of tickers that were immediately banned.

    Side-effect: held positions with critical decay are flagged via emit()
    so the pipeline can force a SELL review even if the LLM would otherwise HOLD.
    """
    # ── SAFETY: This function must NEVER crash the pipeline ──
    try:
        return _run_alpha_decay_purge_inner(watchlist, emit)
    except BaseException as e:
        logger.error(
            "[PIPELINE] alpha_decay: CRITICAL ERROR (swallowed to protect pipeline): %s\n%s",
            e,
            traceback.format_exc(),
        )
        return []


def _run_alpha_decay_purge_inner(
    watchlist: list[str], emit: Callable | None = None
) -> list[str]:
    """Inner implementation — separated so the outer wrapper can catch everything."""
    if emit is None:
        emit = _noop

    if not settings.ALPHA_DECAY_ENABLED:
        logger.info("[PIPELINE] alpha_decay: disabled via config")
        return []

    with get_db() as db:
        # Get active portfolio positions — we do NOT auto-ban these.
        try:
            from app.services.bot_manager import get_active_bot_id

            bid = get_active_bot_id()
        except Exception:
            from app.config import settings as _cfg

            bid = _cfg.BOT_ID
        try:
            pos_rows = db.execute(
                "SELECT DISTINCT ticker FROM positions WHERE qty > 0 AND bot_id = %s",
                [bid],
            ).fetchall()
            positions = {r[0] for r in pos_rows}
        except Exception:
            positions = set()

        banned_tickers = []

        for ticker in watchlist:
            try:
                # 1. Fetch latest fundamentals and price snapshot
                fund_row = db.execute(
                    """
                    SELECT f.debt_to_equity, f.current_ratio,
                           f.week_52_high, f.week_52_low, tm.sector
                    FROM fundamentals f
                    LEFT JOIN ticker_metadata tm ON f.ticker = tm.ticker
                    WHERE f.ticker = %s
                    ORDER BY f.snapshot_date DESC LIMIT 1
                    """,
                    [ticker],
                ).fetchone()

                price_row = db.execute(
                    "SELECT close FROM price_history "
                    "WHERE ticker = %s ORDER BY date DESC LIMIT 1",
                    [ticker],
                ).fetchone()

                # Skip tickers with no data — don't ban on missing info
                if not fund_row or not price_row:
                    continue

                debt_to_equity, current_ratio, week_52_high, week_52_low, sector = (
                    fund_row
                )
                current_price = price_row[0]

                if current_price is None or current_price <= 0:
                    continue

                reasons = []

                # ── Check 1: Penny Stock Rot ──
                if current_price < settings.ALPHA_PENNY_FLOOR:
                    reasons.append(
                        f"Price (${current_price:.2f}) < "
                        f"Penny Floor (${settings.ALPHA_PENNY_FLOOR:.2f})"
                    )

                # ── Check 2: Debt Spiral ──
                # Only flags truly catastrophic debt — many large-caps (ORCL,
                # energy companies) legitimately run D/E > 5.0 via buybacks.
                if (
                    debt_to_equity is not None
                    and debt_to_equity > settings.ALPHA_MAX_DEBT_TO_EQUITY
                    and not _is_exempt_sector(sector)
                ):
                    reasons.append(
                        f"D/E Ratio ({debt_to_equity:.2f}) > "
                        f"Max ({settings.ALPHA_MAX_DEBT_TO_EQUITY}) "
                        f"in {sector or 'Unknown'} sector"
                    )

                # ── Check 3: Current Ratio Collapse ──
                if (
                    current_ratio is not None
                    and current_ratio < settings.ALPHA_MIN_CURRENT_RATIO
                ):
                    reasons.append(
                        f"Current Ratio ({current_ratio:.2f}) < "
                        f"Min ({settings.ALPHA_MIN_CURRENT_RATIO})"
                    )

                # ── Check 4: 52-Week Death Spiral ──
                if week_52_high is not None and week_52_high > 0:
                    drawdown = (week_52_high - current_price) / week_52_high
                    if drawdown > settings.ALPHA_MAX_52_WK_DRAWDOWN:
                        reasons.append(
                            f"Drawdown ({drawdown * 100:.1f}%) > "
                            f"Max ({settings.ALPHA_MAX_52_WK_DRAWDOWN * 100:.1f}%)"
                        )

                # ── Evaluation: Require MULTIPLE failures ──
                # A single high D/E or single low current ratio is NOT enough
                # to ban a stock. We need at least MIN_FAILURES_TO_BAN
                # concurrent violations to avoid false positives.
                if len(reasons) >= MIN_FAILURES_TO_BAN:
                    reason_str = " | ".join(reasons)

                    if ticker in positions:
                        # Held position with critical decay: do NOT silently skip.
                        # Flag it loudly so the pipeline forces a SELL review.
                        logger.warning(
                            "[PIPELINE] alpha_decay: CRITICAL DECAY on open position %s: %s "
                            "— flagging for FORCED SELL review",
                            ticker,
                            reason_str,
                        )
                        emit(
                            "analyzing",
                            f"alpha_decay_force_review_{ticker}",
                            f"⚠️ {ticker}: CRITICAL DECAY on held position — "
                            f"forced SELL review triggered ({reason_str})",
                            status="warning",
                        )
                        # Do NOT ban (we still hold shares), but DO track it so
                        # the analysis pipeline can inject a sell bias.
                        if not hasattr(run_alpha_decay_purge, "_decay_positions"):
                            run_alpha_decay_purge._decay_positions = []
                        run_alpha_decay_purge._decay_positions.append(
                            {"ticker": ticker, "reasons": reason_str}
                        )
                        continue

                    # Ban it — import lazily to avoid circular import issues
                    from app.trading.watchlist import ban_ticker

                    logger.info(
                        "[PIPELINE] alpha_decay: BANNING %s: %s", ticker, reason_str
                    )
                    try:
                        ban_ticker(ticker, reason=f"[ALPHA DECAY] {reason_str}")
                        banned_tickers.append(ticker)
                    except Exception as ban_err:
                        logger.error(
                            "[PIPELINE] alpha_decay: ban_ticker failed for %s: %s",
                            ticker,
                            ban_err,
                        )

            except Exception as e:
                logger.error(
                    "[PIPELINE] alpha_decay: Error evaluating %s: %s", ticker, e
                )

        if banned_tickers:
            logger.info(
                "[PIPELINE] alpha_decay: Banned %d toxic assets: %s",
                len(banned_tickers),
                ", ".join(banned_tickers),
            )

        return banned_tickers


def get_decay_flagged_positions() -> list[dict]:
    """Return held positions flagged with critical decay during the last purge.

    Each entry has {"ticker": ..., "reasons": ...}.
    Callers can use this to inject sell-bias context into the analysis prompt.
    """
    flagged = getattr(run_alpha_decay_purge, "_decay_positions", [])
    return list(flagged)


def clear_decay_flagged_positions() -> None:
    """Reset the per-cycle accumulator (call at cycle start)."""
    run_alpha_decay_purge._decay_positions = []
