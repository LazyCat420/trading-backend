"""
Passive Collector Daemon
Continuously rotates through active watchlist and portfolio tickers to refresh
data in the background, ensuring trading cycles can run analysis instantly
without waiting for scraping.

Runs as an asyncio.create_task() in the main lifespan — lives forever.

Pacing:
  - Market hours (Mon-Fri 6:00 AM – 5:00 PM PT): rotate every 3 hours
  - Off-hours / weekends: sleep 6 hours between rotations
  - Per-ticker freshness: skip if data was collected < 2 hours ago
  - RSS general sweep: only runs every 2nd rotation to avoid waste
"""

import asyncio
import logging
import pytz
from datetime import datetime, timezone, timedelta

from app.db.connection import get_db
from app.trading.watchlist import get_active
from app.trading.portfolio import get_current_state

logger = logging.getLogger(__name__)

# ── Timing constants (seconds) ──
MARKET_HOURS_SLEEP = 3 * 3600      # 3 hours during market hours
OFF_HOURS_SLEEP = 6 * 3600         # 6 hours outside market hours
TICKER_FRESHNESS_HOURS = 2         # skip ticker if data < 2h old
INTER_TICKER_DELAY = 10            # seconds between per-ticker API calls
RSS_SWEEP_CADENCE = 2              # run RSS sweep every Nth rotation

# Module-level status tracking for the frontend
_collector_status = {
    "running": False,
    "state": "idle",            # "collecting", "sleeping", "idle"
    "next_rotation_at": None,   # ISO timestamp — when next rotation starts
    "last_rotation_at": None,
    "current_ticker": None,
    "tickers_in_rotation": 0,
    "total_rotations": 0,
}


def get_collector_status() -> dict:
    """Return current passive collector status for frontend."""
    return dict(_collector_status)


def _is_market_hours() -> bool:
    """Check if we're within extended US market hours (6 AM – 5 PM PT, weekdays).

    Wider than strict 9:30-4 ET to capture pre-market and after-hours news.
    """
    pt = pytz.timezone("America/Los_Angeles")
    now = datetime.now(pt)
    if now.weekday() >= 5:  # Saturday / Sunday
        return False
    return 6 <= now.hour < 17


def _is_ticker_fresh(ticker: str, max_age_hours: float) -> bool:
    """Return True if we already have recent data for this ticker.

    Prevents redundant API calls when the previous rotation's data
    is still within the freshness window.
    """
    try:
        with get_db() as db:
            row = db.execute(
                "SELECT MAX(collected_at) FROM news_articles WHERE ticker = %s",
                [ticker],
            ).fetchone()
        if row and row[0]:
            age = (datetime.now(timezone.utc) - row[0]).total_seconds() / 3600
            return age < max_age_hours
    except Exception as e:
        logger.debug("[PASSIVE] Freshness check failed for %s: %s", ticker, e)
    return False


def _set_state(state: str, next_rotation_at: datetime | None = None) -> None:
    """Update the collector state atomically."""
    _collector_status["state"] = state
    if next_rotation_at:
        _collector_status["next_rotation_at"] = next_rotation_at.isoformat()
    elif state != "collecting":
        _collector_status["next_rotation_at"] = None


async def _passive_collect_ticker(ticker: str):
    """Collect news and reddit for a single ticker (no prices — too noisy)."""
    _collector_status["current_ticker"] = ticker
    try:
        from app.collectors.news_collector import collect_for_ticker as collect_news
        from app.collectors.reddit_collector import collect_for_ticker as collect_reddit

        logger.debug("[PASSIVE] Collecting for %s...", ticker)

        # Run news and reddit concurrently per ticker
        results = await asyncio.gather(
            collect_news(ticker),
            collect_reddit(ticker),
            return_exceptions=True,
        )

        news_count = results[0] if isinstance(results[0], int) else 0
        reddit_count = results[1] if isinstance(results[1], int) else 0

        if news_count or reddit_count:
            logger.info(
                "[PASSIVE] %s: %d news, %d reddit",
                ticker, news_count, reddit_count,
            )
    except Exception as e:
        logger.warning("[PASSIVE] Failed to collect for %s: %s", ticker, e)
    finally:
        _collector_status["current_ticker"] = None


async def run_passive_collector_loop():
    """Background loop that slowly iterates over all relevant tickers."""
    logger.info("[PASSIVE] Starting passive background collector loop.")
    _collector_status["running"] = True

    # Wait for server startup tasks to finish first
    await asyncio.sleep(30)

    while True:
        try:
            # ── Gate: skip collection if system is paused ──
            from app.pipeline.orchestration.cycle_control import cycle_control
            if cycle_control.is_paused:
                logger.debug("[PASSIVE] System is PAUSED — skipping rotation.")
                _set_state("sleeping",
                           next_rotation_at=datetime.now(timezone.utc) + timedelta(seconds=60))
                await asyncio.sleep(60)
                continue

            # ── Gate: decide sleep duration based on market hours ──
            if _is_market_hours():
                sleep_seconds = MARKET_HOURS_SLEEP
            else:
                logger.info("[PASSIVE] Outside market hours. Sleeping %d hours.",
                            OFF_HOURS_SLEEP // 3600)
                wake_at = datetime.now(timezone.utc) + timedelta(seconds=OFF_HOURS_SLEEP)
                _set_state("sleeping", next_rotation_at=wake_at)
                await asyncio.sleep(OFF_HOURS_SLEEP)
                continue

            # 1. Get targets
            state = get_current_state()
            portfolio_tickers = [p["ticker"] for p in state.get("positions", [])]
            watchlist_tickers = [w["ticker"] for w in get_active()]

            target_tickers = list(set(portfolio_tickers + watchlist_tickers))
            _collector_status["tickers_in_rotation"] = len(target_tickers)

            if not target_tickers:
                logger.debug("[PASSIVE] No tickers to track. Sleeping 5 min.")
                _set_state("sleeping",
                           next_rotation_at=datetime.now(timezone.utc) + timedelta(seconds=300))
                await asyncio.sleep(300)
                continue

            _set_state("collecting")
            rotation_num = _collector_status["total_rotations"] + 1
            logger.info(
                "[PASSIVE] Starting rotation #%d: %d tickers", rotation_num, len(target_tickers)
            )

            # 2. General market RSS sweep (only every Nth rotation to reduce waste)
            if rotation_num % RSS_SWEEP_CADENCE == 1 or rotation_num == 1:
                try:
                    from app.collectors.news_collector import collect_all
                    count = await collect_all(limit_feeds=10)  # Top 10 feeds only
                    if count:
                        logger.info("[PASSIVE] RSS sweep: %d articles", count)
                except Exception as e:
                    logger.warning("[PASSIVE] RSS sweep failed: %s", e)
                await asyncio.sleep(5)
            else:
                logger.info("[PASSIVE] Skipping RSS sweep (runs every %d rotations)",
                            RSS_SWEEP_CADENCE)

            # 3. Per-ticker collection with gentle pacing + freshness gate
            skipped = 0
            collected = 0
            for ticker in target_tickers:
                if _is_ticker_fresh(ticker, TICKER_FRESHNESS_HOURS):
                    logger.debug("[PASSIVE] %s: data is fresh, skipping", ticker)
                    skipped += 1
                    continue

                await _passive_collect_ticker(ticker)
                collected += 1
                # Sleep between tickers to avoid rate limits
                await asyncio.sleep(INTER_TICKER_DELAY)

            # 4. Rotation complete
            _collector_status["last_rotation_at"] = datetime.now(
                timezone.utc
            ).isoformat()
            _collector_status["total_rotations"] = rotation_num

            wake_at = datetime.now(timezone.utc) + timedelta(seconds=sleep_seconds)
            _set_state("sleeping", next_rotation_at=wake_at)

            logger.info(
                "[PASSIVE] Rotation #%d complete (%d collected, %d skipped-fresh). "
                "Sleeping %d hours.",
                rotation_num, collected, skipped, sleep_seconds // 3600,
            )
            await asyncio.sleep(sleep_seconds)

        except asyncio.CancelledError:
            logger.info("[PASSIVE] Loop cancelled.")
            _collector_status["running"] = False
            _set_state("idle")
            break
        except Exception as e:
            logger.error("[PASSIVE] Error in main loop: %s", e, exc_info=True)
            _set_state("sleeping",
                       next_rotation_at=datetime.now(timezone.utc) + timedelta(seconds=60))
            await asyncio.sleep(60)
