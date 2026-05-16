"""
Sector metadata collector — fetches sector/industry/cap-tier from yfinance & DB.

Populates ticker_metadata table. Used by graph_queries for sector peers
and by context_builder for the RELATIONSHIP MAP section.

Usage:
    from app.graph.sector_collector import collect_metadata
    updated = await collect_metadata(["NVDA", "AAPL", "AMD"])
"""

import asyncio
import logging

from app.db.connection import get_db

logger = logging.getLogger(__name__)

_CAP_TIERS = [
    (200_000_000_000, "mega"),
    (10_000_000_000, "large"),
    (2_000_000_000, "mid"),
    (300_000_000, "small"),
    (0, "micro"),
]


def classify_cap(market_cap: int | float | None) -> str:
    """Classify market cap into tier: mega/large/mid/small/micro."""
    if not market_cap or market_cap <= 0:
        return "micro"
    for threshold, tier in _CAP_TIERS:
        if market_cap >= threshold:
            return tier
    return "micro"


def classify_asset(ticker: str) -> str:
    """Classify asset type from ticker symbol conventions.

    NOTE: Canonical version now lives in app.config_tickers.classify_asset.
    This wrapper is kept for backward compatibility with any external imports.
    """
    from app.config.config_tickers import classify_asset as _classify

    return _classify(ticker)


async def collect_metadata(tickers: list[str]) -> int:
    """Fetch sector/industry metadata for tickers via yfinance.

    Skips tickers already in ticker_metadata updated within 7 days.
    Returns count of tickers updated.
    """
    with get_db() as db:
        updated = 0

        # Find which tickers need updating
        needs_update = []
        for t in tickers:
            t = t.upper().strip()
            try:
                row = db.execute(
                    "SELECT updated_at FROM ticker_metadata "
                    "WHERE ticker = %s AND updated_at > CURRENT_TIMESTAMP - INTERVAL '7 days'",
                    [t],
                ).fetchone()
                if not row:
                    needs_update.append(t)
            except Exception:
                needs_update.append(t)

        if not needs_update:
            logger.info("sector_collector: all %d tickers up to date", len(tickers))
            return 0

        logger.info(
            "sector_collector: updating %d/%d tickers", len(needs_update), len(tickers)
        )

        for t in needs_update:
            try:
                info = await _fetch_yfinance_info(t)
                if not info:
                    continue

                sector = info.get("sector", "Unknown")
                industry = info.get("industry", "Unknown")
                name = info.get("longName") or info.get("shortName") or t
                market_cap = info.get("marketCap") or 0
                cap_tier = classify_cap(market_cap)
                asset_class = classify_asset(t)

                # Check for ETFs
                qt = info.get("quoteType", "")
                if qt == "ETF":
                    asset_class = "etf"
                    sector = "ETF"
                    industry = info.get("category", "Unknown")

                db.execute(
                    "INSERT INTO ticker_metadata "
                    "(ticker, name, sector, industry, market_cap, "
                    "market_cap_tier, asset_class, updated_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP) "
                    "ON CONFLICT (ticker) DO NOTHING",
                    [t, name, sector, industry, market_cap, cap_tier, asset_class],
                )
                updated += 1

                # Rate limit (yfinance is strict)
                await asyncio.sleep(0.3)

            except Exception as e:
                logger.warning("sector_collector: %s failed: %s", t, e)
                continue

        logger.info("sector_collector: updated %d tickers", updated)
        return updated


async def _fetch_yfinance_info(ticker: str) -> dict | None:
    """Fetch yfinance info dict in a thread (yfinance is blocking)."""
    try:
        import yfinance as yf

        loop = asyncio.get_event_loop()
        info = await loop.run_in_executor(None, lambda: yf.Ticker(ticker).info)
        return info if isinstance(info, dict) else None
    except Exception as e:
        logger.warning("yfinance info failed for %s: %s", ticker, e)
        return None


def get_metadata(ticker: str) -> dict | None:
    """Get cached metadata for a ticker."""
    with get_db() as db:
        try:
            row = db.execute(
                "SELECT ticker, name, sector, industry, market_cap, "
                "market_cap_tier, asset_class FROM ticker_metadata "
                "WHERE ticker = %s",
                [ticker.upper().strip()],
            ).fetchone()
            if not row:
                return None
            return {
                "ticker": row[0],
                "name": row[1],
                "sector": row[2],
                "industry": row[3],
                "market_cap": row[4],
                "market_cap_tier": row[5],
                "asset_class": row[6],
            }
        except Exception:
            return None


def get_sector_peers(ticker: str, limit: int = 10) -> list[dict]:
    """Get tickers in the same sector/industry."""
    with get_db() as db:
        try:
            meta = get_metadata(ticker)
            if not meta or not meta["sector"]:
                return []

            rows = db.execute(
                "SELECT ticker, name, industry, market_cap, market_cap_tier "
                "FROM ticker_metadata "
                "WHERE sector = %s AND ticker != %s "
                "ORDER BY market_cap DESC LIMIT %s",
                [meta["sector"], ticker.upper().strip(), limit],
            ).fetchall()
            return [
                {
                    "ticker": r[0],
                    "name": r[1],
                    "industry": r[2],
                    "market_cap": r[3],
                    "market_cap_tier": r[4],
                }
                for r in rows
            ]
        except Exception:
            return []
