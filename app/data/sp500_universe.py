"""
S&P 500 Universe Loader — Uses hardcoded constituent list.

Loads from app/data/sp500_constituents.py (permanent, no network required).
Optionally enriches market cap via yfinance.
"""

import logging

from app.db.connection import get_db

logger = logging.getLogger(__name__)


async def load_sp500_universe(enrich: bool = False):
    """
    Loads S&P 500 tickers from the hardcoded constituent list into ticker_metadata.

    This is instant and network-free. The list is stored in
    app/data/sp500_constituents.py and updated periodically via
    scripts/gen_sp500_list.py.

    Args:
        enrich: If True, also fetch market cap data from yfinance.
                This is SLOW (~500 API calls) and should only be done
                on explicit user request, not on startup.
    """
    from app.data.sp500_constituents import SP500_TICKERS

    logger.info(
        "Loading S&P 500 universe from hardcoded list (%d tickers)...",
        len(SP500_TICKERS),
    )

    with get_db() as db:
        # Reset sp500 flag for all stocks first
        db.execute("UPDATE ticker_metadata SET sp500 = FALSE")

        loaded = 0
        for i, entry in enumerate(SP500_TICKERS):
            ticker = entry["ticker"]
            name = entry["name"]
            sector = entry["sector"]
            industry = entry["industry"]
            market_cap = None
            market_cap_tier = None

            # Grab existing enrichment data if it exists
            existing = db.execute(
                "SELECT market_cap, market_cap_tier FROM ticker_metadata WHERE ticker = %s",
                (ticker,),
            ).fetchone()
            if existing:
                market_cap = existing[0]
                market_cap_tier = existing[1]

            if enrich:
                try:
                    import yfinance as yf

                    ticker_obj = yf.Ticker(ticker)
                    info = ticker_obj.info
                    sector = info.get("sector", sector)
                    industry = info.get("industry", industry)
                    market_cap = info.get("marketCap", market_cap)
                    name = info.get("shortName", name)

                    if market_cap:
                        if market_cap >= 200e9:
                            market_cap_tier = "mega"
                        elif market_cap >= 10e9:
                            market_cap_tier = "large"
                        elif market_cap >= 2e9:
                            market_cap_tier = "mid"
                        elif market_cap >= 300e6:
                            market_cap_tier = "small"
                        else:
                            market_cap_tier = "micro"
                except Exception as e:
                    logger.debug("Failed to enrich %s via yfinance: %s", ticker, e)

            query = """
                INSERT INTO ticker_metadata (ticker, name, sector, industry, market_cap, market_cap_tier, asset_class, sp500, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, 'stock', TRUE, CURRENT_TIMESTAMP)
                ON CONFLICT (ticker) DO UPDATE SET
                    name = EXCLUDED.name,
                    sector = EXCLUDED.sector,
                    industry = EXCLUDED.industry,
                    market_cap = COALESCE(EXCLUDED.market_cap, ticker_metadata.market_cap),
                    market_cap_tier = COALESCE(EXCLUDED.market_cap_tier, ticker_metadata.market_cap_tier),
                    sp500 = TRUE,
                    updated_at = CURRENT_TIMESTAMP
            """
            db.execute(
                query, (ticker, name, sector, industry, market_cap, market_cap_tier)
            )
            loaded += 1

            if (i + 1) % 100 == 0:
                logger.info(
                    "Loaded %d/%d S&P 500 tickers...", i + 1, len(SP500_TICKERS)
                )

    logger.info("Successfully loaded %d S&P 500 tickers into universe.", loaded)
    return loaded
