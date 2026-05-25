"""
yfinance Collector — Fetches OHLCV, fundamentals, financials, balance sheet.

Pure data collector. No LLM calls. No processing.
Writes to: price_history, fundamentals, financial_history, balance_sheet
"""

import logging

logger = logging.getLogger(__name__)


import datetime
import asyncio
import yfinance as yf
from app.db.connection import get_db


async def fetch_ohlcv_dataframe(ticker: str, period: str = "6mo"):
    """Fetch OHLCV history as a DataFrame without writing to DB."""
    stock = yf.Ticker(ticker)
    try:
        df = await asyncio.to_thread(stock.history, period=period, auto_adjust=True)
        if df is None or df.empty:
            logger.info(f"[yfinance] No price data for {ticker}")
            return None
        return df
    except Exception as e:
        logger.info(f"[yfinance] Error fetching price history for {ticker}: {e}")
        return None


async def collect_price_history(ticker: str, period: str = "6mo") -> int:
    """
    Fetch OHLCV history and upsert into price_history table.
    Returns number of rows inserted.
    """
    df = await fetch_ohlcv_dataframe(ticker, period)
    if df is None:
        return 0

    from app.validation.schema import PriceHistorySchema
    import pandera.errors

    try:
        df = PriceHistorySchema.validate(df)
    except pandera.errors.SchemaError as e:
        logger.error(f"[yfinance] Validation failed for {ticker}: {e}")
        return 0

    rows = []
    for date, row in df.iterrows():
        rows.append(
            [
                ticker,
                date.date(),
                float(row["Open"]),
                float(row["High"]),
                float(row["Low"]),
                float(row["Close"]),
                int(row["Volume"]),
            ]
        )

    if rows:

        def _insert():
            with get_db() as db:
                db.executemany(
                    """
                    INSERT INTO price_history (ticker, date, open, high, low, close, volume, source)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, 'yfinance')
                    ON CONFLICT (ticker, date, source) DO NOTHING
                """,
                    rows,
                )

        await asyncio.to_thread(_insert)

    count = len(rows)

    logger.info(f"[yfinance] {ticker}: {count} price rows written")
    return count


async def fetch_fundamentals_dict(ticker: str) -> dict | None:
    """Fetch fundamentals dictionary without writing to DB."""
    stock = yf.Ticker(ticker)
    try:
        info = await asyncio.to_thread(lambda: stock.info)
        if not info or "symbol" not in info:
            logger.info(f"[yfinance] No fundamentals for {ticker}")
            return None
        return info
    except Exception as e:
        logger.info(f"[yfinance] Error fetching fundamentals for {ticker}: {e}")
        return None


async def collect_fundamentals_finnhub(ticker: str) -> bool:
    """
    Fetch fundamentals snapshot from Finnhub as a tertiary fallback.
    Returns True if data was written.
    """
    import os
    from app.config.config import settings
    
    api_key = settings.FINNHUB_API_KEY or os.environ.get("FINNHUB_API_KEY", "")
    if not api_key:
        logger.info(f"[finnhub] API key not set, skipping fundamentals fallback for {ticker}")
        return False

    try:
        import finnhub
        client = finnhub.Client(api_key=api_key)
        
        # Run Finnhub client API calls in threads since they are blocking I/O calls
        profile = await asyncio.to_thread(client.company_profile2, symbol=ticker)
        financials = await asyncio.to_thread(client.company_basic_financials, ticker, 'all')
        
        if not profile and not financials:
            logger.info(f"[finnhub] No profile/financials data returned for {ticker}")
            return False
            
        metric = financials.get("metric", {}) if financials else {}
        
        today = datetime.date.today()
        # Market Cap from profile is in Millions, convert to dollars
        mkt_cap_m = profile.get("marketCapitalization")
        mkt_cap = float(mkt_cap_m) * 1_000_000 if mkt_cap_m else None
        
        pe = metric.get("peNormalizedTTM") or metric.get("peExclExtraTTM")
        beta = metric.get("beta") or profile.get("beta")
        
        # Map fields to fundamentals table structure
        with get_db() as db:
            db.execute(
                """
                INSERT INTO fundamentals (
                    ticker, snapshot_date, source, market_cap, pe_ratio, forward_pe, peg_ratio,
                    price_to_book, price_to_sales, ev_to_ebitda, profit_margin,
                    roe, roa, revenue, revenue_growth, net_income,
                    debt_to_equity, current_ratio, beta,
                    week_52_high, week_52_low, short_float_pct
                ) VALUES (%s, %s, 'finnhub', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker, snapshot_date) DO NOTHING
                """,
                [
                    ticker.upper(),
                    today,
                    mkt_cap,
                    pe,
                    None,
                    None,
                    metric.get("bookValuePerShareAnnual"), # pb proxy
                    metric.get("psTTM"),
                    None,
                    metric.get("netProfitMarginTTM"),
                    metric.get("roeTTM"),
                    metric.get("roaTTM"),
                    None,
                    None,
                    None,
                    metric.get("debtEquityTTM"),
                    metric.get("currentRatioAnnual"),
                    beta,
                    metric.get("52WeekHigh"),
                    metric.get("52WeekLow"),
                    None,
                ],
            )
        logger.info(f"[finnhub] Successfully stored fundamentals fallback for {ticker}")
        return True
    except Exception as e:
        logger.info(f"[finnhub] Error fetching fundamentals fallback for {ticker}: {e}")
        return False


async def collect_fundamentals(ticker: str) -> bool:
    """
    Fetch fundamentals snapshot and upsert into fundamentals table.
    Returns True if data was written.
    """
    # 1. Try yfinance
    info = await fetch_fundamentals_dict(ticker)
    if info and info.get("marketCap"):
        today = datetime.date.today()
        with get_db() as db:
            db.execute(
                """
                INSERT INTO fundamentals (
                    ticker, snapshot_date, source, market_cap, pe_ratio, forward_pe, peg_ratio,
                    price_to_book, price_to_sales, ev_to_ebitda, profit_margin,
                    roe, roa, revenue, revenue_growth, net_income,
                    debt_to_equity, current_ratio, beta,
                    week_52_high, week_52_low, short_float_pct
                ) VALUES (%s, %s, 'yfinance', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ticker, snapshot_date) DO NOTHING
            """,
                [
                    ticker,
                    today,
                    info.get("marketCap"),
                    info.get("trailingPE"),
                    info.get("forwardPE"),
                    info.get("pegRatio"),
                    info.get("priceToBook"),
                    info.get("priceToSalesTrailing12Months"),
                    info.get("enterpriseToEbitda"),
                    info.get("profitMargins"),
                    info.get("returnOnEquity"),
                    info.get("returnOnAssets"),
                    info.get("totalRevenue"),
                    info.get("revenueGrowth"),
                    info.get("netIncomeToCommon"),
                    info.get("debtToEquity"),
                    info.get("currentRatio"),
                    info.get("beta"),
                    info.get("fiftyTwoWeekHigh"),
                    info.get("fiftyTwoWeekLow"),
                    info.get("shortPercentOfFloat"),
                ],
            )
        logger.info(
            f"[yfinance] {ticker}: fundamentals written (mkt_cap={info.get('marketCap')})"
        )
        return True

    logger.info(f"[yfinance] Failed to collect fundamentals for {ticker}, trying fallbacks...")

    # 2. Try FMP Fallback
    try:
        from app.collectors.fmp_collector import collect_fundamentals as collect_fmp
        fmp_ok = await collect_fmp(ticker)
        if fmp_ok:
            logger.info(f"[fundamentals] Fallback to FMP succeeded for {ticker}")
            return True
    except Exception as e:
        logger.warning(f"[fundamentals] Fallback to FMP failed for {ticker}: {e}")

    # 3. Try Finnhub Fallback
    try:
        finnhub_ok = await collect_fundamentals_finnhub(ticker)
        if finnhub_ok:
            logger.info(f"[fundamentals] Fallback to Finnhub succeeded for {ticker}")
            return True
    except Exception as e:
        logger.warning(f"[fundamentals] Fallback to Finnhub failed for {ticker}: {e}")

    return False


async def collect_financials(ticker: str) -> int:
    """
    Fetch income statement (quarterly + annual) and upsert into financial_history.
    Returns number of rows inserted.
    """
    stock = yf.Ticker(ticker)
    count = 0
    try:
        sources = await asyncio.to_thread(
            lambda: [
                ("quarterly", stock.quarterly_income_stmt),
                ("annual", stock.income_stmt),
            ]
        )
    except Exception as e:
        logger.info(f"[yfinance] Error fetching financials for {ticker}: {e}")
        return 0

    rows = []
    for period_type, financials in sources:
        if financials is None or financials.empty:
            continue

        for col in financials.columns:
            period_end = col.date() if hasattr(col, "date") else col
            data = financials[col]
            rows.append(
                [
                    ticker,
                    period_type,
                    period_end,
                    _safe_float(data, "Total Revenue"),
                    _safe_float(data, "Gross Profit"),
                    _safe_float(data, "Operating Income"),
                    _safe_float(data, "Net Income"),
                    _safe_float(data, "Basic EPS"),
                    None,  # FCF from cash flow statement, not income stmt
                ]
            )

    if rows:

        def _insert():
            with get_db() as db:
                db.executemany(
                    """
                    INSERT INTO financial_history (
                        ticker, period_type, period_end,
                        revenue, gross_profit, operating_income,
                        net_income, eps, free_cash_flow
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker, period_type, period_end) DO NOTHING
                """,
                    rows,
                )

        await asyncio.to_thread(_insert)

    count = len(rows)

    logger.info(f"[yfinance] {ticker}: {count} financial history rows written")
    return count


async def collect_balance_sheet(ticker: str) -> int:
    """
    Fetch balance sheet and upsert into balance_sheet table.
    Returns number of rows inserted.
    """
    stock = yf.Ticker(ticker)
    try:
        bs = await asyncio.to_thread(lambda: stock.balance_sheet)
        if bs is None or bs.empty:
            logger.info(f"[yfinance] No balance sheet for {ticker}")
            return 0
    except Exception as e:
        logger.info(f"[yfinance] Error fetching balance sheet for {ticker}: {e}")
        return 0

    count = 0

    rows = []
    for col in bs.columns:
        period_end = col.date() if hasattr(col, "date") else col
        data = bs[col]

        rows.append(
            [
                ticker,
                period_end,
                _safe_float(data, "Total Assets"),
                _safe_float(data, "Total Liabilities Net Minority Interest"),
                _safe_float(data, "Stockholders Equity"),
                _safe_float(data, "Cash And Cash Equivalents"),
                _safe_float(data, "Total Debt"),
                _safe_float(data, "Working Capital"),
            ]
        )

    if rows:

        def _insert():
            with get_db() as db:
                db.executemany(
                    """
                    INSERT INTO balance_sheet (
                        ticker, period_end, total_assets, total_liabilities,
                        total_equity, cash, total_debt, working_capital
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ticker, period_end) DO NOTHING
                """,
                    rows,
                )

        await asyncio.to_thread(_insert)

    count = len(rows)

    logger.info(f"[yfinance] {ticker}: {count} balance sheet rows written")
    return count


async def collect_all(ticker: str) -> dict:
    """Run all yfinance collectors for a single ticker."""
    prices = await collect_price_history(ticker)
    fundies = await collect_fundamentals(ticker)
    financials = await collect_financials(ticker)
    balance = await collect_balance_sheet(ticker)

    return {
        "ticker": ticker,
        "price_rows": prices,
        "fundamentals": fundies,
        "financial_rows": financials,
        "balance_rows": balance,
    }


async def collect_news(ticker: str) -> int:
    """
    Fetch ticker-specific news from yfinance and scrape full article bodies.
    Proxy to the robust news_collector.py to ensure full body extraction and quality gating.
    """
    try:
        from app.collectors.news_collector import collect_yfinance_news
        logger.info(f"[yfinance] Proxying collect_news({ticker}) to robust news_collector...")
        return await collect_yfinance_news(ticker)
    except Exception as e:
        logger.error(f"[yfinance] Proxy call error for {ticker}: {e}")
        return 0


def _safe_float(series, key: str) -> float | None:
    """Safely extract a float from a pandas Series, handling missing keys."""
    try:
        val = series.get(key)
        if val is not None and str(val) != "nan":
            return float(val)
    except (KeyError, TypeError, ValueError):
        pass
    return None
