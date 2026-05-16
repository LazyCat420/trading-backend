"""
FMP Collector — Financial Modeling Prep.

Pure data collector. No LLM calls.
Writes to: congress_trades (proper JSON API, not HTML scraping)
Requires: FMP_API_KEY in .env (free tier = 250 calls/day)
"""

import logging

logger = logging.getLogger(__name__)


import hashlib
import datetime
import httpx
from app.config import settings
from app.db.connection import get_db

BASE_URL = "https://financialmodelingprep.com/api/v4"


def _get_key() -> str:
    key = settings.FMP_API_KEY
    if not key:
        raise ValueError(
            "FMP_API_KEY not set in .env — get free key at financialmodelingprep.com"
        )
    return key


async def collect_congress_trades(ticker: str | None = None) -> int:
    """
    Fetch congressional stock trades from FMP (proper JSON API).
    Returns number of rows inserted.
    """
    key = _get_key()
    with get_db() as db:
        count = 0

        params = {"apikey": key}
        url = f"{BASE_URL}/senate-trading"

        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(url, params=params)
            if r.status_code == 401:
                logger.info("[fmp] Invalid API key or unauthorized")
                return 0
            if r.status_code == 403:
                logger.info(
                    "[fmp] Congress trades requires premium FMP plan — falling back to CapitolTrades"
                )
                return 0
            r.raise_for_status()
            trades = r.json()

        if not trades:
            logger.info("[fmp] No congress trades returned")
            return 0

        for tx in trades:
            tx_ticker = (
                tx.get("ticker", tx.get("asset_description", "")).upper().strip()
            )
            if not tx_ticker:
                continue

            if ticker and tx_ticker != ticker.upper():
                continue

            trade_id = hashlib.md5(
                f"{tx.get('senator', tx.get('representative', ''))}"
                f"{tx_ticker}{tx.get('transaction_date', '')}"
                f"{tx.get('type', tx.get('transaction_type', ''))}".encode()
            ).hexdigest()

            trade_date = _parse_date(
                tx.get("transaction_date", tx.get("transactionDate", ""))
            )
            disclosure_date = _parse_date(
                tx.get("disclosure_date", tx.get("disclosureDate", ""))
            )

            days = None
            if trade_date and disclosure_date:
                days = (disclosure_date - trade_date).days

            politician = tx.get(
                "senator", tx.get("representative", tx.get("firstName", "Unknown"))
            )
            chamber = "Senate" if "senator" in tx else "House"

            db.execute(
                """
                INSERT INTO congress_trades
                (id, politician, party, chamber, state, ticker,
                 transaction_type, amount_range, trade_date,
                 disclosure_date, days_to_disclose)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """,
                [
                    trade_id,
                    politician,
                    tx.get("party", ""),
                    chamber,
                    tx.get("state", ""),
                    tx_ticker,
                    tx.get("type", tx.get("transaction_type", "")),
                    tx.get("amount", ""),
                    trade_date,
                    disclosure_date,
                    days,
                ],
            )
            count += 1

        logger.info(
            f"[fmp] {count} congress trades written"
            f"{f' (filtered: {ticker})' if ticker else ''}"
        )
        return count


async def collect_all(ticker: str | None = None) -> dict:
    """Run all FMP collectors."""
    trades = await collect_congress_trades(ticker)

    if ticker:
        prices = await collect_price_history(ticker)
        fundies = await collect_fundamentals(ticker)
        financials = await collect_financials(ticker)
        balance = await collect_balance_sheet(ticker)
        return {
            "ticker": ticker,
            "congress_trades": trades,
            "price_rows": prices,
            "fundamentals": fundies,
            "financial_rows": financials,
            "balance_rows": balance,
            "source": "fmp",
        }

    return {"congress_trades": trades, "source": "fmp"}


def _parse_date(date_str: str) -> datetime.date | None:
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d %b %Y"):
        try:
            return datetime.datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None


async def collect_price_history(ticker: str, days_back: int = 365) -> int:
    """Fetch OHLCV history and upsert into price_history table."""
    try:
        api_key = _get_key()
    except ValueError:
        return 0

    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}"

    from app.services.request_utils import SmartClient

    async with SmartClient(base_delay=1.0, max_retries=3) as client:
        resp = await client.get(url, params={"apikey": api_key})
        if resp.status_code != 200:
            logger.info(
                f"[fmp] Error fetching price history for {ticker}: HTTP {resp.status_code}"
            )
            return 0
        data = resp.json()

    historical = data.get("historical", [])
    if not historical:
        logger.info(f"[fmp] No price data for {ticker}")
        return 0

    cutoff = datetime.date.today() - datetime.timedelta(days=days_back)

    with get_db() as db:
        count = 0
        for day in historical:
            try:
                date_obj = datetime.date.fromisoformat(day.get("date", ""))
                if date_obj < cutoff:
                    continue

                db.execute(
                    """
                    INSERT INTO price_history (ticker, date, open, high, low, close, volume, source)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, 'fmp')
                    ON CONFLICT (ticker, date, source) DO NOTHING
                    """,
                    [
                        ticker,
                        date_obj,
                        float(day.get("open", 0)),
                        float(day.get("high", 0)),
                        float(day.get("low", 0)),
                        float(day.get("close", 0)),
                        int(day.get("volume", 0)),
                    ],
                )
                count += 1
            except Exception as e:
                continue

        logger.info(f"[fmp] {ticker}: {count} price rows written")
        return count


async def collect_fundamentals(ticker: str) -> bool:
    """Fetch fundamentals snapshot and upsert into fundamentals table."""
    try:
        api_key = _get_key()
    except ValueError:
        return False

    profile_url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}"
    metrics_url = f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{ticker}"

    from app.services.request_utils import SmartClient

    async with SmartClient(base_delay=1.0, max_retries=3) as client:
        prof_resp = await client.get(profile_url, params={"apikey": api_key})
        metrics_resp = await client.get(
            metrics_url, params={"apikey": api_key, "limit": 1}
        )

        if prof_resp.status_code != 200 or metrics_resp.status_code != 200:
            logger.info(f"[fmp] Error fetching fundamentals for {ticker}")
            return False

        prof_data = prof_resp.json()
        metrics_data = metrics_resp.json()

    if not prof_data:
        logger.info(f"[fmp] No profile data for {ticker}")
        return False

    prof = prof_data[0]
    metrics = metrics_data[0] if metrics_data else {}

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
            ) VALUES (%s, %s, 'fmp', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker, snapshot_date) DO NOTHING
            """,
            [
                ticker,
                today,
                prof.get("mktCap"),
                metrics.get("peRatioTTM"),
                None,  # Forward PE not directly in profile/ttm
                metrics.get("pegRatioTTM"),
                metrics.get("pbRatioTTM"),
                metrics.get("priceToSalesRatioTTM"),
                metrics.get("enterpriseValueOverEBITDATTM"),
                metrics.get("netIncomePerEBT"),  # proxy
                metrics.get("roeTTM"),
                metrics.get("returnOnTangibleAssetsTTM"),
                None,  # Need income statement
                None,
                None,
                metrics.get("debtToEquityTTM"),
                metrics.get("currentRatioTTM"),
                prof.get("beta"),
                None,  # 52w high/low requires another endpoint
                None,
                None,
            ],
        )

        logger.info(f"[fmp] {ticker}: fundamentals written")
        return True


async def collect_financials(ticker: str) -> int:
    """Fetch income statement (quarterly) and upsert into financial_history."""
    try:
        api_key = _get_key()
    except ValueError:
        return 0

    url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}"

    from app.services.request_utils import SmartClient

    async with SmartClient(base_delay=1.0, max_retries=3) as client:
        resp = await client.get(
            url, params={"period": "quarter", "limit": 4, "apikey": api_key}
        )
        if resp.status_code != 200:
            if resp.status_code == 403:
                logger.debug(
                    f"[fmp] {ticker}: Financials unavailable (HTTP 403) - likely an ETF or missing permissions"
                )
            else:
                logger.info(
                    f"[fmp] Error fetching financials for {ticker}: HTTP {resp.status_code}"
                )
            return 0
        data = resp.json()

    if not data:
        return 0

    with get_db() as db:
        count = 0
        for stmt in data:
            try:
                period_end = datetime.date.fromisoformat(stmt.get("date", "")[:10])
                db.execute(
                    """
                    INSERT INTO financial_history (
                        ticker, period_type, period_end,
                        revenue, gross_profit, operating_income,
                        net_income, eps, free_cash_flow
                    ) VALUES (%s, 'quarterly', %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ticker, period_type, period_end) DO NOTHING
                    """,
                    [
                        ticker,
                        period_end,
                        stmt.get("revenue"),
                        stmt.get("grossProfit"),
                        stmt.get("operatingIncome"),
                        stmt.get("netIncome"),
                        stmt.get("eps"),
                        None,  # FCF is in cash flow statement
                    ],
                )
                count += 1
            except Exception:
                pass

        logger.info(f"[fmp] {ticker}: {count} financial history rows written")
        return count


async def collect_balance_sheet(ticker: str) -> int:
    """Fetch balance sheet (quarterly) and upsert into balance_sheet table."""
    try:
        api_key = _get_key()
    except ValueError:
        return 0

    url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}"

    from app.services.request_utils import SmartClient

    async with SmartClient(base_delay=1.0, max_retries=3) as client:
        resp = await client.get(
            url, params={"period": "quarter", "limit": 4, "apikey": api_key}
        )
        if resp.status_code != 200:
            if resp.status_code == 403:
                logger.debug(
                    f"[fmp] {ticker}: Balance sheet unavailable (HTTP 403) - likely an ETF or missing permissions"
                )
            else:
                logger.info(
                    f"[fmp] Error fetching balance sheet for {ticker}: HTTP {resp.status_code}"
                )
            return 0
        data = resp.json()

    if not data:
        return 0

    with get_db() as db:
        count = 0

        for bs in data:
            try:
                period_end = datetime.date.fromisoformat(bs.get("date", "")[:10])
                db.execute(
                    """
                    INSERT INTO balance_sheet (
                        ticker, period_end, total_assets, total_liabilities,
                        total_equity, cash, total_debt, working_capital
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ticker, period_end) DO NOTHING
                    """,
                    [
                        ticker,
                        period_end,
                        bs.get("totalAssets"),
                        bs.get("totalLiabilities"),
                        bs.get("totalStockholdersEquity"),
                        bs.get("cashAndCashEquivalents"),
                        bs.get("totalDebt"),
                        bs.get("totalWorkingCapital", 0),  # sometimes missing
                    ],
                )
                count += 1
            except Exception:
                pass

        logger.info(f"[fmp] {ticker}: {count} balance sheet rows written")
        return count
