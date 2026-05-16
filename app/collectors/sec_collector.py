"""
SEC EDGAR Collector — Fetches 13F institutional holdings via edgartools.

Pure data collector. No LLM calls. No processing.
Writes to: sec_13f_holdings
Library: edgartools (free, no API key needed)

Schedule: Run once per quarter (13F filings are released 45 days after quarter end)

NOTE: edgartools is synchronous — we wrap in asyncio.to_thread to avoid blocking.
"""

import logging

logger = logging.getLogger(__name__)


import asyncio
import datetime
import pandas as pd
from edgar import Company, set_identity
from app.db.connection import get_db

# SEC EDGAR requires a User-Agent header identifying you
set_identity("TradingBot analysis@example.com")

# Top 20 hedge funds / institutional investors by AUM
# CIK numbers verified from SEC EDGAR
# List updated quarterly — ranked by assets under management
TRACKED_FUNDS = [
    # -- Mega Funds (>$100B AUM) --
    ("Berkshire Hathaway", "0001067983"),
    ("Bridgewater Associates", "0001350694"),
    ("Citadel Advisors", "0001423053"),
    ("Renaissance Technologies", "0001037389"),
    ("D.E. Shaw", "0001009207"),
    ("Two Sigma Investments", "0001179392"),
    ("Millennium Management", "0001273087"),
    ("AQR Capital Management", "0001167557"),
    # -- Large Funds ($10-100B AUM) --
    ("Tiger Global Management", "0001167483"),
    ("Point72 Asset Management", "0001603466"),
    ("Soros Fund Management", "0001029160"),
    ("Pershing Square Capital", "0001336528"),
    ("Viking Global Investors", "0001103804"),
    ("Baupost Group", "0001061768"),
    ("Elliott Investment Management", "0001791786"),
    ("Appaloosa Management", "0001656456"),
    # -- Notable / High Conviction --
    ("Druckenmiller (Duquesne)", "0001536411"),
    ("Greenlight Capital", "0001079114"),
    ("Third Point", "0001040273"),
    ("Coatue Management", "0001535392"),
]

# Per-fund timeout for edgartools (seconds)
EDGAR_TIMEOUT = 60


def _fetch_holdings_sync(filer_name: str, cik: str) -> tuple[list[dict], str]:
    """
    Synchronous function that does the slow edgartools work.
    Returns (holdings_list, filing_quarter).
    """
    company = Company(cik)
    # Get only recent filings (limit=5 prevents downloading full history)
    filings = company.get_filings(form="13F-HR").latest(5)

    if not filings or len(filings) == 0:
        logger.info(f"[sec] No 13F filings found for {filer_name} (CIK: {cik})")
        return [], ""

    # Get the latest filing
    latest = filings[0]

    # Determine the COVERED quarter (not the filing date quarter).
    # 13F filings are due 45 days after quarter end:
    #   Filed Jan-Mar → covers Q4 of previous year
    #   Filed Apr-Jun → covers Q1 of same year
    #   Filed Jul-Sep → covers Q2 of same year
    #   Filed Oct-Dec → covers Q3 of same year
    filed_date = latest.filing_date
    if hasattr(filed_date, "date"):
        filed_date = filed_date.date()
    elif isinstance(filed_date, str):
        filed_date = datetime.date.fromisoformat(filed_date)
    filing_q = (filed_date.month - 1) // 3 + 1
    # The covered quarter is one quarter BEFORE the filing quarter
    if filing_q == 1:
        covered_year = filed_date.year - 1
        covered_q = 4
    else:
        covered_year = filed_date.year
        covered_q = filing_q - 1
    filing_quarter = f"{covered_year}Q{covered_q}"

    # Parse the 13F filing to get holdings
    filing_obj = latest.obj()
    if not hasattr(filing_obj, "infotable") or filing_obj.infotable is None:
        logger.info(f"[sec] {filer_name}: no holdings table in latest 13F")
        return [], filing_quarter

    df = filing_obj.infotable
    if not isinstance(df, pd.DataFrame):
        if hasattr(df, "to_dataframe"):
            df = df.to_dataframe()
        else:
            logger.info(
                f"[sec] {filer_name}: cannot parse holdings format ({type(df)})"
            )
            return [], filing_quarter

    holdings = []
    for _, row in df.iterrows():
        ticker = str(row.get("Ticker", ""))
        if not ticker or ticker == "nan":
            ticker = str(row.get("Issuer", "UNKNOWN"))[:20]

        shares = int(row.get("SharesPrnAmount", 0) or 0)
        # SEC EDGAR reports 13F values in THOUSANDS of dollars
        value = float(row.get("Value", 0) or 0) * 1000

        holdings.append(
            {
                "cik": cik,
                "ticker": ticker,
                "filing_quarter": filing_quarter,
                "shares": shares,
                "value": value,
            }
        )

    return holdings, filing_quarter


async def collect_fund_holdings(
    filer_name: str,
    cik: str,
) -> int:
    """
    Fetch latest 13F holdings for a fund and upsert into sec_13f_holdings.
    Runs edgartools in a thread with timeout to avoid blocking.
    Returns number of holdings rows inserted.
    """
    try:
        # Run with timeout — edgartools can hang on large filings
        holdings, filing_quarter = await asyncio.wait_for(
            asyncio.to_thread(_fetch_holdings_sync, filer_name, cik),
            timeout=EDGAR_TIMEOUT,
        )

        if not holdings:
            return 0

        with get_db() as db:
            # Upsert filer to ensure it exists
            db.execute(
                """
                INSERT INTO sec_13f_filers (cik, filer_name)
                VALUES (%s, %s)
                ON CONFLICT (cik) DO UPDATE SET filer_name = EXCLUDED.filer_name
                """,
                [cik, filer_name],
            )

            for h in holdings:
                db.execute(
                    """
                    INSERT INTO sec_13f_holdings
                    (cik, ticker, filing_quarter, shares, value_usd,
                     pct_change, is_new_position, is_exit)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (cik, ticker, filing_quarter) DO NOTHING
                """,
                    [
                        h["cik"],
                        h["ticker"],
                        h["filing_quarter"],
                        h["shares"],
                        h["value"],
                        None,
                        False,
                        False,
                    ],
                )

            logger.info(
                f"[sec] {filer_name}: {len(holdings)} holdings written (Q: {filing_quarter})"
            )
            return len(holdings)

    except asyncio.TimeoutError:
        logger.info(f"[sec] {filer_name}: TIMEOUT after {EDGAR_TIMEOUT}s (CIK: {cik})")
        return 0
    except Exception as e:
        logger.info(f"[sec] {filer_name} error: {e}")
        return 0


async def collect_all_funds() -> dict:
    """Fetch 13F holdings for all tracked funds. Isolates errors per fund."""
    results = {}
    total_holdings = 0
    for name, cik in TRACKED_FUNDS:
        count = await collect_fund_holdings(name, cik)
        results[name] = count
        total_holdings += count
        # Pace between funds to avoid hammering EDGAR
        await asyncio.sleep(2)
    logger.info(
        f"[sec] Total: {total_holdings} holdings across {len(TRACKED_FUNDS)} funds "
        f"({sum(1 for v in results.values() if v > 0)} succeeded)"
    )
    return results


async def collect_ticker_institutional(ticker: str) -> int:
    """Fetch institutional holders for a specific ticker via yfinance.

    This is MORE RELIABLE than edgartools for per-ticker data because:
    - No CIK lookup needed
    - Works for any publicly traded ticker
    - Returns structured holder/shares/value data
    - Free, no rate limiting issues

    Returns number of holders inserted.
    """
    try:
        import os

        # Redirect yfinance cache to /tmp to avoid Permission Denied on /home/appusr
        os.environ["YFINANCE_CACHE_DIR"] = "/tmp/yfinance"
        import yfinance as yf

        t = yf.Ticker(ticker)
        ih = await asyncio.to_thread(lambda: t.institutional_holders)

        if ih is None or len(ih) == 0:
            logger.info(f"[sec] {ticker}: no yfinance institutional data")
            return 0

        with get_db() as db:
            now = datetime.datetime.now()
            quarter = f"{now.year}Q{(now.month - 1) // 3 + 1}"
            count = 0

            for _, row in ih.iterrows():
                holder = str(row.get("Holder", "Unknown"))
                shares = int(row.get("Shares", 0) or 0)
                value = float(row.get("Value", 0) or 0)

                import hashlib

                holder_hash = hashlib.md5(holder.encode()).hexdigest()[:10]
                pseudo_cik = f"yf_{holder_hash}"

                # Upsert filer
                db.execute(
                    """
                    INSERT INTO sec_13f_filers (cik, filer_name)
                    VALUES (%s, %s)
                    ON CONFLICT (cik) DO UPDATE SET filer_name = EXCLUDED.filer_name
                    """,
                    [pseudo_cik, holder],
                )

                # Insert holdings
                db.execute(
                    """
                    INSERT INTO sec_13f_holdings
                    (cik, ticker, filing_quarter, shares, value_usd,
                     pct_change, is_new_position, is_exit)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (cik, ticker, filing_quarter) DO NOTHING
                """,
                    [
                        pseudo_cik,
                        ticker,
                        quarter,
                        shares,
                        value,
                        None,
                        False,
                        False,
                    ],
                )
                count += 1

            logger.info(f"[sec] {ticker}: {count} institutional holders via yfinance")
            return count

    except ImportError:
        logger.info("[sec] yfinance not installed")
        return 0
    except Exception as e:
        logger.info(f"[sec] {ticker} yfinance error: {e}")
        return 0


async def collect_all_tickers_institutional(tickers: list[str]) -> dict:
    """Fetch institutional holders for a list of tickers via yfinance."""
    results = {}
    for ticker in tickers:
        count = await collect_ticker_institutional(ticker)
        results[ticker] = count
        await asyncio.sleep(1)  # Rate limit
    return results
