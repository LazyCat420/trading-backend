"""
Finviz Scraper — Unauthenticated fallback for basic fundamentals.
"""

import logging
import datetime
from bs4 import BeautifulSoup
from app.db.connection import get_db
from app.services.request_utils import SmartClient

logger = logging.getLogger(__name__)


async def collect_fundamentals(ticker: str) -> bool:
    """Scrape basic fundamentals from Finviz."""
    url = f"https://finviz.com/quote.ashx?t={ticker}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    async with SmartClient(base_delay=2.0, max_retries=2) as client:
        resp = await client.get(url, headers=headers)
        if resp.status_code != 200:
            logger.info(f"[finviz] Error scraping {ticker}: HTTP {resp.status_code}")
            return False

    soup = BeautifulSoup(resp.text, "html.parser")

    # Finviz has a main snapshot table with class 'snapshot-table2'
    table = soup.find("table", class_="snapshot-table2")
    if not table:
        logger.info(f"[finviz] No snapshot table found for {ticker}")
        return False

    data = {}
    for row in table.find_all("tr"):
        cols = row.find_all("td")
        for i in range(0, len(cols), 2):
            if i + 1 < len(cols):
                key = cols[i].text.strip()
                val = cols[i + 1].text.strip()
                data[key] = val

    def parse_val(v_str: str):
        if not v_str or v_str == "-":
            return None
        v_str = v_str.replace(",", "")
        mult = 1
        if v_str.endswith("B"):
            mult = 1_000_000_000
            v_str = v_str[:-1]
        elif v_str.endswith("M"):
            mult = 1_000_000
            v_str = v_str[:-1]
        elif v_str.endswith("%"):
            v_str = v_str[:-1]
            # percentages usually parsed as float
        try:
            return float(v_str) * mult
        except ValueError:
            return None

    # Map finviz keys to DB schema
    market_cap = parse_val(data.get("Market Cap"))
    pe = parse_val(data.get("P/E"))
    fpe = parse_val(data.get("Forward P/E"))
    peg = parse_val(data.get("PEG"))
    pb = parse_val(data.get("P/B"))
    ps = parse_val(data.get("P/S"))
    roe = parse_val(data.get("ROE"))
    if roe:
        roe /= 100.0
    roa = parse_val(data.get("ROA"))
    if roa:
        roa /= 100.0
    profit_margin = parse_val(data.get("Profit Margin"))
    if profit_margin:
        profit_margin /= 100.0
    debt_eq = parse_val(data.get("Debt/Eq"))
    cr = parse_val(data.get("Current Ratio"))
    beta = parse_val(data.get("Beta"))
    short_float = parse_val(data.get("Short Float"))
    if short_float:
        short_float /= 100.0

    today = datetime.date.today()
    with get_db() as db:
        db.execute(
            """
            INSERT INTO fundamentals (
                ticker, snapshot_date, source, market_cap, pe_ratio, forward_pe, peg_ratio,
                price_to_book, price_to_sales, profit_margin,
                roe, roa, debt_to_equity, current_ratio, beta, short_float_pct
            ) VALUES (%s, %s, 'finviz', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker, snapshot_date) DO NOTHING
            """,
            [
                ticker,
                today,
                market_cap,
                pe,
                fpe,
                peg,
                pb,
                ps,
                profit_margin,
                roe,
                roa,
                debt_eq,
                cr,
                beta,
                short_float,
            ],
        )

        logger.info(f"[finviz] {ticker}: fundamentals scraped successfully")
        return True
