"""
Congress Trades Collector -- Fetches congressional stock trade disclosures.

Pure data collector. No LLM calls. No processing.
Writes to: congress_trades

Source: CapitolTrades.com (free, no auth, public HTML scraping).
Covers both House and Senate financial disclosures.

Schedule: Run daily -- disclosures are filed within 45 days of trade.

CapitolTrades HTML structure (as of March 2026):
  td[0] = Politician + Party + Chamber + State (combined, e.g. "Mitch McConnellRepublicanSenateKY")
  td[1] = Company Name + Ticker:Exchange (e.g. "Wells Fargo & CoWFC:US")
  td[2] = Published date (e.g. "19 Mar2026")
  td[3] = Trade date (e.g. "1 Mar2026")
  td[4] = Days to disclose (e.g. "days18")
  td[5] = Owner (e.g. "Spouse", "Self")
  td[6] = Transaction type (e.g. "buy", "sell")
  td[7] = Size range (e.g. "1K-15K")
  td[8] = Price (e.g. "N/A")
  td[9] = Link cell
"""

import logging

logger = logging.getLogger(__name__)


import hashlib
import datetime
import re
from app.db.connection import get_db
from app.services.request_utils import SmartClient

BASE_URL = "https://www.capitoltrades.com/trades"

# Party keywords to extract from combined politician cell
PARTIES = {"Republican", "Democrat", "Independent"}
CHAMBERS = {"Senate", "House"}
# All US state codes
STATE_CODES = {
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
    "DC",
    "PR",
    "GU",
    "VI",
    "AS",
    "MP",
}


async def collect_trades(
    pages: int = 3,
    ticker_filter: str | None = None,
) -> int:
    """
    Scrape trade data from CapitolTrades.com and upsert into congress_trades.
    Returns number of rows inserted.
    """
    with get_db() as db:
        total_count = 0

        async with SmartClient(base_delay=2.0, max_retries=3) as client:
            for page in range(1, pages + 1):
                params = {
                    "page": page,
                    "txType": ["buy", "sell"],
                    "assetType": "stock",
                }
                r = await client.get(BASE_URL, params=params)
                if r.status_code != 200:
                    logger.info(f"[congress] Page {page}: HTTP {r.status_code}")
                    break

                try:
                    from bs4 import BeautifulSoup

                    soup = BeautifulSoup(r.text, "lxml")
                except ImportError:
                    from bs4 import BeautifulSoup

                    soup = BeautifulSoup(r.text, "html.parser")

                rows = soup.select("table tbody tr")

                if not rows:
                    logger.info(f"[congress] Page {page}: no rows found")
                    break

                page_count = 0
                for row in rows:
                    trade = _parse_row(row)
                    if not trade or not trade.get("ticker"):
                        continue

                    # Filter by ticker if requested
                    if ticker_filter and trade["ticker"] != ticker_filter.upper():
                        continue

                    # Build unique ID
                    trade_id = hashlib.md5(
                        f"{trade['politician']}{trade['ticker']}"
                        f"{trade['trade_date']}{trade['transaction_type']}".encode()
                    ).hexdigest()

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
                            trade["politician"],
                            trade["party"],
                            trade["chamber"],
                            trade["state"],
                            trade["ticker"],
                            trade["transaction_type"],
                            trade["amount_range"],
                            trade["trade_date"],
                            trade["disclosure_date"],
                            trade["days_to_disclose"],
                        ],
                    )
                    total_count += 1
                    page_count += 1

                logger.info(
                    f"[congress] Page {page}: {len(rows)} rows, {page_count} valid trades"
                )
                await client.pace(2.0, 4.0)

        logger.info(
            f"[congress] {total_count} trades written"
            f"{f' (filtered: {ticker_filter})' if ticker_filter else ''}"
        )
        return total_count


async def collect_trades_for_ticker(ticker: str) -> int:
    """Fetch congress trades filtered to a specific ticker."""
    return await collect_trades(pages=5, ticker_filter=ticker)


async def collect_all() -> dict:
    """Fetch recent congress trades (3 pages). Returns summary."""
    count = await collect_trades(pages=3)
    return {"total_trades": count, "source": "capitoltrades"}


def _parse_row(row) -> dict | None:
    """Parse a single table row from CapitolTrades.

    Column mapping (verified from live HTML March 2026):
      td[0] = PoliticianPartyChmaberState (combined text)
      td[1] = CompanyNameTICKER:EXCHANGE
      td[2] = Published date
      td[3] = Trade date
      td[4] = "daysNN" (filing delay)
      td[5] = Owner
      td[6] = Transaction type
      td[7] = Size range
      td[8] = Price
    """
    tds = row.find_all("td")
    if len(tds) < 8:
        return None

    try:
        # td[0]: Parse politician + party + chamber + state from combined text
        raw_politician = tds[0].get_text(strip=True)
        politician, party, chamber, state = _parse_politician_cell(raw_politician)

        # td[1]: Parse ticker from "Company NameTICKER:EXCHANGE"
        raw_issuer = tds[1].get_text(strip=True)
        ticker = _extract_ticker_from_issuer(raw_issuer)

        if not ticker:
            return None

        # td[2]: Published/disclosure date
        disclosure_date = _parse_date(tds[2].get_text(strip=True))

        # td[3]: Trade date
        trade_date = _parse_date(tds[3].get_text(strip=True))

        # td[4]: Days to disclose (e.g. "days18" -> 18)
        days_text = tds[4].get_text(strip=True)
        days_match = re.search(r"(\d+)", days_text)
        days_to_disclose = int(days_match.group(1)) if days_match else None

        # td[5]: Owner
        # (not stored in current schema but available)

        # td[6]: Transaction type
        tx_type = tds[6].get_text(strip=True).lower() if len(tds) > 6 else ""

        # td[7]: Amount range (normalize Unicode dashes to ASCII)
        amount = tds[7].get_text(strip=True) if len(tds) > 7 else ""
        amount = amount.replace("\u2013", "-").replace(
            "\u2014", "-"
        )  # en-dash, em-dash

        return {
            "politician": politician[:100],
            "party": party,
            "chamber": chamber,
            "state": state,
            "ticker": ticker,
            "transaction_type": tx_type,
            "amount_range": amount,
            "trade_date": trade_date,
            "disclosure_date": disclosure_date,
            "days_to_disclose": days_to_disclose,
        }
    except (IndexError, ValueError) as e:
        logger.info(f"[congress] Parse error: {e}")
        return None


def _parse_politician_cell(text: str) -> tuple[str, str, str, str]:
    """Parse combined politician cell like 'Mitch McConnellRepublicanSenateKY'.

    Strategy: Look for known keywords (party, chamber, state) and extract them,
    leaving the politician name.
    """
    remaining = text
    party = ""
    chamber = ""
    state = ""

    # Extract party
    for p in PARTIES:
        if p in remaining:
            party = p
            remaining = remaining.replace(p, "", 1)
            break

    # Extract chamber
    for c in CHAMBERS:
        if c in remaining:
            chamber = c
            remaining = remaining.replace(c, "", 1)
            break

    # Extract state code (2 uppercase letters at end)
    state_match = re.search(r"([A-Z]{2})$", remaining)
    if state_match and state_match.group(1) in STATE_CODES:
        state = state_match.group(1)
        remaining = remaining[: state_match.start()]

    # What's left is the politician name
    politician = remaining.strip()

    return politician, party, chamber, state


def _extract_ticker_from_issuer(text: str) -> str | None:
    """Extract ticker from issuer cell like 'Wells Fargo & CoWFC:US'.

    The ticker is the uppercase segment before :EXCHANGE at the end.
    Pattern: "Company NameTICKER:XX" or just "Company NameTICKER"
    """
    # Try to find TICKER:EXCHANGE pattern
    match = re.search(r"([A-Z]{1,5}):([A-Z]{2})$", text)
    if match:
        return match.group(1)

    # Try to find trailing uppercase sequence (the ticker)
    match = re.search(r"([A-Z]{1,5})$", text)
    if match:
        ticker = match.group(1)
        # Validate it's not just an abbreviation
        if len(ticker) >= 1:
            return ticker

    return None


def _parse_date(date_str: str) -> datetime.date | None:
    """Parse date string from CapitolTrades like '19 Mar2026' or '1 Mar2026'."""
    if not date_str:
        return None
    # Clean up and normalize spacing
    date_str = " ".join(date_str.split())

    # CapitolTrades format: "19 Mar2026" (no space between month and year)
    # Fix by inserting space before 4-digit year
    date_str = re.sub(r"(\D)(\d{4})", r"\1 \2", date_str)

    for fmt in ("%d %b %Y", "%b %d, %Y", "%B %d, %Y", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None
