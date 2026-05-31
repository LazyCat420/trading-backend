"""
Smart Ticker Extractor — Shared module for extracting stock tickers from text.

Single source of truth used by all collectors. Replaces the 4 separate implementations
across news_collector, youtube_collector, reddit_collector.

Architecture:
  Layer 0: CompanyRegistry (S&P 500 + aliases pre-loaded, cached in PostgreSQL)
  Layer 1: Regex + Registry Match (instant, free)
  Layer 2: Exclusion + Context Scoring (instant, free)
  Layer 3: yfinance Validation (1 API call per unknown symbol, cached forever)
  Layer 4: LLM Batch Disambiguation (background job, not in this file)
"""

import re
import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────


@dataclass
class Company:
    """A verified company with bidirectional symbol ↔ name mapping."""

    symbol: str  # "V"
    name: str  # "Visa Inc"
    aliases: list[str]  # ["visa", "visa inc"]
    sector: str = ""  # "Financials"
    market_cap: float = 0  # from yfinance
    is_sp500: bool = False
    single_letter: bool = False  # V, X, T, C → needs context


@dataclass
class TickerMatch:
    """Result of ticker extraction — always has both symbol AND company_name."""

    symbol: str  # "V"
    company_name: str  # "Visa Inc"
    confidence: float  # 0.0 - 1.0
    source: str  # "$pattern", "company_name", "registry", "yfinance", "bare_caps"
    context_snippet: str = ""  # surrounding text for debugging


# ─────────────────────────────────────────────
# FALSE_TICKERS — consolidated from all collectors
# ─────────────────────────────────────────────

FALSE_TICKERS = {
    # Internet slang / Reddit
    "AI",
    "CEO",
    "IPO",
    "ETF",
    "ATH",
    "ATL",
    "GDP",
    "CPI",
    "FED",
    "DD",
    "WSB",
    "YOLO",
    "FD",
    "TLDR",
    "IMO",
    "LMAO",
    "LOL",
    "STFU",
    "NOT",
    "THE",
    "AND",
    "FOR",
    "ARE",
    "BUT",
    "HAS",
    "ITS",
    "ALL",
    "NEW",
    "OLD",
    "NOW",
    "BIG",
    "RED",
    "LOW",
    "HIGH",
    "UP",
    "GO",
    "RIP",
    "OG",
    "DM",
    "PM",
    "AM",
    "IT",
    "US",
    "UK",
    "EU",
    "USD",
    "EDIT",
    "TIL",
    "ELI",
    "FAQ",
    "PSA",
    "FYI",
    "NSFW",
    "OP",
    "AI",
    "NEW",
    "READY",
    "YOU",
    "FOR",
    # Common English words that appear as bare caps
    "ON",
    "OR",
    "AS",
    "AT",
    "IS",
    "IN",
    "NO",
    "SO",
    "AN",
    "DO",
    "OF",
    "WE",
    "VS",
    "BE",
    "HE",
    "ME",
    "IF",
    "MY",
    "BY",
    "LIVE",
    "ILL",
    "ISNT",
    "OPEN",
    "SHOW",
    "BEST",
    "REAL",
    "NEXT",
    "MORE",
    "JUST",
    "ONLY",
    "YOUR",
    "WILL",
    "EVERY",
    "INTO",
    "OUT",
    "OUR",
    "ALSO",
    "BEEN",
    "EVEN",
    "MOST",
    "MUCH",
    "THAN",
    "THEM",
    "THEY",
    "THIS",
    "VERY",
    "WELL",
    "WERE",
    "WHAT",
    "WHEN",
    "WILL",
    "WITH",
    "PUTS",
    "DAY",
    "NEAR",
    "WAR",
    "HAPPY",
    "WORLD",
    "TO",
    # Additional English words caught leaking in 13.5h cycle
    "SOON",
    "KEEP",
    "MARCH",
    "SWOT",
    "ANNA",
    "HEAT",
    "LION",
    "OSINT",
    "PRE",
    "MAVAN",
    "TAM",
    "ONE",
    "TWO",
    "TAKE",
    "SAID",
    "OVER",
    "LIKE",
    "SOME",
    "BEEN",
    "COULD",
    "WOULD",
    "BACK",
    "MAKE",
    "MADE",
    "YEAR",
    "LAST",
    "WEEK",
    "LOOK",
    "HARD",
    "FAST",
    "HALF",
    "FULL",
    "FIND",
    "GOOD",
    "HUGE",
    "SURE",
    "DOES",
    "DONE",
    "DOWN",
    "EACH",
    "HAVE",
    "HERE",
    "HOPE",
    "IDEA",
    "KNOW",
    "LEFT",
    "LETS",
    "LOTS",
    "PLAY",
    "STAY",
    "STOP",
    "THAT",
    "SAME",
    "SAFE",
    "RISK",
    "GAVE",
    "TRUE",
    "TURN",
    "USED",
    "WANT",
    "ZERO",
    # Government / organizations
    "SEC",
    "DOJ",
    "FBI",
    "CIA",
    "IRS",
    "IMF",
    "WEF",
    "FDA",
    "EPA",
    "WHO",
    "NATO",
    "OPEC",
    "UN",
    "EU",
    "CDC",
    # Financial acronyms
    "PE",
    "EPS",
    "ROE",
    "ROI",
    "DCF",
    "AUM",
    "NAV",
    "PNL",
    "ER",
    "IV",
    "OI",
    "DTE",
    "ITM",
    "OTM",
    "ATM",
    "YTD",
    "QOQ",
    "YOY",
    "MOM",
    "EOD",
    "EOW",
    # Common English words that look like tickers
    "HOLD",
    "SELL",
    "BUY",
    "LONG",
    "SHORT",
    "CALL",
    "PUT",
    "BULL",
    "BEAR",
    "PUMP",
    "DUMP",
    "DIP",
    "RUN",
    "GAP",
    "TOP",
    "BOT",
    "MID",
    "MAX",
    "MIN",
    "AVG",
    "PDF",
    "API",
    "URL",
    "USB",
    "CPU",
    "GPU",
    "RAM",
    "SSD",
    "INC",
    "LLC",
    "LTD",
    "CORP",
    "EST",
    "FUND",
    "CEO",
    "CFO",
    "CTO",
    "COO",
    "CMO",
    "CIO",
    "FYI",
    "TBD",
    "TBA",
    "WIP",
    "EOD",
    # ── Media / Tech / Common abbreviations ──
    # These are very common in general news articles and almost never
    # refer to the stock. E.g., "TV" = television, not Grupo Televisa;
    # "HD" = high definition, not Home Depot (which is matched by company name).
    "TV",
    "HD",
    "PC",
    "DVD",
    "VR",
    "AR",
    "HR",
    "PR",
    "AD",
    "ADS",
    "APP",
    "GPS",
    "USB",
    "LED",
    "LCD",
    "VPN",
    "RSS",
    "NFT",
    "SUV",
    "AC",
    # ── Titles / Roles ──
    "VP",
    "EVP",
    "SVP",
    "PHD",
    "DR",
    "MD",
    # ── Geography / Places ──
    "NYC",
    "LA",
    "SF",
    "DC",
    "UAE",
    # ── More common words that leak through ──
    "LIFE",
    "PLAN",
    "FREE",
    "HOME",
    "CARE",
    "SAVE",
    "POST",
    "RATE",
    "LINE",
    "FUEL",
    "LAND",
    "RIDE",
    "TECH",
    "GROW",
    "ROCK",
    "CASH",
    "DEAL",
    "WIRE",
    "TALK",
    "WORK",
    "MOVE",
    "GAIN",
    "LOSS",
    "VOTE",
    "JOBS",
    "PAYS",
    "HITS",
    "GETS",
    "SAYS",
    "GOES",
    "SEES",
    "USES",
    "WINS",
    "PICK",
    "GAVE",
    "RISE",
    "FELL",
    "FUND",
    "GOLD",
    # Single letters that need context
    "I",
    "A",
    "PI",
}

# Market cap threshold: if a FALSE_TICKER symbol has market cap >= this, allow it
# through to context scoring instead of hard-blocking it.
# e.g. OPEN (Opendoor, $1.5B), FAST (Fastenal, $40B), RUN (Sunrun, $3B)
MIN_MARKET_CAP_OVERRIDE = 1_000_000_000  # $1B


def _is_hard_blocked(sym: str, registry: "CompanyRegistry") -> bool:
    """
    Check if a symbol should be hard-blocked.

    Returns False (= allow through) if:
      - Symbol is NOT in FALSE_TICKERS, OR
      - Symbol IS in FALSE_TICKERS but is a verified stock with market_cap >= $1B

    Returns True (= block) if:
      - Symbol is in FALSE_TICKERS and NOT a known high-cap stock
    """
    if sym not in FALSE_TICKERS:
        return False  # not blocked
    # It's in FALSE_TICKERS — check if it's also a real high-cap stock
    company = registry.lookup_symbol(sym)
    if company and (company.is_sp500 or company.market_cap >= MIN_MARKET_CAP_OVERRIDE):
        return False  # high-cap override — allow through
    return True  # blocked


# Financial context keywords — if near a bare caps word, boost confidence
FINANCIAL_CONTEXT = {
    "stock",
    "shares",
    "price",
    "earnings",
    "revenue",
    "profit",
    "market",
    "cap",
    "bullish",
    "bearish",
    "buy",
    "sell",
    "hold",
    "target",
    "analyst",
    "upgrade",
    "downgrade",
    "rating",
    "calls",
    "puts",
    "options",
    "dividend",
    "split",
    "ipo",
    "merger",
    "acquisition",
    "guidance",
    "forecast",
    "quarterly",
    "q1",
    "q2",
    "q3",
    "q4",
    "eps",
    "pe",
    "valuation",
    "growth",
    "rally",
    "crash",
    "surge",
    "plunge",
    "bounce",
    "moon",
    "squeeze",
    "short",
    "long",
    "position",
    "portfolio",
    "sector",
    "overweight",
    "underweight",
    "outperform",
    "underperform",
    "yolo",
    "dd",
    "fd",
    "bags",
    "holding",
    "avg",
    "dca",
    "cost basis",
    "support",
    "resistance",
    "breakout",
    "premiums",
    "volume",
    "report",
    "reporting",
}

# ─────────────────────────────────────────────
# Anti-Patterns — Common English phrases that DISQUALIFY ticker matches.
# If a 2-4 letter ticker candidate ONLY appears inside these patterns
# (and never in financial framing), it gets a heavy confidence penalty.
# This stops "TV", "HR", "PR", "AD" etc. from leaking through when
# they're used as ordinary English abbreviations.
# ─────────────────────────────────────────────

# Map: ticker symbol → list of regex patterns where that word is used
# as a common English abbreviation, NOT as a stock ticker.
# Each pattern must match case-insensitively around the ticker word.
COMMON_WORD_ANTI_PATTERNS: dict[str, list[str]] = {
    "TV": [
        r"(?i)\bon\s+TV\b",
        r"(?i)\bTV\s+(?:show|series|host|channel|station|network|advert|commercial|programme|program|screen|set|appearance|personality|star|presenter|executive|studios|production|broadcast|news|drama|comedy|episode|season|viewer|audience|reality)",
        r"(?i)\breality\s+TV\b",
        r"(?i)\bcable\s+TV\b",
        r"(?i)\blive\s+TV\b",
        r"(?i)\bsatellite\s+TV\b",
        r"(?i)\bsmart\s+TV\b",
        r"(?i)\b(?:watch|watched|watching)\s+(?:on\s+)?TV\b",
        r"(?i)\bTV\s+(?:ad|ads|advert|adverts|advertising|campaign)\b",
        r"(?i)\bchance\s+to\s+be\s+on\s+TV\b",
    ],
    "HD": [
        r"(?i)\bHD\s+(?:video|quality|resolution|display|screen|camera|content|streaming|format|recording)",
        r"(?i)\b(?:full|ultra|1080p|720p|4K)\s+HD\b",
        r"(?i)\bHD\s+(?:TV|monitor)\b",
    ],
    "PC": [
        r"(?i)\bPC\s+(?:game|games|gaming|user|users|hardware|software|version|platform|desktop|laptop|computer|build|market|sales)\b",
        r"(?i)\bgaming\s+PC\b",
        r"(?i)\bdesktop\s+PC\b",
    ],
    "VR": [
        r"(?i)\bVR\s+(?:headset|game|games|gaming|experience|content|device|world|technology|goggles)\b",
        r"(?i)\bvirtual\s+reality\b",
    ],
    "AR": [
        r"(?i)\bAR\s+(?:glasses|experience|app|apps|technology|feature|content|device|headset)\b",
        r"(?i)\baugmented\s+reality\b",
    ],
    "HR": [
        r"(?i)\bHR\s+(?:department|manager|team|director|policy|policies|software|platform|consultant|professional|issue|officer)\b",
        r"(?i)\bhuman\s+resources?\b",
    ],
    "PR": [
        r"(?i)\bPR\s+(?:firm|agency|team|campaign|strategy|manager|stunt|disaster|crisis|statement|representative|department)\b",
        r"(?i)\bpublic\s+relations?\b",
        r"(?i)\bpress\s+release\b",
    ],
    "AD": [
        r"(?i)\b(?:TV|video|online|digital|print|display|banner|pop.?up|targeted|political)\s+ad\b",
        r"(?i)\bad\s+(?:campaign|revenue|spend|spending|network|blocker|tech|targeting|budget|industry|market|placement)\b",
    ],
}


def _check_anti_patterns(sym: str, full_text: str) -> float:
    """Check if a ticker candidate appears only in common English anti-patterns.

    Returns a confidence PENALTY (0.0 = no penalty, -0.40 = heavy penalty).
    The penalty fires only when ALL occurrences of the symbol match anti-patterns
    and NONE appear in financial contexts.
    """
    patterns = COMMON_WORD_ANTI_PATTERNS.get(sym)
    if not patterns:
        return 0.0

    # Count total bare occurrences of the symbol
    total_mentions = len(re.findall(rf"\b{re.escape(sym)}\b", full_text))
    if total_mentions == 0:
        return 0.0

    # Count how many occurrences are inside anti-patterns
    anti_pattern_count = 0
    for pattern in patterns:
        anti_pattern_count += len(re.findall(pattern, full_text))

    # If most/all occurrences are in anti-patterns, this is likely NOT a stock
    # We cap at total_mentions to avoid double-counting overlapping patterns
    anti_ratio = min(anti_pattern_count, total_mentions) / total_mentions

    if anti_ratio >= 0.8:
        # Almost all uses are common English — heavy penalty
        return -0.40
    elif anti_ratio >= 0.5:
        # Mixed usage — moderate penalty
        return -0.25
    return 0.0


# Regex for very strong direct syntax (applied directly to the symbol string)
DIRECT_SYNTAX_VERBS = r"(?i)\b(?:bought|sold|buy|sell|short|shorting|long|longing|holding|calls on|puts on|shares of)\s+"
DIRECT_SYNTAX_PRICE = (
    r"\s+(?:up|down|\+|-)?[0-9]+(?:\.[0-9]+)?%|\s+to\s+\$?[0-9]+|\s+at\s+\$?[0-9]+"
)


# ─────────────────────────────────────────────
# CompanyRegistry — Bidirectional name ↔ symbol
# ─────────────────────────────────────────────


class CompanyRegistry:
    """
    Single source of truth for ticker ↔ company name mapping.

    Pre-loaded with S&P 500 from Wikipedia + manual aliases.
    Cached in PostgreSQL `company_registry` table for persistence.
    """

    def __init__(self):
        self._by_symbol: dict[str, Company] = {}  # "V" → Company
        self._by_name: dict[str, Company] = {}  # "visa" → Company
        self._by_alias: dict[str, Company] = {}  # "visa inc" → Company
        self._rejected: set[str] = set()  # symbols confirmed not real
        self._loaded = False

    def lookup_symbol(self, sym: str) -> Company | None:
        """Look up by ticker symbol (case-insensitive)."""
        return self._by_symbol.get(sym.upper())

    def lookup_name(self, name: str) -> Company | None:
        """Look up by company name or alias (case-insensitive)."""
        key = name.lower().strip()
        return self._by_name.get(key) or self._by_alias.get(key)

    def is_known(self, sym: str) -> bool:
        """Is this symbol in the registry?"""
        return sym.upper() in self._by_symbol

    def is_rejected(self, sym: str) -> bool:
        """Was this symbol confirmed as not a real stock?"""
        return sym.upper() in self._rejected

    def is_single_letter(self, sym: str) -> bool:
        """Single-letter tickers need extra context to confirm."""
        c = self._by_symbol.get(sym.upper())
        return c.single_letter if c else len(sym) == 1

    def add_company(self, company: Company):
        """Register a company in all lookup indexes."""
        self._by_symbol[company.symbol.upper()] = company
        self._by_name[company.name.lower()] = company
        for alias in company.aliases:
            self._by_alias[alias.lower()] = company

    def add_rejected(self, sym: str):
        """Mark a symbol as confirmed not a real stock."""
        self._rejected.add(sym.upper())

    @property
    def size(self) -> int:
        return len(self._by_symbol)

    def load(self):
        """Load registry from PostgreSQL cache or scrape fresh."""
        if self._loaded:
            return

        # Try loading from DB cache first
        count = self._load_from_db()
        if count >= 400:  # S&P 500 minus a few edge cases
            self._add_manual_aliases()
            self._add_nasdaq_extras()
            self._loaded = True
            logger.info(
                "[ticker_extractor] Registry loaded from DB: %d companies", count
            )
            return

        # Scrape S&P 500 from Wikipedia
        sp_count = self._scrape_sp500()
        # Add manual aliases
        self._add_manual_aliases()
        # Add NASDAQ extras not in S&P
        self._add_nasdaq_extras()
        # Save to DB
        self._save_to_db()
        self._loaded = True
        logger.info(
            "[ticker_extractor] Registry built: %d companies (%d from S&P 500)",
            self.size,
            sp_count,
        )

    def _load_from_db(self) -> int:
        """Load from company_registry table if it exists."""
        try:
            from app.db.connection import get_db

            with get_db() as db:
                # Check if table exists
                tables = [
                    r[0]
                    for r in db.execute(
                        "SELECT tablename FROM pg_catalog.pg_tables "
                        "WHERE schemaname = 'public'"
                    ).fetchall()
                ]
                if "company_registry" not in tables:
                    return 0

                rows = db.execute("""
                    SELECT symbol, company_name, aliases, sector, market_cap,
                           is_sp500, verified, rejected
                    FROM company_registry
                """).fetchall()

            for row in rows:
                sym, name, aliases_json, sector, mcap, is_sp, verified, rejected = row
                aliases = json.loads(aliases_json) if aliases_json else []
                if rejected:
                    self._rejected.add(sym.upper())
                    # Auto-ban: also add to runtime FALSE_TICKERS set
                    FALSE_TICKERS.add(sym.upper())
                    continue
                c = Company(
                    symbol=sym,
                    name=name,
                    aliases=aliases,
                    sector=sector or "",
                    market_cap=mcap or 0,
                    is_sp500=bool(is_sp),
                    single_letter=len(sym) == 1,
                )
                self.add_company(c)
            banned = len([s for s in self._rejected])
            if banned:
                logger.info(
                    "[ticker_extractor] Loaded %d banned symbols from DB into FALSE_TICKERS",
                    banned,
                )
            return len(self._by_symbol)
        except Exception as e:
            logger.warning(f"Failed to load registry from DB: {e}")
            return 0

    def _scrape_sp500(self) -> int:
        """Scrape S&P 500 list from Wikipedia."""
        import requests

        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            r = requests.get(url, timeout=15)
            if r.status_code != 200:
                logger.debug(
                    "[ticker_extractor] Wikipedia S&P 500 fetch failed: HTTP %d",
                    r.status_code,
                )
                return 0

            # Parse the first table (S&P 500 constituents)
            import pandas as pd

            tables = pd.read_html(r.text)
            if not tables:
                logger.debug(
                    "[ticker_extractor] No tables found on Wikipedia S&P 500 page"
                )
                return 0

            df = tables[0]
            count = 0
            for _, row in df.iterrows():
                symbol = str(row.get("Symbol", "")).strip().replace(".", "-")
                security = str(row.get("Security", "")).strip()
                sector = str(row.get("GICS Sector", "")).strip()

                if not symbol or symbol == "nan":
                    continue

                # Build aliases from company name
                aliases = [security.lower()]
                # Add short forms: "Apple Inc." → "apple"
                short = security.lower().replace(" inc.", "").replace(" inc", "")
                short = short.replace(" corp.", "").replace(" corp", "")
                short = short.replace(" co.", "").replace(" co", "")
                short = short.replace(" group", "").replace(" holdings", "")
                short = short.replace(" ltd.", "").replace(" ltd", "")
                short = short.replace(",", "").strip()
                if short != security.lower():
                    aliases.append(short)

                c = Company(
                    symbol=symbol,
                    name=security,
                    aliases=aliases,
                    sector=sector,
                    market_cap=0,
                    is_sp500=True,
                    single_letter=len(symbol) == 1,
                )
                self.add_company(c)
                count += 1

            return count
        except Exception as e:
            logger.debug("[ticker_extractor] S&P 500 scrape failed: %s", e)
            return 0

    def _add_manual_aliases(self):
        """Add common aliases not in the Wikipedia table."""
        manual = {
            "GOOGL": ["google", "alphabet"],
            "META": ["facebook", "fb", "meta platforms"],
            "TSLA": ["tesla"],
            "AMZN": ["amazon"],
            "MSFT": ["microsoft"],
            "AAPL": ["apple"],
            "NVDA": ["nvidia"],
            "JPM": ["jp morgan", "jpmorgan", "chase"],
            "BAC": ["bank of america", "bofa"],
            "GS": ["goldman sachs", "goldman"],
            "WFC": ["wells fargo"],
            "XOM": ["exxon", "exxon mobil", "exxonmobil"],
            "CVX": ["chevron"],
            "BA": ["boeing"],
            "LMT": ["lockheed", "lockheed martin"],
            "RTX": ["raytheon"],
            "CRM": ["salesforce"],
            "ADBE": ["adobe"],
            "ORCL": ["oracle"],
            "NOW": ["servicenow"],
            "PANW": ["palo alto", "palo alto networks"],
            "COIN": ["coinbase"],
            "HOOD": ["robinhood"],
            "UBER": ["uber"],
            "ABNB": ["airbnb"],
            "SNOW": ["snowflake"],
            "CRWD": ["crowdstrike"],
            "TSM": ["tsmc", "taiwan semi", "taiwan semiconductor"],
            "ARM": ["arm holdings"],
            "F": ["ford", "ford motor"],
            "C": ["citigroup", "citi"],
            "V": ["visa"],
            "T": ["at&t", "att"],
            "X": ["u.s. steel", "us steel", "united states steel"],
            "M": ["macy's", "macys"],
            "K": ["kellanova", "kellogg", "kellogg's"],
        }
        for sym, aliases in manual.items():
            existing = self._by_symbol.get(sym)
            if existing:
                for a in aliases:
                    if a.lower() not in self._by_alias:
                        existing.aliases.append(a)
                        self._by_alias[a.lower()] = existing
            else:
                # Not in S&P 500, add as new entry
                c = Company(
                    symbol=sym,
                    name=aliases[0].title(),
                    aliases=aliases,
                    sector="",
                    market_cap=0,
                    is_sp500=True,
                    single_letter=len(sym) == 1,
                )
                self.add_company(c)

    def _add_nasdaq_extras(self):
        """Add major NASDAQ stocks not in S&P 500."""
        extras = {
            "SMCI": "Super Micro Computer",
            "MRVL": "Marvell Technology",
            "PLTR": "Palantir Technologies",
            "SOFI": "SoFi Technologies",
            "DDOG": "Datadog",
            "ZS": "Zscaler",
            "FTNT": "Fortinet",
            "TTD": "The Trade Desk",
            "SNAP": "Snap Inc",
            "PINS": "Pinterest",
            "ROKU": "Roku",
            "SQ": "Block Inc",
            "SHOP": "Shopify",
            "RBLX": "Roblox",
            "U": "Unity Software",
            "RIVN": "Rivian",
            "LCID": "Lucid Motors",
            "NIO": "NIO Inc",
        }
        for sym, name in extras.items():
            if sym not in self._by_symbol:
                c = Company(
                    symbol=sym,
                    name=name,
                    aliases=[name.lower()],
                    sector="",
                    market_cap=0,
                    is_sp500=False,
                    single_letter=len(sym) == 1,
                )
                self.add_company(c)

    def _save_to_db(self):
        """Persist registry to PostgreSQL."""
        try:
            from app.db.connection import get_db

            with get_db() as db:
                # Clear and re-insert
                db.execute("DELETE FROM company_registry")
                for c in self._by_symbol.values():
                    db.execute(
                        """
                        INSERT INTO company_registry
                            (symbol, company_name, aliases, sector, market_cap, is_sp500, verified, source)
                        VALUES (%s, %s, %s, %s, %s, %s, TRUE, 'sp500_load')
                        ON CONFLICT (symbol) DO NOTHING
                    """,
                        [
                            c.symbol,
                            c.name,
                            json.dumps(c.aliases),
                            c.sector,
                            c.market_cap,
                            c.is_sp500,
                        ],
                    )
                # Also save rejected
                for sym in self._rejected:
                    db.execute(
                        """
                        INSERT INTO company_registry
                            (symbol, company_name, aliases, rejected, source)
                        VALUES (%s, %s, '[]', TRUE, 'yfinance_reject')
                        ON CONFLICT (symbol) DO NOTHING
                    """,
                        [sym, f"REJECTED_{sym}"],
                    )
            logger.info(
                "[ticker_extractor] Saved %d companies + %d rejected to DB",
                self.size,
                len(self._rejected),
            )
        except Exception as e:
            logger.warning(f"Failed to save registry to DB: {e}")


# ─────────────────────────────────────────────
# Module-level registry singleton
# ─────────────────────────────────────────────

_registry = CompanyRegistry()


def get_registry() -> CompanyRegistry:
    """Get the global CompanyRegistry, loading if needed."""
    if not _registry._loaded:
        _registry.load()
    return _registry


# ─────────────────────────────────────────────
# Regex patterns
# ─────────────────────────────────────────────

DOLLAR_TICKER = re.compile(r"\$([A-Z]{1,5})\b")  # $NVDA, $V
BARE_CAPS = re.compile(r"\b([A-Z]{2,5})\b")  # NVDA, ARM
SINGLE_LETTER = re.compile(r"\b([A-Z])\b")  # V, X, T, C


# ─────────────────────────────────────────────
# Layer 1+2: Extract + Score
# ─────────────────────────────────────────────


def extract_tickers(
    text: str,
    title: str | None = None,
    source: str = "unknown",
) -> list[TickerMatch]:
    """
    Extract stock tickers from text with confidence scoring.

    Runs Layers 1+2:
      - Layer 1: Regex + registry matching
      - Layer 2: Exclusion filtering + context boosting

    Does NOT call yfinance or LLM. Those are separate functions.
    Returns list of TickerMatch sorted by confidence descending.
    """
    registry = get_registry()
    candidates: dict[str, TickerMatch] = {}  # symbol → best match
    full_text = f"{title or ''} {text}"
    text_lower = full_text.lower()

    # ── Layer 1a: $TICKER pattern (highest confidence) ──
    # $TICKER always signals explicit financial intent, so we only block
    # truly impossible symbols (I, A, etc.) not ambiguous high-cap ones
    for match in DOLLAR_TICKER.finditer(full_text):
        sym = match.group(1).upper()
        if sym in {"I", "A", "IT", "IS", "IF", "AM", "PM", "BY", "MY", "US", "UK"}:
            continue
        company = registry.lookup_symbol(sym)
        name = company.name if company else sym
        snippet = full_text[max(0, match.start() - 30) : match.end() + 30]
        candidates[sym] = TickerMatch(
            symbol=sym,
            company_name=name,
            confidence=0.95,
            source="$pattern",
            context_snippet=snippet.strip(),
        )

    # ── Layer 1b: Company name matching ──
    for name_key, company in registry._by_name.items():
        if name_key in text_lower:
            sym = company.symbol
            if sym not in candidates or candidates[sym].confidence < 0.90:
                idx = text_lower.index(name_key)
                snippet = full_text[max(0, idx - 20) : idx + len(name_key) + 20]
                candidates[sym] = TickerMatch(
                    symbol=sym,
                    company_name=company.name,
                    confidence=0.90,
                    source="company_name",
                    context_snippet=snippet.strip(),
                )

    for alias_key, company in registry._by_alias.items():
        if len(alias_key) > 2 and alias_key in text_lower:
            sym = company.symbol
            if sym not in candidates or candidates[sym].confidence < 0.90:
                idx = text_lower.index(alias_key)
                snippet = full_text[max(0, idx - 20) : idx + len(alias_key) + 20]
                candidates[sym] = TickerMatch(
                    symbol=sym,
                    company_name=company.name,
                    confidence=0.90,
                    source="company_name",
                    context_snippet=snippet.strip(),
                )

    # ── Layer 1c: Bare caps in registry ──
    for match in BARE_CAPS.finditer(full_text):
        sym = match.group(1).upper()
        if sym in candidates:
            continue
        if registry.is_rejected(sym):
            continue
        if _is_hard_blocked(sym, registry):
            continue
        company = registry.lookup_symbol(sym)
        if company:
            # If it's in FALSE_TICKERS but passed the market-cap override,
            # use lower base confidence — context scoring will boost if legit
            if sym in FALSE_TICKERS:
                base = 0.50  # ambiguous — needs context boost to survive
            elif company.single_letter:
                base = 0.40
            else:
                base = 0.85
            snippet = full_text[max(0, match.start() - 30) : match.end() + 30]
            candidates[sym] = TickerMatch(
                symbol=sym,
                company_name=company.name,
                confidence=base,
                source="registry",
                context_snippet=snippet.strip(),
            )

    # ── Layer 1d: Bare caps NOT in registry (low confidence) ──
    for match in BARE_CAPS.finditer(full_text):
        sym = match.group(1).upper()
        if _is_hard_blocked(sym, registry) or sym in candidates:
            continue
        if registry.is_rejected(sym):
            continue
        if not registry.is_known(sym) and len(sym) >= 2:
            snippet = full_text[max(0, match.start() - 30) : match.end() + 30]
            candidates[sym] = TickerMatch(
                symbol=sym,
                company_name=sym,
                confidence=0.30,
                source="bare_caps",
                context_snippet=snippet.strip(),
            )

    # ── Layer 2: Context boosting ──
    for sym, tm in candidates.items():
        # Skip already high-confidence matches
        if tm.confidence >= 0.90:
            continue

        boost = 0.0

        # ── Anti-Pattern Check (Penalty for common English usage) ──
        # Must run BEFORE positive boosts so that "on TV" in a culture
        # article doesn't accidentally get boosted by coincidental
        # financial words like "market" or "growth" elsewhere in the text.
        anti_penalty = _check_anti_patterns(sym, full_text)
        if anti_penalty < 0:
            logger.debug(
                "[ticker_extractor] %s: anti-pattern penalty %.2f",
                sym,
                anti_penalty,
            )

        # Direct Syntax Check (Massive boost for explicit financial framing)
        # e.g., "bought ON", "ON +5%", "(ON)"
        direct_syntax_matches = [
            # Verbs before: "bought ON"
            rf"{DIRECT_SYNTAX_VERBS}{re.escape(sym)}\b",
            # Price action after: "ON +5%" or "ON dropped to $10"
            rf"\b{re.escape(sym)}{DIRECT_SYNTAX_PRICE}",
            # Parenthetical: "(ON)" or "[ON]"
            rf"[\(\[]{re.escape(sym)}[\)\]]",
            # Ticker prefix: "ticker ON"
            rf"(?i)\bticker\s+{re.escape(sym)}\b",
        ]

        has_direct_syntax = any(
            re.search(pattern, full_text) for pattern in direct_syntax_matches
        )

        if has_direct_syntax:
            boost = max(boost, 0.40)

        # Financial context words nearby
        # Check within 100 chars of any mention
        for m in re.finditer(rf"\b{re.escape(sym)}\b", full_text):
            # Safe slice window of 100 characters before and after
            start_idx = max(0, m.start() - 100)
            end_idx = min(len(full_text), m.end() + 100)
            window = full_text[start_idx:end_idx].lower()

            # Avoid counting the symbol itself if it overlaps with a context word
            context_hits = sum(
                1 for kw in FINANCIAL_CONTEXT if re.search(rf"\b{kw}\b", window)
            )
            if context_hits >= 1:
                boost = max(boost, 0.20)
            if context_hits >= 3:
                boost = max(boost, 0.30)

        # Frequency bonus: mentioned 3+ times
        count = len(re.findall(rf"\b{re.escape(sym)}\b", full_text))
        if count >= 3:
            boost += 0.15
        elif count >= 2:
            boost += 0.05

        # Title bonus
        if title and sym in title.upper():
            boost += 0.10

        # Apply anti-pattern penalty AFTER positive boosts.
        # If direct financial syntax was found (e.g., "$TV +5%"),
        # the anti-pattern penalty is suppressed — explicit financial
        # framing overrides common English usage.
        if has_direct_syntax:
            anti_penalty = 0.0  # Direct syntax = definitely a ticker

        tm.confidence = min(1.0, tm.confidence + boost + anti_penalty)

    # ── Filter: reject < 0.40 and excluded ──
    results = [tm for tm in candidates.values() if tm.confidence >= 0.40]

    # Sort by confidence descending
    results.sort(key=lambda t: t.confidence, reverse=True)

    return results


# ─────────────────────────────────────────────
# Layer 3: yfinance Validation
# ─────────────────────────────────────────────


async def validate_unknown_tickers(tickers: list[str]) -> dict[str, bool]:
    """
    Validate unknown ticker symbols via yfinance.

    For each symbol:
      - If yfinance returns a marketCap → verified (True)
      - If yfinance returns empty/error → rejected (False)

    Results are cached in the CompanyRegistry and PostgreSQL.
    Only call this for symbols NOT already in the registry.
    """
    import yfinance as yf
    import asyncio

    registry = get_registry()
    results: dict[str, bool] = {}

    for sym in tickers:
        sym = sym.upper()

        # Skip if already known
        if registry.is_known(sym):
            results[sym] = True
            continue
        if registry.is_rejected(sym):
            results[sym] = False
            continue

        # ── Crypto bypass: known crypto tickers skip yfinance entirely ──
        # yfinance returns quoteType=CRYPTOCURRENCY for these, which would
        # trigger the TRADEABLE_TYPES gate below. But crypto tickers are
        # valid assets handled by CoinGecko (asset_prices table).
        from app.config.config_tickers import CRYPTO_TICKERS as _CRYPTO_SET

        if sym in _CRYPTO_SET:
            results[sym] = True
            logger.info(
                "[ticker_extractor] %s: known crypto ticker, skipping yfinance validation",
                sym,
            )
            continue

        try:

            def _fetch_yf_data():
                ticker_obj = yf.Ticker(sym)
                _info = ticker_obj.info
                _df = None
                _quote_type = _info.get("quoteType", "")
                if _quote_type in {"EQUITY", "ETF"}:
                    try:
                        _df = ticker_obj.history(period="5d")
                    except Exception:
                        pass
                return _info, _df

            info, df = await asyncio.wait_for(
                asyncio.to_thread(_fetch_yf_data),
                timeout=20.0
            )

            mcap = info.get("marketCap", 0) or 0
            name = info.get("shortName") or info.get("longName") or sym
            sector = info.get("sector", "")
            quote_type = info.get("quoteType", "")

            # Gate 1: Reject non-tradeable instrument types
            # EQUITY = common stock, ETF = exchange-traded fund (tradeable)
            # Everything else (MUTUALFUND, INDEX, CURRENCY, FUTURE, OPTION,
            # CRYPTOCURRENCY, BOND) is not a stock we should analyze.
            TRADEABLE_TYPES = {"EQUITY", "ETF"}
            if quote_type and quote_type not in TRADEABLE_TYPES:
                registry.add_rejected(sym)
                FALSE_TICKERS.add(sym)
                results[sym] = False
                logger.info(
                    "[ticker_extractor] yfinance rejected: %s (quoteType=%s, not tradeable)",
                    sym,
                    quote_type,
                )
                _save_rejected_to_db(sym)
                continue

            # Gate 2: Verify actual price data exists (catches delisted instruments)
            # Some instruments have metadata but zero OHLCV rows (e.g. matured bonds)
            if quote_type in TRADEABLE_TYPES and (df is None or df.empty):
                registry.add_rejected(sym)
                FALSE_TICKERS.add(sym)
                results[sym] = False
                logger.info(
                    "[ticker_extractor] yfinance rejected: %s (no price data — likely delisted)",
                    sym,
                )
                _save_rejected_to_db(sym)
                continue

            if mcap > 0:
                # Real stock — add to registry
                c = Company(
                    symbol=sym,
                    name=name,
                    aliases=[name.lower()] if name != sym else [],
                    sector=sector,
                    market_cap=mcap,
                    is_sp500=False,
                    single_letter=len(sym) == 1,
                )
                registry.add_company(c)
                results[sym] = True
                logger.info(
                    "[ticker_extractor] yfinance verified: %s = %s ($%.1fB, type=%s)",
                    sym,
                    name,
                    mcap / 1e9,
                    quote_type,
                )

                # Save to DB
                _save_verified_to_db(c)
            else:
                # Not a real stock
                registry.add_rejected(sym)
                # Auto-ban: add to runtime FALSE_TICKERS so it's blocked immediately
                FALSE_TICKERS.add(sym)
                results[sym] = False
                logger.info(
                    "[ticker_extractor] yfinance rejected: %s (no market cap, added to ban list)",
                    sym,
                )
                _save_rejected_to_db(sym)

        except Exception as e:
            logger.warning(f"yfinance lookup failed for {sym}: {e}")
            results[sym] = False  # Assume not real on error

        await asyncio.sleep(0.3)  # Rate limit yfinance while yielding event loop

    return results


def _save_verified_to_db(company: Company):
    """Save a yfinance-verified company to the DB cache."""
    try:
        from app.db.connection import get_db

        with get_db() as db:
            db.execute(
                """
                INSERT INTO company_registry
                    (symbol, company_name, aliases, sector, market_cap, is_sp500, verified, rejected, source)
                VALUES (%s, %s, %s, %s, %s, FALSE, TRUE, FALSE, 'yfinance')
                ON CONFLICT (symbol) DO NOTHING
            """,
                [
                    company.symbol,
                    company.name,
                    json.dumps(company.aliases),
                    company.sector,
                    company.market_cap,
                ],
            )
    except Exception as e:
        logger.warning(f"Failed to save verified {company.symbol} to DB: {e}")


def _save_rejected_to_db(sym: str):
    """Save a rejected symbol to the DB cache."""
    try:
        from app.db.connection import get_db

        with get_db() as db:
            db.execute(
                """
                INSERT INTO company_registry
                    (symbol, company_name, aliases, rejected, source)
                VALUES (%s, %s, '[]', TRUE, 'yfinance_reject')
                ON CONFLICT (symbol) DO NOTHING
            """,
                [sym, f"REJECTED_{sym}"],
            )
    except Exception as e:
        logger.warning(f"Failed to save rejected {sym} to DB: {e}")


# ─────────────────────────────────────────────
# High-level convenience functions
# ─────────────────────────────────────────────


def extract_and_validate(
    text: str,
    title: str | None = None,
    source: str = "unknown",
) -> list[TickerMatch]:
    """
    Full pipeline: extract tickers.
    NOTE: Validation step is now managed separately since it requires async.

    Args:
        text: Body text to extract from
        title: Optional title/headline (gets extra boost)
        source: Origin identifier for debugging

    Returns: List of TickerMatch with confidence >= 0.40, sorted descending.
    """
    matches = extract_tickers(text, title=title, source=source)

    # Final filter: only return >= 0.40
    return [m for m in matches if m.confidence >= 0.40]


def get_ticker_symbols(text: str, title: str | None = None) -> list[str]:
    """Simple wrapper: returns just the symbol strings (no metadata)."""
    matches = extract_tickers(text, title=title)
    return [m.symbol for m in matches if m.confidence >= 0.60]
