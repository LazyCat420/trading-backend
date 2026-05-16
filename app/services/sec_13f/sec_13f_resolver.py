"""SEC 13F Resolver - handles heuristic CUSIP/Name to Ticker resolution."""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Well-known CUSIP -> ticker mapping (top holdings)
CUSIP_MAP: dict[str, str] = {
    "594918104": "MSFT",
    "037833100": "AAPL",
    "02079K305": "GOOG",
    "02079K107": "GOOGL",
    "023135106": "AMZN",
    "67066G104": "NVDA",
    "30303M102": "META",
    "88160R101": "TSLA",
    "46625H100": "JPM",
    "92826C839": "V",
    "91324P102": "UNH",
    "17275R102": "CSCO",
    "478160104": "JNJ",
    "00724F101": "ADBE",
    "532457108": "LLY",
    "742718109": "PG",
    "931142103": "WMT",
    "58933Y105": "MRK",
    "20030N101": "CMCSA",
    "87612E106": "TGT",
    "22160K105": "COST",
    "31428X106": "FDX",
    "254687106": "DIS",
    "260557103": "DOW",
    "111320107": "BA",
    "09247X101": "BLK",
    "02005N100": "ALLY",
    "172967424": "C",
    "084670702": "BRK-B",
    "78462F103": "SPY",
    "464287655": "IWM",
    "808513105": "SCHW",
    "369604103": "GE",
    "459200101": "IBM",
    "31620M106": "FANG",
    "48203R104": "JNPR",
    "585055106": "MDT",
    "571903202": "MA",
    "00206R102": "T",
    "92343V104": "VZ",
    "12504L109": "CSIQ",
    "002824100": "ABT",
    "026874784": "AIG",
    "00287Y109": "ABBV",
    "718172109": "PFE",
    "68389X105": "ORCL",
    "11135F101": "CRM",
    "64110L106": "NFLX",
    "007903107": "AMD",
    "458140100": "INTC",
    "747525103": "QCOM",
    "70450Y103": "PYPL",
    "191216100": "KO",
    "713448108": "PEP",
    "166764100": "CVX",
    "30231G102": "XOM",
}

# Try to extract from issuer name (e.g., "APPLE INC" -> search via heuristics)
NAME_MAP: dict[str, str] = {
    "APPLE": "AAPL",
    "MICROSOFT": "MSFT",
    "AMAZON": "AMZN",
    "ALPHABET": "GOOGL",
    "GOOGLE": "GOOGL",
    "META PLATFORMS": "META",
    "FACEBOOK": "META",
    "NVIDIA": "NVDA",
    "TESLA": "TSLA",
    "BERKSHIRE": "BRK-B",
    "JPMORGAN": "JPM",
    "JOHNSON": "JNJ",
    "UNITEDHEALTH": "UNH",
    "VISA": "V",
    "PROCTER": "PG",
    "ELI LILLY": "LLY",
    "MASTERCARD": "MA",
    "WALMART": "WMT",
    "BROADCOM": "AVGO",
    "COSTCO": "COST",
    "CISCO": "CSCO",
    "ABBVIE": "ABBV",
    "PFIZER": "PFE",
    "ORACLE": "ORCL",
    "SALESFORCE": "CRM",
    "NETFLIX": "NFLX",
    "ADOBE": "ADBE",
    "AMD": "AMD",
    "INTEL": "INTC",
    "QUALCOMM": "QCOM",
    "PAYPAL": "PYPL",
    "BOEING": "BA",
    "DISNEY": "DIS",
    "COCA-COLA": "KO",
    "PEPSICO": "PEP",
    "MERCK": "MRK",
    "CHEVRON": "CVX",
    "EXXON": "XOM",
    "ALLY": "ALLY",
    "GENERAL ELECTRIC": "GE",
    "GENERAL MOTORS": "GM",
    "CITIGROUP": "C",
    "BANK OF AMERICA": "BAC",
    "WELLS FARGO": "WFC",
    "GOLDMAN": "GS",
    "MORGAN STANLEY": "MS",
    "COCA COLA": "KO",
    "HOME DEPOT": "HD",
    "MCDONALD": "MCD",
    "NIKE": "NKE",
    "STARBUCKS": "SBUX",
    "UBER": "UBER",
    "AIRBNB": "ABNB",
    "SNOWFLAKE": "SNOW",
    "PALANTIR": "PLTR",
    "CROWDSTRIKE": "CRWD",
    "DATADOG": "DDOG",
    "SERVICENOW": "NOW",
    "SHOPIFY": "SHOP",
    "ADVANCED MICRO": "AMD",
    "TAIWAN SEMI": "TSM",
    "ASML": "ASML",
    "CATERPILLAR": "CAT",
    "DEERE": "DE",
    "LOCKHEED": "LMT",
    "RAYTHEON": "RTX",
    "AMERICAN EXPRESS": "AXP",
    "CAPITAL ONE": "COF",
    "T-MOBILE": "TMUS",
    "VERIZON": "VZ",
    "AT&T": "T",
    "COMCAST": "CMCSA",
    "TARGET": "TGT",
    "FEDEX": "FDX",
    "SCHWAB": "SCHW",
    "BLACKROCK": "BLK",
}


class SEC13FResolver:
    """Best-effort CUSIP/name -> ticker symbol resolution."""

    def __init__(self) -> None:
        self._cusip_cache: dict[str, str] = {}

    def resolve_ticker(self, cusip: str, name: str, title: str) -> str:
        """Strategy:
        1. Check memory cache
        2. Check hardcoded CUSIP map (most common large-cap)
        3. Check company name map (well-known names)
        """
        clean_cusip = cusip.strip() if cusip else ""
        upper_name = name.upper() if name else ""

        cache_key = f"{clean_cusip}_{upper_name}"
        if cache_key in self._cusip_cache:
            return self._cusip_cache[cache_key]

        if clean_cusip and clean_cusip in CUSIP_MAP:
            self._cusip_cache[cache_key] = CUSIP_MAP[clean_cusip]
            return CUSIP_MAP[clean_cusip]

        for pattern, tick in NAME_MAP.items():
            if pattern in upper_name:
                self._cusip_cache[cache_key] = tick
                return tick

        self._cusip_cache[cache_key] = ""
        return ""

    def process_holdings_and_report(
        self, holdings: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], int]:
        """Resolves tickers for a list of holdings.
        Returns: (Valid Holdings, int count_of_empty_tickers)
        """
        valid: list[dict[str, Any]] = []
        empty_count = 0

        for h in holdings:
            cusip = h.get("cusip", "")
            name = h.get("name_of_issuer", "")
            title = h.get("title_of_class", "")

            # Only resolve if it doesn't already have one
            ticker = h.get("ticker", "")
            if not ticker:
                ticker = self.resolve_ticker(cusip, name, title)

            if ticker:
                h["ticker"] = ticker
                valid.append(h)
            else:
                empty_count += 1

        return valid, empty_count
