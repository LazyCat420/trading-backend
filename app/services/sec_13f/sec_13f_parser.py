"""SEC 13F Parser - handles XML/HTML parsing and quarter derivation."""

import logging
import re
import warnings
from datetime import datetime
from typing import Any

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

logger = logging.getLogger(__name__)

# Suppress XML-parsed-as-HTML warnings from BeautifulSoup
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"

# 13F filing schedule: filed ~45 days after quarter end
_QUARTER_FILING_DEADLINES = {
    1: (5, 15),  # Q1 holdings → due May 15
    2: (8, 14),  # Q2 holdings → due Aug 14
    3: (11, 14),  # Q3 holdings → due Nov 14
    4: (2, 14),  # Q4 holdings → due Feb 14 (of next year)
}


class SEC13FParser:
    """Parses SEC 13F EDGAR indices, info tables, and handles quarter derivation."""

    def __init__(self) -> None:
        self.stats_amended_skipped = 0
        self.stats_parsed_filings = 0

    def next_filing_date(self, current_quarter: str) -> str:
        """Calculate the next expected 13F filing date based on the quarter."""
        try:
            year = int(current_quarter[:4])
            q = int(current_quarter[-1])
        except (ValueError, IndexError):
            # Fallback: 45 days from now
            from datetime import timedelta

            return (datetime.now() + timedelta(days=45)).strftime("%Y-%m-%d")

        # The NEXT quarter after current_quarter
        next_q = (q % 4) + 1
        next_year = year + 1 if next_q <= q else year
        if q == 4:
            next_year = year + 1

        month, day = _QUARTER_FILING_DEADLINES[next_q]
        if next_q == 4:
            filing_year = next_year + 1
        else:
            filing_year = next_year

        return f"{filing_year}-{month:02d}-{day:02d}"

    def find_latest_13f(
        self,
        submissions: dict[str, Any],
        cik: str,
    ) -> dict[str, Any] | None:
        """Find the most recent 13F-HR filing from submissions."""
        filings = self.find_13f_filings(submissions, cik, max_filings=1)
        return filings[0] if filings else None

    def find_13f_filings(
        self,
        submissions: dict[str, Any],
        cik: str,
        max_filings: int = 8,
    ) -> list[dict[str, Any]]:
        """Find up to `max_filings` 13F-HR filings from submissions."""
        recent = submissions.get("filings", {}).get("recent", {})
        if not recent:
            return []

        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        accession_numbers = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        results: list[dict[str, Any]] = []
        seen_quarters: set[str] = set()

        for i in range(len(forms)):
            if forms[i] not in ("13F-HR", "13F-HR/A"):
                continue
            if i >= len(filing_dates) or i >= len(accession_numbers):
                continue

            filing_date_str = filing_dates[i]
            accession = accession_numbers[i]
            primary_doc = primary_docs[i] if i < len(primary_docs) else ""

            try:
                dt = datetime.strptime(filing_date_str, "%Y-%m-%d")
                if dt.month <= 3:
                    q_year, q_num = dt.year - 1, 4
                elif dt.month <= 6:
                    q_year, q_num = dt.year, 1
                elif dt.month <= 9:
                    q_year, q_num = dt.year, 2
                else:
                    q_year, q_num = dt.year, 3
            except ValueError:
                continue

            quarter = f"{q_year}Q{q_num}"
            if quarter in seen_quarters:
                self.stats_amended_skipped += 1
                continue  # Skip amendments for same quarter
            seen_quarters.add(quarter)

            file_accession = accession.replace("-", "")
            stripped_cik = cik.lstrip("0")

            results.append(
                {
                    "accession": accession,
                    "filing_date": filing_date_str,
                    "quarter": quarter,
                    "primary_doc": primary_doc,
                    "index_url": (
                        f"{SEC_ARCHIVES_URL}/{stripped_cik}/"
                        f"{file_accession}/"
                        f"{accession}-index.htm"
                    ),
                    "filing_url": (
                        f"{SEC_ARCHIVES_URL}/{stripped_cik}/"
                        f"{file_accession}/{primary_doc}"
                    ),
                    "cik": stripped_cik,
                    "file_accession": file_accession,
                }
            )
            self.stats_parsed_filings += 1

            if len(results) >= max_filings:
                break

        return results

    def get_info_table_url_from_json(
        self, json_data: dict[str, Any], cik: str, file_accession: str
    ) -> str | None:
        """Find the info table URL from index.json API."""
        try:
            items = json_data.get("directory", {}).get("item", [])
            xml_files = [
                item["name"]
                for item in items
                if item.get("name", "").endswith(".xml")
                and item.get("name") != "primary_doc.xml"
            ]
            if xml_files:
                stripped_cik = cik.lstrip("0")
                return (
                    f"{SEC_ARCHIVES_URL}/{stripped_cik}/{file_accession}/{xml_files[0]}"
                )
        except Exception:
            pass
        return None

    def get_info_table_url_from_html(
        self, html_text: str, cik: str, file_accession: str
    ) -> str | None:
        """Find the info table URL via HTML fallback scrape."""
        try:
            soup = BeautifulSoup(html_text, "lxml")
            stripped_cik = cik.lstrip("0")
            for link in soup.find_all("a", href=True):
                href = link["href"]
                text = link.get_text(strip=True).lower()
                if any(
                    x in text or x in href.lower()
                    for x in (
                        "infotable",
                        "information table",
                        "informationtable",
                    )
                ):
                    if href.startswith("/"):
                        return f"https://www.sec.gov{href}"
                    elif href.startswith("http"):
                        return href
                    else:
                        return (
                            f"{SEC_ARCHIVES_URL}/{stripped_cik}/{file_accession}/{href}"
                        )

            # Fallback: grab any XML that isn't primary_doc.xml
            for link in soup.find_all("a", href=True):
                href = link["href"]
                fname = href.split("/")[-1] if "/" in href else href
                if (
                    fname.endswith(".xml")
                    and fname != "primary_doc.xml"
                    and not fname.startswith("R")
                ):
                    if href.startswith("/"):
                        return f"https://www.sec.gov{href}"
                    elif href.startswith("http"):
                        return href
                    else:
                        return (
                            f"{SEC_ARCHIVES_URL}/{stripped_cik}/"
                            f"{file_accession}/{fname}"
                        )
        except Exception:
            pass
        return None

    def parse_info_table(self, content: str) -> list[dict[str, Any]]:
        """Parse a 13F information table (XML) into holdings dicts."""
        holdings: list[dict[str, Any]] = []

        try:
            soup = BeautifulSoup(content, "xml")
            info_entries = soup.find_all("infoTable")
            if info_entries:
                for entry in info_entries:
                    holding = self._parse_xml_entry(entry)
                    if holding:
                        holdings.append(holding)
                if holdings:
                    return holdings
        except Exception as e:
            logger.debug("[SEC 13F] XML parser failed: %s", e)

        soup = BeautifulSoup(content, "lxml")
        info_entries = soup.find_all(re.compile(r"infotable", re.IGNORECASE))
        if info_entries:
            for entry in info_entries:
                holding = self._parse_xml_entry(entry)
                if holding:
                    holdings.append(holding)
            return holdings

        rows = soup.find_all("tr")
        for row in rows:
            cells = row.find_all("td")
            if len(cells) >= 4:
                holding = self._parse_html_row(cells)
                if holding:
                    holdings.append(holding)

        return holdings

    def _parse_xml_entry(self, entry: Any) -> dict[str, Any] | None:
        """Parse a single <infoTable> XML entry."""

        def _get_text(tag_name: str) -> str:
            tag = entry.find(re.compile(tag_name, re.IGNORECASE))
            return tag.get_text(strip=True) if tag else ""

        name = _get_text("nameofissuer")
        cusip = _get_text("cusip")
        value_str = _get_text("value")
        shares_str = _get_text(r"sshprnamt$")
        share_type = _get_text("sshprnamttype")
        title = _get_text("titleofclass")

        try:
            value_usd = float(value_str.replace(",", "")) if value_str else 0
        except ValueError:
            value_usd = 0

        try:
            shares = int(shares_str.replace(",", "")) if shares_str else 0
        except ValueError:
            shares = 0

        if not name:
            return None

        return {
            "name_of_issuer": name,
            "cusip": cusip,
            "value_usd": value_usd,
            "shares": shares,
            "share_type": share_type or "SH",
            "title_of_class": title,  # Use as part of resolution later
            "ticker": "",  # To be resolved later
        }

    def _parse_html_row(self, cells: list[Any]) -> dict[str, Any] | None:
        """Parse a holdings row from an HTML table."""
        try:
            name = cells[0].get_text(strip=True)
            title = cells[1].get_text(strip=True) if len(cells) > 1 else ""
            cusip = cells[2].get_text(strip=True) if len(cells) > 2 else ""
            value_str = cells[3].get_text(strip=True) if len(cells) > 3 else "0"
            shares_str = cells[4].get_text(strip=True) if len(cells) > 4 else "0"
            share_type = cells[5].get_text(strip=True) if len(cells) > 5 else "SH"

            value_usd = float(value_str.replace(",", "")) if value_str else 0
            shares = int(shares_str.replace(",", "")) if shares_str else 0

            return {
                "name_of_issuer": name,
                "cusip": cusip,
                "value_usd": value_usd,
                "shares": shares,
                "share_type": share_type,
                "title_of_class": title,
                "ticker": "",
            }
        except (ValueError, IndexError):
            return None
