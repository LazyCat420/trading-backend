"""SEC 13F Repository - handles all database operations."""

import logging
from typing import Any

from app.db.connection import get_db

logger = logging.getLogger(__name__)

# Constants shared by default seeding
DEFAULT_FILERS: list[dict[str, str]] = [
    {"cik": "0001067983", "name": "Berkshire Hathaway"},
    {"cik": "0001423053", "name": "Citadel Advisors"},
    {"cik": "0001037389", "name": "Renaissance Technologies"},
    {"cik": "0001350694", "name": "Bridgewater Associates"},
    {"cik": "0001336528", "name": "Pershing Square Capital"},
    {"cik": "0001791786", "name": "Elliott Investment Management"},
    {"cik": "0001179392", "name": "Two Sigma Investments"},
    {"cik": "0001603466", "name": "Point72 Asset Management"},
    {"cik": "0001009207", "name": "DE Shaw & Co"},
    {"cik": "0001167557", "name": "AQR Capital Management"},
    {"cik": "0001273087", "name": "Millennium Management"},
    {"cik": "0001103804", "name": "Viking Global Investors"},
    {"cik": "0001536411", "name": "Druckenmiller (Duquesne Family Office)"},
    {"cik": "0001135730", "name": "Coatue Management"},
    {"cik": "0001167483", "name": "Tiger Global Management"},
]


class SEC13FRepository:
    """Manages database CRUD operations for SEC 13F filers and holdings."""

    def __init__(self) -> None:
        pass

    def ensure_filers(self) -> None:
        """Seed default filers into DB if not present."""
        with get_db() as db:
            for filer in DEFAULT_FILERS:
                try:
                    db.execute(
                        """
                        INSERT INTO sec_13f_filers (cik, filer_name)
                        VALUES (%s, %s)
                        ON CONFLICT (cik) DO NOTHING
                        """,
                        [filer["cik"], filer["name"]],
                    )
                except Exception:
                    pass  # Already exists

    def check_collected_today(self) -> int:
        """Return number of holdings collected today."""
        with get_db() as db:
            row = db.execute(
                "SELECT COUNT(*) FROM sec_13f_holdings WHERE collected_at >= CURRENT_DATE"
            ).fetchone()
            return row[0] if row else 0

    def get_active_filers(self) -> list[tuple[str, str]]:
        """Get list of active (cik, filer_name) tuples."""
        with get_db() as db:
            return db.execute(
                "SELECT cik, filer_name FROM sec_13f_filers WHERE is_active = TRUE"
            ).fetchall()

    def get_filer_schedule(self, cik: str) -> tuple[Any, Any]:
        """Return (next_expected_filing, latest_quarter) for a filer."""
        with get_db() as db:
            sched_row = db.execute(
                "SELECT next_expected_filing, latest_quarter "
                "FROM sec_13f_filers WHERE cik = %s",
                [cik],
            ).fetchone()
            if sched_row:
                return sched_row[0], sched_row[1]
            return None, None

    def update_filer_last_checked(self, cik: str) -> None:
        """Mark filer as checked just now."""
        with get_db() as db:
            db.execute(
                "UPDATE sec_13f_filers SET last_checked = CURRENT_TIMESTAMP WHERE cik = %s",
                [cik],
            )

    def update_filer_schedule(self, cik: str, quarter: str, next_filing: str) -> None:
        """Update last checking time, latest quarter, and next expected filing."""
        with get_db() as db:
            db.execute(
                "UPDATE sec_13f_filers "
                "SET last_checked = CURRENT_TIMESTAMP, "
                "    latest_quarter = %s, "
                "    next_expected_filing = %s "
                "WHERE cik = %s",
                [quarter, next_filing, cik],
            )

    def check_quarter_exists(self, cik: str, quarter: str) -> int:
        """Return count of holdings for a given CIK and quarter."""
        with get_db() as db:
            existing = db.execute(
                "SELECT COUNT(*) FROM sec_13f_holdings WHERE cik = %s AND filing_quarter = %s",
                [cik, quarter],
            ).fetchone()
            return existing[0] if existing else 0

    def save_holdings(
        self, cik: str, holdings: list[dict[str, Any]], quarter: str, filing_date: str
    ) -> tuple[int, int]:
        """Save holdings to DB, capping to prevent massive records.
        Returns (saved_count, capped_count).
        """
        with get_db() as db:
            # Filter out holdings without tickers (already filtered mostly, but sanity check)
            valid_holdings = [h for h in holdings if h.get("ticker")]

            # Cap holdings per filer
            capped_count = 0
            max_holdings = 500  # Hardcoded max or use settings
            if len(valid_holdings) > max_holdings:
                capped_count = len(valid_holdings) - max_holdings
                valid_holdings.sort(key=lambda h: h.get("value_usd", 0), reverse=True)
                valid_holdings = valid_holdings[:max_holdings]

            saved = 0
            for h in valid_holdings:
                try:
                    db.execute(
                        """
                        INSERT INTO sec_13f_holdings
                            (cik, ticker, name_of_issuer, cusip, value_usd,
                             shares, share_type, filing_quarter, filing_date,
                             collected_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
                ON CONFLICT (cik, ticker, filing_quarter) DO NOTHING)
                        ON CONFLICT (cik, ticker, filing_quarter) DO UPDATE SET
                            value_usd = EXCLUDED.value_usd,
                            shares = EXCLUDED.shares,
                            collected_at = EXCLUDED.collected_at
                        """,
                        [
                            cik,
                            h.get("ticker", ""),
                            h.get("name_of_issuer", ""),
                            h.get("cusip", ""),
                            h.get("value_usd", 0),
                            h.get("shares", 0),
                            h.get("share_type", "SH"),
                            quarter,
                            filing_date,
                        ],
                    )
                    saved += 1
                except Exception as e:
                    logger.debug(
                        "[SEC 13F] Insert failed for %s: %s", h.get("ticker"), e
                    )

            return saved, capped_count

    def update_latest_quarter_if_newer(self, cik: str, newest_q: str) -> None:
        with get_db() as db:
            try:
                db.execute(
                    """
                    UPDATE sec_13f_filers 
                    SET latest_quarter = CASE 
                        WHEN latest_quarter IS NULL THEN %s
                        WHEN %s > latest_quarter THEN %s
                        ELSE latest_quarter
                    END
                    WHERE cik = %s
                    """,
                    [newest_q, newest_q, newest_q, cik],
                )
            except Exception as e:
                logger.error(
                    "[SEC 13F] Failed to update latest_quarter for CIK %s: %s", cik, e
                )

    def get_scored_tickers(self, limit: int = 50) -> list[dict[str, Any]]:
        """Build dict list from recent 13F holdings in DB."""
        with get_db() as db:
            rows = db.execute(
                """
                SELECT ticker, COUNT(DISTINCT cik) as inst_count,
                       SUM(value_usd) as total_value
                FROM sec_13f_holdings
                WHERE ticker != '' AND ticker IS NOT NULL
                GROUP BY ticker
                ORDER BY inst_count DESC
                LIMIT %s
                """,
                [limit],
            ).fetchall()

            return [
                {"ticker": r[0], "inst_count": r[1], "total_value": r[2]} for r in rows
            ]

    def get_holdings_for_ticker(self, ticker: str) -> list[dict[str, Any]]:
        """Get institutional holders for a specific ticker."""
        with get_db() as db:
            rows = db.execute(
                """
                SELECT h.cik, f.filer_name, h.value_usd, h.shares,
                       h.share_type, h.filing_quarter, h.filing_date
                FROM sec_13f_holdings h
                LEFT JOIN sec_13f_filers f ON h.cik = f.cik
                WHERE h.ticker = %s
                ORDER BY h.value_usd DESC
                """,
                [ticker],
            ).fetchall()

            return [
                {
                    "cik": r[0],
                    "filer_name": r[1],
                    "value_usd": r[2],
                    "shares": r[3],
                    "share_type": r[4],
                    "filing_quarter": r[5],
                    "filing_date": str(r[6]) if r[6] else None,
                }
                for r in rows
            ]
