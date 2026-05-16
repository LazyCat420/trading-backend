from typing import Any, List, Tuple


class TrackerRepo:
    def __init__(self, db: Any):
        self.db = db

    def get_fund_list(self) -> List[Tuple]:
        return self.db.execute(
            """
            SELECT f.cik, f.filer_name, f.latest_quarter, f.is_active,
                   COUNT(h.ticker) AS holding_count, COALESCE(SUM(h.value_usd), 0) AS total_value
            FROM sec_13f_filers f
            LEFT JOIN sec_13f_holdings h ON f.cik = h.cik AND h.filing_quarter = f.latest_quarter
            GROUP BY f.cik, f.filer_name, f.latest_quarter, f.is_active
            """
        ).fetchall()

    def get_filer_info(self, cik: str) -> Tuple | None:
        return self.db.execute(
            "SELECT filer_name, latest_quarter FROM sec_13f_filers WHERE cik = %s",
            [cik],
        ).fetchone()

    def get_holdings_for_quarter(self, cik: str, quarter: str) -> List[Tuple]:
        return self.db.execute(
            "SELECT ticker, name_of_issuer, shares, value_usd, cusip FROM sec_13f_holdings WHERE cik = %s AND filing_quarter = %s",
            [cik, quarter],
        ).fetchall()

    def get_holdings_summary_for_quarter(self, cik: str, quarter: str) -> List[Tuple]:
        return self.db.execute(
            "SELECT ticker, shares, value_usd FROM sec_13f_holdings WHERE cik = %s AND filing_quarter = %s",
            [cik, quarter],
        ).fetchall()

    def get_all_holding_history(self, cik: str) -> List[Tuple]:
        return self.db.execute(
            "SELECT ticker, filing_quarter, shares FROM sec_13f_holdings WHERE cik = %s ORDER BY filing_quarter ASC",
            [cik],
        ).fetchall()

    def get_fund_overlap(self, min_funds: int) -> List[Tuple]:
        return self.db.execute(
            """
            SELECT h.ticker, COUNT(DISTINCT h.cik) AS fund_count, SUM(h.value_usd) AS total_value,
                   SUM(h.shares) AS total_shares, STRING_AGG(DISTINCT f.filer_name, ', ') AS fund_names,
                   MAX(h.name_of_issuer) AS name_of_issuer
            FROM sec_13f_holdings h
            JOIN sec_13f_filers f ON h.cik = f.cik
            WHERE h.filing_quarter = f.latest_quarter AND h.ticker != '' AND h.ticker IS NOT NULL
            GROUP BY h.ticker HAVING COUNT(DISTINCT h.cik) >= %s
            ORDER BY fund_count DESC, total_value DESC LIMIT 100
            """,
            [min_funds],
        ).fetchall()

    def get_ticker_history_for_fund(self, cik: str, ticker: str) -> List[Tuple]:
        return self.db.execute(
            "SELECT filing_quarter, filing_date, shares, value_usd FROM sec_13f_holdings WHERE cik = %s AND ticker = %s ORDER BY filing_quarter ASC",
            [cik, ticker.upper()],
        ).fetchall()
