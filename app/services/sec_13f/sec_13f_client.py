"""SEC 13F Client - handles HTTP communication with the SEC APIs."""

import logging
import time
from typing import Any

import requests

from app.config import settings

logger = logging.getLogger(__name__)

SEC_BASE_URL = "https://data.sec.gov"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"
RATE_LIMIT_SECS = 0.15  # 6-7 req/sec, well within 10/s limit


class SEC13FClient:
    """Network client for SEC API interactions with built-in rate-limiting."""

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": settings.SEC_USER_AGENT,
                "Accept-Encoding": "gzip, deflate",
            }
        )

    def get_submissions(self, cik: str) -> dict[str, Any] | None:
        """Fetch company submissions JSON from SEC EDGAR."""
        padded_cik = cik.lstrip("0").zfill(10)
        url = f"{SEC_BASE_URL}/submissions/CIK{padded_cik}.json"

        time.sleep(RATE_LIMIT_SECS)
        try:
            resp = self._session.get(url, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            logger.warning(
                "[SEC 13F] Submissions %s returned %d", url, resp.status_code
            )
        except Exception as e:
            logger.error("[SEC 13F] Submissions request failed for CIK %s: %s", cik, e)
        return None

    def get_json(self, url: str) -> dict[str, Any] | None:
        """Fetch a JSON document with rate limiting."""
        time.sleep(RATE_LIMIT_SECS)
        try:
            resp = self._session.get(url, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            logger.warning("[SEC 13F] get_json %s returned %d", url, resp.status_code)
        except Exception as e:
            logger.warning("[SEC 13F] get_json failed for %s: %s", url, e)
        return None

    def get_text(self, url: str) -> str | None:
        """Fetch a text/html document with rate limiting."""
        time.sleep(RATE_LIMIT_SECS)
        try:
            resp = self._session.get(url, timeout=30)
            if resp.status_code == 200:
                return resp.text
            logger.warning("[SEC 13F] get_text %s returned %d", url, resp.status_code)
        except Exception as e:
            logger.error("[SEC 13F] get_text failed for %s: %s", url, e)
        return None
