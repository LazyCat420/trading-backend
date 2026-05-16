"""
Request utilities -- shared across all collectors.
Handles: rate limiting, user-agent rotation, exponential backoff, pacing,
         proxy rotation, per-domain rate limits.

Usage:
    async with SmartClient(base_delay=1.0) as client:
        r = await client.get("https://example.com/api")

Proxy rotation:
    Set PROXY_LIST="http://proxy1:port,http://proxy2:port" in .env
    If not set, requests go direct (no proxy).
"""

import asyncio
import logging
import random
import time
import os
from urllib.parse import urlparse
import httpx

logger = logging.getLogger(__name__)

# Realistic browser user-agents for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
]


class SmartClient:
    """httpx.AsyncClient wrapper with built-in anti-rate-limiting.

    Features:
    - Exponential backoff on 429/503 responses
    - User-agent rotation per request
    - Proxy rotation (if PROXY_LIST env var set)
    - Per-domain rate limiting (1.5s between requests to same domain)
    - Max retries with jitter
    """

    def __init__(
        self,
        base_delay: float = 1.0,
        max_retries: int = 3,
        timeout: float = 15.0,
    ):
        self.base_delay = base_delay
        self.max_retries = max_retries
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._domain_last_request: dict[str, float] = {}
        self._min_domain_interval = 1.5  # seconds between requests to same domain

        # Proxy rotation: set PROXY_LIST="http://p1:port,http://p2:port" in .env
        proxy_str = os.environ.get("PROXY_LIST", "")
        self._proxies = [p.strip() for p in proxy_str.split(",") if p.strip()]
        self._proxy_index = 0

    def _get_proxy(self) -> str | None:
        """Rotate through proxy list if available."""
        if not self._proxies:
            return None
        proxy = self._proxies[self._proxy_index % len(self._proxies)]
        self._proxy_index += 1
        return proxy

    async def __aenter__(self):
        proxy = self._get_proxy()
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=True,
            proxy=proxy,
        )
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_headers(self, extra_headers: dict | None = None) -> dict:
        """Build headers with rotated user-agent."""
        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        if extra_headers:
            headers.update(extra_headers)
        return headers

    async def _enforce_domain_rate_limit(self, url: str):
        """Ensure minimum interval between requests to the same domain."""
        domain = urlparse(url).netloc
        now = time.monotonic()
        last = self._domain_last_request.get(domain, 0)
        elapsed = now - last
        if elapsed < self._min_domain_interval:
            await asyncio.sleep(self._min_domain_interval - elapsed)
        self._domain_last_request[domain] = time.monotonic()

    async def get(
        self,
        url: str,
        params: dict | None = None,
        headers: dict | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        """GET with retry, exponential backoff, and user-agent rotation."""
        assert self._client is not None, "Use 'async with SmartClient() as client:'"

        # Per-domain rate limiting
        await self._enforce_domain_rate_limit(url)

        # Allow per-request timeout override
        req_timeout = httpx.Timeout(timeout) if timeout else None

        last_response = None
        for attempt in range(self.max_retries):
            req_headers = self._get_headers(headers)
            try:
                r = await self._client.get(
                    url, params=params, headers=req_headers, timeout=req_timeout
                )
                last_response = r

                if r.status_code == 200:
                    return r

                if r.status_code in (429, 503):
                    wait = (2**attempt) + random.uniform(0.5, 1.5)
                    logger.warning(
                        "[smart_client] %s: HTTP %d, backing off %.1fs (attempt %d/%d)",
                        url,
                        r.status_code,
                        wait,
                        attempt + 1,
                        self.max_retries,
                    )
                    await asyncio.sleep(wait)
                    continue

                # Other non-200 status -- don't retry
                logger.debug("[smart_client] %s: HTTP %d", url, r.status_code)
                return r

            except httpx.TimeoutException:
                wait = (2**attempt) + random.uniform(0.5, 1.5)
                logger.warning(
                    "[smart_client] %s: timeout, retrying in %.1fs (attempt %d/%d)",
                    url,
                    wait,
                    attempt + 1,
                    self.max_retries,
                )
                await asyncio.sleep(wait)
            except httpx.HTTPError as e:
                logger.warning("[smart_client] %s: error %s", url, e)
                return httpx.Response(status_code=0, request=httpx.Request("GET", url))

        # All retries exhausted
        if last_response is not None:
            return last_response
        return httpx.Response(status_code=0, request=httpx.Request("GET", url))

    async def pace(self, min_delay: float = 1.0, max_delay: float = 3.0):
        """Random delay between requests to avoid detection."""
        await asyncio.sleep(random.uniform(min_delay, max_delay))
