"""
API Rate Limiter — shared per-service semaphores for safe parallel scraping.

When multiple tickers collect data in parallel, each ticker fires off
requests to yfinance, finnhub, Reddit, YouTube, etc. Without rate
limiting, 5 tickers × 5 sources = 25 concurrent HTTP requests, which
can trigger IP bans and API rate limits.

This module provides a centralized set of per-API semaphores so that
even with aggressive parallelism, we never exceed safe limits for
any single service.

Usage:
    from app.services.api_rate_limiter import rate_limiter

    async with rate_limiter.acquire("yfinance"):
        await collect_price_history(ticker)

    # Or use the decorator
    @rate_limiter.limit("reddit")
    async def collect_reddit_posts(ticker):
        ...
"""

import asyncio
import functools
import logging
import contextlib
from typing import Any, Callable

from app.config import settings

logger = logging.getLogger(__name__)


class APIRateLimiter:
    """Per-service semaphore manager.

    Each external API gets its own asyncio.Semaphore that caps
    concurrent requests across ALL parallel ticker collections.
    """

    def __init__(self):
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._burst_mode: bool = False
        self._limits: dict[str, int] = {
            "yfinance": settings.YFINANCE_MAX_CONCURRENT,
            "finnhub": settings.FINNHUB_MAX_CONCURRENT,
            "reddit": settings.REDDIT_MAX_CONCURRENT,
            "youtube": settings.YOUTUBE_MAX_CONCURRENT,
            "yf_news": settings.YFINANCE_MAX_CONCURRENT,  # shares yfinance limit
        }

    def enable_burst_mode(self, enabled: bool = True):
        """Temporarily boost limits for micro-intensity runs."""
        self._burst_mode = enabled
        if enabled:
            logger.info("[rate_limiter] Burst mode enabled. API semaphores bypassed.")
        else:
            logger.info("[rate_limiter] Burst mode disabled.")

    def _get_semaphore(self, service: str) -> asyncio.Semaphore:
        """Lazy-init semaphore for a service."""
        if service not in self._semaphores:
            limit = self._limits.get(service, 3)  # default: 3 concurrent
            self._semaphores[service] = asyncio.Semaphore(limit)
            logger.debug(
                "[rate_limiter] Created semaphore for %s (max=%d)",
                service,
                limit,
            )
        return self._semaphores[service]

    @contextlib.asynccontextmanager
    async def acquire(self, service: str):
        """Get the semaphore for use in `async with`.

        Usage:
            async with rate_limiter.acquire("yfinance"):
                await do_yfinance_call()
        """
        if self._burst_mode:
            yield
            return
            
        sem = self._get_semaphore(service)
        async with sem:
            yield

    def limit(self, service: str) -> Callable:
        """Decorator to rate-limit an async function.

        Usage:
            @rate_limiter.limit("reddit")
            async def scrape_reddit(ticker):
                ...
        """

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                async with self.acquire(service):
                    return await func(*args, **kwargs)

            return wrapper

        return decorator

    def status(self) -> dict[str, dict]:
        """Return current state of all semaphores for monitoring."""
        result = {}
        for service, limit in self._limits.items():
            sem = self._semaphores.get(service)
            if sem:
                # Semaphore._value shows remaining slots
                # (not part of public API but stable in CPython)
                available = getattr(sem, "_value", "?")
                result[service] = {
                    "max": limit,
                    "available": available,
                    "in_use": limit - available if isinstance(available, int) else "?",
                }
            else:
                result[service] = {"max": limit, "available": limit, "in_use": 0}
        return result


# Singleton — import this everywhere
rate_limiter = APIRateLimiter()
