"""
Priority Analysis Queue — Drop-in replacement for asyncio.Queue
================================================================

Wraps asyncio.PriorityQueue so that callers can continue using
`await queue.put(ticker)` and `await queue.get()` with plain strings,
while internally ordering by priority.

Priority levels (lower = higher priority):
  0: Portfolio holdings — need stop-loss and exit analysis ASAP
  1: Deep-tier tickers — flagged for intensive research
  2: Standard watchlist — routine monitoring
  3: Newly discovered tickers — fresh from discovery
  4: Glance-tier tickers — quick check only

Sentinel (None) is always lowest priority (999) to ensure workers
drain all real tickers before shutting down.
"""

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Priority constants
PRIORITY_PORTFOLIO = 0
PRIORITY_DEEP = 1
PRIORITY_WATCHLIST = 2
PRIORITY_DISCOVERED = 3
PRIORITY_GLANCE = 4
PRIORITY_SENTINEL = 999


class PriorityAnalysisQueue:
    """Drop-in wrapper around asyncio.PriorityQueue.

    Callers use the same `put()`, `put_nowait()`, `get()`, `qsize()`,
    `task_done()`, and `join()` interface as asyncio.Queue.
    Items are plain strings (tickers) or None (sentinel).
    Priority is determined by `classify()` or `put_with_priority()`.
    """

    def __init__(
        self,
        position_tickers: set[str] | None = None,
        deep_tickers: set[str] | None = None,
        glance_tickers: set[str] | None = None,
    ):
        self._q: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._seq = 0  # Tie-breaker for same-priority items (FIFO within tier)
        self._position_tickers = position_tickers or set()
        self._deep_tickers = deep_tickers or set()
        self._glance_tickers = glance_tickers or set()

    def classify(self, ticker: str | None) -> int:
        """Determine the priority of a ticker based on known sets."""
        if ticker is None:
            return PRIORITY_SENTINEL
        if ticker in self._position_tickers:
            return PRIORITY_PORTFOLIO
        if ticker in self._deep_tickers:
            return PRIORITY_DEEP
        if ticker in self._glance_tickers:
            return PRIORITY_GLANCE
        return PRIORITY_WATCHLIST

    def put_nowait(self, ticker: str | None, priority: int | None = None) -> None:
        """Put a ticker into the queue with automatic or explicit priority."""
        prio = priority if priority is not None else self.classify(ticker)
        self._seq += 1
        self._q.put_nowait((prio, self._seq, ticker))

    async def put(self, ticker: str | None, priority: int | None = None) -> None:
        """Async put with automatic or explicit priority."""
        prio = priority if priority is not None else self.classify(ticker)
        self._seq += 1
        await self._q.put((prio, self._seq, ticker))

    async def get(self) -> str | None:
        """Get the highest-priority ticker (lowest priority number)."""
        _prio, _seq, ticker = await self._q.get()
        return ticker

    def qsize(self) -> int:
        return self._q.qsize()

    def empty(self) -> bool:
        return self._q.empty()

    def task_done(self) -> None:
        self._q.task_done()

    async def join(self) -> None:
        await self._q.join()

    def update_sets(
        self,
        position_tickers: set[str] | None = None,
        deep_tickers: set[str] | None = None,
        glance_tickers: set[str] | None = None,
    ) -> None:
        """Update the classification sets (e.g. after triage completes)."""
        if position_tickers is not None:
            self._position_tickers = position_tickers
        if deep_tickers is not None:
            self._deep_tickers = deep_tickers
        if glance_tickers is not None:
            self._glance_tickers = glance_tickers
