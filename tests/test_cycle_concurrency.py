"""
Tests for trading cycle concurrency — queue pipelining behavior.

These tests are self-contained and do NOT import any trading-service modules
that require database connections. They verify the concurrent queue design.
"""

import asyncio
import time
import pytest


@pytest.mark.asyncio
async def test_queue_concurrency_nonblocking_ingestion():
    """Verify that analysis queue does not block on ticker collection
    and can ingest/process concurrently."""
    queue = asyncio.Queue()

    # Simulating the scraper pushing tickers quickly
    async def simulate_collection():
        for ticker in ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]:
            await asyncio.sleep(0.01)  # fast collection
            await queue.put(ticker)
        await queue.put(None)  # sentinel

    # Simulating the worker processing tickers
    processed = []

    async def simulate_worker():
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break
            processed.append(item)
            await asyncio.sleep(0.02)  # worker time
            queue.task_done()

    await asyncio.gather(simulate_collection(), simulate_worker())
    assert processed == ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]


@pytest.mark.asyncio
async def test_multiple_workers_consume_concurrently():
    """Verify that multiple workers pull from the queue simultaneously,
    resulting in faster total processing than a single worker."""
    queue = asyncio.Queue()
    tickers = [f"T{i}" for i in range(10)]
    processed = []
    lock = asyncio.Lock()

    WORKER_COUNT = 3

    # Push all tickers + sentinels
    for t in tickers:
        queue.put_nowait(t)
    for _ in range(WORKER_COUNT):
        queue.put_nowait(None)

    async def worker(wid: int):
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break
            await asyncio.sleep(0.01)  # simulate work
            async with lock:
                processed.append((wid, item))
            queue.task_done()

    start = time.monotonic()
    await asyncio.gather(*[worker(i) for i in range(WORKER_COUNT)])
    elapsed = time.monotonic() - start

    # All tickers should be processed
    assert len(processed) == len(tickers)
    assert set(item for _, item in processed) == set(tickers)

    # Multiple workers should have participated
    worker_ids = set(wid for wid, _ in processed)
    assert len(worker_ids) > 1, (
        f"Only worker(s) {worker_ids} processed items — expected concurrency"
    )

    # With 3 workers and 10 items @ 10ms each, total should be ~40ms not ~100ms
    assert elapsed < 0.08, (
        f"Took {elapsed:.3f}s — workers may not be running concurrently"
    )


@pytest.mark.asyncio
async def test_worker_receives_tickers_as_they_arrive():
    """Verify that a worker starts processing a ticker BEFORE
    all tickers have been pushed (true streaming behavior)."""
    queue = asyncio.Queue()
    events = []  # Track order of events

    async def slow_collector():
        for i, ticker in enumerate(["AAPL", "MSFT", "GOOGL"]):
            await asyncio.sleep(0.03)  # slow scraping
            events.append(f"pushed_{ticker}")
            await queue.put(ticker)
        await queue.put(None)

    async def worker():
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break
            events.append(f"started_{item}")
            await asyncio.sleep(0.01)  # fast analysis
            events.append(f"done_{item}")
            queue.task_done()

    await asyncio.gather(slow_collector(), worker())

    # The worker should start processing AAPL before MSFT is pushed
    pushed_msft_idx = events.index("pushed_MSFT")
    started_aapl_idx = events.index("started_AAPL")
    assert started_aapl_idx < pushed_msft_idx, (
        f"Worker didn't start AAPL before MSFT was pushed. Events: {events}"
    )


@pytest.mark.asyncio
async def test_sentinel_stops_worker_gracefully():
    """Verify that a None sentinel stops the worker without losing items."""
    queue = asyncio.Queue()
    processed = []

    queue.put_nowait("AAPL")
    queue.put_nowait("MSFT")
    queue.put_nowait(None)  # sentinel

    async def worker():
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break
            processed.append(item)
            queue.task_done()

    await worker()
    assert processed == ["AAPL", "MSFT"]
    assert queue.qsize() == 0
