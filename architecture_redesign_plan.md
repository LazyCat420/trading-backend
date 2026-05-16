# Pipeline Streaming & Targeted Data Sniping Redesign Plan

Based on the issues outlined (bottlenecks from batched processing, mass data scraping generating junk data, and the cycle hanging indefinitely), here is a phased redesign plan to completely decouple the pipeline into a fast, streaming, on-demand agentic loop.

---

## 🛑 Problem Summary & Investigation Findings

1. **Mass Scraping Overload:** 
   * **Why it happens:** In the current `data_phase.py`, data collection runs as a massive batch process (`run_global_collection` and `run_perticker_collection`). All tickers selected by the `TickerSelector` (which pulls positions, watchlist, and random discovery tickers) are immediately blasted with API calls for news, technicals, and institutional data *before* any LLM analysis begins.
   * **The result:** Huge amounts of junk data are collected for tickers the LLM might not even care about.
2. **Phase Bottlenecks:** Analysis runs concurrently per ticker, but the **Trading Phase** waits for ALL analysis to finish (`await analysis_task` in the orchestrator) before executing trades. By the time it finishes, the market has moved.
3. **Zombie Async Loops:** 
   * **Why it happens:** At the end of the cycle, `run_autoresearch` is fired off as a background task. It triggers the `EvolutionRouter` (`run_debate`), which launches up to 3 rounds of LLM debates per issue. Because these `llm.chat` calls lack strict asyncio timeouts, if the LLM backend stalls, the event loop stays alive indefinitely (the "10-hour hang").
4. **Harness Evolution Starting from Scratch:**
   * **Why it happens:** The Test & Prove environment (`test_prove.py`) currently only does a *dry run* (syntax check / AST parse) of proposed scrapers. It doesn't actually execute them to see if they pull data. If a bad scraper is deployed, it breaks. When AutoResearch tries to fix it, it often overwrites it with another untested script instead of rolling back to the last known working function and iterating from there.

---

## 🛠️ Phase 1: Targeted "Data Sniping" (On-Demand Retrieval)

**Goal:** Move away from scraping *everything* in Phase 2 to an "on-demand" agentic collection model based on actual holdings or strict criteria.

*   **1.1: Ticker Triage & Seed Selection:** 
    *   Instead of pulling a massive list of tickers, the `TickerSelector` should only return high-conviction seeds: Active Holdings (e.g., AAPL), open limit orders, and user-specified watchlists.
*   **1.2: Agentic Tool-Driven Collection:**
    *   Remove the monolithic "Mass Collector" (`data_phase.py`). 
    *   Give the `Analyzer Agent` specific "Sniping Tools" (e.g., `get_company_filings(ticker)`, `get_latest_news(ticker)`).
    *   When the Analyzer evaluates AAPL, *it* decides what data it needs to fetch in real-time. If it needs news, it fires the tool. This eliminates fetching junk data that the agent never reads.
*   **1.3: Data Janitor Streamlining:**
    *   Instead of cleaning massive datasets in batch, the Janitor agents act as middleware. When the Analyzer calls `get_latest_news()`, the Janitor intercepts the raw response, sanitizes/summarizes it, and returns the clean packet directly to the Analyzer.

---

## 🚀 Phase 2: Full Streaming Architecture (Eliminate Batch Bottlenecks)

**Goal:** Trade as soon as a single ticker is analyzed, rather than waiting for the entire market batch.

*   **2.1: Per-Ticker Execution Loop:**
    *   In `orchestrator_core.py`, decouple Phase 5 (Trading). 
    *   Instead of `await analysis_task` blocking the trade phase, the `analysis_queue` should feed directly into a `trading_queue`.
*   **2.2: Continuous Flow:**
    *   `Collection` (Sniping) ➡️ `Analysis` ➡️ `Decision` ➡️ `Trade Execution`.
    *   As soon as AAPL finishes being analyzed, the trading engine reads the result and executes immediately. Meanwhile, NVDA is still being analyzed. This ensures trades hit the market instantly.

---

## 🧹 Phase 3: Resolving Zombie Async Loops (Lifecycle Management)

**Goal:** Ensure the pipeline cleanly resolves and shuts down idle tasks after the cycle finishes.

*   **3.1: Strict LLM Timeouts:**
    *   Wrap all background LLM calls in `debate.py` and `autoresearch.py` with `asyncio.wait_for(..., timeout=120)`. If they hang, they will automatically cancel and release the event loop.
*   **3.2: Explicit End-of-Cycle Shutdown:**
    *   Track all background operations in an `asyncio.TaskGroup`. At the end of `_execute_cycle`, explicitly cancel trailing routines that exceed the cycle time limit.

---

## 🧬 Phase 4: Active Harness Evolution & Versioning (The Feedback Loop)

**Goal:** Ensure the bot only deploys scrapers that actually work, and builds upon successful versions instead of starting from scratch.

*   **4.1: Live Sandbox Testing:**
    *   Modify `test_prove.py`. Instead of just parsing AST syntax, run the proposed scraper function in an isolated sandbox with a dummy ticker (e.g., AAPL). If it throws an error or returns no data, **fail the test** and send the error trace back to the LLM to try again.
*   **4.2: Harness Version Control & Rollback:**
    *   Implement a "Working State Registry" for tools. When a scraper succeeds in production, flag that code version as `STABLE`. 
    *   If an Evolution Debate proposes a fix that fails in production, the system automatically rolls back the function to the last `STABLE` version. The next debate will then use the `STABLE` version as its starting point rather than the broken one.

---

## ❓ Questions for Collaboration:
1. **Ticker Selection:** If we switch entirely to "sniping", how do you want the bot to discover *new* tickers to trade? Should we keep a small, separate background scout that just looks for new tickers and adds them to a watchlist?
2. **AutoResearch Testing:** For Phase 4 (Live Sandbox Testing), is it okay if the Evolution Router actually executes the generated scraper code on the Jetson/DGX to test it? (It will use live network access to test pulling data).
3. **Data Availability:** If we switch to Agentic Data Sniping (on-demand), the LLM will take slightly longer per ticker (since it has to wait for API calls in real-time). Are you okay with trading off *initial collection time* for *targeted, high-quality data*?
