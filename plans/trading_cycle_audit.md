# Trading Cycle Audit Report (Cycle `cycle-1779914510`)

## Executive Summary
This audit reports on the active trading cycle (`cycle-1779914510`) initiated on **2026-05-27 20:41:50 UTC** evaluating **30 tickers** with V2 cognition. 

While the Python worker processes are successfully running and moving tickers from collection through standard analysis phases, a **critical database schema mismatch** is silently preventing any analysis results from being saved to the database. Additionally, several system service timeouts and microservice communication errors are degrading pipeline efficiency.

---

## 1. Critical Bugs & Database Issues

### 💥 Issue 1: Database Write Transaction Rollback on Ticker Completion
* **File:** `app/pipeline/analysis/decision_engine.py` (lines 1228–1257)
* **Error Message:** 
  `[ERROR] [PIPELINE] [decision_engine] log failed: there is no unique or exclusion constraint matching the ON CONFLICT specification`
* **Root Cause:**
  At the end of ticker analysis, `_log_decision()` wraps the DB writes to `analysis_results` and `cycle_summaries` inside an atomic `with db.transaction():` block.
  The insert statement for `cycle_summaries` specifies:
  `ON CONFLICT (ticker, cycle_id) DO UPDATE SET ...`
  However, the `cycle_summaries` table was created without a `PRIMARY KEY` or a unique constraint on `(ticker, cycle_id)`.
  This is due to a mismatch between:
  - `schema_pg.sql` (defines `PRIMARY KEY (ticker, cycle_id)`)
  - `app/utils/db_migrations.py` (defines table creation without any `PRIMARY KEY` or unique constraints).
* **Impact:** 
  The transaction is aborted and rolled back. No records are saved to `analysis_results` or `cycle_summaries`. This makes the database empty of decisions, meaning **no automated trades can execute** and **no results are visible in the UI** for completed tickers.

---

## 2. Infrastructure & Scraping Bottlenecks

### 🔌 Issue 2: PRISM Service Connection Failures & Timeouts
* **Error Message:**
  `[WARNING] [PRISM] API call to http://10.0.0.16:7777/chat?stream=false timed out. Raising immediately to trigger fast fallback.`
  `[WARNING] [PrismAgentCaller] Prism routing failed for smart_janitor, falling back to local: ...`
* **Root Cause:**
  The PRISM service running on host `10.0.0.16:7777` is frequently timing out (exceeding connection thresholds) or throwing `500 Internal Server Error` under heavy load.
* **Impact:**
  The pipeline is forced to fallback to local vLLM generation. This wastes network resources, slows down evaluation, and puts excessive concurrent load on local GPU endpoints (like `dgx_spark` or `jetson`).

### ⏱️ Issue 3: Reddit Scraper Service JIT Timeouts
* **Error Message:**
  `[ERROR] [PIPELINE]   [Reddit] <TICKER> TIMEOUT`
* **Root Cause:**
  Reddit post collection JIT checks are routinely hitting their 120s timeout limit (e.g., on AMP, AOS, CHMP, CMPX, COHR, HIMS, IBKR, KKR, LTM, NDAQ, NXT, PAM).
* **Impact:**
  Each timeout stalls the data collection phase, extending the total cycle runtime and contributing to overall worker queue delay.

---

## 3. Worker Timeout & Quality Degradation

### ⏳ Issue 4: MetaOrchestrator and Worker Timeouts
* **Error Message:**
  `[WARNING] [V2] MetaOrchestrator TIMEOUT for <TICKER> (60s) — injecting fallback context`
  `[ERROR] Analysis TIMEOUT for <TICKER>: LLM Timeout after 360s`
* **Root Cause:**
  Under high concurrent load, the 60-second limit for sub-agent routing in `MetaOrchestrator` and the 360-second limit for worker analysis in `execute_v2_pipeline` are breached.
* **Impact:**
  - **MetaOrchestrator timeouts:** Force tickers to run in "evidence-only mode", bypassing adversarial debate completely to save GPU time. This degrades the depth of the trading thesis.
  - **Worker timeouts:** Cause the worker thread to completely drop standard completion, aborting the ticker analysis and forcing a fallback `HOLD @ 0%` decision.

---

## 4. Recommended Fixes (To Be Executed Post-Cycle)

### 🛠️ Step 1: Fix Database Schema Constraint (COMPLETED & VERIFIED)
* **Status:** Resolved and verified during the active cycle.
* **Fix details:**
  - Added a dynamic database migration in `app/db/migrations.py` to automatically deduplicate and create the `PRIMARY KEY (ticker, cycle_id)` constraint on the live database.
  - Updated the inline table creation query in `app/utils/db_migrations.py` so future environments construct the table correctly.
* **Verification:** Confirmed that tickers completing after the migration (such as `NXT`) are successfully written to both tables without transaction rollbacks. Regression tests in `tests/regression/test_schema_alignment.py` and `tests/integration/test_db_constraints.py` also pass successfully.

### 🛠️ Step 2: Optimise PRISM Client & Fallbacks (PENDING)
* Adjust PRISM API call client timeout thresholds or implement a queue/backoff system so it doesn't cause immediate cascade timeouts.

### 🛠️ Step 3: Optimise Reddit Scraper & JIT Caching (PENDING)
* Pre-fetch Reddit threads in the background rather than executing JIT (Just-In-Time) fetches during critical analysis cycles.

