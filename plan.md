# Trading Pipeline Debugging & Refactoring Plan

> **Status**: Implementation COMPLETE — all 6 phases built.  
> **Last Updated**: 2026-05-16

---

## 1. How LLM Requests are Triggered (The Architecture)

The routing of LLM requests to the DGX Spark and Jetson boxes is controlled primarily by the following files:

*   **`app/services/vllm_client.py`**: Heart of LLM routing. Priority queue (HIGH, NORMAL, LOW) per hardware endpoint. Dynamically probes endpoints for active model names.
*   **`app/services/prism_client.py`**: Client for Prism Gateway. Routes through `/agent` proxy or falls back to offline shadow logging.
*   **`app/pipeline/analysis/decision_engine.py`**: Triggers LLM workloads (specialist agents, RLM analysis, Glance Tier checks).

**How Routing Works in `vllm_client.py`:**
*   **ALL endpoints** (including Jetson) now route through Prism when `PRISM_AGENT_ROUTING=True` *(Phase 3 change)*
*   When `PRISM_AGENT_ROUTING=False`: direct vLLM + offline shadow-log to Prism

---

## 2. Why the pipeline was processing without hitting Prism for 10 hours

### A. The Bypasses & Fallbacks (Direct Routing) — **FIXED in Phase 3**
1.  ~~**Jetson Bypass**: Jetson requests explicitly bypassed Prism~~ → **Now routes through Prism**
2.  **Health Check Fallback**: If `prism_client.check_health()` fails, falls back to direct vLLM
3.  ~~**Shadow Log Failures**: Silently swallowed at debug level~~ → **Now surfaces as warnings**

### B. Aggressive Caching & Triage Tiers
1.  **Data Sufficiency Gate**: Bypasses scraping when data is sufficient
2.  **Glance Tier Skip**: Returns cached thesis when no material change detected

### C. The Queue Dispatcher is Stuck
1.  **KV Cache Protection**: Pauses when `cache_usage >= 0.90`
2.  **Circuit Breaker**: Pauses after consecutive batch timeouts

---

## 3. How the Logic *Should* Be Working

1.  **Triggering**: Trading cycle starts → tickers triaged into tiers (Deep, Standard, Glance)
2.  **Data Collection**: Tickers collect data concurrently. Cache hits skip scraping.
3.  **Analysis**: Tickers hit `decision_engine.py`
    *   Glance: cheap LLM change-detection check
    *   Standard/Deep: specialist agents → RLM Config C → optionally Debate Config D
4.  **Execution**: All requests go through `vllm_client.py` → routed via Prism or direct

---

## 4. Current State of the Agentic Loop & Subagents

### What's implemented:
*   **`app/tools/subagent_tools.py`**: Primary agents can spawn research subagents with **dynamic tool provisioning** *(Phase 4)*
*   **`app/tools/prism_agent_harness.py`**: Prism-delegated agent loop with local fallback *(Phase 6)*
*   **`app/tools/script_sandbox.py`**: Sandboxed Python execution for quant equations *(Phase 5)*
*   **`app/tools/executor.py`**: Local agentic loop (used as fallback when Prism is unavailable)

### How subagent tool provisioning works (Phase 4):
Parent agents can now pass `enabled_tools: ["search_web", "scrape_url", "get_market_data"]` when spawning a subagent. The subagent only receives those tools, preventing context bloat.

### How the Onion Layered Architecture works (Phase 6):
```
Layer 1 (trading-cycle-backend) → Defines tools, holds data state
Layer 2 (Prism Gateway)         → Runs agentic loop, tracks everything
Layer 3 (Hermes/vLLM)           → Executes raw LLM completions
```

---

## 5. Phased Implementation (COMPLETED)

### ✅ Phase 1: Diagnostic Script
*   **File**: `diagnose_pipeline.py`
*   **What it does**: Checks endpoint health, Prism status, recent LLM logs, endpoint call distribution, cycle audit events, and Glance skip counts.
*   **Run**: `python diagnose_pipeline.py`

### ✅ Phase 2: Verify Triage & Cache Gates
*   **Covered by**: `diagnose_pipeline.py` (queries `cycle_audit_log` for Glance SKIPs)

### ✅ Phase 3: Unify Prism Telemetry
*   **File**: `app/services/vllm_client.py`
*   **Changes**:
    - Removed Jetson Prism bypass — ALL endpoints now route through Prism when `PRISM_AGENT_ROUTING=True`
    - Shadow log errors surfaced as warnings instead of debug-level silencing

### ✅ Phase 4: Dynamic Subagent Tool Provisioning
*   **File**: `app/tools/subagent_tools.py`
*   **Changes**:
    - `spawn_research_subagent` now accepts optional `enabled_tools` parameter
    - Parent agents specify exactly which tools a subagent receives
    - Falls back to all tools if no valid tools match the whitelist

### ✅ Phase 5: Custom Agentic Tool Creation (Python Sandbox)
*   **Files**: `app/tools/script_sandbox.py`, `app/tools/sandbox_runner.py`
*   **Changes**:
    - New `execute_quant_script` tool with Pydantic validation
    - DATA injection — agent passes data dict, code operates on it
    - RestrictedPython guardrails: no filesystem, no network, no imports beyond math/json/statistics
    - Old `run_python_script` preserved as deprecated alias

### ✅ Phase 6: Onion Layered Architecture (Prism Agent Harness)
*   **File**: `app/tools/prism_agent_harness.py` (NEW)
*   **Changes**:
    - `run_prism_agent()` delegates the full agentic loop to Prism `/agent`
    - Prism runs the loop, tracks every step natively
    - Transparent fallback to local `executor.py` when Prism is unhealthy
    - `subagent_tools.py` updated to use Prism-first routing

---

## 6. Files Changed

| File | Phase | Change |
|------|-------|--------|
| `diagnose_pipeline.py` | 1 | Diagnostic script |
| `app/services/vllm_client.py` | 3 | Removed Jetson bypass, surfaced shadow log errors |
| `app/tools/subagent_tools.py` | 4, 6 | Dynamic tool provisioning + Prism-first routing |
| `app/tools/sandbox_runner.py` | 5 | Upgraded sandbox with DATA injection & blocked imports |
| `app/tools/script_sandbox.py` | 5 | New `execute_quant_script` tool with Pydantic models |
| `app/tools/prism_agent_harness.py` | 6 | NEW: Prism-delegated agent harness |
| `app/tools/__init__.py` | 5, 6 | Registered new tools in exports |
| `plan.md` | — | This file |

---

## 7. How to Test

```bash
# Phase 1: Run diagnostics
python diagnose_pipeline.py

# Phase 5: Test sandbox guardrails (should block)
python -c "
from app.tools.sandbox_runner import run_isolated
run_isolated('import os; print(os.listdir(\".\"))')
"

# Phase 5: Test sandbox happy path
python -c "
from app.tools.sandbox_runner import run_isolated
import json
run_isolated(
    'prices = DATA[\"prices\"]; avg = sum(prices)/len(prices); print(json.dumps({\"avg_price\": avg}))',
    json.dumps({\"prices\": [100, 102, 98, 105]})
)
"
```

---

## 8. Configuration Reference

| Setting | Default | Effect |
|---------|---------|--------|
| `PRISM_ENABLED` | `True` | Master toggle for Prism integration |
| `PRISM_AGENT_ROUTING` | `False` | `True` = route ALL endpoints through Prism `/agent` |
| `PRISM_AGENT` | `CUSTOM_MARKET_ALPHA` | Prism agent persona for routing |
