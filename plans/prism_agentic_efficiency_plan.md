# Plan: Optimizing Prism Agentic Efficiency & Token Usage

This document outlines the architectural plan to optimize how `trading-service` interacts with `prism-service`. It introduces a hybrid data-access pattern to reduce token consumption, eliminate attention dilution, and unlock dynamic, emergent tool-calling capabilities.

---

## 1. Executive Summary

Currently, the pipeline dumps massive blocks of pre-fetched database records (e.g., financial tables, news sentiment logs) into LLM system and user prompts. This approach causes:
1.  **Token Bloat:** Substantial input token costs on every cycle run.
2.  **Attention Dilution:** The "lost in the middle" effect, where LLMs fail to spot critical outliers in large tables.
3.  **High Latency:** Long contexts slow down response times and spike TTFT (Time to First Token).

This plan shifts structured/math-based data lookup from **static upfront prompting** to **dynamic, on-demand query tools**, while maintaining full documents for narrative and narrative-style contexts.

---

## 2. The Hybrid Context Model

We partition data and LLM routing into two separate tracks:

```
                  ┌───────────────────────────────┐
                  │          Data Type            │
                  └──────────────┬────────────────┘
                                 │
         ┌───────────────────────┴───────────────────────┐
         ▼                                               ▼
┌──────────────────┐                            ┌──────────────────┐
│ Narrative / Story│                            │ Numerical / Math │
│  (News, Reddit)  │                            │  (P/E, RSI, Cash)│
└────────┬─────────┘                            └────────┬─────────┘
         │                                               │
         ▼                                               ▼
┌──────────────────┐                            ┌──────────────────┐
│ Upfront Context  │                            │ Precision Query  │
│  (Read Full text)│                            │  (On-Demand Tools│
└────────┬─────────┘                            └────────┬─────────┘
         │                                               │
         ▼                                               ▼
┌──────────────────┐                            ┌──────────────────┐
│ /chat Endpoint   │                            │ /agent Endpoint  │
│(Fast Direct Call)│                            │(Agentic Tool Loop│
└──────────────────┘                            └──────────────────┘
```

### Track A: Narrative Context (Stories, News, Sentiment)
*   **Data Characteristics:** Textual flow, sentiment shifts, historical context, thematic evolution.
*   **Method:** Pre-fetch and include full (but cleaned/summarized) articles in the prompt context.
*   **Routing:** Route to the `/chat` endpoint (direct proxy) to get fast, single-turn summaries without running server-side loops.

### Track B: Numerical Context (Financial Metrics, Indicators, Transactions)
*   **Data Characteristics:** Standardized numerical values, ratios, prices, and volumes.
*   **Method:** Provide the agent with fine-grained query tools. The agent starts with a blank slate and dynamically fetches only the metrics it needs.
*   **Routing:** Route to the `/agent` endpoint (agentic loop) to let Prism manage the tool execution and planning state.

---

## 3. Precision Tool Specifications

To support Track B, we define three core precision tools that agents can call:

### 1. `query_financial_metrics`
Fetches specific financial statements, ratios, or multiples for a ticker.
*   **Arguments:**
    *   `ticker` (str): Entity symbol (e.g., `"AAPL"`).
    *   `metrics` (list[str]): Specific metrics to retrieve (e.g., `["pe_ratio", "debt_to_equity", "fcf"]`).
    *   `period` (str, optional): Timeframe range (e.g., `"Q1_2026"`, `"FY_2025"`).
*   **Response Shape:**
    ```json
    {
      "pe_ratio": { "value": 28.5, "unit": "ratio", "period": "Q1_2026" },
      "debt_to_equity": { "value": 1.2, "unit": "ratio", "period": "Q1_2026" }
    }
    ```

### 2. `query_technical_indicator`
Fetches technical indicators and momentum values.
*   **Arguments:**
    *   `ticker` (str): Symbol.
    *   `indicator` (str): E.g., `"RSI"`, `"MACD"`, `"SMA"`.
    *   `timeframe` (str): E.g., `"daily"`, `"weekly"`.
*   **Response Shape:**
    ```json
    {
      "indicator": "RSI",
      "value": 37.8,
      "status": "OVERSOLD",
      "timestamp": "2026-05-25T16:00:00Z"
    }
    ```

### 3. `search_database_facts`
A semantic/keyword RAG tool to search unstructured context for specific mentions.
*   **Arguments:**
    *   `ticker` (str): Symbol.
    *   `query` (str): Topic to search (e.g., `"insider buying"`, `"congressional trades"`).
*   **Response Shape:**
    ```json
    {
      "results": [
        { "source": "news", "text": "CEO bought 10,000 shares on 2026-05-12", "date": "2026-05-12" }
      ]
    }
    ```

---

## 4. Edge Cases & Mitigations

During execution, the following safeguards must be built into the query tools:

### A. The "Out of Context" Scale Error
*   **Problem:** The tool returns `{"revenue": 100}`. Without knowing if this is $100, $100 Thousand, or $100 Million, the agent's math fails.
*   **Mitigation:** Every tool response MUST return self-describing metadata: value, multiplier (Millions/Billions), currency (USD), and period.

### B. The Sequential Loop Tax (Latency)
*   **Problem:** If the agent needs 5 metrics and queries them one by one in 5 turns, latency explodes.
*   **Mitigation:**
    1.  The query tools accept arrays (`metrics: list[str]`) so multiple metrics are fetched in a single request.
    2.  Prism/vLLM parallel tool calling is enabled, allowing the agent to emit multiple tool calls in a single turn.

### C. Mismatched Timeframes (Temporal Drift)
*   **Problem:** The agent compares 2026 P/E with 2025 Revenue because it didn't specify periods.
*   **Mitigation:** The tools default all queries to the active cycle's target period and validate that retrieved datetimes are consistent.

### D. Synonym & Schema Mapping
*   **Problem:** Agent queries `P/E Ratio` but database column is `price_to_earnings`.
*   **Mitigation:** The database connector uses a mapping dict / synonym resolver (e.g., `"pe"`, `"p/e"`, `"pe_ratio"` -> `price_to_earnings`).

---

## 5. Execution Roadmap

*   **Phase 1 (Tool Registration):** Write the precision query tools in `app/tools` and register them in `app/tools/registry.py`.
*   **Phase 2 (Swarm Configuration):** Modify [tool_whitelists.py](file:///home/lazycat/github/projects/sun/trading-service/app/agents/tool_whitelists.py) to bind these tools to the respective analyst roles.
*   **Phase 3 (Prompt Reduction):** Edit the analyst templates in `debate_coordinator.py` to remove the default dumping of raw structured tables, replacing them with a guide instructing the agent to use tools for numerical values.
*   **Phase 4 (Validation):** Run regression tests to verify that agents successfully query metrics and parse the JSON responses.
