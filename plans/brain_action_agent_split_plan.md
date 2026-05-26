# Plan: Split-Agent Architecture (Brain vs. Action Agent)

This document details the architectural plan to split our agent pipeline into a two-tier model: a high-level **Brain Agent (Planner/Reasoner)** and a temporary, isolated **Action Agent (Executor)**. This design addresses token-exhaustion issues, leverages rolling context windows, and maintains high reasoning performance by isolating raw tool executions.

---

## 🛑 Problem Statement

Under the current monolithic agent loop (`run_agent_loop` in `agent_loop.py`):
1. **Context Window Inflation:** When an agent decides to execute a series of tools (e.g., retrieving technicals, reading news articles, running database queries), the raw outputs are appended to the agent's active conversation history.
2. **Token Exhaustion:** If tool outputs are verbose or multi-turn sequence outputs are large, the agent rapidly fills up its context length. This causes the agent to run out of tokens before it can generate its final trading decision.
3. **Attention Dilution:** Carrying pages of database outputs, SQL logs, or article summaries degrades the model's high-level planning and reasoning capability (the "lost-in-the-middle" problem).

---

## 💡 The Brain-Action Split Architecture

Instead of running a single agent that reasons, plans, and executes tools in the same conversation, we split these tasks into two specialized agents running in separate conversation sessions:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        MAIN ORCHESTRATION LOOP                         │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        BRAIN AGENT (Reasoner)                          │
│                                                                        │
│  - Prompts: Persona, target ticker, overall goal                       │
│  - State: Full conversation history, high-level planning               │
│  - Action: Decides *what* data is needed; emits a structured query     │
│  - CONTEXT: Clean of raw tool outputs (High-level only)                │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼ (Orchestrator Intercepts query JSON)
┌────────────────────────────────────────────────────────────────────────┐
│                    SPAWN FRESH SESSION / NEW CHET                      │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        ACTION AGENT (Executor)                         │
│                                                                        │
│  - Prompts: Instruction payload from Brain, tool definitions           │
│  - State: Temporary, short-lived session (Clean slate)                 │
│  - Action: Executes tool(s), catches errors, processes raw tables      │
│  - CONTEXT: Disposable; only lives during the tool execution loop      │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼ (Compiles raw data into tight summary)
┌────────────────────────────────────────────────────────────────────────┐
│                       AGGREGATOR / CONDENSER                           │
│                                                                        │
│  - Formats results into a clean, dense, high-level JSON summary        │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼ (Feeds dense summary back to Brain)
┌────────────────────────────────────────────────────────────────────────┐
│                        BRAIN AGENT (Reasoner)                          │
│                                                                        │
│  - Resumes execution with the summarized answer as input               │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Detailed Handoff & Execution Flow

1. **Planning Step:** The **Brain Agent** evaluates its instructions. When it needs data or actions, it does not call tools directly. Instead, it generates a JSON block detailing the requested sub-task.
2. **Orchestration Interception:** The Python coordinator parses this JSON and pauses the Brain Agent's loop.
3. **Worker Spawning:** The coordinator spins up the **Action Agent** in a brand-new, clean chat session.
4. **Tool Execution:** The Action Agent receives the instruction, identifies the right tools to use, calls them, handles any retries/errors, and runs until completion.
5. **Summarization:** The Action Agent takes the raw tool output (e.g., a 100-line database table or a long webpage scrape) and produces a concise summary (e.g., "The P/E is 21.3; 3 directors sold a total of 15k shares last week; RSI is 38.5 (oversold)").
6. **Brain Resumption:** The condensed summary is sent back to the Brain Agent as a mock tool response. The Brain Agent uses this summary to refine its analysis and make the final trading decision.

---

## ⚙️ Handoff Schemas (Technical Specs)

### 1. Brain Agent Request Schema
When the Brain decides to query data, it outputs a JSON command targeting a specific executor:

```json
{
  "executor": "financial_analyst_worker",
  "task_description": "Verify the P/E ratio, net debt-to-equity ratio, and latest quarterly free cash flow trend for Apple.",
  "ticker": "AAPL",
  "required_keys": ["pe_ratio", "debt_to_equity", "fcf_trend"]
}
```

### 2. Action Agent Response Schema
After completing the tools loop, the Action Agent compiles and outputs its findings:

```json
{
  "status": "success",
  "summary": "Verified AAPL metrics: P/E is currently 29.5 (slightly above 5-year average of 26.2). Net debt-to-equity is 1.15 (healthy leverage). FCF trend shows continuous growth over the last 3 quarters ($23.2B -> $25.1B -> $27.4B).",
  "data": {
    "pe_ratio": 29.5,
    "debt_to_equity": 1.15,
    "fcf_trend": ["$23.2B", "$25.1B", "$27.4B"]
  }
}
```

---

## 🧠 Advantages & Rolling Window Leverage

1. **Context Safety:** The Brain Agent's history contains only high-level dialogue and dense summaries. Raw data dumps never clutter its prompt, preventing it from running out of tokens or losing focus.
2. **Maximized Output Space:** The Action Agent operates in a fresh session with `0` previous message history. The model has its entire context window and max output token limit available to process raw data and formulate the execution logic.
3. **Self-Healing Isolation:** If the Action Agent makes a bad tool call or generates python code that throws a syntax error, it can self-heal and retry *within its isolated thread* without polluting the Brain Agent's history with long error logs and code drafts.
4. **Parallelizable Executions:** The Brain can issue multiple data-gathering requests at once. The coordinator can run these Action Agents in parallel, minimizing total latency.

---

## ⚠️ Potential Pitfalls & Mitigations

| Pitfall | Impact | Mitigation |
| :--- | :--- | :--- |
| **Increased Latency** | Decoupling requests into new agent sessions adds API roundtrip latency. | Use parallel execution for independent requests; keep the Action Agent's system prompt extremely lightweight to lower TTFT (Time to First Token). |
| **Context Blindness** | The Action Agent does not know the "why" behind the task, which could lead to rigid or incorrect parameter selections. | The Brain must supply the `task_description` and `required_keys` explicitly in the delegation schema. |
| **Multi-Turn Ping-Pong** | If the Action Agent's summary is insufficient, the Brain has to spawn another query, creating an expensive back-and-forth loop. | Require the Action Agent to validate its output against the Brain's `required_keys` before finishing. |
| **Debugging Complexity** | Tracing issues across multiple disconnected chat sessions is difficult. | Group all Action Agent child sessions under the parent Brain Agent's `cycle_id` and track them with linked traces in the database. |

---

## 🛠️ Implementation Strategy in `trading-service`

To implement this without breaking existing functionality, we propose the following changes:

1. **Create Action Executor Persona:** Register a new agent role in `app/services/vllm_client.py` and `app/agents/tool_whitelists.py` called `action_executor`. This executor is whitelisted for all data-fetching and python execution tools.
2. **Update `agent_loop.py`:** Add a helper function `run_isolated_action_agent(...)` that spins up a fresh, short-lived session with only the action instruction and executing tools.
3. **Introduce Delegation Tools:** Create a new tool called `delegate_task(executor, task_description, required_keys)` and register it in `app/tools/registry.py`. When the Brain Agent calls `delegate_task`, `agent_loop.py` handles spawning the worker agent, captures its output, and returns it to the Brain.
4. **Modify Prompt Templates:** Update the Brain Agents' (e.g. `quant_research_agent`, `pre_trade_agent`) system instructions to guide them to call `delegate_task` whenever they need to fetch or parse large chunks of raw data, rather than calling the raw collection tools directly.

---

## 🧪 A/B Test Prototyping Results

We ran a live A/B test simulation (`scratch/test_tool_selection_efficiency.py`) routing requests to the local vLLM endpoint (`Kbenkhaled/Qwen3.5-35B-A3B-quantized.w4a16`) using the 16-tool pool of the `retriever` agent.

### Task Tested:
> *"Verify AAPL stock's recent performance. Retrieve its 14-day RSI (relative strength index) indicator, find its current market price, and check the latest news headlines."*

### Empirical Comparison:

| Metric | Scenario A (Control - Monolithic) | Scenario B (Variant - Split Agent) | Difference / Savings |
| :--- | :--- | :--- | :--- |
| **Total Tokens** | 1,781 tokens | 1,203 tokens | **-578 tokens (32.45% reduction)** |
| **- Selection Step** | *N/A* | 578 tokens | - |
| **- Action Step** | *N/A* | 625 tokens | - |
| **Execution Latency** | 6.24 seconds | 4.22 seconds | **-2.02 seconds (32.37% faster)** |
| **- Selection Step** | *N/A* | 1.31 seconds | - |
| **- Action Step** | *N/A* | 2.90 seconds | - |
| **Tool Calling Accuracy**| `['get_technical_indicators', 'get_finnhub_news', 'get_market_data']` | `['get_technical_indicators', 'get_finnhub_news', 'get_market_data']` | **100% Identical Output** |

### Key Takeaways:
1. **Zero Accuracy Decay:** The variant selected and called the exact same tools as the control.
2. **Substantial Token Savings:** A **32.45% reduction** in context footprint. On multi-turn loops, this savings compounds heavily, keeping the agents well within their safety thresholds.
3. **Faster Execution:** Shorter prompt payloads directly translate to lower processing time and faster Time-to-First-Token.

---

## ❓ Questions for Collaboration

1. **Tool Access:** Should the Brain Agent have *any* direct data-retrieval tools, or should it be completely stripped of tools other than `delegate_task`? (Completely stripping tools enforces the clean-context model but prevents quick queries).
2. **Concurrency Limits:** Since we are in WSL and deploying to Synology NAS containers, running parallel Action Agents will increase concurrent load on the vLLM server (Jetson/DGX). Are we okay with configuring a strict concurrent request cap specifically for background workers?
3. **Execution Mode:** Should the Action Agent run with full multi-turn capability (allowing it to run a loop of tools in its thread), or should it be a fast, single-turn tool router where it generates all tool calls at once and summarizes the results immediately?

