# Trading Cycle Backend

Autonomous trading cycle engine, isolated from the web frontend.

## Architecture

```
sun/
├── vllm-trading-bot/          ← Frontend: FastAPI + Next.js dashboard
│   ├── app/                   ← Shared Python codebase
│   ├── frontend/              ← Next.js source
│   └── Dockerfile             ← Builds frontend container
│
├── trading-cycle-backend/     ← THIS REPO: cycle engine
│   ├── cycle_main.py          ← Standalone cycle entrypoint
│   ├── Dockerfile             ← Builds backend container
│   ├── entrypoint.sh          ← Docker entrypoint
│   └── README.md
│
└── docker-compose.yml         ← Runs both containers
```

## What runs here

The full trading cycle: **Collect → Analyze → Trade**

| Phase | What it does |
|-------|-------------|
| Phase 1 | System health checks |
| Phase 2 | Data collection (news, Reddit, YouTube, yfinance, RSS) |
| Phase 3 | Macro environment scan |
| Phase 4 | AI analysis (evidence → thesis → debate → decision) |
| Phase 5 | Trade execution (paper trading) |
| Phase 6 | Post-cycle (learning, benchmarks, cleanup) |

## Usage

### Run locally (dev)

```bash
# From the sun/ directory
cd vllm-trading-bot
source .venv/bin/activate

# Run one cycle for AAPL
python ../trading-cycle-backend/cycle_main.py --once --tickers AAPL

# Run scheduler (every 30 minutes)
python ../trading-cycle-backend/cycle_main.py --interval 30
```

### Run with Docker

```bash
# Build both containers
cd sun/
docker compose build

# Start both
docker compose up -d

# View cycle backend logs
docker logs -f trading-cycle-backend

# Run one debug cycle
docker compose run --rm trading-cycle-backend \
    python cycle_main.py --once --tickers AAPL

# Restart only the cycle backend (frontend stays up)
docker compose restart trading-cycle-backend
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CYCLE_INTERVAL` | `30` | Minutes between cycles |
| `CYCLE_TICKERS` | auto-select | Comma-separated ticker override |
| `CYCLE_ONCE` | `false` | Set to `true` for single-shot mode |
| `DATABASE_URL` | — | PostgreSQL connection string |

## Debugging

The whole point of this separation: you can restart, debug, and iterate
on the cycle backend without touching the dashboard.

```bash
# Attach to the cycle backend and watch it run
docker logs -f trading-cycle-backend

# Run a single cycle interactively
docker compose run --rm trading-cycle-backend \
    python cycle_main.py --once --tickers AAPL 2>&1 | tee debug.log

# Check if the cycle is writing to the database
docker compose exec trading-cycle-backend \
    python -c "from app.db.connection import get_db; db=get_db().__enter__(); print(db.execute('SELECT COUNT(*) FROM news_articles').fetchone())"
```

## vLLM Multi-Endpoint Routing

The trading pipeline distributes LLM requests across multiple hardware endpoints.
Each endpoint runs its **own model** — model names are NOT interchangeable between boxes.

### Hardware Endpoints

| Endpoint | IP | Role | What it runs |
|----------|-----|------|-------------|
| `jetson` | `10.0.0.30:8000` | `collector` | Lightweight tasks (summarization, curation, agents) |
| `dgx_spark` | `10.0.0.141:8000` | `trader` | Heavy tasks (trading decisions, debate) |
| `dgx_spark_2` | `10.0.0.103:8000` | `analyst` | Heavy tasks (deep analysis, RLM) |

### The Golden Rule: Endpoint-First Routing

> **NEVER pick a model name first, then find an endpoint for it.**
> **ALWAYS pick the endpoint first, then use that endpoint's model.**

This is the single most important invariant in `vllm_client.py`. Breaking it causes
404 errors because a model name from one box (e.g., Qwen on Jetson) gets sent to a
different box (e.g., DGX running Nemotron) which doesn't host that model.

### How Routing Works

```
1. chat() / chat_with_tools() / chat_stream() called
2. Pick target endpoint:
   - endpoint_override? → use that endpoint
   - model_override?    → find endpoint hosting that model (_pick_best_endpoint)
   - neither?           → pick least-loaded endpoint (_pick_best_endpoint)
3. Set payload model = target_ep.model  ← ALWAYS from the endpoint
4. Enqueue to that endpoint's queue
5. Dispatcher drains queue → sends to endpoint URL with correct model name
```

### What Each Endpoint Discovers

At startup (and every 120s via rediscovery), `discover_roles()` queries each
endpoint's `/v1/models` API:

```
GET http://10.0.0.30:8000/v1/models   → "Kbenkhaled/Qwen3.5-35B-A3B-quantized.w4a16"
GET http://10.0.0.141:8000/v1/models  → "nvidia/Nemotron-4-340B" (example)
GET http://10.0.0.103:8000/v1/models  → "nvidia/Nemotron-4-340B" (example)
```

The discovered model name is stored in `ep.model` and is the ONLY name used
when sending requests to that endpoint. The `ACTIVE_MODEL` env var is just a
seed fallback — it is always overridden by live discovery.

### Concurrency Limits

Each DGX Spark is capped at **8 concurrent requests**. Beyond 8, token generation
drops below 3 tok/s which causes batch timeouts and pipeline stalls.

| Setting | Value | Why |
|---------|-------|-----|
| `DGX_MAX_CONCURRENT` | 8 | >8 drops below 3 tok/s |
| `DGX_SPARK_2_MAX_CONCURRENT` | 8 | Same hardware, same limit |
| `DGX_BATCH_SIZE` | 8 | Match max_concurrent (no over-draining) |
| `DGX_SPARK_2_BATCH_SIZE` | 8 | Same |
| `JETSON_MAX_CONCURRENT` | 24 | Jetson handles 24 concurrent fine |
| `ADAPTIVE_MAX_CONCURRENCY` | 8 | Dynamic ceiling matches DGX cap |

### Common Mistakes That Break Routing

1. **Using `get_least_busy_model()` to pick a model name** — This returns the
   model from whichever endpoint is least busy. If the Jetson is idle, it returns
   the Jetson's model name. That name then leaks into a request destined for the
   DGX → 404.

2. **Setting `payload["model"]` before choosing the endpoint** — The model name in
   the payload MUST come from `target_ep.model`, never from a cross-endpoint lookup.

3. **Injecting `chat_template_kwargs` for non-Qwen models** — Only Qwen models
   support `enable_thinking`. Check `_is_qwen_model(effective_model)` AFTER
   `effective_model` is set from the target endpoint, not before.
