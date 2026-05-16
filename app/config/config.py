"""
Central configuration for the vLLM Trading Bot.
All settings loaded from .env — no hardcoded values anywhere else.
"""

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings


ROOT_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT_DIR / ".env"

# Ensure data directories exist
DATA_DIR = ROOT_DIR / "data"
MEMORY_DB_PATH = DATA_DIR / "memory.db"


class Settings(BaseSettings):
    # ── Jetson/DGX vLLM ──
    JETSON_VLLM_URL: str = "http://10.0.0.30:8000"
    DGX_SPARK_VLLM_URL: str = "http://10.0.0.141:8000"
    DGX_SPARK_2_VLLM_URL: str = "http://10.0.0.103:8000"
    ACTIVE_MODEL: str = ""  # Auto-discovered from vLLM /v1/models at startup

    # ── Concurrency (tuned from saturation benchmarks — see tests/benchmarks/outputs/) ──
    JETSON_MAX_CONCURRENT: int = 24  # user verified max
    DGX_MAX_CONCURRENT: int = 8  # capped: >8 concurrent drops below 3 tok/s
    DGX_SPARK_2_MAX_CONCURRENT: int = 8  # capped: >8 concurrent drops below 3 tok/s
    RLM_MAX_CONCURRENT: int = (
        2  # max concurrent RLM sessions (uses own client, occupies slots)
    )

    # ── Batch Dispatch (prevents queue overload) ──
    # Items are drained from the queue in batches, not one-at-a-time.
    # Each batch completes before the next is dispatched.
    JETSON_BATCH_SIZE: int = 24       # Jetson Orin AGX 64GB — 24 concurrent max
    DGX_BATCH_SIZE: int = 8            # DGX Spark — matched to max_concurrent
    DGX_SPARK_2_BATCH_SIZE: int = 8    # DGX Spark 2 — matched to max_concurrent
    BATCH_TIMEOUT: int = 60           # 60s per batch (Jetson inference is 5-20s; prevents queue backup)
    BATCH_CIRCUIT_BREAKER_THRESHOLD: int = 3  # consecutive failed batches → disable endpoint 60s

    # ── Adaptive Concurrency (caller-side LLM throttling) ──
    ADAPTIVE_MIN_CONCURRENCY: int = 4   # floor when KV cache pressure is high (>80%)
    ADAPTIVE_MAX_CONCURRENCY: int = 8   # ceiling when cache pressure is low (<60%)

    # ── Pipeline ──
    MAX_ANALYSIS_TICKERS: int = 30  # hard cap on tickers per cycle
    MAX_CYCLE_TICKERS: int = 0  # 0 = unlimited; 1-N caps total tickers for fast testing
    MIN_MARKET_CAP: float = 50_000_000  # $50M floor — reject OTC/penny
    CYCLE_TIMEOUT_MINUTES: int = 120  # 2-hour hard cap per cycle
    V2_TICKER_CONCURRENCY: int = (
        3  # parallel tickers — sized for Jetson (3 tickers × ~44 calls ≈ 12 in-flight, fits 24 slots)
    )
    VLLM_FUTURE_TIMEOUT: int = 180  # seconds before a hung LLM future is killed (aligned with batch timeout)
    ANALYSIS_WORKER_TIMEOUT_SECONDS: int = (
        300  # 5-min hard cap per ticker — aligned with debate timeout
    )
    BOT_ID: str = "lazy-trader-v4"
    COLLECTION_MAX_CONCURRENT: int = 5  # parallel per-ticker scrapers

    # Pipeline modes:
    #   "scout"      — wait for all data, run macro scout in parallel, then analyze (recommended)
    #   "sequential" — wait for all data, then analyze (no macro scout)
    #   "overlap"    — start analysis as each ticker finishes collection (legacy)
    PIPELINE_MODE: str = "scout"
    PIPELINE_VERSION: str = "v2"  # "v1" | "v2" | "ab"
    PIPELINE_BENCHMARK_GROUP: str = "baseline"
    MACRO_SCOUT_ENABLED: bool = True  # enable/disable macro strategy scout

    # ── Queue & Utility ──
    PIPELINE_QUEUE_HIGH_WATERMARK: int = 200
    PIPELINE_QUEUE_LOW_WATERMARK: int = 100
    EMBEDDING_SERVER_URL: str = "http://localhost:8001/embed"
    REDIS_URL: str = "redis://localhost:6379"

    # ── Per-API Concurrency Limits ──
    # Caps concurrent requests to each external service when multiple
    # tickers collect data in parallel. Prevents IP bans and API rate limits.
    YFINANCE_MAX_CONCURRENT: int = 3  # yfinance HTTP (no auth, IP-based)
    FINNHUB_MAX_CONCURRENT: int = 5  # finnhub API (60 calls/min free tier)
    REDDIT_MAX_CONCURRENT: int = 2  # reddit public JSON (no auth, conservative)
    YOUTUBE_MAX_CONCURRENT: int = 2  # yt-dlp subprocess (CPU + network heavy)

    # ── LLM Curation (Pass 2.7) ──
    LLM_CURATION_ENABLED: bool = True  # toggle on/off
    LLM_CURATION_MAX_PROMOTE: int = 5  # max tickers promoted per cycle
    LLM_CURATION_FALLBACK: str = "pass_all"  # "pass_all" | "block_all" on failure

    # ── Watchlist Health & Auto-Purge ──
    WATCHLIST_PURGE_ENABLED: bool = True  # toggle on/off
    WATCHLIST_MAX_PURGE: int = 2  # max tickers purged per cycle
    WATCHLIST_PURGE_MIN_SCORE: int = 30  # only purge below this health score
    WATCHLIST_GRACE_CYCLES: int = 3  # new tickers get N cycles before scoring

    # ── Smart Ticker Triage ──
    TRIAGE_ENABLED: bool = True  # toggle triage on/off (flat list if disabled)
    TRIAGE_GLANCE_HOURS: int = 24  # analyzed within N hours → Glance tier
    TRIAGE_DEEP_HOURS: int = 72  # not analyzed in N hours → Deep tier
    TRIAGE_NEGLECT_MAX_DAYS: int = 5  # flag neglected after N days
    TRIAGE_MAX_CONSECUTIVE_GLANCE: int = 5  # force Standard after N Glance skips
    TRIAGE_DEEP_NEWS_VOLUME: int = 5  # >= N news articles in 24h → Deep tier

    # ── Alpha Decay Purge (Mathematical Pruning) ──
    ALPHA_DECAY_ENABLED: bool = True  # toggle fundamental math purge
    ALPHA_MAX_DEBT_TO_EQUITY: float = (
        50.0  # purge > 5000% debt (catches true rot, not leverage)
    )
    ALPHA_MIN_CURRENT_RATIO: float = (
        0.3  # purge if assets can't cover 30% short-term liabilities
    )
    ALPHA_MAX_52_WK_DRAWDOWN: float = 0.85  # purge if down 85% from 52-week high
    ALPHA_PENNY_FLOOR: float = (
        3.00  # actively purge if price crashes below $3 (deep OTC)
    )
    ALPHA_EXEMPT_DEBT_SECTORS: list[
        str
    ] = [  # these inherently run high debt; ignore D/E rules
        "Financial Services",
        "Real Estate",
        "Banks",
        "Utilities",
        "Energy",
    ]

    # ── Paper Trading ──
    STARTING_CASH: float = 100000.0

    # ── Janitor Agent (Data Hygiene) ──
    AUDIT_LOG_TTL_DAYS: int = 14  # Delete llm_audit_logs older than this
    NEWS_DUPLICATE_TTL_DAYS: int = 30  # Delete duplicate news older than this
    LESSON_CONSOLIDATION_THRESHOLD: int = 50  # Consolidate when lessons exceed this

    # ── Database ──
    DATABASE_URL: str = (
        "postgresql://localhost:5432/trading_bot"
    )
    TEST_DATABASE_URL: str = (
        "postgresql://localhost:5432/trading_bot_test"
    )

    # ── Finnhub ──
    FINNHUB_API_KEY: str = ""

    # ── FRED (Federal Reserve) ──
    FRED_API_KEY: str = ""

    # ── Financial Modeling Prep ──
    FMP_API_KEY: str = ""

    # ── EIA (Energy Information Administration) ──
    EIA_API_KEY: str = ""

    # ── News API Rotator Keys ──
    MARKETAUX_API_KEY: str = ""
    NEWSAPI_API_KEY: str = ""
    ALPHAVANTAGE_API_KEY: str = ""
    POLYGON_API_KEY: str = ""
    MASSIVE_API_KEY: str = ""  # Polygon rebranded to Massive — same API
    GNEWS_API_KEY: str = ""
    CURRENTS_API_KEY: str = ""
    THENEWSAPI_KEY: str = ""
    WORLDNEWSAPI_KEY: str = ""
    STOCKDATA_API_KEY: str = ""
    TWELVEDATA_API_KEY: str = ""

    # ── AISStream (real-time vessel tracking) ──
    AISSTREAM_API_KEY: str = ""

    # ── War/Oil Intelligence Map ──
    GDELT_POLL_INTERVAL_MIN: int = 15
    AIS_POLL_INTERVAL_MIN: int = 5
    WAR_CONTEXT_ENABLED: bool = True

    # ── Prism AI Gateway (MongoDB mirror) ──
    PRISM_URL: str = "http://localhost:7777"
    PRISM_PROJECT: str = "vllm-trading-bot"
    PRISM_USERNAME: str = "lazy-trader"
    PRISM_ENABLED: bool = True
    PRISM_AGENT: str = "CUSTOM_MARKET_ALPHA"  # Routes through the CUSTOM_MARKET_ALPHA persona in Prism — custom agent with tailored trading tools
    PRISM_AGENT_ROUTING: bool = False  # False = direct vLLM + offline sync (avoids Prism's agentic loop conflicting with local tool execution)
    PRISM_MONGO_URI: str = "mongodb://10.0.0.16:27017/?directConnection=true"
    PRISM_MONGO_DB: str = "prism"
    OFFLINE_SYNC_ENABLED: bool = True  # Toggle offline shadow-logging to Prism (can also be toggled at runtime via API)

    # ── SEC 13F Tracking ──
    SEC_USER_AGENT: str = "vllm-trading-bot analysis@example.com"
    SEC_13F_MAX_FILERS: int = 0  # 0 means scrape all

    # ── Prism Working Memory ──
    WORKING_MEMORY_MAX_SLOTS: int = 18

    # ── Tool Calling Bypass ──
    USE_TOOL_CALLING: bool = False

    # ── Hermes Agent (Hub-and-Spoke) ──
    JETSON_HERMES_HOST: str = "10.0.0.30"
    JETSON_HERMES_PORT: int = 8642

    DGX_SPARK_HERMES_HOST: str = "10.0.0.141"
    DGX_SPARK_HERMES_PORT: int = 8642

    DGX_SPARK_2_HERMES_HOST: str = "10.0.0.103"
    DGX_SPARK_2_HERMES_PORT: int = 8642

    API_SERVER_KEY: str = "change-me-local-dev"

    @field_validator("API_SERVER_KEY")
    def warn_default_key(cls, v):
        if v == "change-me-local-dev":
            import warnings

            warnings.warn(
                "API_SERVER_KEY is set to the default insecure value!", stacklevel=2
            )
        return v

    @property
    def HERMES_ENDPOINT_MAP(self) -> dict[str, str]:
        """Map endpoint keys (jetson, dgx_spark, dgx_spark_2) to their Hermes URLs."""
        mapping = {}
        if self.JETSON_HERMES_HOST:
            mapping["jetson"] = (
                f"http://{self.JETSON_HERMES_HOST}:{self.JETSON_HERMES_PORT}/v1/chat/completions"
            )
        if self.DGX_SPARK_HERMES_HOST:
            mapping["dgx_spark"] = (
                f"http://{self.DGX_SPARK_HERMES_HOST}:{self.DGX_SPARK_HERMES_PORT}/v1/chat/completions"
            )
        if self.DGX_SPARK_2_HERMES_HOST:
            mapping["dgx_spark_2"] = (
                f"http://{self.DGX_SPARK_2_HERMES_HOST}:{self.DGX_SPARK_2_HERMES_PORT}/v1/chat/completions"
            )
        return mapping

    # ── JIT Scraper Queue ──
    SCRAPER_MAX_QUEUE_SIZE: int = 1000
    SCRAPER_JIT_PRIORITY: int = 1  # highest priority (blocks analysis)
    SCRAPER_ROUTINE_PRIORITY: int = 5  # routine sweep priority
    SCRAPER_MAX_RETRIES: int = 3
    SCRAPER_WORKER_POLL_SECS: int = 5  # how often workers poll for new requests

    # ── Data Lifecycle ──
    RAW_DATA_TTL_HOURS: int = 72  # raw content kept for 72h
    ARCHIVE_TTL_DAYS: int = 30  # archived summaries kept for 30d
    MAX_ANALYSES_PER_RECORD: int = 5  # multi-angle re-analysis cap per record

    # ── Re-Analysis ──
    REANALYSIS_ENABLED: bool = False  # gate: disabled until Phase 3 verified
    REANALYSIS_SLOT_PCT: float = 0.60  # % of analysis slots for re-analysis
    FRESH_DATA_SLOT_PCT: float = 0.40  # % of slots for fresh data analysis

    # ── Strategy Ranking ──
    MIN_TRADES_BEFORE_BENCH: int = 10  # need N trades before benching
    WIN_RATE_BENCH_THRESHOLD: float = 0.40  # bench prompts below 40% win rate
    WIN_RATE_BONUS_THRESHOLD: float = 0.55  # bonus confidence for >55% win rate

    # ── Meta-Agent ──
    META_AGENT_ENABLED: bool = False  # gate: disabled until Phase 6 verified
    META_AGENT_INTERVAL_HOURS: int = 6  # how often the meta-agent runs
    MAX_ACTIVE_GENERATED_PROMPTS: int = 20  # cap on active generated lenses

    # ── P&L Evaluation Intervals ──
    TRADE_EVAL_INTERVALS_DAYS: list[int] = [1, 3, 7, 14]

    model_config = {
        "env_file": str(ENV_FILE),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
