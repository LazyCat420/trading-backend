"""
Centralized constants for the vLLM Trading Bot.
"""

from pathlib import Path

APP_VERSION = "1.0.0"
LOG_DIR = Path("logs")
PIPELINE_LOG_PATH = str(LOG_DIR / "backend_terminal.log")


class TradingAction:
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class PipelinePhase:
    COLLECTING = "collecting"
    ANALYZING = "analyzing"
    TRADING = "trading"
    STARTING = "starting"
    PURGE = "purge"
    STOPPED = "stopped"
    PAUSED = "paused"
    RESUMED = "resumed"


class Status:
    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    SKIPPED = "skipped"
    OK = "ok"


class Source:
    YFINANCE = "yfinance"
    FINNHUB = "finnhub"
    REDDIT = "reddit"
    YOUTUBE = "youtube"
    RSS = "rss"
    OPENBB = "openbb"
    FRED = "fred"
    COINGECKO = "coingecko"
    CONGRESS = "congress"
    SEC = "sec"


class AssetClass:
    STOCK = "stock"
    CRYPTO = "crypto"
    COMMODITY = "commodity"


class Table:
    PRICE_HISTORY = "price_history"
    FUNDAMENTALS = "fundamentals"
    TECHNICALS = "technicals"
    FINANCIAL_HISTORY = "financial_history"
    BALANCE_SHEET = "balance_sheet"
    ASSET_PRICES = "asset_prices"
    MACRO_INDICATORS = "macro_indicators"
    NEWS_ARTICLES = "news_articles"
    REDDIT_POSTS = "reddit_posts"
    YOUTUBE_TRANSCRIPTS = "youtube_transcripts"
    SEC_13F_HOLDINGS = "sec_13f_holdings"
    CONGRESS_TRADES = "congress_trades"
    WATCHLIST = "watchlist"
    POSITIONS = "positions"
    ORDERS = "orders"
    TRADE_FILLS = "trade_fills"
    POSITION_LOTS = "position_lots"
    LOT_CLOSURES = "lot_closures"
    BOTS = "bots"
    LLM_AUDIT_LOGS = "llm_audit_logs"
    ANALYSIS_RESULTS = "analysis_results"


# ══════════════════════════════════════════
# ASI-EVOLVE STRATEGY EVOLUTION CONSTANTS
# ══════════════════════════════════════════
EVOLVE_MODE = "paper"  # "paper" | "disabled" (never "live")
EVOLVE_EPSILON = 0.2  # exploration rate for epsilon-greedy sampling
EVOLVE_SAMPLE_K = 5  # number of past nodes to sample each round
EVOLVE_COGNITION_K = 5  # number of cognition lessons to retrieve
EVOLVE_BACKTEST_WINDOW_START = "2023-01-01"  # fixed OOS backtest window start
EVOLVE_BACKTEST_WINDOW_END = "2024-01-01"  # fixed OOS backtest window end
EVOLVE_BACKTEST_TICKERS = ["SPY", "QQQ"]  # default tickers for backtest
EVOLVE_MIN_WIN_RATE = 0.45  # KEEP gate: minimum win-rate
EVOLVE_MIN_TRADES = 10  # KEEP gate: minimum number of trades
EVOLVE_SIGNAL_ENABLED = True  # inject best_strategy signal into live context
EMBED_SERVER_URL = ""  # e.g. "http://10.0.0.20:8001" — set in .env

# ══════════════════════════════════════════
# AUTO-RESEARCH V2: TRADING CONSTRAINTS
# ══════════════════════════════════════════
TRADING_CONSTRAINTS = """
STRICT TRADING CONSTRAINTS:
1. Fees: Assume 0.005% fee per trade.
2. Slippage: Assume 0.1% slippage on entry and exit.
You MUST explicitly acknowledge these constraints in your rationale if you recommend a BUY.
"""

CYCLE_READONLY_TABLES = [
    "price_history",
    "fundamentals",
    "technicals",
    "financial_history",
    "balance_sheet",
    "macro_indicators",
]

CYCLE_READONLY_FILES = [
    "app/constants.py",
    "app/pipeline/decision_engine.py",
    "app/pipeline/debate_engine.py",
    "app/pipeline/data_phase.py",
]
