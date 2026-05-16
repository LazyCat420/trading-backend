"""
Shared ticker classifications — single source of truth.

Fix #9: Consolidates CRYPTO_TICKERS and COMMODITY_TICKERS that were
duplicated across decision_engine.py and context_builder.py.

Import from here:
    from app.config.config_tickers import CRYPTO_TICKERS, COMMODITY_TICKERS, ALT_ASSET_TICKERS
"""

# Static Data - Not overridable via environment variables
# Known crypto tickers (no fundamentals like P/E, D/E)
# NOTE: "ETH" IS included. While ETH is also the Grayscale Ethereum Mini
# Trust ETF on US exchanges, yfinance resolves ETH as quoteType=CRYPTOCURRENCY
# and the pipeline primarily discovers ETH from crypto-context articles
# (e.g. "whale opens $90M long bets as ETH price chart eyes $3.2K").
# CoinGecko stores Ethereum prices in asset_prices. If the ETH *ETF* needs
# tracking, use its full ticker or an alias system.
CRYPTO_TICKERS = {"BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX"}

# Commodity tickers (tradeable via paper trader)
COMMODITY_TICKERS = {"GOLD", "OIL", "SILVER", "COPPER", "NATGAS", "WHEAT"}

# Combined: skip fundamental agent for these (no P/E, D/E, etc.)
ALT_ASSET_TICKERS = CRYPTO_TICKERS | COMMODITY_TICKERS


def classify_asset(ticker: str) -> str:
    """Classify asset type from ticker symbol: 'crypto', 'commodity', or 'stock'.

    Previously duplicated inline in:
        - decision_engine.py (ternary)
        - agent_execution.py (ternary)
        - paper_trader.py (ternary)
        - sector_collector.py (function)
        - context_builder.py / context_builder.py (if checks)
    """
    t = ticker.upper()
    if t in CRYPTO_TICKERS:
        return "crypto"
    if t in COMMODITY_TICKERS:
        return "commodity"
    return "stock"
