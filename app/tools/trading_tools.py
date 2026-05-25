import json
import logging
from app.tools.registry import registry, PermissionLevel
from app.trading.paper_trader import buy, sell
from app.trading.watchlist import add_ticker, remove_ticker
from app.config import settings

logger = logging.getLogger(__name__)


@registry.register(
    name="buy_stock",
    description="Execute a buy order for a stock ticker. Requires user confirmation.",
    parameters={
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "The stock ticker symbol (e.g., AAPL).",
            },
            "size_pct": {
                "type": "number",
                "description": "The percentage of available cash to use (e.g., 0.10 for 10%, 1.0 for 100%). Default is 0.10.",
            },
        },
        "required": ["ticker"],
    },
    tier=1,
    source="paper_trader",
    permission=PermissionLevel.WRITE,  # Paper trading — nothing is irreversible
)
async def buy_stock(ticker: str, size_pct: float = 0.10) -> str:
    """Execute a paper buy order."""
    logger.info(
        "[TradingTools] Executing buy order for %s (size: %.2f)", ticker, size_pct
    )
    # Ensure uppercase
    ticker = ticker.upper().strip()

    # Use the dynamically resolved active bot_id (not settings.BOT_ID)
    try:
        from app.services.bot_manager import get_active_bot_id

        bot_id = get_active_bot_id()
    except Exception:
        bot_id = settings.BOT_ID

    try:
        result = await buy(bot_id, ticker, size_pct)
        if "error" in result:
            return json.dumps({"status": "error", "message": result["error"]})
        return json.dumps({"status": "success", "trade": result})
    except Exception as e:
        logger.error("[TradingTools] Buy failed: %s", e)
        return json.dumps({"status": "error", "message": str(e)})


@registry.register(
    name="sell_stock",
    description="Execute a sell order to close a position for a stock ticker. Requires user confirmation.",
    parameters={
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "The stock ticker symbol to sell (e.g., AAPL).",
            }
        },
        "required": ["ticker"],
    },
    tier=1,
    source="paper_trader",
    permission=PermissionLevel.WRITE,  # Paper trading — nothing is irreversible
)
async def sell_stock(ticker: str) -> str:
    """Execute a paper sell order (closes the entire position)."""
    logger.info("[TradingTools] Executing sell order for %s", ticker)
    ticker = ticker.upper().strip()
    # Use the dynamically resolved active bot_id (not settings.BOT_ID)
    try:
        from app.services.bot_manager import get_active_bot_id

        bot_id = get_active_bot_id()
    except Exception:
        bot_id = settings.BOT_ID

    try:
        result = await sell(bot_id, ticker)
        if "error" in result:
            return json.dumps({"status": "error", "message": result["error"]})
        return json.dumps({"status": "success", "trade": result})
    except Exception as e:
        logger.error("[TradingTools] Sell failed: %s", e)
        return json.dumps({"status": "error", "message": str(e)})


@registry.register(
    name="add_to_watchlist",
    description="Add a stock ticker (e.g., from the Discovery list) to the user's active watchlist. Do NOT use this tool to add items TO discovery; discovery is automatic. Requires user confirmation.",
    parameters={
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "The stock ticker symbol to add (e.g., AAPL).",
            }
        },
        "required": ["ticker"],
    },
    tier=1,
    source="watchlist",
    permission=PermissionLevel.WRITE,  # Watchlist edits are reversible
)
def add_to_watchlist(ticker: str) -> str:
    """Add a ticker to watchlist."""
    logger.info("[TradingTools] Executing add to watchlist for %s", ticker)
    ticker = ticker.upper().strip()
    try:
        is_new = add_ticker(ticker, source="chat", notes="Added via AI Strategy Chat")
        msg = f"Added {ticker}" if is_new else f"Reactivated {ticker}"
        return json.dumps({"status": "success", "message": msg, "is_new": is_new})
    except Exception as e:
        logger.error("[TradingTools] Add to watchlist failed: %s", e)
        return json.dumps({"status": "error", "message": str(e)})


@registry.register(
    name="remove_from_watchlist",
    description="Remove a stock ticker from the user's active watchlist. Requires user confirmation.",
    parameters={
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "The stock ticker symbol to remove (e.g., AAPL).",
            }
        },
        "required": ["ticker"],
    },
    tier=1,
    source="watchlist",
    permission=PermissionLevel.WRITE,  # Watchlist edits are reversible
)
def remove_from_watchlist(ticker: str) -> str:
    """Remove a ticker from watchlist."""
    logger.info("[TradingTools] Executing remove from watchlist for %s", ticker)
    ticker = ticker.upper().strip()
    try:
        removed = remove_ticker(ticker)
        if not removed:
            return json.dumps(
                {"status": "error", "message": f"{ticker} not in active watchlist"}
            )
        return json.dumps(
            {"status": "success", "message": f"Removed {ticker} from watchlist"}
        )
    except Exception as e:
        logger.error("[TradingTools] Remove from watchlist failed: %s", e)
        return json.dumps({"status": "error", "message": str(e)})


@registry.register(
    name="get_sec_filings",
    description="Fetch recent SEC filings (10-K, 10-Q, 8-K) for a ticker.",
    parameters={
        "type": "object",
        "properties": {"ticker": {"type": "string"}},
        "required": ["ticker"],
    },
    tier=0,
    source="sec",
)
async def get_sec_filings_tool(ticker: str) -> str:
    from app.collectors.sec_collector import collect_ticker_institutional

    try:
        res = await collect_ticker_institutional(ticker)
        return json.dumps({"status": "success", "holders_collected": res})
    except Exception as e:
        return json.dumps({"error": str(e)})


@registry.register(
    name="get_options_flow",
    description="Fetch unusual options activity and flow for a ticker.",
    parameters={
        "type": "object",
        "properties": {"ticker": {"type": "string"}},
        "required": ["ticker"],
    },
    tier=0,
    source="options",
)
async def get_options_flow_tool(ticker: str) -> str:
    from app.collectors.options_collector import collect_options

    try:
        res = await collect_options(ticker)
        return json.dumps(res)
    except Exception as e:
        return json.dumps({"error": str(e)})


@registry.register(
    name="get_insider_trades",
    description="Fetch recent insider trading activity for a ticker.",
    parameters={
        "type": "object",
        "properties": {"ticker": {"type": "string"}},
        "required": ["ticker"],
    },
    tier=0,
    source="insider",
)
async def get_insider_trades_tool(ticker: str) -> str:
    from app.collectors.insider_collector import collect_insider

    try:
        res = await collect_insider(ticker)
        return json.dumps(res)
    except Exception as e:
        return json.dumps({"error": str(e)})


@registry.register(
    name="get_congress_trades",
    description="Fetch recent congressional trading activity for a ticker.",
    parameters={
        "type": "object",
        "properties": {"ticker": {"type": "string"}},
        "required": ["ticker"],
    },
    tier=0,
    source="congress",
)
async def get_congress_trades_tool(ticker: str) -> str:
    from app.collectors.congress_collector import collect_trades_for_ticker

    try:
        res = await collect_trades_for_ticker(ticker)
        return json.dumps({"status": "success", "trades_collected": res})
    except Exception as e:
        return json.dumps({"error": str(e)})


@registry.register(
    name="get_earnings_data",
    description="Fetch upcoming or recent earnings dates and estimates for a ticker.",
    parameters={
        "type": "object",
        "properties": {"ticker": {"type": "string"}},
        "required": ["ticker"],
    },
    tier=0,
    source="earnings",
)
async def get_earnings_data_tool(ticker: str) -> str:
    from app.collectors.earnings_collector import collect_earnings

    try:
        res = await collect_earnings(ticker)
        return json.dumps(res)
    except Exception as e:
        return json.dumps({"error": str(e)})




@registry.register(
    name="get_finviz_fundamentals",
    description="Fetch fundamental data from Finviz.",
    parameters={
        "type": "object",
        "properties": {"ticker": {"type": "string"}},
        "required": ["ticker"],
    },
    tier=0,
    source="finviz",
)
async def get_finviz_fundamentals_tool(ticker: str) -> str:
    from app.collectors.finviz_scraper import collect_fundamentals
    from app.db.connection import get_db
    import json

    try:
        success = await collect_fundamentals(ticker)
        if success:
            with get_db() as db:
                row = db.execute(
                    "SELECT * FROM fundamentals WHERE ticker = %s ORDER BY snapshot_date DESC LIMIT 1",
                    [ticker],
                ).fetchone()
                if row and db.description:
                    cols = [column[0] for column in db.description]
                    return json.dumps(dict(zip(cols, row)), default=str)
        return json.dumps({"error": "Failed to collect fundamentals from finviz"})
    except Exception as e:
        logger.error("[TradingTools] Finviz fundamentals failed for %s: %s", ticker, e)
        return json.dumps({"error": str(e)})


@registry.register(
    name="get_polygon_price_history",
    description="Fetch historical OHLCV price data from Polygon.",
    parameters={
        "type": "object",
        "properties": {"ticker": {"type": "string"}},
        "required": ["ticker"],
    },
    tier=0,
    source="polygon",
)
async def get_polygon_price_history_tool(ticker: str) -> str:
    from app.collectors.polygon_collector import collect_all

    try:
        res = await collect_all(ticker)
        return json.dumps(res)
    except Exception as e:
        return json.dumps({"error": str(e)})
