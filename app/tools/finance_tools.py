from pydantic import BaseModel, Field
from app.tools.registry import registry
from app.db.connection import get_db
from app.utils.text_utils import format_db_section, fmt_usd


class TickerInput(BaseModel):
    ticker: str = Field(description="The stock ticker symbol (e.g. AAPL)")


@registry.register(
    name="get_market_data",
    description="Get recent price history, fundamentals, financials, and balance sheet for a stock from multiple reliable sources.",
    parameters={
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "The stock ticker symbol (e.g., AAPL)",
            }
        },
        "required": ["ticker"],
    },
    tier=0,
    source="data_rotator",
    input_model=TickerInput,
)
async def get_market_data(ticker: str) -> str:
    from app.collectors.data_rotator import (
        fetch_price_history,
        fetch_fundamentals,
        fetch_financials,
        fetch_balance_sheet,
    )
    from app.services.api_rate_limiter import rate_limiter

    # Still acquire yfinance semaphore just to be safe as it's the primary target
    async with rate_limiter.acquire("yfinance"):
        await fetch_price_history(ticker)
        await fetch_fundamentals(ticker)
        await fetch_financials(ticker)
        await fetch_balance_sheet(ticker)

    with get_db() as db:
        sections = []

        # Fundamentals
        rows = db.execute(
            """
            SELECT snapshot_date, market_cap, pe_ratio, forward_pe, peg_ratio,
                   price_to_book, profit_margin, roe, revenue, revenue_growth,
                   debt_to_equity, beta, week_52_high, week_52_low, short_float_pct
            FROM fundamentals WHERE ticker = %s ORDER BY snapshot_date DESC LIMIT 1
        """,
            [ticker],
        ).fetchall()
        sections.append(
            format_db_section(
                "Fundamentals",
                rows,
                [
                    "Date",
                    "MarketCap",
                    "PE",
                    "ForwardPE",
                    "PEG",
                    "P/B",
                    "ProfitMargin",
                    "ROE",
                    "Revenue",
                    "RevenueGrowth",
                    "D/E",
                    "Beta",
                    "52wHigh",
                    "52wLow",
                    "ShortFloat%",
                ],
            )
        )

        # Financials
        fin_rows = db.execute(
            """
            SELECT period_end, revenue, gross_profit, operating_income, net_income, eps, free_cash_flow
            FROM financial_history WHERE ticker = %s ORDER BY period_end DESC LIMIT 4
        """,
            [ticker],
        ).fetchall()
        if fin_rows:
            fin_lines = ["\n## Recent Financials"]
            for row in fin_rows:
                rev = fmt_usd(row[1]) if row[1] else "N/A"
                ni = fmt_usd(row[4]) if row[4] else "N/A"
                eps = f"EPS=${row[5]:.2f}" if row[5] else ""
                fin_lines.append(f"  {row[0]}: Rev={rev}, Net Income={ni}, {eps}")
            sections.append("\n".join(fin_lines))

    return "\n".join(sections)


@registry.register(
    name="get_finnhub_news",
    description="Get the latest news articles for a stock from Finnhub.",
    parameters={
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "The stock ticker symbol"}
        },
        "required": ["ticker"],
    },
    tier=0,
    source="finnhub",
    input_model=TickerInput,
)
async def get_finnhub_news(ticker: str) -> str:
    from app.collectors.news_collector import collect_finnhub_news
    from app.services.api_rate_limiter import rate_limiter

    async with rate_limiter.acquire("finnhub"):
        await collect_finnhub_news(ticker)

    with get_db() as db:
        rows = db.execute(
            """
            SELECT title, publisher, published_at, COALESCE(llm_summary, summary)
            FROM news_articles WHERE ticker = %s ORDER BY published_at DESC LIMIT 15
        """,
            [ticker],
        ).fetchall()

    if not rows:
        return "No recent news found."

    return format_db_section(
        "Recent News", rows, ["Title", "Publisher", "Date", "Summary"]
    )


@registry.register(
    name="get_technical_indicators",
    description="Get computed technical indicators (RSI, MACD, SMA, Bollinger Bands).",
    parameters={
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "The stock ticker symbol"}
        },
        "required": ["ticker"],
    },
    tier=0,
    source="computed",
    input_model=TickerInput,
)
async def get_technical_indicators(ticker: str) -> str:
    from app.processors.technical_processor import get_signals

    # Assumes price history already populated by yfinance tool OR we trigger it if missing!
    # Wait, technical processor automatically computes it from DB price_history.
    from app.processors.technical_processor import compute_technicals

    try:
        compute_technicals(ticker)
    except Exception:
        pass

    signals = get_signals(ticker)
    return signals if signals else "No technical signals available."
