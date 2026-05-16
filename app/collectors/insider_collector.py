"""
Insider transactions collector — fetches insider buy/sell data from yfinance.

Provides insider trading signals for context enrichment.

Usage:
    from app.collectors.insider_collector import collect_insider
    data = await collect_insider("NVDA")
"""

import asyncio
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


async def collect_insider(ticker: str) -> dict | None:
    """Fetch insider transaction summary for a ticker.

    Returns dict with buy_count, sell_count, net_signal, etc.
    """
    try:
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, lambda: _fetch_insider(ticker))
        if data:
            logger.info(
                "insider_collector: %s buys=%d sells=%d",
                ticker,
                data.get("buy_count", 0),
                data.get("sell_count", 0),
            )
        return data
    except Exception as e:
        logger.warning("insider_collector: %s failed: %s", ticker, e)
        return None


def _fetch_insider(ticker: str) -> dict | None:
    """Synchronous insider fetch via yfinance."""
    try:
        import yfinance as yf

        t = yf.Ticker(ticker)

        # yfinance provides insider_transactions and insider_purchases
        try:
            insiders = t.insider_transactions
        except Exception:
            insiders = None

        if insiders is None or insiders.empty:
            return None

        buys = 0
        sells = 0
        buy_value = 0
        sell_value = 0
        transactions = []

        for _, row in insiders.head(20).iterrows():
            text = str(row.get("Text", "")).lower()
            shares = abs(int(row.get("Shares", 0) or 0))
            value = abs(float(row.get("Value", 0) or 0))
            insider = str(row.get("Insider", "Unknown"))

            if "purchase" in text or "buy" in text or "acquisition" in text:
                buys += 1
                buy_value += value
                action = "BUY"
            elif "sale" in text or "sell" in text or "disposition" in text:
                sells += 1
                sell_value += value
                action = "SELL"
            else:
                action = "OTHER"

            transactions.append(
                {
                    "insider": insider,
                    "action": action,
                    "shares": shares,
                    "value": value,
                }
            )

        total = buys + sells
        if total == 0:
            return None

        buy_ratio = buys / total if total else 0
        net_signal = (
            "BULLISH"
            if buy_ratio > 0.6
            else "BEARISH"
            if buy_ratio < 0.4
            else "NEUTRAL"
        )

        return {
            "ticker": ticker,
            "buy_count": buys,
            "sell_count": sells,
            "buy_value": buy_value,
            "sell_value": sell_value,
            "buy_ratio": round(buy_ratio, 2),
            "net_signal": net_signal,
            "transactions": transactions[:10],
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.warning("insider_collector: yfinance failed for %s: %s", ticker, e)
        return None


def format_insider_context(data: dict) -> str:
    """Format insider data as a context section for the LLM."""
    if not data:
        return ""

    lines = [f"\n## Insider Transactions ({data['ticker']})"]
    lines.append(f"Signal: {data['net_signal']} (buy ratio: {data['buy_ratio']:.0%})")
    lines.append(f"Buys: {data['buy_count']} (${data['buy_value']:,.0f})")
    lines.append(f"Sells: {data['sell_count']} (${data['sell_value']:,.0f})")

    if data.get("transactions"):
        lines.append("Recent Transactions:")
        for tx in data["transactions"][:5]:
            lines.append(
                f"  {tx['insider']}: {tx['action']} "
                f"{tx['shares']:,} shares (${tx['value']:,.0f})"
            )

    return "\n".join(lines) + "\n"
