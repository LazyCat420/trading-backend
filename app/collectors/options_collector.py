"""
Options collector — fetches options chain data from yfinance.

Collects put/call ratio, open interest, and volume for context enrichment.

Usage:
    from app.collectors.options_collector import collect_options
    data = await collect_options("NVDA")
"""

import asyncio
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


async def collect_options(ticker: str) -> dict | None:
    """Fetch options chain summary for a ticker.

    Returns dict with put_call_ratio, total_call_oi, total_put_oi, etc.
    Stores formatted text in context via return value.
    """
    try:
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, lambda: _fetch_options(ticker))
        if data:
            logger.info(
                "options_collector: %s pc_ratio=%.2f",
                ticker,
                data.get("put_call_ratio", 0),
            )
        return data

    except Exception as e:
        logger.warning("options_collector: %s failed: %s", ticker, e)
        return None


def _fetch_options(ticker: str) -> dict | None:
    """Synchronous yfinance options fetch."""
    try:
        import yfinance as yf

        t = yf.Ticker(ticker)

        # Get nearest expiration
        dates = t.options
        if not dates:
            return None

        chain = t.option_chain(dates[0])
        calls = chain.calls
        puts = chain.puts

        if calls.empty and puts.empty:
            return None

        total_call_vol = int(calls["volume"].sum()) if "volume" in calls else 0
        total_put_vol = int(puts["volume"].sum()) if "volume" in puts else 0
        total_call_oi = (
            int(calls["openInterest"].sum()) if "openInterest" in calls else 0
        )
        total_put_oi = int(puts["openInterest"].sum()) if "openInterest" in puts else 0

        pc_ratio = total_put_vol / total_call_vol if total_call_vol > 0 else 0

        # Find highest volume strikes
        top_calls = (
            calls.nlargest(3, "volume")[["strike", "volume", "openInterest"]].to_dict(
                "records"
            )
            if not calls.empty
            else []
        )
        top_puts = (
            puts.nlargest(3, "volume")[["strike", "volume", "openInterest"]].to_dict(
                "records"
            )
            if not puts.empty
            else []
        )

        return {
            "ticker": ticker,
            "expiration": dates[0],
            "put_call_ratio": round(pc_ratio, 3),
            "total_call_volume": total_call_vol,
            "total_put_volume": total_put_vol,
            "total_call_oi": total_call_oi,
            "total_put_oi": total_put_oi,
            "top_calls": top_calls,
            "top_puts": top_puts,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.warning("options_collector: yfinance failed for %s: %s", ticker, e)
        return None


def format_options_context(data: dict) -> str:
    """Format options data as a context section for the LLM."""
    if not data:
        return ""

    lines = [f"\n## Options Flow ({data['ticker']})"]
    lines.append(f"Nearest Expiration: {data['expiration']}")
    lines.append(f"Put/Call Ratio: {data['put_call_ratio']:.3f}")

    sentiment = (
        "BEARISH"
        if data["put_call_ratio"] > 1.0
        else "BULLISH"
        if data["put_call_ratio"] < 0.7
        else "NEUTRAL"
    )
    lines.append(f"Options Sentiment: {sentiment}")

    lines.append(
        f"Total Call Volume: {data['total_call_volume']:,} | OI: {data['total_call_oi']:,}"
    )
    lines.append(
        f"Total Put Volume: {data['total_put_volume']:,} | OI: {data['total_put_oi']:,}"
    )

    if data.get("top_calls"):
        lines.append("Top Calls by Volume:")
        for c in data["top_calls"]:
            lines.append(
                f"  Strike ${c.get('strike', '?')} — Vol: {c.get('volume', 0):,} OI: {c.get('openInterest', 0):,}"
            )

    if data.get("top_puts"):
        lines.append("Top Puts by Volume:")
        for p in data["top_puts"]:
            lines.append(
                f"  Strike ${p.get('strike', '?')} — Vol: {p.get('volume', 0):,} OI: {p.get('openInterest', 0):,}"
            )

    return "\n".join(lines) + "\n"
