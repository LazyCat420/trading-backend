"""
Earnings history collector — fetches earnings vs estimates from yfinance.

Provides beat/miss streak and surprise percentages for context enrichment.

Usage:
    from app.collectors.earnings_collector import collect_earnings
    data = await collect_earnings("NVDA")
"""

import asyncio
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


async def collect_earnings(ticker: str) -> dict | None:
    """Fetch earnings history for a ticker.

    Returns dict with beat_count, miss_count, streak, etc.
    """
    try:
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, lambda: _fetch_earnings(ticker))
        if data:
            logger.info(
                "earnings_collector: %s beats=%d misses=%d streak=%s",
                ticker,
                data.get("beat_count", 0),
                data.get("miss_count", 0),
                data.get("current_streak", "?"),
            )
        return data
    except Exception as e:
        logger.warning("earnings_collector: %s failed: %s", ticker, e)
        return None


def _fetch_earnings(ticker: str) -> dict | None:
    """Synchronous earnings fetch via yfinance."""
    try:
        import yfinance as yf

        t = yf.Ticker(ticker)

        try:
            earnings = t.earnings_history
        except Exception:
            earnings = None

        if earnings is None or earnings.empty:
            return None

        quarters = []
        beats = 0
        misses = 0
        streak_type = None
        streak_count = 0

        for _, row in earnings.iterrows():
            actual = row.get("epsActual")
            estimate = row.get("epsEstimate")
            surprise = row.get("surprisePercent")

            if actual is None or estimate is None:
                continue

            beat = actual > estimate
            quarters.append(
                {
                    "date": str(row.name) if hasattr(row, "name") else "?",
                    "eps_actual": round(float(actual), 3),
                    "eps_estimate": round(float(estimate), 3),
                    "surprise_pct": round(float(surprise * 100), 1) if surprise else 0,
                    "result": "BEAT" if beat else "MISS",
                }
            )

            if beat:
                beats += 1
            else:
                misses += 1

        if not quarters:
            return None

        # Calculate streak (from most recent)
        if quarters:
            streak_type = quarters[0]["result"]
            streak_count = 1
            for q in quarters[1:]:
                if q["result"] == streak_type:
                    streak_count += 1
                else:
                    break

        total = beats + misses
        return {
            "ticker": ticker,
            "beat_count": beats,
            "miss_count": misses,
            "beat_rate": round(beats / total, 2) if total else 0,
            "current_streak": f"{streak_count}x {streak_type}"
            if streak_type
            else "N/A",
            "quarters": quarters[:8],
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.warning("earnings_collector: yfinance failed for %s: %s", ticker, e)
        return None


def format_earnings_context(data: dict) -> str:
    """Format earnings data as a context section for the LLM."""
    if not data:
        return ""

    lines = [f"\n## Earnings History ({data['ticker']})"]
    lines.append(
        f"Beat Rate: {data['beat_rate']:.0%} ({data['beat_count']} beats, {data['miss_count']} misses)"
    )
    lines.append(f"Current Streak: {data['current_streak']}")

    if data.get("quarters"):
        lines.append("Recent Quarters:")
        for q in data["quarters"][:4]:
            lines.append(
                f"  {q['date']}: {q['result']} — "
                f"Actual ${q['eps_actual']:.2f} vs Est ${q['eps_estimate']:.2f} "
                f"({q['surprise_pct']:+.1f}%)"
            )

    return "\n".join(lines) + "\n"
