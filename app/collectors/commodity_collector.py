"""
Commodity Collector -- Fetches commodity prices as macro indicators.

NOT for trading -- commodities serve as predictive signals for stocks/crypto.
When commodity prices hit extremes they indicate macro problems/opportunities.

Writes to: asset_prices (asset_class='commodity')

Key signals:
  - Gold high → risk-off, flight to safety
  - Oil extreme → inflation/earnings impact
  - Copper momentum → global growth proxy ("Dr. Copper")
  - Wheat/NatGas spikes → CPI pressure, geopolitical risk
"""

import logging

logger = logging.getLogger(__name__)


import statistics
import yfinance as yf
from app.db.connection import get_db

# Tracked commodities: (yfinance_ticker, display_symbol, description)
COMMODITIES = [
    ("GC=F", "GOLD", "Gold futures - safe haven"),
    ("CL=F", "OIL", "Crude oil WTI - energy/inflation"),
    ("SI=F", "SILVER", "Silver futures - industrial demand"),
    ("HG=F", "COPPER", "Copper futures - global growth"),
    ("NG=F", "NATGAS", "Natural gas - energy sector"),
    ("ZW=F", "WHEAT", "Wheat futures - food inflation"),
]


async def collect_commodity_prices(days: int = 90) -> int:
    """Fetch historical commodity prices via yfinance.

    Uses Ticker().history() instead of download() to avoid
    multi-index DataFrame columns that cause Series-in-float() errors.

    Stores in asset_prices with asset_class='commodity'.
    Returns total rows written.
    """
    with get_db() as db:
        total = 0

        for yf_ticker, symbol, desc in COMMODITIES:
            try:
                stock = yf.Ticker(yf_ticker)
                data = stock.history(period=f"{days}d", auto_adjust=True)
                if data.empty:
                    logger.info(f"[commodity] {symbol}: no data")
                    continue

                count = 0
                for date, row in data.iterrows():
                    db.execute(
                        """
                        INSERT INTO asset_prices
                        (symbol, asset_class, date, open, high, low, close,
                         volume, currency, source)
                        VALUES (%s, 'commodity', %s, %s, %s, %s, %s, %s, 'USD',
                                'yfinance')
                ON CONFLICT (symbol, asset_class, date) DO NOTHING
                    """,
                        [
                            symbol,
                            date.date(),
                            float(row["Open"]),
                            float(row["High"]),
                            float(row["Low"]),
                            float(row["Close"]),
                            float(row.get("Volume", 0)),
                        ],
                    )
                    count += 1

                logger.info(f"[commodity] {symbol}: {count} price points")
                total += count
            except Exception as e:
                logger.info(f"[commodity] {symbol}: error - {e}")

        return total


def get_commodity_signals() -> dict:
    """Compute commodity z-scores and momentum for LLM context.

    Returns dict of signals per commodity:
      {
        "GOLD": {
          "price": 2650.0,
          "change_30d_pct": 5.2,
          "z_score": 1.8,
          "signal": "ELEVATED",
          "description": "Gold futures - safe haven"
        },
        ...
        "summary": "Gold ELEVATED (z=1.8), Oil NORMAL (z=0.3), ..."
      }
    """
    with get_db() as db:
        signals = {}

        for _, symbol, desc in COMMODITIES:
            rows = db.execute(
                """
                SELECT close FROM asset_prices
                WHERE symbol = %s AND asset_class = 'commodity'
                ORDER BY date DESC
                LIMIT 90
            """,
                [symbol],
            ).fetchall()

            if len(rows) < 10:
                signals[symbol] = {
                    "price": None,
                    "signal": "NO_DATA",
                    "description": desc,
                }
                continue

            closes = [r[0] for r in rows]
            current = closes[0]
            mean = statistics.mean(closes)
            stdev = statistics.stdev(closes) if len(closes) > 1 else 1.0

            z = (current - mean) / stdev if stdev > 0 else 0.0

            # 30-day momentum
            if len(closes) >= 22:
                price_30d_ago = closes[21]
                change_30d = ((current - price_30d_ago) / price_30d_ago) * 100
            else:
                change_30d = 0.0

            # Classify signal
            if z > 2.0:
                signal = "EXTREME_HIGH"
            elif z > 1.0:
                signal = "ELEVATED"
            elif z < -2.0:
                signal = "EXTREME_LOW"
            elif z < -1.0:
                signal = "DEPRESSED"
            else:
                signal = "NORMAL"

            signals[symbol] = {
                "price": round(current, 2),
                "change_30d_pct": round(change_30d, 1),
                "z_score": round(z, 2),
                "signal": signal,
                "description": desc,
            }

        # Build summary line
        parts = []
        for sym in ["GOLD", "OIL", "COPPER", "NATGAS", "SILVER", "WHEAT"]:
            s = signals.get(sym, {})
            if s.get("signal") and s["signal"] != "NO_DATA":
                parts.append(f"{sym} {s['signal']} (z={s.get('z_score', 0)})")
        signals["summary"] = ", ".join(parts)

        return signals


def format_commodity_context() -> str:
    """Format commodity signals as a text section for LLM context."""
    signals = get_commodity_signals()
    lines = [
        "\n## Commodity Macro Indicators",
        "Commodity extremes signal macro stress. "
        "High gold = risk-off. High oil = inflation. "
        "Low copper = slowing growth.",
    ]

    for sym in ["GOLD", "OIL", "COPPER", "NATGAS", "SILVER", "WHEAT"]:
        s = signals.get(sym)
        if not s or s.get("signal") == "NO_DATA":
            continue
        lines.append(
            f"  {sym}: ${s['price']:,.2f} | "
            f"30d: {s['change_30d_pct']:+.1f}% | "
            f"Z-score: {s['z_score']:+.2f} | "
            f"Signal: {s['signal']}"
        )

    summary = signals.get("summary", "")
    if summary:
        lines.append(f"\n  Macro Summary: {summary}")

    return "\n".join(lines) + "\n"


async def collect_all() -> dict:
    """Run commodity collection."""
    rows = await collect_commodity_prices(days=90)
    signals = get_commodity_signals()
    return {"rows_written": rows, "signals": signals}
