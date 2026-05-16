"""
Market Regime Detector -- Classifies current market environment.

Uses SPY trend, VIX level, and breadth to determine:
  BULL, BEAR, or SIDEWAYS regime.

The regime affects how aggressive the bot should be:
  BULL:     Normal position sizing, favor BUY signals
  SIDEWAYS: Reduce position sizes by 50%, favor HOLD
  BEAR:     Reduce position sizes by 70%, favor defensive/SELL
"""

from app.db.connection import get_db


def get_market_regime() -> dict:
    """Classify current market regime from SPY data.

    Logic:
      SPY > SMA200 + VIX < 20 = BULL
      SPY < SMA200 + VIX > 25 = BEAR
      Everything else = SIDEWAYS
    """
    with get_db() as db:
        # Get SPY price and moving averages
        spy_tech = db.execute("""
            SELECT sma_50, sma_200
            FROM technicals WHERE ticker = 'SPY'
            ORDER BY date DESC LIMIT 1
        """).fetchone()

        spy_price_row = db.execute("""
            SELECT close FROM price_history
            WHERE ticker = 'SPY' ORDER BY date DESC LIMIT 1
        """).fetchone()

        # Fallback if no SPY data
        if not spy_price_row or not spy_tech:
            return {
                "regime": "UNKNOWN",
                "confidence": 0,
                "spy_price": None,
                "position_multiplier": 0.5,
                "note": "No SPY data available. Defaulting to cautious mode.",
            }

        spy_price = spy_price_row[0]
        sma_50 = spy_tech[0] or spy_price
        sma_200 = spy_tech[1] or spy_price

        # VIX check (from price_history if we have it, else estimate from ATR)
        vix_row = db.execute("""
            SELECT close FROM price_history
            WHERE ticker = '^VIX' OR ticker = 'VIX'
            ORDER BY date DESC LIMIT 1
        """).fetchone()
        vix = vix_row[0] if vix_row else None

        # Regime classification
        above_sma200 = spy_price > sma_200
        above_sma50 = spy_price > sma_50
        golden_cross = sma_50 > sma_200

        # SPY return over last 20 days
        spy_20d = db.execute("""
            SELECT close FROM price_history
            WHERE ticker = 'SPY' ORDER BY date DESC LIMIT 21
        """).fetchall()

        recent_return = 0
        if len(spy_20d) >= 2:
            recent_return = (spy_20d[0][0] - spy_20d[-1][0]) / spy_20d[-1][0] * 100

        # Score-based classification
        bull_score = 0
        if above_sma200:
            bull_score += 30
        if above_sma50:
            bull_score += 20
        if golden_cross:
            bull_score += 20
        if recent_return > 2:
            bull_score += 15
        elif recent_return > 0:
            bull_score += 5
        if vix and vix < 18:
            bull_score += 15
        elif vix and vix < 25:
            bull_score += 5

        # Classify
        if bull_score >= 70:
            regime = "BULL"
            position_mult = 1.0
        elif bull_score <= 30:
            regime = "BEAR"
            position_mult = 0.3
        else:
            regime = "SIDEWAYS"
            position_mult = 0.5

        return {
            "regime": regime,
            "bull_score": bull_score,
            "spy_price": round(spy_price, 2),
            "sma_50": round(sma_50, 2),
            "sma_200": round(sma_200, 2),
            "above_sma200": above_sma200,
            "golden_cross": golden_cross,
            "vix": round(vix, 2) if vix else None,
            "recent_return_20d": round(recent_return, 2),
            "position_multiplier": position_mult,
        }
