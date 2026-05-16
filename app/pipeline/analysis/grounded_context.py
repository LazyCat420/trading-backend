from dataclasses import dataclass
from typing import List, Dict
import logging
from app.data.market_snapshot import MarketSnapshot
from app.data.market_data import build_snapshot
from app.db.connection import get_db

logger = logging.getLogger(__name__)


@dataclass
class GroundedContext:
    """
    Consolidated context container that guarantees hard separation
    between quantitative numbers (MarketSnapshot) and qualitative text (news).
    """

    snapshot: MarketSnapshot
    news_headlines: List[str]
    recent_trades: List[Dict]

    @classmethod
    async def build(cls, ticker: str, lookback_days: int = 60) -> "GroundedContext":
        """
        Build a grounded context for a ticker. Hard-gates if price is invalid.
        """
        snapshot = await build_snapshot(ticker, lookback_days=lookback_days)
        snapshot.assert_price_valid()

        # Fetch news (text only, no LLM parsing)
        news_headlines = []
        with get_db() as db:
            cur = db.execute(
                """
                SELECT title, publisher, published_at, summary
                FROM news_articles
                WHERE ticker = %s
                  AND (quality_status IS NULL OR quality_status != 'discarded')
                ORDER BY published_at DESC
                LIMIT 10
                """,
                [ticker],
            )
            for row in cur.fetchall():
                title, pub, dt, summary = row
                news_headlines.append(f"[{dt}] {pub}: {title} - {summary}")

        # Fetch recent trades
        recent_trades = []
        with get_db() as db:
            cur = db.execute(
                """
                SELECT side, qty, price, filled_at, realized_pnl
                FROM orders
                WHERE ticker = %s
                ORDER BY filled_at DESC
                LIMIT 5
                """,
                [ticker],
            )
            cols = [desc[0] for desc in cur.description]
            for row in cur.fetchall():
                recent_trades.append(dict(zip(cols, row)))

        return cls(
            snapshot=snapshot,
            news_headlines=news_headlines,
            recent_trades=recent_trades,
        )

    def to_prompt(self) -> str:
        """
        Returns the complete, strict block format prompt.
        """
        lines = []
        lines.append(self.snapshot.to_prompt_block())

        if self.news_headlines:
            lines.append("\n[RECENT NEWS HEADLINES]")
            lines.extend(self.news_headlines)
        else:
            lines.append("\n[RECENT NEWS HEADLINES]\nNone")

        if self.recent_trades:
            lines.append("\n[RECENT TRADES]")
            for t in self.recent_trades:
                lines.append(
                    f"{t['side']} {t['qty']} @ {t['price']} (PNL: {t['realized_pnl']}) at {t['filled_at']}"
                )
        else:
            lines.append("\n[RECENT TRADES]\nNone")

        return "\n".join(lines)
