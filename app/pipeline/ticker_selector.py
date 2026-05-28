"""
Ticker Selector - Smart algorithm for choosing the best N tickers to process
by blending Portfolio, Large Cap, Mid/Small Cap, and Random Discovery slots.

HARD TOTAL CAP: When max_tickers=N, exactly N total tickers are processed.
Positions get priority (filled first) but COUNT AGAINST the cap. This ensures
max_tickers=1 means "process 1 ticker total", not "1 plus however many
positions you have". Remaining slots after positions go to watchlist + discovery.
"""

import logging
import random
from dataclasses import dataclass, field
from typing import List
from app.db.connection import get_db

logger = logging.getLogger(__name__)


@dataclass
class TickerSelectionResult:
    """Structured output from ticker selection so callers know the breakdown."""

    position_tickers: List[str] = field(default_factory=list)
    non_position_tickers: List[str] = field(default_factory=list)

    @property
    def all_tickers(self) -> List[str]:
        """Combined deduped list: positions first, then non-position names."""
        seen = set()
        out = []
        for t in self.position_tickers:
            if t not in seen:
                seen.add(t)
                out.append(t)
        for t in self.non_position_tickers:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out


class TickerSelector:
    @staticmethod
    def select_tickers_for_cycle(requested_tickers: List[str], cap: int) -> List[str]:
        """Convenience wrapper — returns a flat list for backward compatibility."""
        result = TickerSelector.select_tickers_for_cycle_v2(requested_tickers, cap)
        return result.all_tickers

    @staticmethod
    def select_tickers_for_cycle_v2(
        requested_tickers: List[str],
        cap: int,
    ) -> TickerSelectionResult:
        """
        Build the cycle ticker list with a hard total cap.

        `cap` is the HARD CEILING on total tickers processed. Positions get
        priority (filled first), then remaining slots go to non-position
        tickers (watchlist + discovery). This ensures that when the user
        sets max_tickers=1, exactly 1 ticker is processed — not 1 + N positions.

        Returns a TickerSelectionResult with separate position / non-position lists.
        """
        if cap is None or cap < 0:
            cap = 50

        requested = set(t.upper().strip() for t in requested_tickers if t.strip())

        # ── 1. Fetch open positions (ALWAYS included, outside cap) ──
        position_tickers: set[str] = set()
        with get_db() as db:
            try:
                # Resolve active bot for position filtering
                try:
                    from app.services.bot_manager import get_active_bot_id

                    bid = get_active_bot_id()
                except Exception:
                    from app.config import settings as _cfg

                    bid = _cfg.BOT_ID

                # Check if positions table exists
                tbl_check = db.execute(
                    "SELECT 1 FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_name = 'positions'"
                ).fetchone()
                if tbl_check:
                    pos_rows = db.execute(
                        "SELECT ticker FROM position_lots WHERE status = 'open' AND bot_id = %s",
                        [bid],
                    ).fetchall()
                    for r in pos_rows:
                        position_tickers.add(r[0])
                    logger.info(
                        "[SELECTOR] Fetched %d position tickers for bot '%s'",
                        len(position_tickers),
                        bid,
                    )
            except Exception as e:
                logger.warning("[SELECTOR] Failed to fetch open positions: %s", e)

        # ── 1.5. Fetch 24-Hour Cooldown Tickers ──
        recent_analyzed: set[str] = set()
        with get_db() as db:
            try:
                recent_rows = db.execute(
                    "SELECT DISTINCT ticker FROM decision_outcomes WHERE created_at > NOW() - INTERVAL '24 hours'"
                ).fetchall()
                analysis_rows = db.execute(
                    "SELECT DISTINCT ticker FROM analysis_results WHERE created_at > NOW() - INTERVAL '24 hours'"
                ).fetchall()
                for r in recent_rows:
                    recent_analyzed.add(r[0])
                for r in analysis_rows:
                    recent_analyzed.add(r[0])
            except Exception as e:
                logger.warning("[SELECTOR] Failed to fetch 24-hour cooldown tickers: %s", e)

        # ── 1b. Enforce hard cap on positions themselves ──
        # Positions get priority but still count against the total cap.
        if len(position_tickers) > cap:
            logger.warning(
                "[SELECTOR] %d positions exceed hard cap %d — truncating positions!",
                len(position_tickers), cap,
            )
            position_tickers = set(list(position_tickers)[:cap])

        # Remaining slots for non-position tickers
        non_position_slots = cap - len(position_tickers)

        # ── 2. Build non-position set (requested + watchlist), capped ──
        non_position: set[str] = set()

        if non_position_slots <= 0:
            logger.info(
                "[SELECTOR] All %d cap slots filled by positions — no room for non-position tickers",
                cap,
            )
        else:
            # Add manually requested tickers (minus any that are already positions)
            for t in requested:
                if t not in position_tickers and len(non_position) < non_position_slots:
                    non_position.add(t)

            # Add active watchlist (minus positions), up to remaining slots
            # Triage handles the "is there new data?" question — selector just gathers candidates.
            with get_db() as db:
                try:
                    wl_rows = db.execute(
                        "SELECT ticker FROM watchlist WHERE status = 'active'"
                    ).fetchall()
                    for r in wl_rows:
                        t_val = r[0]
                        if t_val not in position_tickers and len(non_position) < non_position_slots:
                            if t_val not in recent_analyzed:
                                non_position.add(t_val)
                            else:
                                logger.info("[SELECTOR] Skipping watchlist ticker %s due to 24-hour cooldown", t_val)
                except Exception as e:
                    logger.warning("[SELECTOR] Failed to fetch watchlist: %s", e)

        # ── 3. Discovery fill (only if non-position set is under its slot allocation) ──
        if len(non_position) < non_position_slots:
            remaining_slots = non_position_slots - len(non_position)

            large_slots = max(1, int(remaining_slots * 0.40))
            mid_slots = max(1, int(remaining_slots * 0.40))
            random_slots = remaining_slots - large_slots - mid_slots
            if random_slots < 0:
                large_slots = remaining_slots
                mid_slots = 0
                random_slots = 0

            # Exclude positions, already-selected non-position tickers, and recently analyzed
            exclude = position_tickers | non_position | recent_analyzed
            if not exclude:
                placeholders = "'___'"
                params: list = []
            else:
                placeholders = ",".join(["%s"] * len(exclude))
                params = list(exclude)

            base_query = """
                SELECT d.ticker, d.score, m.market_cap_tier, m.sp500,
                       COALESCE(MAX(a.created_at), '2000-01-01') as last_analyzed
                FROM discovered_tickers d
                LEFT JOIN ticker_metadata m ON d.ticker = m.ticker
                LEFT JOIN analysis_results a ON d.ticker = a.ticker
                WHERE d.ticker NOT IN ({placeholders})
                  AND (d.validation_status IS NULL OR d.validation_status != 'quarantine')
                GROUP BY d.ticker, d.score, m.market_cap_tier, m.sp500
                ORDER BY
                     CASE
                        WHEN COALESCE(MAX(a.created_at), '2000-01-01') > CURRENT_TIMESTAMP - INTERVAL '24 hours' THEN 0
                        ELSE 1
                     END DESC,
                     d.score DESC
                LIMIT 300
            """
            query = base_query.format(placeholders=placeholders)

            try:
                with get_db() as db:
                    candidates = db.execute(query, params).fetchall()
            except Exception as e:
                logger.warning("[SELECTOR] Failed to query discovered_tickers: %s", e)
                candidates = []

            # Fast Ticker Validation (Discovery Only) ──
            # Strip out numbers, dashes, and known macro acronyms that cause YFinance to fail
            import re
            
            from app.processors.ticker_extractor import FALSE_TICKERS

            def is_valid_ticker_format(t: str) -> bool:
                if not t or len(t) > 5:
                    return False
                if bool(re.search(r"[0-9\-]", t)):
                    return False
                if t in FALSE_TICKERS:
                    return False
                return True

            large_candidates: list[str] = []
            mid_small_candidates: list[str] = []
            mystery_candidates: list[str] = []

            for row in candidates:
                t_ticker = row[0]
                if not is_valid_ticker_format(t_ticker):
                    continue

                tier = row[2]
                sp500 = row[3]
                if sp500 or tier in ("mega", "large"):
                    large_candidates.append(t_ticker)
                elif tier in ("mid", "small", "micro"):
                    mid_small_candidates.append(t_ticker)
                else:
                    mystery_candidates.append(t_ticker)

            def fill_bucket(bucket_list, required_slots):
                picks = []
                while len(picks) < required_slots and bucket_list:
                    picks.append(bucket_list.pop(0))
                return picks

            discovery: list[str] = []
            large_picks = fill_bucket(large_candidates, large_slots)
            discovery.extend(large_picks)
            mid_picks = fill_bucket(mid_small_candidates, mid_slots)
            discovery.extend(mid_picks)

            leftovers = large_candidates + mid_small_candidates + mystery_candidates
            shortfall = non_position_slots - (len(non_position) + len(discovery))
            random_picks: list[str] = []
            if shortfall > 0 and leftovers:
                random.shuffle(leftovers)
                random_picks = fill_bucket(leftovers, shortfall)
                discovery.extend(random_picks)

            for t in discovery:
                non_position.add(t)
        else:
            large_picks = []
            mid_picks = []
            random_picks = []

        # ── 4. Apply hard cap — non-position tickers fill remaining slots only ──
        capped_non_position = list(non_position)[:non_position_slots]

        total = len(position_tickers) + len(capped_non_position)
        logger.info(
            "[TICKER SELECTOR] HARD CAP: %d total. "
            "Positions: %d, Non-position: %d/%d slots. "
            "Large/SP500: %d, Mid/Small: %d, Random: %d. "
            "Final total: %d",
            cap,
            len(position_tickers),
            len(capped_non_position),
            non_position_slots,
            len(large_picks),
            len(mid_picks),
            len(random_picks),
            total,
        )
        assert total <= cap, (
            f"[SELECTOR BUG] Total tickers {total} exceeds hard cap {cap}! "
            f"positions={len(position_tickers)}, non_position={len(capped_non_position)}"
        )

        return TickerSelectionResult(
            position_tickers=list(position_tickers),
            non_position_tickers=capped_non_position,
        )
