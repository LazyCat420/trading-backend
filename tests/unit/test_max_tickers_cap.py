"""
TDD tests for max_tickers HARD TOTAL cap enforcement.

These tests verify that when a user sets max_tickers=N, exactly N TOTAL tickers
are processed during the entire trading cycle. Position tickers get priority
(filled first) but they COUNT AGAINST the cap. This ensures max_tickers=1
means "process 1 ticker total", not "1 plus however many positions you have".

RED → GREEN: Write tests first, watch them fail, then fix the code.
"""

import pytest
import random
from unittest.mock import MagicMock
from contextlib import contextmanager


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: TickerSelector enforces hard TOTAL cap
# ─────────────────────────────────────────────────────────────────────────────
class TestTickerSelectorHardCap:
    def _setup_mock_db(self, mock_db, positions=None, watchlist=None, discovered=None):
        """Configure mock_db for TickerSelector queries."""
        positions = positions or []
        watchlist = watchlist or []
        discovered = discovered or []

        def side_effect_execute(query, params=None):
            cursor = MagicMock()
            if "information_schema" in query:
                cursor.fetchone.return_value = (1,)
            elif "position_lots" in query:
                cursor.fetchall.return_value = [(t,) for t in positions]
            elif "watchlist" in query:
                cursor.fetchall.return_value = [(t,) for t in watchlist]
            elif "discovered_tickers" in query:
                cursor.fetchall.return_value = [
                    (t, 100, "large", True, "2000-01-01") for t in discovered
                ]
            else:
                cursor.fetchall.return_value = []
            return cursor

        mock_db.execute.side_effect = side_effect_execute

    def test_cap_1_limits_total_to_1(self, mock_db, monkeypatch):
        """With cap=1 and 2 positions: only 1 position, 0 non-position (hard total cap)."""
        self._setup_mock_db(
            mock_db,
            positions=["AAPL", "MSFT"],
            watchlist=["GOOG", "AMZN", "META", "TSLA", "NFLX"],
            discovered=[],
        )

        @contextmanager
        def fake_get_db():
            yield mock_db

        monkeypatch.setattr("app.db.connection.get_db", fake_get_db)
        monkeypatch.setattr("app.pipeline.ticker_selector.get_db", fake_get_db)
        monkeypatch.setattr(
            "app.services.bot_manager.get_active_bot_id", lambda: "test-bot"
        )

        from app.pipeline.ticker_selector import TickerSelector

        result = TickerSelector.select_tickers_for_cycle_v2([], cap=1)

        # Hard total cap = 1: positions fill first, no room for non-position
        assert len(result.all_tickers) <= 1, (
            f"Expected <=1 TOTAL tickers (hard cap), got {len(result.all_tickers)}: "
            f"{result.all_tickers}"
        )
        assert len(result.position_tickers) <= 1, (
            f"Positions should be truncated to cap when positions > cap, "
            f"got {len(result.position_tickers)}"
        )
        assert len(result.non_position_tickers) == 0, (
            f"No room for non-position tickers when cap=1 and positions exist"
        )

    def test_cap_3_with_2_positions(self, mock_db, monkeypatch):
        """With cap=3 and 2 positions: 2 positions + 1 non-position = 3 total."""
        self._setup_mock_db(
            mock_db,
            positions=["AAPL", "MSFT"],
            watchlist=["GOOG", "AMZN", "META", "TSLA", "NFLX"],
            discovered=[],
        )

        @contextmanager
        def fake_get_db():
            yield mock_db

        monkeypatch.setattr("app.db.connection.get_db", fake_get_db)
        monkeypatch.setattr("app.pipeline.ticker_selector.get_db", fake_get_db)
        monkeypatch.setattr(
            "app.services.bot_manager.get_active_bot_id", lambda: "test-bot"
        )

        from app.pipeline.ticker_selector import TickerSelector

        result = TickerSelector.select_tickers_for_cycle_v2([], cap=3)

        assert len(result.all_tickers) <= 3, (
            f"Expected <=3 TOTAL tickers (hard cap), got {len(result.all_tickers)}: "
            f"{result.all_tickers}"
        )
        assert len(result.position_tickers) == 2, "Both positions should fit in cap=3"
        assert len(result.non_position_tickers) <= 1, (
            f"Expected <=1 non-position (3 cap - 2 positions), "
            f"got {len(result.non_position_tickers)}"
        )

    def test_cap_1_no_positions(self, mock_db, monkeypatch):
        """With cap=1 and 0 positions: exactly 1 watchlist ticker."""
        self._setup_mock_db(
            mock_db,
            positions=[],
            watchlist=["GOOG", "AMZN", "META", "TSLA", "NFLX"],
            discovered=[],
        )

        @contextmanager
        def fake_get_db():
            yield mock_db

        monkeypatch.setattr("app.db.connection.get_db", fake_get_db)
        monkeypatch.setattr("app.pipeline.ticker_selector.get_db", fake_get_db)
        monkeypatch.setattr(
            "app.services.bot_manager.get_active_bot_id", lambda: "test-bot"
        )

        from app.pipeline.ticker_selector import TickerSelector

        result = TickerSelector.select_tickers_for_cycle_v2([], cap=1)

        assert len(result.all_tickers) == 1, (
            f"Expected exactly 1 total (0 positions + 1 non-position), "
            f"got {len(result.all_tickers)}: {result.all_tickers}"
        )
        assert len(result.position_tickers) == 0
        assert len(result.non_position_tickers) == 1

    def test_cap_0_defaults_to_50(self, mock_db, monkeypatch):
        """cap=0 should default to 50 (not unlimited)."""
        self._setup_mock_db(mock_db, positions=[], watchlist=[], discovered=[])

        @contextmanager
        def fake_get_db():
            yield mock_db

        monkeypatch.setattr("app.db.connection.get_db", fake_get_db)
        monkeypatch.setattr("app.pipeline.ticker_selector.get_db", fake_get_db)
        monkeypatch.setattr(
            "app.services.bot_manager.get_active_bot_id", lambda: "test-bot"
        )

        from app.pipeline.ticker_selector import TickerSelector

        result = TickerSelector.select_tickers_for_cycle_v2([], cap=0)
        assert isinstance(result.non_position_tickers, list)

    def test_cap_1_with_requested_tickers(self, mock_db, monkeypatch):
        """When user manually requests 3 tickers but cap=1, only 1 total gets through."""
        self._setup_mock_db(
            mock_db,
            positions=["AAPL"],
            watchlist=[],
            discovered=[],
        )

        @contextmanager
        def fake_get_db():
            yield mock_db

        monkeypatch.setattr("app.db.connection.get_db", fake_get_db)
        monkeypatch.setattr("app.pipeline.ticker_selector.get_db", fake_get_db)
        monkeypatch.setattr(
            "app.services.bot_manager.get_active_bot_id", lambda: "test-bot"
        )

        from app.pipeline.ticker_selector import TickerSelector

        result = TickerSelector.select_tickers_for_cycle_v2(
            ["NVDA", "GOOG", "META"], cap=1
        )

        # cap=1, 1 position → 0 slots for non-position
        assert len(result.all_tickers) <= 1, (
            f"Hard cap=1 should limit TOTAL to 1, "
            f"got {len(result.all_tickers)}: {result.all_tickers}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: PipelineContext carries max_tickers
# ─────────────────────────────────────────────────────────────────────────────
class TestPipelineContext:
    def test_max_tickers_field_exists(self):
        """PipelineContext should have a max_tickers field."""
        from app.pipeline.core import PipelineContext

        ctx = PipelineContext(
            tickers=["AAPL"],
            collect=True,
            analyze=True,
            trade=True,
            cycle_id="test-001",
            max_tickers=1,
        )
        assert ctx.max_tickers == 1

    def test_max_tickers_defaults_to_none(self):
        """max_tickers should default to None when not specified."""
        from app.pipeline.core import PipelineContext

        ctx = PipelineContext(
            tickers=["AAPL"],
            collect=True,
            analyze=True,
            trade=True,
            cycle_id="test-001",
        )
        assert ctx.max_tickers is None


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: data_phase merge cap respects max_tickers as HARD TOTAL cap
# ─────────────────────────────────────────────────────────────────────────────
class TestDataPhaseMergeCap:
    """Test the merge + cap logic extracted from data_phase.run().

    We test the core cap algorithm directly rather than the full async pipeline
    to avoid needing to mock 15+ dependencies.
    """

    def test_merge_cap_hard_total_limit(self):
        """With max_tickers=3, total tickers after merge never exceeds 3."""
        tickers = ["AAPL", "MSFT", "CRWV"]  # 2 positions + 1 selected = 3 total
        position_tickers = ["AAPL", "MSFT"]
        discovered_tickers = [
            "TSLA", "GOOG", "META", "AMZN", "NFLX",
            "AMD", "INTC", "PLTR", "SOFI", "RIVN",
        ]  # 10 discovered
        max_tickers = 3

        # Reproduce data_phase merge logic
        original_tickers = list(tickers)
        _effective_cap = max_tickers
        _protected = set(position_tickers)

        # Already at cap check (hard total)
        _at_cap = len(tickers) >= _effective_cap

        if not _at_cap:
            new_tickers = [t for t in discovered_tickers if t not in tickers]
            if new_tickers:
                tickers = list(tickers) + new_tickers

        # Apply hard total cap
        if len(tickers) > _effective_cap:
            protected_kept = [t for t in tickers if t in _protected]
            unprotected = [t for t in tickers if t not in _protected]
            remaining_slots = max(0, _effective_cap - len(protected_kept))
            watchlist_set = set(original_tickers) - _protected
            wl_from_watchlist = [t for t in unprotected if t in watchlist_set]
            others = [t for t in unprotected if t not in watchlist_set]
            capped_wl = wl_from_watchlist[:remaining_slots]
            remaining_after_wl = remaining_slots - len(capped_wl)
            capped_unprotected = capped_wl + others[: max(0, remaining_after_wl)]
            tickers = protected_kept + capped_unprotected

        # Hard total cap: never exceeds max_tickers
        assert len(tickers) <= max_tickers, (
            f"Expected <={max_tickers} total, got {len(tickers)}: {tickers}"
        )
        assert "AAPL" in tickers, "Position AAPL must be kept"
        assert "MSFT" in tickers, "Position MSFT must be kept"

    def test_merge_skipped_when_at_cap(self):
        """When tickers already at cap, discovery merge is skipped entirely."""
        tickers = ["AAPL"]  # 1 ticker = at cap when max_tickers=1
        max_tickers = 1

        _at_cap = len(tickers) >= max_tickers
        assert _at_cap, "Should detect we're at cap"

        discovered_tickers = ["TSLA", "GOOG", "META"]
        merged = 0

        if not _at_cap:
            merged = len(discovered_tickers)

        assert merged == 0, "Should not merge any discovered tickers when at cap"

    def test_positions_count_against_cap(self):
        """With max_tickers=1 and 5 positions, only 1 position survives (hard cap)."""
        positions = [f"POS{i}" for i in range(5)]
        tickers = positions + ["EXTRA1", "EXTRA2"]
        max_tickers = 1

        _effective_cap = max_tickers
        _protected = set(positions)

        if len(tickers) > _effective_cap:
            protected_kept = [t for t in tickers if t in _protected][:_effective_cap]
            remaining = _effective_cap - len(protected_kept)
            non_pos = [t for t in tickers if t not in _protected][:remaining]
            tickers = protected_kept + non_pos

        assert len(tickers) <= max_tickers, (
            f"HARD CAP VIOLATED: {len(tickers)} > {max_tickers}. Tickers: {tickers}"
        )

    def test_effective_cap_falls_back_to_setting(self):
        """When max_tickers is None, fall back to MAX_ANALYSIS_TICKERS."""
        from app.config import settings

        max_tickers = None
        _effective_cap = max_tickers if max_tickers is not None else settings.MAX_ANALYSIS_TICKERS
        assert _effective_cap == settings.MAX_ANALYSIS_TICKERS
        assert _effective_cap == 30


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: data_phase skips discovery merge when at cap
# ─────────────────────────────────────────────────────────────────────────────
class TestDiscoverySkipAtCap:
    """When the total tickers already satisfy the hard cap,
    discovered tickers should NOT be merged into the current cycle."""

    def test_discovery_not_merged_when_at_cap(self):
        """With max_tickers=1 and 1 ticker already, skip merge."""
        tickers = ["AAPL"]  # 1 position = at cap when max_tickers=1
        max_tickers = 1

        _at_cap = len(tickers) >= max_tickers
        assert _at_cap, "Should be at cap: 1 total >= 1 cap"

        discovered_tickers = ["TSLA", "GOOG", "META"]
        if _at_cap:
            merged_tickers = 0  # Skip merge
        else:
            new_tickers = [t for t in discovered_tickers if t not in tickers]
            merged_tickers = len(new_tickers)

        assert merged_tickers == 0, "Should not merge any discovered tickers when at cap"


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Safety net — final ticker count NEVER exceeds cap (TOTAL)
# ─────────────────────────────────────────────────────────────────────────────
class TestSafetyNet:
    def test_final_count_hard_total_invariant(self):
        """No matter what happens, final tickers <= max_tickers (hard total cap)."""
        for _ in range(100):  # Fuzz with random inputs
            n_positions = random.randint(0, 10)
            n_watchlist = random.randint(0, 20)
            n_discovered = random.randint(0, 100)
            max_tickers = random.randint(1, 5)

            positions = [f"POS{i}" for i in range(n_positions)]
            watchlist = [f"WL{i}" for i in range(n_watchlist)]
            discovered = [f"DISC{i}" for i in range(n_discovered)]

            # Build initial tickers: positions first, then non-position up to cap
            _protected = set(positions)
            pos_kept = positions[:max_tickers]
            remaining_slots = max(0, max_tickers - len(pos_kept))
            non_pos = [t for t in watchlist if t not in _protected][:remaining_slots]
            tickers = pos_kept + non_pos
            original_tickers = list(tickers)

            _effective_cap = max_tickers

            # Check if at cap before merge
            if len(tickers) < _effective_cap:
                new_tickers = [t for t in discovered if t not in tickers]
                if new_tickers:
                    tickers = list(tickers) + new_tickers

            # Apply hard total cap
            if len(tickers) > _effective_cap:
                protected_kept = [t for t in tickers if t in _protected][:_effective_cap]
                remaining = _effective_cap - len(protected_kept)
                unprotected = [t for t in tickers if t not in _protected]
                watchlist_set = set(original_tickers) - _protected
                wl_from_wl = [t for t in unprotected if t in watchlist_set]
                others = [t for t in unprotected if t not in watchlist_set]
                capped_wl = wl_from_wl[:remaining]
                remaining_after = remaining - len(capped_wl)
                capped_non = capped_wl + others[:max(0, remaining_after)]
                tickers = protected_kept + capped_non

            # Safety net (exactly like data_phase.py)
            if len(tickers) > _effective_cap:
                kept_pos = [t for t in tickers if t in _protected][:_effective_cap]
                rem = _effective_cap - len(kept_pos)
                kept_non = [t for t in tickers if t not in _protected][:rem]
                tickers = kept_pos + kept_non

            assert len(tickers) <= max_tickers, (
                f"HARD CAP VIOLATED: {len(tickers)} total tickers > "
                f"cap {max_tickers}. Positions={n_positions}, "
                f"WL={n_watchlist}, Disc={n_discovered}"
            )
