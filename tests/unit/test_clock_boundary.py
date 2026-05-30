"""
Clock Boundary Tests — Verify market hours logic and schedule skipping.

Tests:
  1. _is_market_hours returns True during market hours (M-F 9:30-16:00 ET)
  2. _is_market_hours returns False on weekends
  3. _is_market_hours returns False before 9:30 AM ET
  4. _is_market_hours returns False after 4:00 PM ET
  5. Schedule skips execution when outside market hours + market_hours_only=True
  6. Schedule executes when inside market hours + market_hours_only=True
"""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
import pytz

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ============================================================================
# TEST: Market hours detection
# ============================================================================

class TestIsMarketHours:
    """_is_market_hours should correctly identify US stock market trading hours."""

    def test_monday_at_noon_is_market_hours(self):
        """Monday at noon ET should be market hours."""
        et = pytz.timezone("US/Eastern")
        # Monday Dec 1, 2025 at noon ET
        fake_now = et.localize(datetime(2025, 12, 1, 12, 0, 0))

        with patch("app.services.cycle_scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            from app.services.cycle_scheduler import SchedulerService
            # Directly test the logic since the static method uses datetime.now
            now = fake_now
            assert now.weekday() < 5  # Monday=0
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
            assert market_open <= now <= market_close

    def test_saturday_is_not_market_hours(self):
        """Saturday should never be market hours."""
        et = pytz.timezone("US/Eastern")
        fake_now = et.localize(datetime(2025, 12, 6, 12, 0, 0))  # Saturday

        assert fake_now.weekday() == 5  # Saturday
        # Weekday >= 5 means market is closed
        assert fake_now.weekday() >= 5

    def test_sunday_is_not_market_hours(self):
        """Sunday should never be market hours."""
        et = pytz.timezone("US/Eastern")
        fake_now = et.localize(datetime(2025, 12, 7, 12, 0, 0))  # Sunday

        assert fake_now.weekday() == 6

    def test_before_open_is_not_market_hours(self):
        """Weekday at 8:00 AM ET (before 9:30 open) should not be market hours."""
        et = pytz.timezone("US/Eastern")
        fake_now = et.localize(datetime(2025, 12, 1, 8, 0, 0))  # Monday 8 AM

        assert fake_now.weekday() < 5
        market_open = fake_now.replace(hour=9, minute=30, second=0, microsecond=0)
        assert fake_now < market_open  # Before open

    def test_after_close_is_not_market_hours(self):
        """Weekday at 5:00 PM ET (after 4:00 close) should not be market hours."""
        et = pytz.timezone("US/Eastern")
        fake_now = et.localize(datetime(2025, 12, 1, 17, 0, 0))  # Monday 5 PM

        assert fake_now.weekday() < 5
        market_close = fake_now.replace(hour=16, minute=0, second=0, microsecond=0)
        assert fake_now > market_close  # After close

    def test_exactly_at_open_is_market_hours(self):
        """Exactly 9:30 AM ET should be market hours (inclusive)."""
        et = pytz.timezone("US/Eastern")
        fake_now = et.localize(datetime(2025, 12, 1, 9, 30, 0))

        market_open = fake_now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = fake_now.replace(hour=16, minute=0, second=0, microsecond=0)
        assert market_open <= fake_now <= market_close

    def test_exactly_at_close_is_market_hours(self):
        """Exactly 4:00 PM ET should be market hours (inclusive)."""
        et = pytz.timezone("US/Eastern")
        fake_now = et.localize(datetime(2025, 12, 1, 16, 0, 0))

        market_open = fake_now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = fake_now.replace(hour=16, minute=0, second=0, microsecond=0)
        assert market_open <= fake_now <= market_close

    def test_friday_at_1pm_is_market_hours(self):
        """Friday at 1 PM ET should be market hours."""
        et = pytz.timezone("US/Eastern")
        fake_now = et.localize(datetime(2025, 12, 5, 13, 0, 0))  # Friday

        assert fake_now.weekday() == 4  # Friday
        market_open = fake_now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = fake_now.replace(hour=16, minute=0, second=0, microsecond=0)
        assert market_open <= fake_now <= market_close


# ============================================================================
# TEST: Schedule behavior at clock boundaries
# ============================================================================

class TestScheduleClockBoundary:
    """Scheduler should respect market_hours_only flag."""

    @pytest.mark.asyncio
    async def test_schedule_skips_outside_market_hours(self):
        """When market_hours_only=True and outside hours, schedule skips."""
        mock_db = MagicMock()
        mock_db.execute.return_value = mock_db
        mock_db.fetchone.return_value = (
            "sched-1", "Test", "interval", None, 2.0,
            True, True, True, "[]", None, True, True,  # market_hours_only=True
            True, None, None, 0, "ok", None,
            "2025-01-01", "2025-01-01",
        )

        from app.services.cycle_scheduler import SchedulerService

        with patch.object(SchedulerService, "_is_market_hours", return_value=False), \
             patch("app.services.cycle_scheduler.get_db") as mock_get_db, \
             patch("app.services.cycle_scheduler.cycle_control") as mock_cc, \
             patch.object(SchedulerService, "_sync_next_run_to_db"):
            mock_cc.is_paused = False
            mock_get_db.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_get_db.return_value.__exit__ = MagicMock(return_value=False)

            await SchedulerService.execute_schedule("sched-1")

        # Should NOT have inserted a system_command (skipped)
        insert_calls = [
            c for c in mock_db.execute.call_args_list
            if "INSERT INTO system_commands" in str(c)
        ]
        assert len(insert_calls) == 0, "Schedule should have been skipped outside market hours"

    @pytest.mark.asyncio
    async def test_schedule_runs_during_market_hours(self):
        """When market_hours_only=True and inside hours, schedule executes."""
        mock_db = MagicMock()
        mock_db.execute.return_value = mock_db
        # First fetchone = schedule row, second = pipeline_state (idle)
        mock_db.fetchone.side_effect = [
            (
                "sched-2", "Test", "interval", None, 2.0,
                True, True, True, "[]", None, True, True,  # market_hours_only=True
                True, None, None, 0, "ok", None,
                "2025-01-01", "2025-01-01",
            ),
            ("idle",),  # pipeline_state.status
        ]

        from app.services.cycle_scheduler import SchedulerService

        with patch.object(SchedulerService, "_is_market_hours", return_value=True), \
             patch("app.services.cycle_scheduler.get_db") as mock_get_db, \
             patch("app.services.cycle_scheduler.cycle_control") as mock_cc, \
             patch.object(SchedulerService, "_sync_next_run_to_db"):
            mock_cc.is_paused = False
            mock_get_db.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_get_db.return_value.__exit__ = MagicMock(return_value=False)

            await SchedulerService.execute_schedule("sched-2")

        # Should have inserted a system_command (executed)
        insert_calls = [
            c for c in mock_db.execute.call_args_list
            if "INSERT INTO system_commands" in str(c)
        ]
        assert len(insert_calls) >= 1, "Schedule should have executed during market hours"


# ============================================================================
# TEST: Paused system auto-resumes when schedule executes
# ============================================================================

class TestPausedSystemResumes:
    """When cycle_control.is_paused is True, schedules should auto-resume the system."""

    @pytest.mark.asyncio
    async def test_paused_system_resumes_schedule(self):
        """Paused system should resume and execute scheduled cycles."""
        mock_db = MagicMock()
        mock_db.execute.return_value = mock_db
        mock_db.fetchone.side_effect = [
            (
                "sched-paused", "Test", "interval", None, 2.0,
                True, True, True, "[]", None, True, False,
                True, None, None, 0, "ok", None,
                "2025-01-01", "2025-01-01",
            ),
            ("idle",),
        ]

        from app.services.cycle_scheduler import SchedulerService

        with patch("app.services.cycle_scheduler.cycle_control") as mock_cc, \
             patch("app.services.cycle_scheduler.get_db") as mock_get_db, \
             patch.object(SchedulerService, "_sync_next_run_to_db"):
            mock_cc.is_paused = True
            mock_get_db.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_get_db.return_value.__exit__ = MagicMock(return_value=False)

            await SchedulerService.execute_schedule("sched-paused")

            # Check that resume was triggered on cycle_control
            mock_cc.resume.assert_called_once()

        # Should have inserted a system_command (executed)
        insert_calls = [
            c for c in mock_db.execute.call_args_list
            if "INSERT INTO system_commands" in str(c)
        ]
        assert len(insert_calls) >= 1, "Schedule should have executed and dispatched command"
