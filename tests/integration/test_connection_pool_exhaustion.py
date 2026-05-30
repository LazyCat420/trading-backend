"""
Connection Pool Exhaustion Tests — Verify get_db() handles pool pressure gracefully.

Tests:
  1. get_db returns a usable cursor and auto-closes on exit
  2. Multiple concurrent get_db calls don't deadlock
  3. Pool timeout triggers GC recovery
  4. PooledCursor leak detection fires on gc
  5. close() is idempotent (calling twice doesn't crash)
"""
import os
import sys
import time
import threading
import pytest
from unittest.mock import patch, MagicMock, PropertyMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ============================================================================
# FIXTURE: Override global autouse patch_get_db
# ============================================================================

@pytest.fixture(autouse=True)
def patch_get_db():
    """Override the global mock db fixture to test the real connection manager logic."""
    yield


# ============================================================================
# TEST: get_db context manager usage
# ============================================================================

class TestGetDbContextManager:
    """get_db should yield a PooledCursor and auto-close on exit."""

    def test_get_db_yields_cursor_and_auto_closes(self):
        """get_db yields a PooledCursor, which auto-closes when the block exits."""
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = MagicMock()
        mock_conn.autocommit = True

        mock_pool = MagicMock()
        mock_pool.getconn.return_value = mock_conn

        with patch("app.db.connection._ensure_pool", return_value=mock_pool):
            from app.db.connection import get_db

            with get_db() as db:
                assert db is not None
                assert not db._closed

            # After exiting the block, close should have been called
            assert db._closed is True
            mock_pool.putconn.assert_called_once_with(mock_conn)

    def test_get_db_closes_on_exception(self):
        """Even if an exception occurs inside with-block, cursor is closed."""
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = MagicMock()
        mock_conn.autocommit = True

        mock_pool = MagicMock()
        mock_pool.getconn.return_value = mock_conn

        with patch("app.db.connection._ensure_pool", return_value=mock_pool):
            from app.db.connection import get_db

            try:
                with get_db() as db:
                    raise ValueError("intentional error")
            except ValueError:
                pass

            assert db._closed is True
            mock_pool.putconn.assert_called_once()


# ============================================================================
# TEST: Pool timeout recovery
# ============================================================================

class TestPoolTimeoutRecovery:
    """When the pool times out, get_db should trigger GC and retry."""

    def test_pool_timeout_triggers_gc_retry(self):
        """First getconn times out, GC runs, second getconn succeeds."""
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = MagicMock()
        mock_conn.autocommit = True

        mock_pool = MagicMock()
        mock_pool.getconn.side_effect = [
            Exception("timeout getting connection"),
            mock_conn,  # Success after GC
        ]

        with patch("app.db.connection._ensure_pool", return_value=mock_pool), \
             patch("gc.collect") as mock_gc:
            from app.db.connection import get_db

            with get_db() as db:
                assert db is not None

            mock_gc.assert_called_once()
            assert mock_pool.getconn.call_count == 2


# ============================================================================
# TEST: PooledCursor leak detection
# ============================================================================

class TestLeakDetection:
    """PooledCursor.__del__ should warn about unclosed cursors."""

    def test_pooled_cursor_del_warns_on_leak(self):
        """If PooledCursor is garbage-collected without close(), it logs a warning."""
        from app.db.connection import PooledCursor

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = MagicMock()

        mock_pool = MagicMock()

        cursor = PooledCursor(mock_conn, pool=mock_pool)
        assert not cursor._closed

        # Simulate garbage collection
        with patch("app.db.connection.logger") as mock_logger:
            cursor.__del__()

        mock_logger.warning.assert_called_once()
        assert "LEAK" in str(mock_logger.warning.call_args)


# ============================================================================
# TEST: close() idempotency
# ============================================================================

class TestCloseIdempotency:
    """Calling close() multiple times should not crash."""

    def test_double_close_is_safe(self):
        """Calling close() twice should not raise or call putconn twice."""
        from app.db.connection import PooledCursor

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = MagicMock()

        mock_pool = MagicMock()

        cursor = PooledCursor(mock_conn, pool=mock_pool)
        cursor.close()
        cursor.close()  # Should not raise

        mock_pool.putconn.assert_called_once()


# ============================================================================
# TEST: Concurrent access doesn't deadlock
# ============================================================================

class TestConcurrentAccess:
    """Multiple threads using get_db should not deadlock."""

    def test_concurrent_get_db_no_deadlock(self):
        """10 threads each calling get_db should all complete within 5 seconds."""
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = MagicMock()
        mock_conn.autocommit = True

        mock_pool = MagicMock()
        mock_pool.getconn.return_value = mock_conn

        results = []
        errors = []

        def worker():
            try:
                with patch("app.db.connection._ensure_pool", return_value=mock_pool):
                    from app.db.connection import get_db
                    with get_db() as db:
                        db.execute("SELECT 1")
                        results.append(True)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker) for _ in range(10)]
        start = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
        elapsed = time.monotonic() - start

        assert len(errors) == 0, f"Errors in threads: {errors}"
        assert len(results) == 10, f"Only {len(results)}/10 threads completed"
        assert elapsed < 5.0, f"Took {elapsed:.1f}s — possible deadlock"


# ============================================================================
# TEST: Execute with error rollback
# ============================================================================

class TestExecuteErrorHandling:
    """PooledCursor.execute should rollback on SQL errors."""

    def test_execute_rolls_back_on_error(self):
        """If execute() raises, it should rollback the connection."""
        from app.db.connection import PooledCursor

        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("syntax error")

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.autocommit = False

        cursor = PooledCursor(mock_conn, pool=MagicMock())

        with pytest.raises(Exception, match="syntax error"):
            cursor.execute("INVALID SQL")

        mock_conn.rollback.assert_called_once()
