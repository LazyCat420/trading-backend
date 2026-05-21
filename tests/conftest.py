"""
Shared test fixtures for trading-cycle-backend tests.

Provides:
  - Mocked settings that don't require .env
  - Isolated ToolRegistry for testing tool provisioning
"""
import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock, patch

# ── Database Fixtures ──────────────────────────────────────────────────

@pytest.fixture(scope="session")
def real_test_db_engine():
    """Return the NAS test database URL if explicitly enabled."""
    if not os.environ.get("TRADING_BOT_TEST_DB"):
        return None

    db_url = "postgresql://trader:trading_bot_pass@10.0.0.16:5433/trading_bot_test"

    try:
        import psycopg
        with psycopg.connect(db_url, autocommit=True, connect_timeout=5) as conn:
            conn.execute("SELECT 1")
        return db_url
    except Exception:
        return None


@pytest.fixture
def real_db(real_test_db_engine):
    """Yield a real database connection pool and truncate tables on exit."""
    if not real_test_db_engine:
        pytest.skip("Test database not enabled. Set TRADING_BOT_TEST_DB=1 to enable.")

    from psycopg_pool import ConnectionPool
    from app.db.connection import PooledCursor

    pool = ConnectionPool(conninfo=real_test_db_engine, min_size=1, max_size=2, kwargs={"autocommit": True})
    pool.wait()

    with pool.connection() as conn:
        cursor = PooledCursor(conn)
        yield cursor

        try:
            tables = conn.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            ).fetchall()
            if tables:
                table_names = ", ".join([f'"{t[0]}"' for t in tables])
                conn.execute(f"TRUNCATE TABLE {table_names} CASCADE")
        except Exception:
            pass

    pool.close()

@pytest.fixture
def patch_real_get_db(real_db):
    """Patch get_db() to return the real test database cursor."""
    from unittest.mock import patch
    from contextlib import contextmanager
    @contextmanager
    def fake_get_db():
        yield real_db
    with patch("app.db.connection.get_db", fake_get_db):
        yield real_db


@pytest.fixture
def mock_db():
    """Provide a mock PooledCursor that behaves like a real DB cursor."""
    cursor = MagicMock()
    cursor.execute.return_value = cursor
    cursor.fetchone.return_value = None
    cursor.fetchall.return_value = []
    cursor.description = None
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    return cursor


@pytest.fixture
def patch_get_db(mock_db):
    """Patch get_db() globally so no real DB connections are created."""
    with patch("app.db.connection.get_db", return_value=mock_db), \
         patch("app.db.connection._ensure_pool"):
        yield mock_db


# ── vLLM Client Fixtures ──────────────────────────────────────────────


@pytest.fixture
def mock_llm():
    """Provide a mock VLLMClient with a pre-configured chat() response."""
    client = MagicMock()
    client.chat = MagicMock(return_value=("mock response", 100, 500))
    client.model = "test-model"
    client.discover_roles = MagicMock(return_value={})
    client.get_least_busy_model = MagicMock(return_value="test-model")
    client.get_trader_model = MagicMock(return_value="test-model")
    client.queue_status.return_value = {
        "jetson": {"active": 0, "max_concurrent": 2, "queued": 0},
        "dgx_spark": {"active": 0, "max_concurrent": 4, "queued": 0},
    }
    return client


@pytest.fixture
def patch_llm(mock_llm):
    """Patch the global llm singleton so no real vLLM calls are made."""
    with patch("app.services.vllm_client.llm", mock_llm):
        yield mock_llm

