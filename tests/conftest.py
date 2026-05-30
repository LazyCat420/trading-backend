"""
Shared test fixtures for trading-cycle-backend tests.

Provides:
  - Mocked settings that don't require .env
  - Isolated ToolRegistry for testing tool provisioning
"""
import os
import sys

# Set execution mode to staging during tests to bypass production API key validation check
os.environ["EXECUTION_MODE"] = "staging"

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

# ── Database Fixtures ──────────────────────────────────────────────────

@pytest.fixture(scope="session")
def real_test_db_engine():
    """Return the NAS test database URL if explicitly enabled."""
    if not os.environ.get("TRADING_BOT_TEST_DB"):
        return None

    from app.config import settings
    db_url = settings.TEST_DATABASE_URL

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

        # Explicitly close the cursor BEFORE cleanup to prevent
        # [DB LEAK] warnings from PooledCursor.__del__
        cursor.close()

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


@pytest.fixture(autouse=True)
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
    client.chat = AsyncMock(return_value=("mock response", 100, 500))
    client.chat_with_tools = AsyncMock(return_value={"text": "mock response", "total_tokens": 100, "elapsed_ms": 500})
    client.model = "test-model"
    client.discover_roles = MagicMock(return_value={})
    client.get_least_busy_model = MagicMock(return_value="test-model")
    client.get_trader_model = MagicMock(return_value="test-model")
    client.queue_status.return_value = {
        "jetson": {"active": 0, "max_concurrent": 2, "queued": 0},
        "dgx_spark": {"active": 0, "max_concurrent": 4, "queued": 0},
    }
    return client


@pytest.fixture(autouse=True)
def patch_llm(mock_llm):
    """Patch the global llm singleton in-place so all modules share the mock."""
    from app.services.vllm_client import llm
    with patch.object(llm, "chat", mock_llm.chat), \
         patch.object(llm, "chat_with_tools", mock_llm.chat_with_tools), \
         patch.object(llm, "discover_roles", mock_llm.discover_roles), \
         patch.object(llm, "get_least_busy_model", mock_llm.get_least_busy_model), \
         patch.object(llm, "get_trader_model", mock_llm.get_trader_model), \
         patch.object(llm, "queue_status", mock_llm.queue_status):
        yield mock_llm


def pytest_unconfigure(config):
    """Force exit to prevent hanging on background threads or connections."""
    import os
    exitstatus = getattr(config, "exitstatus", 0)
    os._exit(exitstatus)


