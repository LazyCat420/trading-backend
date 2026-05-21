"""
Conftest for unit tests that need the tool registry populated.

Mocks the psycopg module so tool imports can succeed without a real database,
then forces the tool registration by importing app.tools.
"""

import sys
import pytest
from unittest.mock import MagicMock


@pytest.fixture(autouse=True, scope="session")
def mock_psycopg():
    """Mock psycopg so tool modules can import without a real DB driver."""
    # Only mock if psycopg isn't already importable
    if "psycopg" not in sys.modules:
        mock_module = MagicMock()
        mock_pool = MagicMock()
        sys.modules["psycopg"] = mock_module
        sys.modules["psycopg.rows"] = MagicMock()
        sys.modules["psycopg_pool"] = mock_pool

    # Now force-import all tool modules so decorators run and register tools
    try:
        import app.tools  # noqa: F401 — triggers all tool registrations
    except Exception:
        pass  # Some tools may fail deeper imports; that's OK for unit tests

    yield
