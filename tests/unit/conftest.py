"""
Conftest for unit tests that need the tool registry populated.

Mocks the psycopg module so tool imports can succeed without a real database,
then forces the tool registration by importing app.tools.
"""

import sys
import pytest
from unittest.mock import MagicMock
@pytest.fixture(autouse=True, scope="session")
def disable_prism_routing_globally():
    """Globally disable Prism routing in unit tests to prevent network/endpoint dependencies."""
    from app.config import settings
    # Pydantic settings are mutable by default, but using object.__setattr__ to bypass any frozen checks safely.
    object.__setattr__(settings, "PRISM_ENABLED", False)
    object.__setattr__(settings, "PRISM_AGENT_ROUTING", False)

@pytest.fixture(autouse=True, scope="session")
def mock_new_agents_globally():
    """Globally mock the new specialist agents that make external LLM calls."""
    from unittest.mock import AsyncMock, patch
    with patch("app.agents.portfolio_allocator_agent.run_portfolio_allocator", new_callable=AsyncMock) as mock_pa, \
         patch("app.agents.post_mortem_auditor_agent.run_post_mortem", new_callable=AsyncMock) as mock_pm:
        mock_pa.return_value = {}
        mock_pm.return_value = None
        yield mock_pa, mock_pm

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
