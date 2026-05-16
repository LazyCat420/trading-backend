"""
Tests for Phase 4: Dynamic Subagent Tool Provisioning
Tests for the ToolRegistry filtering methods used by subagent_tools.py

Tests cover:
  🔴🟢 TDD Unit Tests:
    - get_schemas_by_names returns correct subset
    - get_schemas_by_names with empty list returns empty
    - get_schemas_by_names with unknown names returns empty
    - Intersection logic: requested tools that don't exist are silently skipped
    - get_schemas_by_tier / get_schemas_by_permission filtering
  🔗 Integration Tests:
    - spawn_research_subagent with enabled_tools filters correctly
    - spawn_research_subagent with invalid tools falls back to all
"""

import sys
import os
import importlib.util

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import registry directly (not through app.tools which pulls the full app stack)
_reg_path = os.path.join(os.path.dirname(__file__), "..", "app", "tools", "registry.py")
_spec = importlib.util.spec_from_file_location("registry", _reg_path)
_registry_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_registry_mod)

ToolRegistry = _registry_mod.ToolRegistry
PermissionLevel = _registry_mod.PermissionLevel


# ══════════════════════════════════════════════════════════════
# Fixtures — build an isolated registry for testing
# ══════════════════════════════════════════════════════════════


def _build_test_registry() -> ToolRegistry:
    """Create a fresh registry with known test tools."""
    reg = ToolRegistry()

    @reg.register(
        name="search_web",
        description="Search the web for information.",
        parameters={"type": "object", "properties": {}, "required": []},
        tier=0,
        source="ddgs",
        permission=PermissionLevel.READ_ONLY,
        tags=["search", "web"],
    )
    async def _search_web():
        return "web results"

    @reg.register(
        name="scrape_url",
        description="Scrape a URL and return its text content.",
        parameters={"type": "object", "properties": {}, "required": []},
        tier=0,
        source="trafilatura",
        permission=PermissionLevel.READ_ONLY,
        tags=["scrape"],
    )
    async def _scrape_url():
        return "scraped content"

    @reg.register(
        name="get_market_data",
        description="Get market data for a ticker.",
        parameters={"type": "object", "properties": {}, "required": []},
        tier=0,
        source="yfinance",
        permission=PermissionLevel.READ_ONLY,
        tags=["finance", "market"],
    )
    async def _get_market_data():
        return "market data"

    @reg.register(
        name="buy_stock",
        description="Buy shares of a stock.",
        parameters={"type": "object", "properties": {}, "required": []},
        tier=2,
        source="broker",
        permission=PermissionLevel.DESTRUCTIVE,
        tags=["trading"],
    )
    async def _buy_stock():
        return "bought"

    @reg.register(
        name="spawn_research_subagent",
        description="Spawn a research subagent.",
        parameters={"type": "object", "properties": {}, "required": []},
        tier=1,
        permission=PermissionLevel.WRITE,
    )
    async def _spawn():
        return "spawned"

    return reg


# ══════════════════════════════════════════════════════════════
# 🔴🟢 TDD — get_schemas_by_names
# ══════════════════════════════════════════════════════════════


class TestGetSchemasByNames:
    """Test the whitelist filtering used by Phase 4 dynamic tool provisioning."""

    def test_returns_correct_subset(self):
        reg = _build_test_registry()
        result = reg.get_schemas_by_names(["search_web", "scrape_url"])
        names = [s["function"]["name"] for s in result]
        assert names == ["search_web", "scrape_url"]

    def test_single_tool(self):
        reg = _build_test_registry()
        result = reg.get_schemas_by_names(["get_market_data"])
        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_market_data"

    def test_empty_list_returns_empty(self):
        reg = _build_test_registry()
        result = reg.get_schemas_by_names([])
        assert result == []

    def test_unknown_names_returns_empty(self):
        reg = _build_test_registry()
        result = reg.get_schemas_by_names(["nonexistent_tool", "fake_tool"])
        assert result == []

    def test_mixed_known_unknown(self):
        """Known tools are returned, unknown are silently skipped."""
        reg = _build_test_registry()
        result = reg.get_schemas_by_names(["search_web", "nonexistent", "buy_stock"])
        names = [s["function"]["name"] for s in result]
        assert "search_web" in names
        assert "buy_stock" in names
        assert "nonexistent" not in names
        assert len(names) == 2

    def test_all_tools(self):
        reg = _build_test_registry()
        all_names = [s["function"]["name"] for s in reg.schemas]
        result = reg.get_schemas_by_names(all_names)
        assert len(result) == len(all_names)


# ══════════════════════════════════════════════════════════════
# 🔴🟢 TDD — get_schemas_by_tier
# ══════════════════════════════════════════════════════════════


class TestGetSchemasByTier:
    """Test tier-based filtering."""

    def test_tier_0(self):
        reg = _build_test_registry()
        result = reg.get_schemas_by_tier(0)
        names = [s["function"]["name"] for s in result]
        assert "search_web" in names
        assert "scrape_url" in names
        assert "get_market_data" in names
        assert "buy_stock" not in names

    def test_tier_2(self):
        reg = _build_test_registry()
        result = reg.get_schemas_by_tier(2)
        names = [s["function"]["name"] for s in result]
        assert names == ["buy_stock"]

    def test_nonexistent_tier(self):
        reg = _build_test_registry()
        result = reg.get_schemas_by_tier(99)
        assert result == []


# ══════════════════════════════════════════════════════════════
# 🔴🟢 TDD — get_schemas_by_permission
# ══════════════════════════════════════════════════════════════


class TestGetSchemasByPermission:
    """Test permission-based filtering."""

    def test_read_only(self):
        reg = _build_test_registry()
        result = reg.get_schemas_by_permission(PermissionLevel.READ_ONLY)
        names = [s["function"]["name"] for s in result]
        assert "search_web" in names
        assert "buy_stock" not in names

    def test_destructive(self):
        reg = _build_test_registry()
        result = reg.get_schemas_by_permission(PermissionLevel.DESTRUCTIVE)
        names = [s["function"]["name"] for s in result]
        assert names == ["buy_stock"]


# ══════════════════════════════════════════════════════════════
# 🔴🟢 TDD — Tool Metadata
# ══════════════════════════════════════════════════════════════


class TestToolMeta:
    """Test that tool metadata is correctly stored and retrievable."""

    def test_get_tool_meta(self):
        reg = _build_test_registry()
        meta = reg.get_tool_meta("search_web")
        assert meta is not None
        assert meta.tier == 0
        assert meta.source == "ddgs"
        assert meta.permission == PermissionLevel.READ_ONLY

    def test_get_tool_meta_unknown(self):
        reg = _build_test_registry()
        meta = reg.get_tool_meta("nonexistent")
        assert meta is None

    def test_is_fallback_default_false(self):
        reg = _build_test_registry()
        assert reg.is_fallback("search_web") is False

    def test_check_permission_read_only(self):
        reg = _build_test_registry()
        allowed, reason = reg.check_permission("search_web")
        assert allowed is True

    def test_check_permission_destructive(self):
        reg = _build_test_registry()
        allowed, reason = reg.check_permission("buy_stock")
        assert allowed is False
        assert "DESTRUCTIVE" in reason

    def test_registry_snapshot(self):
        reg = _build_test_registry()
        snapshot = reg.get_registry_snapshot()
        assert len(snapshot) == 5
        names = [s["name"] for s in snapshot]
        assert "search_web" in names


# ══════════════════════════════════════════════════════════════
# 🔗 Integration — Subagent Tool Filtering Logic
# ══════════════════════════════════════════════════════════════


class TestSubagentToolFiltering:
    """Test the filtering logic that subagent_tools.py uses.

    We replicate the exact logic here instead of importing subagent_tools
    (which would pull in the full app stack).
    """

    def test_enabled_tools_filters_correctly(self):
        """Phase 4: enabled_tools whitelist works."""
        reg = _build_test_registry()
        enabled_tools = ["search_web", "scrape_url"]

        # Replicate the logic from subagent_tools.py
        active_schemas = reg.get_schemas_by_names(enabled_tools)
        active_schemas = [
            s for s in active_schemas
            if s["function"]["name"] != "spawn_research_subagent"
        ]

        names = [s["function"]["name"] for s in active_schemas]
        assert names == ["search_web", "scrape_url"]
        assert "spawn_research_subagent" not in names

    def test_enabled_tools_removes_spawn_even_if_requested(self):
        """Safety: spawn tool is always excluded from subagents."""
        reg = _build_test_registry()
        enabled_tools = ["search_web", "spawn_research_subagent"]

        active_schemas = reg.get_schemas_by_names(enabled_tools)
        active_schemas = [
            s for s in active_schemas
            if s["function"]["name"] != "spawn_research_subagent"
        ]

        names = [s["function"]["name"] for s in active_schemas]
        assert "spawn_research_subagent" not in names
        assert "search_web" in names

    def test_no_enabled_tools_gives_all_minus_spawn(self):
        """Phase 4: omitting enabled_tools gives all tools minus spawn."""
        reg = _build_test_registry()

        # Replicate the fallback logic
        active_schemas = [
            s for s in reg.schemas
            if s["function"]["name"] != "spawn_research_subagent"
        ]

        names = [s["function"]["name"] for s in active_schemas]
        assert "spawn_research_subagent" not in names
        assert len(names) == 4  # all 5 minus spawn

    def test_invalid_enabled_tools_falls_back(self):
        """Phase 4: invalid tool names produce empty, triggering fallback."""
        reg = _build_test_registry()
        enabled_tools = ["fake_tool_1", "fake_tool_2"]

        active_schemas = reg.get_schemas_by_names(enabled_tools)
        active_schemas = [
            s for s in active_schemas
            if s["function"]["name"] != "spawn_research_subagent"
        ]

        # Fallback: if empty, use all tools
        if not active_schemas:
            active_schemas = [
                s for s in reg.schemas
                if s["function"]["name"] != "spawn_research_subagent"
            ]

        names = [s["function"]["name"] for s in active_schemas]
        assert len(names) == 4  # all tools minus spawn
