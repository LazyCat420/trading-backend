"""
Plugin Registry — Strangler Fig pattern for modular feature migration.

Allows features to be registered, enabled/disabled at runtime, and
migrated one at a time from the monolithic pipeline into isolated plugins.

Usage:
    from app.plugins.plugin_registry import plugin_registry

    # Registration (in the plugin's __init__.py)
    plugin_registry.register("debate", debate_module, enabled=True)

    # Consumption (in the pipeline)
    if plugin_registry.is_enabled("debate"):
        debate = plugin_registry.get("debate")
        result = await debate.run(context)
"""

import logging
import os
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PluginInfo:
    """Metadata for a registered plugin."""

    name: str
    module: ModuleType | Any  # the actual module or class
    enabled: bool = True
    version: str = "0.1.0"
    description: str = ""
    dependencies: list[str] = field(default_factory=list)


class PluginRegistry:
    """Central registry for all optional trading bot plugins.

    Plugins can be enabled/disabled via:
        1. Code: plugin_registry.register("name", module, enabled=False)
        2. Env var: PLUGINS_DISABLED="debate,intel_map" (comma-separated)
        3. Runtime: plugin_registry.disable("name")
    """

    def __init__(self) -> None:
        self._plugins: dict[str, PluginInfo] = {}
        # Parse PLUGINS_DISABLED env var once at init
        disabled_str = os.environ.get("PLUGINS_DISABLED", "")
        self._env_disabled: set[str] = {
            p.strip().lower() for p in disabled_str.split(",") if p.strip()
        }
        if self._env_disabled:
            logger.info(
                "[PluginRegistry] Env-disabled plugins: %s",
                ", ".join(self._env_disabled),
            )

    def register(
        self,
        name: str,
        module: ModuleType | Any,
        *,
        enabled: bool = True,
        version: str = "0.1.0",
        description: str = "",
        dependencies: list[str] | None = None,
    ) -> None:
        """Register a plugin module."""
        # Env var override takes precedence
        if name.lower() in self._env_disabled:
            enabled = False
            logger.info("[PluginRegistry] '%s' disabled by PLUGINS_DISABLED env", name)

        # Check dependencies
        deps = dependencies or []
        for dep in deps:
            if dep in self._plugins and not self._plugins[dep].enabled:
                logger.warning(
                    "[PluginRegistry] '%s' depends on disabled '%s' — disabling",
                    name,
                    dep,
                )
                enabled = False

        self._plugins[name] = PluginInfo(
            name=name,
            module=module,
            enabled=enabled,
            version=version,
            description=description,
            dependencies=deps,
        )
        status = "✓ enabled" if enabled else "✗ disabled"
        logger.info("[PluginRegistry] Registered '%s' v%s [%s]", name, version, status)

    def get(self, name: str) -> Any | None:
        """Get a plugin module by name. Returns None if not found or disabled."""
        info = self._plugins.get(name)
        if info is None:
            logger.debug("[PluginRegistry] Plugin '%s' not found", name)
            return None
        if not info.enabled:
            logger.debug("[PluginRegistry] Plugin '%s' is disabled", name)
            return None
        return info.module

    def is_enabled(self, name: str) -> bool:
        """Check if a plugin is registered and enabled."""
        info = self._plugins.get(name)
        return info.enabled if info else False

    def enable(self, name: str) -> bool:
        """Enable a plugin at runtime. Returns True if found."""
        info = self._plugins.get(name)
        if info is None:
            logger.warning("[PluginRegistry] Cannot enable '%s': not registered", name)
            return False
        info.enabled = True
        logger.info("[PluginRegistry] Enabled '%s'", name)
        return True

    def disable(self, name: str) -> bool:
        """Disable a plugin at runtime. Returns True if found."""
        info = self._plugins.get(name)
        if info is None:
            logger.warning("[PluginRegistry] Cannot disable '%s': not registered", name)
            return False
        info.enabled = False
        logger.info("[PluginRegistry] Disabled '%s'", name)
        return True

    def list_plugins(self) -> list[dict[str, Any]]:
        """List all registered plugins with their status."""
        return [
            {
                "name": info.name,
                "enabled": info.enabled,
                "version": info.version,
                "description": info.description,
                "dependencies": info.dependencies,
            }
            for info in self._plugins.values()
        ]

    def get_enabled(self) -> list[str]:
        """Get names of all enabled plugins."""
        return [name for name, info in self._plugins.items() if info.enabled]

    def get_disabled(self) -> list[str]:
        """Get names of all disabled plugins."""
        return [name for name, info in self._plugins.items() if not info.enabled]


# Singleton — import this everywhere
plugin_registry = PluginRegistry()
