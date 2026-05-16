"""
Agent Registry — Capability-based lookup for fallback routing.

CORAL Gap 4: When an agent fails with REROUTABLE, the registry finds
which alternate agent can substitute based on:
  1. The `fallback_for` declaration (explicit substitution list)
  2. The `min_context_required` check (can the fallback work with available data?)
  3. The `health` status (is the fallback agent itself healthy?)

Loads from memory/agent_registry.json at startup and maintains
in-memory health toggles updated at runtime.

Usage:
    from app.recovery.registry import agent_registry

    # Find a substitute for a failed agent
    fallback = agent_registry.find_fallback("sentiment_agent")
    # Returns "meta_agent" (declared as fallback in the JSON)

    # Mark an agent as unhealthy
    agent_registry.mark_degraded("fund_flow_agent")

    # Check status
    info = agent_registry.get_agent_info("technical_agent")
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

# Default path for the agent registry JSON
_DEFAULT_REGISTRY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "memory",
    "agent_registry.json",
)


class AgentRegistry:
    """Capability-based agent directory for dynamic fallback routing.

    The registry is loaded once from a JSON file and cached in memory.
    Health status is maintained at runtime and resets each cycle.
    """

    def __init__(self, registry_path: str | None = None):
        self._path = registry_path or _DEFAULT_REGISTRY_PATH
        self._agents: dict[str, dict] = {}
        self._health: dict[str, str] = {}  # Runtime health overrides
        self._loaded = False

    def _ensure_loaded(self):
        """Load the registry JSON if not already loaded."""
        if self._loaded:
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                self._agents = json.load(f)
            # Initialize health from JSON defaults
            for name, info in self._agents.items():
                self._health[name] = info.get("health", "ok")
            self._loaded = True
            logger.info(
                "[REGISTRY] Loaded %d agents from %s",
                len(self._agents),
                self._path,
            )
        except FileNotFoundError:
            logger.warning("[REGISTRY] Registry file not found: %s", self._path)
            self._loaded = True  # Don't retry
        except Exception as e:
            logger.error("[REGISTRY] Failed to load registry: %s", e)
            self._loaded = True

    def find_fallback(
        self,
        failed_agent: str,
        available_context: list[str] | None = None,
    ) -> str | None:
        """Find the best substitute for a failed agent.

        Selection logic:
        1. Look for agents that declare `failed_agent` in their `fallback_for` list
        2. Among those, filter by:
           a. Health status (must be "ok")
           b. Context requirements (if available_context provided)
        3. Prefer agents with the most specific skill overlap
        4. meta_agent is the universal fallback (no context requirements)

        Args:
            failed_agent: Name of the agent that failed.
            available_context: List of available data categories
                               (e.g., ["price_data", "news_items"]).
                               If None, context requirements are not checked.

        Returns:
            Name of the fallback agent, or None if no suitable fallback.
        """
        self._ensure_loaded()

        if failed_agent not in self._agents:
            logger.warning(
                "[REGISTRY] Unknown agent '%s' — no fallback possible",
                failed_agent,
            )
            return None

        candidates = []
        for name, info in self._agents.items():
            # Skip the failed agent itself
            if name == failed_agent:
                continue

            # Must declare it can substitute for the failed agent
            if failed_agent not in info.get("fallback_for", []):
                continue

            # Must be healthy
            if self._health.get(name, "ok") != "ok":
                logger.debug(
                    "[REGISTRY] Skipping %s (health=%s)",
                    name,
                    self._health.get(name),
                )
                continue

            # Check context requirements if we know what's available
            if available_context is not None:
                required = info.get("min_context_required", [])
                if required and not all(r in available_context for r in required):
                    logger.debug(
                        "[REGISTRY] Skipping %s (missing context: %s)",
                        name,
                        [r for r in required if r not in available_context],
                    )
                    continue

            candidates.append(name)

        if not candidates:
            logger.info(
                "[REGISTRY] No fallback found for '%s'",
                failed_agent,
            )
            return None

        # Prefer the candidate with the most specific skills
        # (more skills = more specific analysis, likely better substitute)
        best = max(
            candidates,
            key=lambda n: len(self._agents[n].get("skills", [])),
        )

        logger.info(
            "[REGISTRY] Fallback for '%s' → '%s' (from %d candidates)",
            failed_agent,
            best,
            len(candidates),
        )
        return best

    def mark_degraded(self, agent_name: str):
        """Mark an agent as degraded (won't be selected as fallback)."""
        self._ensure_loaded()
        self._health[agent_name] = "degraded"
        logger.warning("[REGISTRY] Agent '%s' marked DEGRADED", agent_name)

    def mark_healthy(self, agent_name: str):
        """Restore an agent to healthy status."""
        self._ensure_loaded()
        self._health[agent_name] = "ok"
        logger.info("[REGISTRY] Agent '%s' marked HEALTHY", agent_name)

    def is_healthy(self, agent_name: str) -> bool:
        """Check if an agent is currently healthy."""
        self._ensure_loaded()
        return self._health.get(agent_name, "ok") == "ok"

    def get_agent_info(self, agent_name: str) -> dict | None:
        """Get full info for an agent (skills, health, fallback_for, etc.)."""
        self._ensure_loaded()
        info = self._agents.get(agent_name)
        if info is None:
            return None
        return {
            **info,
            "name": agent_name,
            "health": self._health.get(agent_name, "ok"),
        }

    def list_agents(self) -> list[dict]:
        """List all registered agents with their current status.

        Used by the monitoring dashboard.
        """
        self._ensure_loaded()
        return [
            {
                "name": name,
                "skills": info.get("skills", []),
                "health": self._health.get(name, "ok"),
                "fallback_for": info.get("fallback_for", []),
                "min_context": info.get("min_context_required", []),
            }
            for name, info in self._agents.items()
        ]

    def reset_health(self):
        """Reset all agents to healthy status (called at cycle start)."""
        self._ensure_loaded()
        for name in self._agents:
            self._health[name] = "ok"
        logger.info("[REGISTRY] All agent health statuses reset to OK")


# Singleton
agent_registry = AgentRegistry()
