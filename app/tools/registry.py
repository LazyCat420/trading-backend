"""
Tool Registry — Central registry for all agent tools.

Inspired by Claude Code's buildTool() pattern:
  - Every tool has typed metadata (tier, source, permission, size limits)
  - Input validation via Pydantic models (optional per tool)
  - Result truncation prevents context overflow
  - Permission levels gate destructive operations
"""

import inspect
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


# ── Permission Levels (inspired by Claude Code's isReadOnly/isDestructive) ──
class PermissionLevel(str, Enum):
    """Permission level for a tool. Higher levels require more oversight.

    READ_ONLY:   Safe to call freely — no side effects (e.g., get_market_data)
    WRITE:       Creates/modifies data — logged but auto-approved (e.g., write_memory_note)
    DESTRUCTIVE: Irreversible operations — requires human approval (e.g., run_local_command, deploy_fix)
    """

    READ_ONLY = "read_only"
    WRITE = "write"
    DESTRUCTIVE = "destructive"


@dataclass
class ToolMeta:
    """Metadata for a registered tool.

    Modeled after Claude Code's Tool type, which includes isEnabled(),
    isReadOnly(), isDestructive(), isConcurrencySafe(), and maxResultSizeChars.
    """

    tier: int = 0  # 0=collect, 1=analyze, 2=validate
    source: str = ""  # e.g. "yfinance", "hermes", "reddit"
    fallback_only: bool = False  # if True, only invoked when primary tools return empty
    permission: PermissionLevel = PermissionLevel.READ_ONLY
    max_result_chars: int = (
        50_000  # Truncate results beyond this to prevent context overflow
    )
    input_model: type[BaseModel] | None = (
        None  # Optional Pydantic model for input validation
    )
    concurrency_safe: bool = True  # If False, tool should not be called in parallel
    tags: list[str] = field(
        default_factory=list
    )  # Search keywords for future ToolSearch


class ToolRegistry:
    def __init__(self):
        self.tools: dict[str, Callable] = {}
        self.schemas: list[dict] = []
        self._meta: dict[str, ToolMeta] = {}

    def register(
        self,
        func: Callable | None = None,
        name: str | None = None,
        description: str | None = None,
        parameters: dict | None = None,
        *,
        tier: int = 0,
        source: str = "",
        fallback_only: bool = False,
        permission: PermissionLevel = PermissionLevel.READ_ONLY,
        max_result_chars: int = 50_000,
        input_model: type[BaseModel] | None = None,
        concurrency_safe: bool = True,
        tags: list[str] | None = None,
    ):
        """Register an async function as a tool. Can be used as a decorator with or without arguments.

        Args:
            tier: Processing tier (0=collect, 1=analyze, 2=validate).
            source: Data source label (e.g. "yfinance", "hermes").
            fallback_only: If True, only invoke when primary tools return empty results.
            permission: Permission level (read_only, write, destructive).
            max_result_chars: Max chars for tool output before truncation.
            input_model: Optional Pydantic BaseModel for input validation.
            concurrency_safe: If False, this tool should not be called in parallel.
            tags: Search keywords for future ToolSearch feature.
        """

        def decorator(f: Callable):
            tool_name = name or f.__name__
            self.tools[tool_name] = f
            self._meta[tool_name] = ToolMeta(
                tier=tier,
                source=source,
                fallback_only=fallback_only,
                permission=permission,
                max_result_chars=max_result_chars,
                input_model=input_model,
                concurrency_safe=concurrency_safe,
                tags=tags or [],
            )

            self.schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": description
                        or f.__doc__
                        or "No description provided.",
                        "parameters": parameters
                        or {"type": "object", "properties": {}, "required": []},
                    },
                }
            )
            return f

        if func is None:
            return decorator
        else:
            return decorator(func)

    def is_fallback(self, name: str) -> bool:
        """Check if a tool is marked as fallback-only."""
        meta = self._meta.get(name)
        return meta.fallback_only if meta else False

    def get_tool_meta(self, name: str) -> ToolMeta | None:
        """Get metadata for a registered tool."""
        return self._meta.get(name)

    def get_primary_schemas(self) -> list[dict]:
        """Get schemas for non-fallback tools only (Layer 1 structured APIs)."""
        return [s for s in self.schemas if not self.is_fallback(s["function"]["name"])]

    def get_fallback_schemas(self) -> list[dict]:
        """Get schemas for fallback-only tools (Layer 2 Hermes)."""
        return [s for s in self.schemas if self.is_fallback(s["function"]["name"])]

    def get_schemas_by_names(self, names: list[str]) -> list[dict]:
        """Get schemas for specific tools by name (whitelist filtering)."""
        return [s for s in self.schemas if s["function"]["name"] in names]

    def get_schemas_by_tier(self, tier: int) -> list[dict]:
        """Get schemas for tools at a specific processing tier."""
        return [
            s
            for s in self.schemas
            if self._meta.get(s["function"]["name"], ToolMeta()).tier == tier
        ]

    def get_schemas_by_permission(self, permission: PermissionLevel) -> list[dict]:
        """Get schemas for tools at a specific permission level."""
        return [
            s
            for s in self.schemas
            if self._meta.get(s["function"]["name"], ToolMeta()).permission
            == permission
        ]

    def _validate_input(self, func_name: str, kwargs: dict) -> dict:
        """Validate tool input against Pydantic model if one is registered.

        Returns validated kwargs (potentially with defaults filled in).
        Raises ValidationError if input is invalid.
        """
        meta = self._meta.get(func_name)
        if not meta or not meta.input_model:
            return kwargs

        try:
            validated = meta.input_model(**kwargs)
            return validated.model_dump()
        except ValidationError as e:
            logger.warning(
                "[ToolRegistry] Input validation FAILED for %s: %s | raw kwargs: %s",
                func_name,
                e.error_count(),
                kwargs,
            )
            raise

    def _truncate_result(self, func_name: str, result: str) -> str:
        """Truncate tool result if it exceeds max_result_chars.

        Prevents context window overflow when tools return huge payloads
        (e.g., scraping a full webpage, dumping entire price histories).
        """
        meta = self._meta.get(func_name)
        max_chars = meta.max_result_chars if meta else 50_000

        if len(result) <= max_chars:
            return result

        # Keep first 80% + last 10%, with a truncation notice in the middle
        head_chars = int(max_chars * 0.8)
        tail_chars = int(max_chars * 0.1)
        truncated_count = len(result) - head_chars - tail_chars
        notice = (
            f"\n\n[TRUNCATED: {truncated_count:,} characters removed. "
            f"Original size: {len(result):,} chars.]\n\n"
        )
        truncated = result[:head_chars] + notice + result[-tail_chars:]

        logger.info(
            "[ToolRegistry] Truncated %s output: %d → %d chars (-%d)",
            func_name,
            len(result),
            len(truncated),
            truncated_count,
        )
        return truncated

    def check_permission(self, func_name: str) -> tuple[bool, str]:
        """Check if a tool is allowed to execute based on its permission level.

        Returns (allowed: bool, reason: str).
        DESTRUCTIVE tools are blocked by default — caller must handle approval.
        """
        meta = self._meta.get(func_name)
        if not meta:
            return True, "no metadata (legacy tool, allowed)"

        if meta.permission == PermissionLevel.DESTRUCTIVE:
            return False, (
                f"Tool '{func_name}' is DESTRUCTIVE (permission={meta.permission.value}). "
                f"Requires human approval before execution."
            )

        return True, f"permission={meta.permission.value}"

    async def execute_tool_call(
        self,
        tool_call: dict,
        *,
        skip_permission_check: bool = False,
        agent_name: str = "",
        ticker: str = "",
        cycle_id: str = "",
        tool_cache: dict | None = None,
        enforce_ticker: bool = False,
    ) -> dict:
        """Execute a single tool call from the LLM and return the formatted result.

        Args:
            tool_call: The tool call dict from the LLM response.
            skip_permission_check: If True, bypass permission checks (for pre-approved calls).
            agent_name: The agent requesting this tool (for usage tracking).
            ticker: Current ticker context (for usage tracking).
            cycle_id: Current cycle context (for usage tracking).
            tool_cache: Pre-fetched tool results to intercept redundant executions.
            enforce_ticker: If True, block tool calls where the 'ticker' argument
                            doesn't match the context ticker. Used during debates
                            to prevent cross-ticker data contamination.
        """
        tool_call_id = tool_call.get("id")
        function_info = tool_call.get("function", {})
        func_name = function_info.get("name")
        arguments_json = function_info.get("arguments", "{}")

        # ── Robust Normalization of Tool Name & Arguments ──
        if func_name:
            # Strip trailing tags like </Function
            if "</" in func_name:
                func_name = func_name.split("</")[0].strip()
            
            # Map capitalized names with spaces/hyphens to registered lowercase snake_case names
            def clean_str(s: str) -> str:
                return s.lower().replace(" ", "").replace("_", "").replace("-", "")
            
            target_cleaned = clean_str(func_name)
            matched_name = None
            for registered_name in self.tools:
                reg_cleaned = clean_str(registered_name)
                # Check for exact normalized match
                if reg_cleaned == target_cleaned:
                    matched_name = registered_name
                    break
                # Check with get_ prefix variation (e.g. "Technical Indicators" -> "get_technical_indicators")
                if clean_str("get_" + registered_name) == target_cleaned or reg_cleaned == clean_str("get_" + target_cleaned):
                    matched_name = registered_name
                    break

            if matched_name:
                if matched_name != func_name:
                    logger.info("[ToolRegistry] Normalized tool call name: '%s' -> '%s'", func_name, matched_name)
                func_name = matched_name
                # Update function info representation
                function_info["name"] = func_name

        # ── Parse and Normalize JSON arguments ──
        try:
            # Clean up trailing tags from arguments string if present
            if arguments_json and "</" in arguments_json:
                arguments_json = arguments_json.split("</")[0].strip()
                
            kwargs = json.loads(arguments_json)
            
            # Convert all argument keys to lowercase (e.g. {"Ticker": "IP"} -> {"ticker": "IP"})
            kwargs = {k.lower(): v for k, v in kwargs.items()}
            
            # Update the parsed arguments representation
            arguments_json = json.dumps(kwargs)
            function_info["arguments"] = arguments_json
            
            normalized_args = json.dumps(kwargs, sort_keys=True, separators=(',', ':'))
            cache_key = f"{func_name}:{normalized_args}"
        except Exception as e:
            logger.error(
                "[ToolRegistry] Failed to parse JSON arguments for %s: %s | raw: %s",
                func_name,
                e,
                arguments_json[:200],
            )
            self._log_usage(
                func_name or "unknown",
                agent_name,
                ticker,
                cycle_id,
                False,
                0,
                f"JSON parse error: {e}",
            )
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": func_name,
                "content": json.dumps(
                    {
                        "error": f"Invalid JSON arguments: {e}",
                        "hint": "Please provide valid JSON arguments.",
                    }
                ),
            }

        # ── Ticker-Lock Guardrail ──
        if ticker and "ticker" in kwargs:
            tool_ticker = str(kwargs["ticker"]).upper().strip()
            context_ticker = str(ticker).upper().strip()
            if tool_ticker and tool_ticker != context_ticker:
                logger.warning(
                    "[ToolRegistry] CROSS-CONTAMINATION BLOCKED: %s requested ticker %s but context is %s",
                    func_name,
                    tool_ticker,
                    context_ticker,
                )
                self._log_usage(
                    func_name, agent_name, ticker, cycle_id, False, 0, f"Cross-contamination blocked: {tool_ticker} != {context_ticker}"
                )
                return {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": func_name,
                    "content": json.dumps({
                        "error": f"Unauthorized ticker access. You are analyzing {context_ticker}, but you requested {tool_ticker}.",
                        "hint": f"Only query data for the assigned ticker ({context_ticker})."
                    })
                }

        # ── Tool existence check ──
        import os
        use_lazy_tool = os.environ.get("USE_LAZY_TOOL_SERVICE", "false").lower() == "true"
        is_remote_tool = False

        if func_name not in self.tools:
            if use_lazy_tool:
                logger.info("[ToolRegistry] Tool '%s' not found locally, treating as remote tool.", func_name)
                is_remote_tool = True
            else:
                self._log_usage(
                    func_name or "unknown",
                    agent_name,
                    ticker,
                    cycle_id,
                    False,
                    0,
                    f"Tool '{func_name}' not found",
                )
                return {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": func_name,
                    "content": json.dumps({"error": f"Tool '{func_name}' not found."}),
                }

        # ── Cache interception check ──
        if tool_cache:
            cached_content = None
            hit_key = None
            if cache_key in tool_cache:
                cached_content = tool_cache[cache_key]
                hit_key = cache_key
            elif func_name in tool_cache:
                cached_content = tool_cache[func_name]
                hit_key = func_name

            if cached_content is not None:
                logger.info(
                    "[ToolRegistry] Intercepted %s via tool_cache (%s) to prevent redundant execution.",
                    func_name,
                    hit_key,
                )
                # Log cache hit usage (0ms execution)
                self._log_usage(
                    func_name, agent_name, ticker, cycle_id, True, 0, "Cache Hit"
                )
                return {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": func_name,
                    "content": cached_content,
                }

        if not is_remote_tool:
            # ── Permission check ──
            if not skip_permission_check:
                allowed, reason = self.check_permission(func_name)
                if not allowed:
                    logger.warning(
                        "[ToolRegistry] PERMISSION DENIED: %s — %s", func_name, reason
                    )
                    self._log_usage(
                        func_name,
                        agent_name,
                        ticker,
                        cycle_id,
                        False,
                        0,
                        f"Permission denied: {reason}",
                    )
                    return {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": func_name,
                        "content": json.dumps(
                            {
                                "error": reason,
                                "requires_approval": True,
                                "pending_command": arguments_json,
                                "action_required": "This tool requires human approval. The request has been logged for review.",
                            }
                        ),
                    }

            # ── Input validation (Pydantic) ──
            try:
                kwargs = self._validate_input(func_name, kwargs)
            except ValidationError as e:
                error_details = e.errors()
                self._log_usage(
                    func_name,
                    agent_name,
                    ticker,
                    cycle_id,
                    False,
                    0,
                    f"Validation failed: {len(error_details)} errors",
                )
                return {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": func_name,
                    "content": json.dumps(
                        {
                            "error": f"Input validation failed: {len(error_details)} error(s)",
                            "details": [
                                {"field": err.get("loc", []), "message": err.get("msg", "")}
                                for err in error_details[:5]
                            ],
                            "hint": "Fix the arguments and try again.",
                        }
                    ),
                }

        logger.info(
            "[ToolRegistry] Executing tool: %s with args: %s", func_name, kwargs
        )

        # ── Execute ──
        t0 = time.monotonic()
        try:
            import os
            service_source = "trading-service"
            if use_lazy_tool and is_remote_tool:
                import httpx
                base_url = os.environ.get("LAZY_TOOL_SERVICE_URL", "http://10.0.0.16:5591")
                url = f"{base_url}/execute/{func_name}"
                logger.info("[ToolRegistry] Forwarding execution of '%s' to lazy-tool-service: %s", func_name, url)
                service_source = "lazy-tool-service"
                async with httpx.AsyncClient(timeout=300.0) as client:
                    resp = await client.post(url, json=kwargs)
                    if resp.status_code != 200:
                        raise RuntimeError(f"lazy-tool-service returned status code {resp.status_code}: {resp.text}")
                    resp_json = resp.json()
                    result = resp_json.get("content", "")
            else:
                func = self.tools[func_name]
                if inspect.iscoroutinefunction(func):
                    result = await func(**kwargs)
                else:
                    result = func(**kwargs)

            if not isinstance(result, str):
                result = json.dumps(result)

            # ── Result truncation ──
            result = self._truncate_result(func_name, result)

            elapsed_ms = int((time.monotonic() - t0) * 1000)
            self._log_usage(func_name, agent_name, ticker, cycle_id, True, elapsed_ms, service_source=service_source)

            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": func_name,
                "content": result,
                "service_source": service_source,
            }
        except Exception as e:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            logger.exception("[ToolRegistry] Tool execution failed for %s", func_name)
            import os
            service_source = "lazy-tool-service" if os.environ.get("USE_LAZY_TOOL_SERVICE", "false").lower() == "true" else "trading-service"
            self._log_usage(
                func_name, agent_name, ticker, cycle_id, False, elapsed_ms, str(e), service_source=service_source
            )
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": func_name,
                "content": json.dumps({"error": str(e)}),
                "service_source": service_source,
            }

    def _log_usage(
        self,
        tool_name: str,
        agent_name: str = "",
        ticker: str = "",
        cycle_id: str = "",
        success: bool = True,
        execution_ms: int = 0,
        error_message: str | None = None,
        service_source: str = "trading-service",
    ) -> None:
        """Log a tool usage event to PostgreSQL (fire-and-forget)."""
        try:
            from app.db.connection import get_db

            with get_db() as db:
                db.execute(
                    "INSERT INTO tool_usage_stats "
                    "(tool_name, agent_name, ticker, cycle_id, success, execution_ms, error_message, service_source) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    [
                        tool_name,
                        agent_name,
                        ticker,
                        cycle_id,
                        success,
                        execution_ms,
                        error_message,
                        service_source,
                    ],
                )
        except Exception as e:
            # Never let usage tracking break tool execution
            logger.debug("[ToolRegistry] Usage log failed (non-fatal): %s", e)

    def get_registry_snapshot(self) -> list[dict]:
        """Return a snapshot of all registered tools with their metadata.

        Used by the /tools API endpoint for frontend introspection.
        """
        snapshot = []
        for schema in self.schemas:
            func_info = schema.get("function", {})
            name = func_info.get("name", "")
            meta = self._meta.get(name, ToolMeta())
            snapshot.append(
                {
                    "name": name,
                    "description": func_info.get("description", ""),
                    "parameters": func_info.get("parameters", {}),
                    "tier": meta.tier,
                    "source": meta.source,
                    "permission": meta.permission.value,
                    "fallback_only": meta.fallback_only,
                    "concurrency_safe": meta.concurrency_safe,
                    "max_result_chars": meta.max_result_chars,
                    "tags": meta.tags,
                }
            )
        return snapshot


registry = ToolRegistry()
