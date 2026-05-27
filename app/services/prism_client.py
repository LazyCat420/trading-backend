"""
Prism Gateway Client — isolated component for interacting with the Prism AI proxy.
Provides request formatting, session management, and metric tracking.
Can be plugged into other applications easily.
"""

import asyncio
import logging
import uuid
from typing import Any

import httpx
from httpx import RequestError, HTTPStatusError
from app.config import settings
logger = logging.getLogger(__name__)


def normalize_prism_model(model_name: str) -> str:
    """Normalize dynamic model IDs from vLLM to canonical Prism names.
    E.g., 'Qwen/Qwen3.5-122B-A10B-FP8' -> 'qwen3.5-122b-a10b'
    """
    # Return model_name directly without modifications so that Prism matches the loaded vLLM model key.
    return model_name


class PrismClient:
    """
    Standalone client for routing LLM requests through Prism Gateway.
    Handles session tracking, timeouts, and payload enrichment for Prism's /agent endpoint.
    """

    def __init__(self):
        self._sessions: dict[str, str] = {}
        self._conversations: dict[str, str] = {}
        self._client: httpx.AsyncClient | None = None
        self._is_healthy = False
        self._last_health_check = 0.0
        self._url: str | None = None
        self._project: str | None = None
        self._username: str | None = None
        self._enabled: bool | None = None
        self._agent: str | None = None

    @property
    def url(self) -> str:
        return self._url if self._url is not None else settings.PRISM_URL

    @url.setter
    def url(self, value: str):
        self._url = value

    @property
    def project(self) -> str:
        return self._project if self._project is not None else settings.PRISM_PROJECT

    @project.setter
    def project(self, value: str):
        self._project = value

    @property
    def username(self) -> str:
        return self._username if self._username is not None else settings.PRISM_USERNAME

    @username.setter
    def username(self, value: str):
        self._username = value

    @property
    def enabled(self) -> bool:
        return self._enabled if self._enabled is not None else settings.PRISM_ENABLED

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    @property
    def agent(self) -> str:
        return self._agent if self._agent is not None else settings.PRISM_AGENT

    @agent.setter
    def agent(self, value: str):
        self._agent = value

    async def check_health(self) -> bool:
        """Dynamically check if Prism is available.
        Caches the result for 30 seconds to avoid latency on every request.
        """
        import time
        now = time.monotonic()

        if not self.enabled:
            return False

        # Return cached health if we checked within the last 30 seconds
        if now - self._last_health_check < 30.0:
            return self._is_healthy

        try:
            client = await self._get_client()
            r = await client.get(f"{self.url}/health", timeout=2.0)
            is_up = r.status_code == 200
        except Exception:
            is_up = False

        if self._is_healthy and not is_up:
            logger.warning("[PRISM] ⚠️ Prism Gateway at %s is unreachable. Marking as unhealthy.", self.url)
        elif not self._is_healthy and is_up:
            logger.info("[PRISM] ✅ Prism Gateway at %s is healthy again.", self.url)

        self._is_healthy = is_up
        self._last_health_check = now

        return is_up

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy-init a persistent async client for connection reuse."""
        if self._client is not None and not self._client.is_closed:
            try:
                asyncio.get_event_loop()
            except RuntimeError:
                self._client = None
        if self._client is None or self._client.is_closed:
            limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
            self._client = httpx.AsyncClient(timeout=120.0, limits=limits)
        return self._client

    def _get_or_create_session(self, group_key: str) -> tuple[str | None, bool]:
        """Return (session_id, is_new) for the given group key."""
        if not group_key:
            return None, False
        if group_key in self._sessions:
            return self._sessions[group_key], False
        session_id = str(uuid.uuid4())
        self._sessions[group_key] = session_id
        return session_id, True

    def end_session(self, group_key: str):
        """Clear a tracked session."""
        removed = self._sessions.pop(group_key, None)
        removed_conv = self._conversations.pop(group_key, None)
        if removed or removed_conv:
            logger.debug("[PRISM] Ended session for %s", group_key[:16])

    async def _call_endpoint(
        self,
        client: httpx.AsyncClient,
        url: str,
        json_payload: dict,
        headers: dict | None = None,
    ) -> httpx.Response:
        """Internal API wrapper with explicit timeout and retry logic."""
        from app.utils.text_utils import sanitize_surrogates
        json_payload = sanitize_surrogates(json_payload)

        max_retries = 3
        backoff = 1.0

        for attempt in range(max_retries):
            try:
                r = await client.post(
                    url,
                    json=json_payload,
                    headers=headers,
                    timeout=120.0,
                )
                r.raise_for_status()
                return r
            except (RequestError, HTTPStatusError) as e:
                # Do not retry on 4xx client errors
                if (
                    isinstance(e, HTTPStatusError)
                    and 400 <= e.response.status_code < 500
                ):
                    raise
                if attempt == max_retries - 1:
                    logger.debug(
                        "[PRISM] API call to %s failed after %d attempts: %s",
                        url,
                        max_retries,
                        e,
                    )
                    raise
                await asyncio.sleep(backoff)
                backoff *= 2.0

        raise RuntimeError(
            f"Failed to call endpoint {url} after {max_retries} attempts"
        )

    def _format_tools(self, tools: list[dict] | None) -> list[dict] | None:
        """Format standard OpenAI tool schemas into Prism's expected flat format."""
        if not tools:
            return None
        formatted_tools = []
        for t in tools:
            if isinstance(t, dict) and t.get("type") == "function" and "function" in t:
                func_data = t["function"]
                formatted_tool = {
                    "name": func_data.get("name"),
                    "description": func_data.get("description", ""),
                }
                if "parameters" in func_data:
                    formatted_tool["parameters"] = func_data["parameters"]
                if "_isCustom" in t:
                    formatted_tool["_isCustom"] = t["_isCustom"]
                elif "_isCustom" in func_data:
                    formatted_tool["_isCustom"] = func_data["_isCustom"]
                formatted_tools.append(formatted_tool)
            else:
                formatted_tools.append(t)
        return formatted_tools

    def _enrich_payload_with_tools_and_prefixes(
        self,
        payload: dict[str, Any],
        tools: list[dict] | None,
        system_prompt: str,
        messages: list[dict]
    ) -> None:
        """Enriches the payload with enabledTools and prefixes tool names in system_prompt/messages."""
        import re
        from app.tools.registry import registry
        
        # 1. Collect all tool names for prefixing
        tool_names = set()
        for name in registry.tools.keys():
            tool_names.add(name)
            
        mcp_prefix = "mcp__lazy-tool-service__"
        
        # 2. Build enabledTools list based on whitelist if provided, otherwise all registry tools
        enabled_tools = []
        if tools is not None:
            # We have a specific whitelist of tools
            whitelist_names = set()
            for t in tools:
                if isinstance(t, dict):
                    if t.get("type") == "function" and "function" in t:
                        name = t["function"].get("name")
                        if name:
                            whitelist_names.add(name)
                    elif t.get("name"):
                        whitelist_names.add(t["name"])
            for name in whitelist_names:
                enabled_tools.append(name)
                enabled_tools.append(f"{mcp_prefix}{name}")
        else:
            # Fall back to all registry tools
            for name in tool_names:
                enabled_tools.append(name)
                enabled_tools.append(f"{mcp_prefix}{name}")
            
        # Add core built-in tools
        built_ins = [
            "execute_python", "search_web", "read_file", "write_file",
            "str_replace_file", "file_info", "file_diff", "browser_action",
            "browser_script", "precise_calculator"
        ]
        for bi in built_ins:
            if bi not in enabled_tools:
                enabled_tools.append(bi)
                
        payload["enabledTools"] = enabled_tools
        
        # 3. Prefix prompt and messages
        def prefix_text(text: str) -> str:
            if not text:
                return text
            # Sort by length descending to avoid partial matches
            sorted_names = sorted(list(tool_names), key=len, reverse=True)
            for name in sorted_names:
                pattern = r'\b(?<!' + re.escape(mcp_prefix) + r')' + re.escape(name) + r'\b'
                text = re.sub(pattern, mcp_prefix + name, text)
            return text
            
        prefixed_system_prompt = prefix_text(system_prompt)
        payload["systemPrompt"] = prefixed_system_prompt[:15000]
        if "conversationMeta" in payload:
            payload["conversationMeta"]["systemPrompt"] = prefixed_system_prompt[:15000]
            
        prefixed_messages = []
        for msg in messages:
            content = msg.get("content")
            new_msg = {**msg}
            if isinstance(content, str):
                new_msg["content"] = prefix_text(content)
            
            # Prefix name field for tool responses
            if msg.get("role") == "tool":
                tool_name = msg.get("name")
                if tool_name and tool_name in tool_names:
                    new_msg["name"] = f"{mcp_prefix}{tool_name}"
                    
            # Prefix tool calls
            if "tool_calls" in msg:
                prefixed_calls = []
                for tc in msg["tool_calls"]:
                    func_info = tc.get("function", {})
                    func_name = func_info.get("name")
                    if func_name and func_name in tool_names:
                        prefixed_calls.append({
                            **tc,
                            "function": {
                                **func_info,
                                "name": f"{mcp_prefix}{func_name}"
                            }
                        })
                    else:
                        prefixed_calls.append(tc)
                new_msg["tool_calls"] = prefixed_calls
            prefixed_messages.append(new_msg)
            
        payload["messages"] = prefixed_messages

    def _resolve_prism_agent_id(self, agent_name: str) -> str:
        """Map local trading service agent names to custom agent IDs registered in Prism."""
        from app.services.prism_agent_registry import resolve_agent_id
        return resolve_agent_id(agent_name, default_agent=self.agent)

    def get_chat_payload_and_url(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        system_prompt: str,
        agent_name: str,
        ticker: str,
        cycle_id: str,
        enable_thinking: bool,
        tools: list[dict] | None = None,
        is_qwen_model: bool = False,
        agentic_mode: bool = True,
        provider: str = "vllm",
    ) -> tuple[dict, str, dict]:
        """
        Returns (payload, url, headers) formatted for Prism /agent (non-streaming).
        Routes through the configured agent persona (CUSTOM_MARKET_ALPHA).

        Args:
            agentic_mode: When True (default), enables coordinator tools (team_create,
                send_message, stop_agent) and the agentic loop. When False, Prism acts
                as a simple LLM proxy without spawning worker agents. Pipeline calls
                should pass agentic_mode=False to prevent the coordinator doom loop.
        """
        model = normalize_prism_model(model)
        title_parts = [agent_name]
        if ticker:
            title_parts.append(ticker)
        if cycle_id:
            title_parts.append(cycle_id[:12])
        title = " · ".join(title_parts)

        if cycle_id:
            if agent_name == "user_chat":
                group_key = cycle_id
            else:
                ticker_part = f"-{ticker}" if ticker else ""
                group_key = f"{cycle_id}{ticker_part}-{agent_name}"
        else:
            group_key = f"chat-{agent_name}" if agent_name == "user_chat" else ""
        session_id, is_new = self._get_or_create_session(group_key)

        # Reuse conversation ID for the same group key (e.g. cycle/ticker/agent) only in agentic mode
        if agentic_mode and group_key:
            if group_key not in self._conversations:
                self._conversations[group_key] = str(uuid.uuid4())
            conversation_id = self._conversations[group_key]
        else:
            conversation_id = str(uuid.uuid4())

        payload: dict[str, Any] = {
            "provider": provider,
            "model": model,
            "messages": messages,
            "maxTokens": max_tokens,
            "temperature": temperature,
            "conversationId": conversation_id,
            "project": self.project,
            "username": self.username,
            "agent": self._resolve_prism_agent_id(agent_name),
            "functionCallingEnabled": agentic_mode or bool(tools),
            "agenticLoopEnabled": agentic_mode,
            "autoApprove": settings.PRISM_AUTO_APPROVE,
            "systemPrompt": system_prompt[:15000],
            "conversationMeta": {
                "title": title,
                "systemPrompt": system_prompt[:15000],
                "settings": {
                    "provider": provider,
                    "model": model,
                },
            },
        }
        if agentic_mode:
            self._enrich_payload_with_tools_and_prefixes(payload, tools, system_prompt, messages)
        if is_qwen_model or (model and "qwen" in model.lower()):
            payload["thinkingEnabled"] = enable_thinking
        # Forward tools if present.
        if tools:
            payload["tools"] = self._format_tools(tools)

        if is_new:
            payload["createSession"] = True
        elif session_id:
            payload["sessionId"] = session_id

        # Route to agentic loop endpoint only when agentic_mode is True, otherwise use standard chat completion
        if agentic_mode:
            target_url = f"{self.url}/agent?stream=false"
        else:
            target_url = f"{self.url}/chat?stream=false"
        headers = {
            "Content-Type": "application/json",
            "x-project": self.project,
            "x-username": self.username,
        }

        return payload, target_url, headers

    def get_stream_payload_and_url(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        system_prompt: str,
        agent_name: str,
        ticker: str,
        enable_thinking: bool,
        tools: list[dict] | None = None,
        is_qwen_model: bool = False,
        agentic_mode: bool = True,
        provider: str = "vllm",
    ) -> tuple[dict, str, dict]:
        """
        Returns (payload, url, headers) formatted for Prism /agent streaming.
        Routes through the configured agent persona (CUSTOM_MARKET_ALPHA).
        """
        model = normalize_prism_model(model)
        title_parts = [agent_name]
        if ticker:
            title_parts.append(ticker)
        title = " · ".join(title_parts)

        group_key = f"chat-{agent_name}" if agent_name == "user_chat" else ""
        session_id, is_new = self._get_or_create_session(group_key)

        # Reuse conversation ID for the same group key only in agentic mode
        if agentic_mode and group_key:
            if group_key not in self._conversations:
                self._conversations[group_key] = str(uuid.uuid4())
            conversation_id = self._conversations[group_key]
        else:
            conversation_id = str(uuid.uuid4())

        payload: dict[str, Any] = {
            "provider": provider,
            "model": model,
            "messages": messages,
            "maxTokens": max_tokens,
            "temperature": temperature,
            "conversationId": conversation_id,
            "project": self.project,
            "username": self.username,
            "agent": self._resolve_prism_agent_id(agent_name),
            "functionCallingEnabled": agentic_mode or bool(tools),
            "agenticLoopEnabled": agentic_mode,
            "autoApprove": settings.PRISM_AUTO_APPROVE,
            "systemPrompt": system_prompt[:15000],
            "conversationMeta": {
                "title": title,
                "systemPrompt": system_prompt[:15000],
                "settings": {
                    "provider": provider,
                    "model": model,
                },
            },
        }
        if agentic_mode:
            self._enrich_payload_with_tools_and_prefixes(payload, tools, system_prompt, messages)
        if is_qwen_model or (model and "qwen" in model.lower()):
            payload["thinkingEnabled"] = enable_thinking
        if tools:
            payload["tools"] = self._format_tools(tools)

        if is_new:
            payload["createSession"] = True
        elif session_id:
            payload["sessionId"] = session_id

        # Route to agentic loop endpoint only when agentic_mode is True, otherwise use standard chat completion
        if agentic_mode:
            target_url = f"{self.url}/agent"
        else:
            target_url = f"{self.url}/chat"
        headers = {
            "Content-Type": "application/json",
            "x-project": self.project,
            "x-username": self.username,
        }

        return payload, target_url, headers

    async def register_or_update_custom_agent(
        self,
        name: str,
        identity: str,
        guidelines: str = "",
        enabled_tools: list[str] | None = None,
        project: str = "vllm-trading-bot",
    ) -> str:
        """Register a custom agent in Prism, or update it if it already exists.
        Returns the custom agent ID (e.g. 'CUSTOM_BEAR_MACRO_SENTIMENT_T2_AGENT').
        """
        # Clean agent name to match uppercase identifier slug
        # e.g., "bear_macro_sentiment_t2_agent" -> "BEAR_MACRO_SENTIMENT_T2_AGENT"
        slug = name.upper().replace(" ", "_").replace("-", "_").strip("_")
        agent_id = f"CUSTOM_{slug}" if not slug.startswith("CUSTOM_") else slug

        # Fetch list of existing agents to check for duplicates and get the database ID
        client = await self._get_client()
        headers = {
            "Content-Type": "application/json",
            "x-project": self.project,
            "x-username": self.username,
        }

        agent_db_id = None
        try:
            r = await client.get(f"{self.url}/custom-agents", headers=headers, timeout=10.0)
            r.raise_for_status()
            existing_agents = r.json()
            for agent in existing_agents:
                if agent.get("agentId") == agent_id:
                    agent_db_id = agent.get("_id")
                    break
        except Exception as e:
            logger.warning("[PRISM] Failed to query existing custom agents: %s", e)

        # Standardize display name (e.g. "bear_macro_sentiment_t2_agent" -> "Bear Macro Sentiment T2 Agent")
        display_name = name.replace("_", " ").title() if "_" in name else name

        payload = {
            "name": display_name,
            "identity": identity,
            "guidelines": guidelines,
            "enabledTools": enabled_tools or [],
            "project": project,
            "usesDirectoryTree": False,
            "usesCodingGuidelines": False,
        }

        if agent_db_id:
            # Update existing custom agent
            try:
                logger.info("[PRISM] Updating existing custom agent %s (db_id: %s)", agent_id, agent_db_id)
                r = await client.put(f"{self.url}/custom-agents/{agent_db_id}", json=payload, headers=headers, timeout=10.0)
                r.raise_for_status()
            except Exception as e:
                logger.error("[PRISM] Failed to update custom agent %s: %s", agent_id, e)
                raise
        else:
            # Create a new custom agent
            try:
                logger.info("[PRISM] Creating new custom agent %s", agent_id)
                r = await client.post(f"{self.url}/custom-agents", json=payload, headers=headers, timeout=10.0)
                r.raise_for_status()
            except Exception as e:
                logger.error("[PRISM] Failed to create custom agent %s: %s", agent_id, e)
                raise

        return agent_id

