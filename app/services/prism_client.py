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
from app.services.prism_mongo import log_request_to_mongo

logger = logging.getLogger(__name__)


class PrismClient:
    """
    Standalone client for routing LLM requests through Prism Gateway.
    Handles session tracking, timeouts, and payload enrichment for Prism's /agent endpoint.
    """

    def __init__(self):
        self.url = settings.PRISM_URL
        self.project = settings.PRISM_PROJECT
        self.username = settings.PRISM_USERNAME
        self.enabled = settings.PRISM_ENABLED
        self.agent = settings.PRISM_AGENT
        self._sessions: dict[str, str] = {}
        self._client: httpx.AsyncClient | None = None
        self._is_healthy = False
        self._last_health_check = 0.0

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
        if removed:
            logger.debug("[PRISM] Ended session for %s", group_key[:16])

    async def _call_endpoint(
        self,
        client: httpx.AsyncClient,
        url: str,
        json_payload: dict,
        headers: dict | None = None,
    ) -> httpx.Response:
        """Internal API wrapper with explicit timeout and retry logic."""
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
        title_parts = [agent_name]
        if ticker:
            title_parts.append(ticker)
        if cycle_id:
            title_parts.append(cycle_id[:12])
        title = " · ".join(title_parts)

        conversation_id = str(uuid.uuid4())
        group_key = cycle_id or (
            f"chat-{agent_name}" if agent_name == "user_chat" else ""
        )
        session_id, is_new = self._get_or_create_session(group_key)

        payload: dict[str, Any] = {
            "provider": "vllm",
            "model": model,
            "messages": messages,
            "maxTokens": max_tokens,
            "temperature": temperature,
            "conversationId": conversation_id,
            "project": self.project,
            "username": self.username,
            "agent": self.agent,
            "functionCallingEnabled": agentic_mode,
            "agenticLoopEnabled": agentic_mode,
            "conversationMeta": {
                "title": title,
                "systemPrompt": system_prompt[:3000],
                "settings": {
                    "provider": "vllm",
                    "model": model,
                },
            },
        }
        if is_qwen_model:
            payload["thinkingEnabled"] = enable_thinking
        # Only include tools for interactive chat (agentic_mode=True).
        # Pipeline calls must NOT forward tools to Prism — they are
        # executed locally by the trading bot's agent_loop.py.
        if tools and agentic_mode:
            payload["tools"] = tools

        if is_new:
            payload["createSession"] = True
        elif session_id:
            payload["sessionId"] = session_id

        target_url = f"{self.url}/agent?stream=false"
        headers = {
            "Content-Type": "application/json",
            "x-project": self.project,
            "x-username": self.username,
        }

        return payload, target_url, headers

    # ─── Offline / Shadow Logging ─────────────────────────────────────

    async def offline_log(
        self,
        messages: list[dict],
        response_text: str,
        model_name: str,
        agent_name: str = "user_chat",
        ticker: str = "",
        system_prompt: str = "",
        cycle_id: str = "",
        # Telemetry fields for MongoDB request logging
        input_tokens: int = 0,
        output_tokens: int = 0,
        elapsed_ms: int = 0,
        endpoint_name: str = "",
    ):
        """
        Send an offline/shadow log to Prism's Conversation API to record interactions
        that bypass the Prism Gateway proxy (e.g. Hermes, pipeline direct calls).

        Also writes a request telemetry document to MongoDB so the Retina dashboard
        analytics (stats, models, timeline, costs) reflect these calls.

        When cycle_id is provided, all pipeline calls within the same trading cycle
        are grouped under one Prism session for easier auditing.
        """
        if not self.enabled:
            return

        conversation_id = str(uuid.uuid4())
        request_id = str(uuid.uuid4())

        # Session grouping: use cycle_id so all calls in a cycle share one session
        group_key = cycle_id or ""
        session_id, is_new = (
            self._get_or_create_session(group_key) if group_key else (None, False)
        )

        # Build the messages array by appending the assistant's response to the inputs
        full_messages = messages.copy()
        full_messages.append(
            {
                "role": "assistant",
                "content": response_text,
                "provider": "vllm",
                "model": model_name,
            }
        )

        title_parts = [f"Offline Sync: {agent_name}"]
        if ticker:
            title_parts.append(ticker)
        if cycle_id:
            title_parts.append(cycle_id[:12])
        title = " · ".join(title_parts)

        payload: dict[str, Any] = {
            "messages": full_messages,
            "conversationMeta": {
                "title": title,
                "systemPrompt": system_prompt[:3000] if system_prompt else "",
            },
        }

        # Attach session info for cycle grouping in Prism
        if is_new:
            payload["createSession"] = True
        elif session_id:
            payload["sessionId"] = session_id

        headers = {
            "Content-Type": "application/json",
            "x-project": self.project,
            "x-username": self.username,
        }

        # 1. Write conversation message to Prism HTTP API (existing behaviour)
        try:
            client = await self._get_client()
            await self._call_endpoint(
                client=client,
                url=f"{self.url}/conversations/{conversation_id}/messages",
                json_payload=payload,
                headers=headers,
            )
            logger.debug(
                "[PRISM] Offline log saved: %s · %s", agent_name, ticker or "no-ticker"
            )
        except Exception as e:
            logger.warning("[PRISM] Offline logging failed for %s: %s", agent_name, e)

        # 2. Write request telemetry to MongoDB (for dashboard analytics)
        total_time_sec = elapsed_ms / 1000.0 if elapsed_ms > 0 else 0.0
        try:
            await log_request_to_mongo(
                request_id=request_id,
                conversation_id=conversation_id,
                model=model_name,
                agent_name=agent_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_time_sec=total_time_sec,
                messages=messages,
                response_text=response_text,
                project=self.project,
                username=self.username,
                endpoint_name=endpoint_name,
                ticker=ticker,
                cycle_id=cycle_id,
            )
        except Exception as e:
            logger.warning("[PRISM] MongoDB telemetry write failed: %s", e)

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
    ) -> tuple[dict, str, dict]:
        """
        Returns (payload, url, headers) formatted for Prism /agent streaming.
        Routes through the configured agent persona (CUSTOM_MARKET_ALPHA).
        """
        title_parts = [agent_name]
        if ticker:
            title_parts.append(ticker)
        title = " · ".join(title_parts)

        conversation_id = str(uuid.uuid4())
        group_key = f"chat-{agent_name}" if agent_name == "user_chat" else ""
        session_id, is_new = self._get_or_create_session(group_key)

        payload: dict[str, Any] = {
            "provider": "vllm",
            "model": model,
            "messages": messages,
            "maxTokens": max_tokens,
            "temperature": temperature,
            "conversationId": conversation_id,
            "project": self.project,
            "username": self.username,
            "agent": self.agent,
            "functionCallingEnabled": True,
            "agenticLoopEnabled": True,
            "conversationMeta": {
                "title": title,
                "systemPrompt": system_prompt[:3000],
                "settings": {
                    "provider": "vllm",
                    "model": model,
                },
            },
        }
        if is_qwen_model:
            payload["thinkingEnabled"] = enable_thinking
        if tools:
            payload["tools"] = tools

        if is_new:
            payload["createSession"] = True
        elif session_id:
            payload["sessionId"] = session_id

        target_url = f"{self.url}/agent"
        headers = {
            "Content-Type": "application/json",
            "x-project": self.project,
            "x-username": self.username,
        }

        return payload, target_url, headers
