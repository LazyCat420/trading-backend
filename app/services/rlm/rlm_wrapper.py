"""
RLM Wrapper — Recursive Language Model for deep context analysis.

Uses the official alexzhang13/rlm library (pip install rlms).
Points the OpenAI backend at the Jetson vLLM endpoint.

Key integration details:
- VLLMClient subclasses the library's OpenAIClient to inject
  chat_template_kwargs for enable_thinking control on Qwen3.
- We monkey-patch rlm.core.rlm.get_client so the RLM class
  returns our VLLMClient when backend="vllm".
- Compact custom system prompt fits Qwen3's context limits.

Paper: MIT CSAIL 2025 — RLM(GPT-5-mini) > GPT-5 on long-context tasks
Repo:  https://github.com/alexzhang13/rlm
"""

import json
import re
import asyncio
import logging
from typing import Any

from rlm import RLM
from rlm.clients.openai import OpenAIClient
from rlm.clients.base_lm import BaseLM
from rlm.logger import RLMLogger
from app.config import settings
from app.services.rlm.rlm_tools import TRADING_TOOLS
from app.utils.text_utils import strip_think_tags, parse_json_response, sanitize_ascii, parse_trading_decision
from app.services.vllm_client import _is_qwen_model

logger = logging.getLogger(__name__)

# RLM concurrency limiter — RLM uses its own OpenAI client (bypasses our
# priority queue), so we cap concurrent sessions to leave slots for the queue.
_rlm_semaphore = asyncio.Semaphore(settings.RLM_MAX_CONCURRENT)


# ---------------------------------------------------------------------------
# VLLMClient — OpenAI client with Qwen3 thinking control
# ---------------------------------------------------------------------------
class VLLMClient(OpenAIClient):
    """OpenAI-compatible client for vLLM that injects enable_thinking.

    The rlm library's OpenAIClient passes extra_body={} to the OpenAI API.
    We override completion/acompletion to add chat_template_kwargs.
    """

    def __init__(
        self,
        enable_thinking: bool = False,
        thinking_budget: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget
        logger.debug(
            "[VLLMClient] enable_thinking=%s, thinking_budget=%s",
            enable_thinking,
            thinking_budget,
        )

    def _build_extra_body(self) -> dict:
        """Build extra_body with thinking control + optional budget.

        Only injects chat_template_kwargs for Qwen models.
        Non-Qwen models (Nemotron, Llama) don't support this param.
        """
        # Check if the model is Qwen-family
        model_name = getattr(self, "model_name", "") or ""
        if not _is_qwen_model(model_name):
            return {}  # Non-Qwen models: no extra body

        extra = {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}
        if self.thinking_budget and self.enable_thinking:
            extra["thinking"] = {"budget_tokens": self.thinking_budget}
        return extra

    def completion(
        self, prompt: str | list[dict[str, Any]], model: str | None = None
    ) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(
            isinstance(item, dict) for item in prompt
        ):
            messages = prompt
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        model = model or self.model_name
        extra_body = self._build_extra_body()

        response = self.client.chat.completions.create(
            model=model, messages=messages, extra_body=extra_body
        )
        self._track_cost(response, model)
        return response.choices[0].message.content

    async def acompletion(
        self, prompt: str | list[dict[str, Any]], model: str | None = None
    ) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(
            isinstance(item, dict) for item in prompt
        ):
            messages = prompt
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        model = model or self.model_name
        extra_body = self._build_extra_body()

        response = await self.async_client.chat.completions.create(
            model=model, messages=messages, extra_body=extra_body
        )
        self._track_cost(response, model)
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Monkey-patch rlm.core.rlm.get_client to use VLLMClient for vllm backend
# ---------------------------------------------------------------------------
_original_get_client = None  # Stored on first call to _build_rlm


def _patched_get_client(backend: str, backend_kwargs: dict[str, Any]) -> BaseLM:
    """Intercept the rlm library's get_client to inject VLLMClient."""
    if backend == "vllm":
        # Copy to avoid mutating the RLM's stored backend_kwargs
        kw = backend_kwargs.copy()
        enable_thinking = kw.pop("enable_thinking", False)
        thinking_budget = kw.pop("thinking_budget", None)
        return VLLMClient(
            enable_thinking=enable_thinking, thinking_budget=thinking_budget, **kw
        )
    # For all other backends, use the original
    return _original_get_client(backend, backend_kwargs)


def _ensure_patched():
    """One-time monkey-patch of rlm.core.rlm.get_client."""
    global _original_get_client
    if _original_get_client is not None:
        return  # Already patched

    import rlm.core.rlm as rlm_module

    _original_get_client = rlm_module.get_client
    rlm_module.get_client = _patched_get_client
    # Safety assertion: verify the patch actually took effect
    assert rlm_module.get_client is _patched_get_client, (
        "rlm monkey-patch failed! The rlm library may have changed its internals. "
        "Pin rlms to a known-good version in requirements.txt."
    )


from app.services.rlm.rlm_prompts import build_rlm_prompt


def _build_rlm(
    enable_thinking: bool = False,
    max_iterations: int = 4,
    thinking_budget: int | None = None,
    ticker: str = "",
    max_depth: int = 1,
    is_escalation: bool = False,
    target_role: str = "analyst",
    endpoint_override: str | None = None,
    system_prompt_override: str | None = None,
    bot_id: str = "",
) -> RLM:
    """Build an RLM instance with vLLM thinking control.

    Injects frozen memory snapshot and per-ticker skill (if any)
    into the system prompt.

    Args:
        target_role: Which endpoint role to use for this RLM session.
            "analyst" = standard analysis (DGX Spark)
            "trader"  = final trading decisions (most capable model)
    """
    _ensure_patched()

    # Resolve the best endpoint + model based on target_role
    from app.services.vllm_client import llm as _llm_singleton

    if target_role == "trader":
        target_model = _llm_singleton.get_trader_model()
        target_url = _llm_singleton.get_trader_url()
        role_label = "TRADER"
    else:
        target_model, target_endpoint_name = _llm_singleton.get_analyst_model_balanced()
        if endpoint_override:
            target_endpoint_name = endpoint_override
        ep = _llm_singleton._find_endpoint_by_name(target_endpoint_name)
        target_url = ep.url if ep else None
        role_label = "ANALYST"

    if target_model and target_url:
        logger.info(
            "[RLM] Using %s endpoint: %s @ %s", role_label, target_model, target_url
        )
    else:
        target_model = settings.ACTIVE_MODEL
        target_url = settings.JETSON_VLLM_URL
        logger.info(
            "[RLM] %s unavailable, falling back to Jetson: %s @ %s",
            role_label,
            target_model,
            target_url,
        )

    # backend_kwargs will be passed to _patched_get_client → VLLMClient
    # We include enable_thinking + thinking_budget as kwargs popped by our patch
    backend_kwargs = {
        "model_name": target_model,
        "base_url": f"{target_url}/v1",
        "api_key": "not-needed",
        "enable_thinking": enable_thinking,
        "thinking_budget": thinking_budget,
    }

    # Fix #7: RLM internal timeout MUST be shorter than the asyncio.wait_for(300s)
    # in decision_engine.py, otherwise the thread runs past the "timeout" invisibly.
    # 290s gives 10s buffer before the 300s asyncio cap fires (was previously 120s/110s).
    timeout = 290.0 if enable_thinking else 300.0

    # Disk logging: writes JSONL files showing every iteration, code block, and tool call
    import os

    log_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "data",
        "rlm_logs",
    )
    os.makedirs(log_dir, exist_ok=True)

    full_system_prompt = build_rlm_prompt(
        ticker, is_escalation, system_prompt_override, bot_id=bot_id
    )

    return RLM(
        backend="vllm",
        backend_kwargs=backend_kwargs,
        environment="local",
        max_depth=max_depth,
        max_iterations=max_iterations,
        max_timeout=timeout,
        max_errors=3,
        custom_system_prompt=full_system_prompt,
        custom_tools=TRADING_TOOLS,
        verbose=True,
        logger=RLMLogger(log_dir=log_dir),
    )




async def rlm_analyze(
    ticker: str,
    context: str,
    max_iterations: int = 4,
    enable_thinking: bool = False,
    thinking_budget: int | None = None,
    cycle_id: str = "",
    bot_id: str = "",
    max_depth: int = 1,
    is_escalation: bool = False,
    target_role: str = "analyst",
    endpoint_override: str | None = None,
    system_prompt_override: str | None = None,
    agent_step: str = "analysis",
) -> dict[str, Any]:
    """
    Run RLM analysis on a large context blob.

    Args:
        enable_thinking: If True, Qwen3 generates <think> chains (slower, deeper).
                        If False, disables thinking via chat_template_kwargs.
        thinking_budget: Max tokens for <think> block (only used when enable_thinking=True).
        target_role: "analyst" or "trader" — which endpoint role to use.

    Returns dict with: action, confidence, rationale, iterations, tokens_used, method
    """
    # Acquire RLM slot (caps concurrent RLM sessions to leave queue slots open)
    async with _rlm_semaphore:
        logger.debug("[RLM] %s acquired RLM slot (role=%s)", ticker, target_role)

        # Isolate Prompt Construction Failures
        try:
            rlm_instance = _build_rlm(
                enable_thinking=enable_thinking,
                max_iterations=max_iterations,
                thinking_budget=thinking_budget,
                ticker=ticker,
                max_depth=max_depth,
                is_escalation=is_escalation,
                target_role=target_role,
                endpoint_override=endpoint_override,
                system_prompt_override=system_prompt_override,
                bot_id=bot_id,
            )
        except Exception as build_e:
            logger.error("[RLM] %s prompt construction failed: %s", ticker, build_e)
            logger.debug("[RLM] traceback (build):", exc_info=True)
            return {
                "action": "HOLD",
                "confidence": 0,
                "rationale": f"RLM prompt construction failed: {str(build_e)}",
                "iterations": 0,
                "tokens_used": 0,
                "method": "rlm",
            }

        try:
            # Resolve the active model for logging
            from app.services.vllm_client import llm as _llm_singleton

            if target_role == "trader":
                active_model = _llm_singleton.get_trader_model()
            else:
                active_model, target_endpoint_name = (
                    _llm_singleton.get_analyst_model_balanced()
                )
                if endpoint_override:
                    target_endpoint_name = endpoint_override
                ep = _llm_singleton._find_endpoint_by_name(target_endpoint_name)

            if not active_model:
                active_model = settings.ACTIVE_MODEL

            root_query = (
                f"Analyze {ticker} for a BUY/SELL/HOLD trading decision. "
                f"Context has {len(context):,} characters of market data. "
                f"Inspect it via REPL, then FINAL your JSON decision."
            )

            # Sanitize context: RLM LocalREPL writes to file with cp1252 on Windows
            # Unicode chars like ↑↓→←• crash it. Replace with ASCII equivalents.
            safe_context = sanitize_ascii(context)

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: rlm_instance.completion(
                    prompt=safe_context, root_prompt=root_query
                ),
            )

            response_text = result.response or ""
            logger.debug(
                "[RLM] %s response (%d chars): %s",
                ticker,
                len(response_text),
                response_text[:300],
            )
            decision = parse_trading_decision(response_text)
            logger.debug("[RLM] %s decision: %s", ticker, decision)

            tokens_used = 0
            if result.usage_summary:
                tokens_used = (
                    result.usage_summary.total_input_tokens
                    + result.usage_summary.total_output_tokens
                )

            iterations = 0
            if result.metadata and hasattr(result.metadata, "iterations"):
                iterations = len(result.metadata.iterations)

            action = decision.get("action", "HOLD").upper()
            if action not in ("BUY", "SELL", "HOLD"):
                action = "HOLD"

            # Infrastructure Audit Trail: Log to PostgreSQL (with context dedup)
            try:
                from app.services.rlm.rlm_audit import log_rlm_audit_trail

                # Resolve endpoint name for telemetry
                ep_name = ""
                if target_role == "trader":
                    ep_name = "dgx_spark"  # trader role default
                else:
                    try:
                        _, ep_n = _llm_singleton.get_analyst_model_balanced()
                        ep_name = ep_n or ""
                    except Exception:
                        pass

                # RLM usage has total input/output split
                p_tokens = 0
                c_tokens = 0
                if result.usage_summary:
                    p_tokens = result.usage_summary.total_input_tokens
                    c_tokens = result.usage_summary.total_output_tokens

                log_rlm_audit_trail(
                    cycle_id=cycle_id,
                    bot_id=bot_id,
                    ticker=ticker,
                    context=context,
                    trading_system_prompt=build_rlm_prompt(
                        ticker, is_escalation, system_prompt_override, bot_id=bot_id
                    ),
                    active_model=active_model,
                    response_text=response_text,
                    tokens_used=tokens_used,
                    execution_time=result.execution_time,
                    agent_step=agent_step,
                    endpoint_name=ep_name,
                    prompt_tokens=p_tokens,
                    completion_tokens=c_tokens,
                )
            except Exception as db_e:
                logger.error(
                    "[RLM] [DB] Audit log un-writable for %s: %s", ticker, db_e
                )

            return {
                "action": action,
                "confidence": int(decision.get("confidence", 0)),
                "rationale": decision.get("rationale", response_text[:300]),
                "iterations": iterations,
                "tokens_used": tokens_used,
                "execution_time_s": round(result.execution_time, 2),
                "method": "rlm",
            }

        except Exception as e:
            logger.error("[RLM] %s failed: %s", ticker, e)
            logger.debug("[RLM] traceback:", exc_info=True)
            return {
                "action": "HOLD",
                "confidence": 0,
                "rationale": f"RLM analysis failed: {str(e)}",
                "iterations": 0,
                "tokens_used": 0,
                "method": "rlm",
            }
