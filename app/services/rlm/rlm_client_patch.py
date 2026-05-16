"""
OpenAI-compatible client for vLLM that injects enable_thinking.

Monkey-patches rlm.core.rlm.get_client to use VLLMClient for 'vllm' backend.
This allows injecting 'chat_template_kwargs' (e.g. enable_thinking) into the OpenAI client for Qwen thinking capabilities.
"""

import logging
from typing import Any
from rlm.clients.openai import OpenAIClient
from rlm.clients.base_lm import BaseLM

logger = logging.getLogger(__name__)


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
        """Build extra_body with thinking control + optional budget."""
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
_original_get_client = None  # Stored on first call to ensure_patched


def _patched_get_client(backend: str, backend_kwargs: dict[str, Any]) -> BaseLM:
    """Intercept the rlm library's get_client to inject VLLMClient."""
    global _original_get_client
    if backend == "vllm":
        # Copy to avoid mutating the RLM's stored backend_kwargs
        kw = backend_kwargs.copy()
        enable_thinking = kw.pop("enable_thinking", False)
        thinking_budget = kw.pop("thinking_budget", None)
        return VLLMClient(
            enable_thinking=enable_thinking, thinking_budget=thinking_budget, **kw
        )
    # For all other backends, use the original
    if _original_get_client is not None:
        return _original_get_client(backend, backend_kwargs)
    raise RuntimeError("Original get_client not stored!")


def ensure_patched():
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
