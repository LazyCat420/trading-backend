import asyncio
from deepeval.models import DeepEvalBaseLLM
from app.services.vllm_client import llm, Priority


class VLLMDeepEvalWrapper(DeepEvalBaseLLM):
    def __init__(self):
        # We pass no additional config since we use the singleton
        super().__init__()

    def load_model(self):
        # The model is loaded via vllm_client singleton, return it
        return llm

    def generate(self, prompt: str) -> str:
        """Synchronous generate for DeepEval."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We are in a running loop, so we cannot do run_until_complete easily
                # unless we run it in an executor, but vllm_client is async.
                # In FastAPI/async contexts, deepeval calls a_generate anyway,
                # but if it falls back here we must handle it.
                future = asyncio.run_coroutine_threadsafe(self.a_generate(prompt), loop)
                return future.result()
            else:
                return loop.run_until_complete(self.a_generate(prompt))
        except RuntimeError:
            return asyncio.run(self.a_generate(prompt))

    async def a_generate(self, prompt: str) -> str:
        """Asynchronous generate for DeepEval."""
        response, _, _ = await llm.chat(
            system="You are an expert impartial evaluator determining factual alignment.",
            user=prompt,
            temperature=0.0,
            max_tokens=2048,
            priority=Priority.HIGH,  # Evaluators need high priority so they don't block
            agent_name="deepeval_judge",
        )
        return response

    def get_model_name(self) -> str:
        return llm.model
