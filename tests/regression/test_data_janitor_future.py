import pytest
import json
from unittest.mock import AsyncMock, patch

from app.pipeline.data.data_janitor import evaluate_relevance

@pytest.mark.asyncio
async def test_data_janitor_future_earnings():
    """
    Test that the data janitor correctly classifies future-dated earnings guidance 
    as relevant and the critic validates it, instead of hallucinatory flagging.
    """
    future_text = "NVIDIA expects Q4 2029 revenue to be $50 billion."
    context = "CURRENT DATE: 2026-05-14\nCONTEXT: The user's active portfolio/watchlist currently tracks these tickers: NVDA. News related to these companies is HIGHLY RELEVANT."

    # Mock the LLM to simulate behavior
    with patch("app.pipeline.data.data_janitor.llm") as mock_llm:
        # Mock responses
        # Attempt 1: Janitor says relevant, confidence 80 (so critic is called)
        janitor_res = '{"status": "relevant", "reason": "Future guidance", "confidence": 80}'
        # Critic says VALID
        critic_res = '{"verdict": "VALID", "critique": ""}'
        
        mock_llm.chat = AsyncMock(side_effect=[
            (janitor_res, 100, 100),
            (critic_res, 100, 100)
        ])
        
        result = await evaluate_relevance(future_text, context)
        
        assert result["status"] == "relevant"
        assert result["confidence"] == 80
        
        # Verify prompts had CURRENT DATE
        calls = mock_llm.chat.call_args_list
        assert len(calls) == 2
        
        # Janitor call
        janitor_call = calls[0]
        assert "CURRENT DATE: 2026-05-14" in janitor_call.kwargs["user"]
        
        # Critic call
        critic_call = calls[1]
        assert "CURRENT DATE: 2026-05-14" in critic_call.kwargs["user"]
