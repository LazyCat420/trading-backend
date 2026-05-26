import pytest
from unittest.mock import MagicMock, patch
from app.services.memory.cycle_closer import CycleCloser
from app.services.rlm.rlm_audit import log_rlm_audit_trail

def test_cycle_closer_sanitizes_surrogates():
    """Verify that CycleCloser sanitizes surrogates from rationales and texts before DB/memory writes."""
    closer = CycleCloser()
    
    # We mock episodic_memory_store and semantic_memory_store
    mock_episodic = MagicMock()
    mock_semantic = MagicMock()
    
    # Input with surrogate character (e.g. \ud83d - half of an emoji surrogate pair)
    surrogate_rationale = "Bullish catalyst found \ud83d\udcbb"
    
    results = [
        {
            "ticker": "AAPL",
            "action": "BUY",
            "confidence": 80,
            "rationale": surrogate_rationale,
            "trade_executed": None,
            "trade_skipped": None
        }
    ]
    
    with patch("app.services.memory.cycle_closer.episodic_memory_store", mock_episodic), \
         patch("app.services.memory.semantic_memory.semantic_memory_store", mock_semantic):
         
        # Run close_cycle
        import asyncio
        asyncio.run(closer.close_cycle(
            cycle_id="test-cycle",
            tickers=["AAPL"],
            mode="test",
            summary={},
            results=results
        ))
        
        # Verify write_episode was called and doesn't contain surrogates
        mock_episodic.write_episode.assert_called_once()
        args, kwargs = mock_episodic.write_episode.call_args
        summary_arg = kwargs.get("summary") or args[2]
        
        # Check that surrogate characters are stripped
        assert "\ud83d" not in summary_arg
        assert "\udcbb" not in summary_arg
        
        # Verify write_semantic was called and content doesn't contain surrogates
        mock_semantic.write_semantic.assert_called_once()
        s_args, s_kwargs = mock_semantic.write_semantic.call_args
        content_arg = s_kwargs.get("content") or s_args[2]
        assert "\ud83d" not in content_arg
        assert "\udcbb" not in content_arg

def test_log_rlm_audit_trail_sanitizes_surrogates():
    """Verify that log_rlm_audit_trail sanitizes surrogates and doesn't throw UnicodeEncodeError."""
    # Mock get_db to return a mock DB cursor that doesn't actually hit postgres
    mock_db = MagicMock()
    
    surrogate_text = "Analysis report with surrogates \ud83d\udcbb"
    
    with patch("app.services.rlm.rlm_audit.get_db", return_value=mock_db):
        try:
            log_rlm_audit_trail(
                cycle_id="test-cycle",
                bot_id="test-bot",
                ticker="AAPL",
                context=surrogate_text,
                trading_system_prompt="System prompt \ud83d",
                active_model="model",
                response_text=surrogate_text,
                tokens_used=100,
                execution_time=1.0,
                completion_tokens=50
            )
        except UnicodeEncodeError as e:
            pytest.fail(f"log_rlm_audit_trail raised UnicodeEncodeError: {e}")
            
        # Verify db.execute was called with sanitized strings
        assert mock_db.__enter__.return_value.execute.call_count >= 3
        
        # Ensure all execute arguments are free of surrogates
        for call in mock_db.__enter__.return_value.execute.call_args_list:
            args, _ = call
            for arg in args[1] if len(args) > 1 else []:
                if isinstance(arg, str):
                    assert "\ud83d" not in arg
                    assert "\udcbb" not in arg
