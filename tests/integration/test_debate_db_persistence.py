import pytest
from app.cognition.debate.debate_coordinator import run_adversarial_debate
from app.db.connection import get_db

@pytest.mark.asyncio
async def test_run_adversarial_debate_persistence(patch_llm, patch_get_db, mock_db):
    """
    Test that run_adversarial_debate successfully persists its results to the debate_history table.
    We mock the underlying llm calls to simulate a debate completion.
    """
    # Mock LLM calls:
    # 1. Bull persona fast debate
    # 2. Bear persona fast debate
    # 3. Cross-examiner
    
    bull_response = '{"action": "BUY", "claims": ["Revenue up 20% [fundamentals:RevGrowth=0.20]"]}'
    bear_response = '{"action": "SELL", "claims": ["P/E is 80 [market_data:PE=80.0]"]}'
    cross_exam_response = '{"summary": "Evidence verified", "verified_bull_claims": ["Revenue up 20% [fundamentals:RevGrowth=0.20]"], "verified_bear_claims": ["P/E is 80 [market_data:PE=80.0]"], "unverified_bull_claims": [], "unverified_bear_claims": []}'
    judge_response = '{"action": "HOLD", "confidence": 50, "winning_side": "bull", "key_deciding_factor": "Revenue growth", "rejected_claim_impact": "None", "rationale": "Revenue up 20%"}'

    # We mock the chat method on the LLM singleton. Since it is imported before the patch,
    # we patch the attribute on the imported modules.
    
    from unittest.mock import AsyncMock, patch
    
    from app.cognition.contracts.evidence import EvidencePacket
    packet = EvidencePacket(entity_id="TEST", ticker="TEST", structured_facts=[])
    
    mock_run_biased = AsyncMock()
    mock_run_biased.side_effect = [
        (bull_response, 100, []), # Bull T1
        (bull_response, 100, []), # Bull T2
        (bull_response, 100, []), # Bull T3
        (bull_response, 100, []), # Bull T4 (if any)
        (bear_response, 100, []), # Bear T1
        (bear_response, 100, []), # Bear T2
        (bear_response, 100, []), # Bear T3
        (bear_response, 100, []), # Bear T4 (if any)
    ] * 5 # Enough for all permutations

    async def mock_chat_fn(*args, **kwargs):
        agent_name = kwargs.get("agent_name", "")
        if "cross_examiner" in agent_name:
            return (cross_exam_response, 100, 100)
        elif agent_name == "debate_judge":
            return (judge_response, 100, 100)
        return (cross_exam_response, 100, 100)
    
    with patch("app.cognition.debate.debate_coordinator._run_biased_agent", mock_run_biased), \
         patch("app.cognition.debate.debate_coordinator.llm.chat", new_callable=AsyncMock) as mock_chat1, \
         patch("app.cognition.debate.debate_judge.llm.chat", new_callable=AsyncMock) as mock_chat2:
        
        mock_chat1.side_effect = mock_chat_fn
        mock_chat2.side_effect = mock_chat_fn
        
        result = await run_adversarial_debate(
            ticker="TEST",
            packet=packet,
            cycle_id="cycle_123",
            bot_id="bot_abc",
            position_context={
                "held": True,
                "qty": 10,
                "avg_entry": 150.00,
                "current_price": 160.00,
                "unrealized_pnl": 100.00,
                "unrealized_pnl_pct": 6.67,
                "holding_days": 14,
                "stop_loss_pct": 8.0,
                "stop_price": 138.00,
                "original_thesis": "Bullish breakout",
                "original_thesis_date": "2026-05-10",
                "original_thesis_conf": 85,
            }
        )
        
    assert result.judge_action == "HOLD"
    
    # Check that database persistence was called
    db_calls = mock_db.execute.call_args_list
    
    # Look for inserts into debate_history
    history_inserts = [
        call for call in db_calls if "INSERT INTO debate_history" in call[0][0]
    ]
    
    # It should save 1 entry representing the entire debate (with persona outcomes)
    assert len(history_inserts) == 1, "Expected 1 debate_history insert"
    
    # Let's verify the cross-examiner entry
    final_insert = history_inserts[0]
    params = final_insert[0][1]
    
    # params structure from debate_coordinator:
    # (id, ticker, cycle_id, pro_argument, con_argument, winner, final_action, final_confidence, persona_outcomes)
    assert params[1] == "TEST"
    assert params[2] == "cycle_123"
    assert "Revenue up 20%" in params[3]  # pro_argument
    assert "P/E is 80" in params[4]       # con_argument
    assert "HOLD" in params[6]            # final_action
    
    # Verify persona_outcomes json
    import json
    persona_outcomes = json.loads(params[8])
    assert "Fundamental" in persona_outcomes
