import asyncio
import logging
from pprint import pprint

from app.cognition.evidence.packet_builder import build_evidence_packet
from app.cognition.orchestration.meta_orchestrator import MetaOrchestrator
from app.cognition.verification.sufficiency_gate import check_data_sufficiency
from app.agents.task_board import task_board
from app.cognition.debate.debate_coordinator import run_adversarial_debate
from app.cognition.debate.thesis_agent import generate_thesis

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def main():
    ticker = "TMO"
    cycle_id = "test_cycle_01"
    bot_id = "test_bot_01"

    print(f"\n--- 1. Testing Evidence Packet Builder for {ticker} ---")
    packet = await build_evidence_packet(ticker)
    
    print(f"Missing Fields: {packet.missing_fields}")
    print(f"Number of Structured Facts: {len(packet.structured_facts)}")
    if len(packet.structured_facts) > 0:
        print(f"Sample Fact: {packet.structured_facts[0]}")
    
    print(f"Number of Claims Extracted: {len(packet.claims)}")
    print(f"Source Summaries: {len(packet.source_summaries)}")
    print(f"Freshness: {packet.freshness_summary}")

    print("\n--- 2. Testing Sufficiency Gate ---")
    sufficiency = check_data_sufficiency(ticker, packet)
    print(f"Sufficiency Status: {sufficiency.status}")
    print(f"Sufficiency Blockers: {sufficiency.blockers}")

    print("\n--- 3. Testing MetaOrchestrator (Specialized Agents) ---")
    agent_insights, tokens = await MetaOrchestrator.orchestrate(
        entity_id=ticker,
        packet=packet,
        sufficiency=sufficiency,
        cycle_id=cycle_id,
        bot_id=bot_id,
    )
    print(f"Agent Insights Keys: {list(agent_insights.keys())}")
    
    findings = await task_board.get_findings(ticker=ticker, cycle_id=cycle_id)
    print(f"TaskBoard Findings Posted: {len(findings)}")
    for f in findings:
        print(f" - [{f['source_agent']}] {f['category']}: {f['content'][:100]}...")

    if not agent_insights:
        agent_insights = {}
    
    agent_insights["team_findings"] = (
        f"# TEAM FINDINGS FROM SPECIALIST AGENTS\n"
        f"{len(findings)} findings shared by team:\n"
        f"{[f['content'] for f in findings]}"
    )

    print("\n--- 4. Testing Adversarial Debate ---")
    debate_result = await run_adversarial_debate(
        ticker=ticker,
        packet=packet,
        cycle_id=cycle_id,
        bot_id=bot_id,
        agent_insights=agent_insights,
    )
    
    if debate_result:
        print(f"Debate Integrity: {debate_result.integrity_status}")
        print(f"Debate Verdict: {debate_result.judge_action} @ {debate_result.judge_confidence}%")
        print(f"Verified Bull Claims: {len(debate_result.verified_bull_claims)}")
        print(f"Verified Bear Claims: {len(debate_result.verified_bear_claims)}")
        print(f"Unverified Claims: {len(debate_result.unverified_claims)}")
        print(f"Judge Rationale: {debate_result.judge_rationale}")
        
        extra_context_parts = []
        if debate_result.judge_rationale:
            debate_summary = (
                f"# ADVERSARIAL DEBATE RESULT\n"
                f"**Verdict:** {debate_result.judge_action} @ {debate_result.judge_confidence}%\n"
                f"**Rationale:** {debate_result.judge_rationale}\n"
                f"**Key Factor:** {debate_result.key_deciding_factor}\n"
                f"**Data Quality Warning:** {debate_result.rejected_claim_impact}\n\n"
                f"### Verified Winning Claims:\n"
            )
            extra_context_parts.append(debate_summary)
            
        extra_context = "\n\n".join(extra_context_parts)
    else:
        print("Debate was skipped or disabled.")
        extra_context = ""

    print("\n--- 5. Testing Final Thesis Generation ---")
    try:
        thesis, thesis_tokens = await generate_thesis(
            entity_id=ticker,
            packet=packet,
            cycle_id=cycle_id,
            bot_id=bot_id,
            extra_context=extra_context,
            held=False,
        )
        print(f"Thesis Action: {thesis.action}")
        print(f"Thesis Confidence: {thesis.confidence}")
        print(f"Thesis Claims: {len(thesis.core_claims)}")
        print(f"Thesis Rationale: {thesis.rationale}")
        
        if thesis.confidence == 0 and not thesis.core_claims:
            print("⚠️ EMPTY_SIGNAL DETECTED!")
    except Exception as e:
        print(f"Thesis Generation Failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
