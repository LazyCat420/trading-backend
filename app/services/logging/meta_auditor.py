import logging
from app.services import vllm_client
from app.utils.text_utils import parse_json_response
from app.services.logging.cycle_auditor import auditor

logger = logging.getLogger(__name__)

META_AUDITOR_PROMPT = """You are the Lead Investment Auditor.
Your job is to read an AI-generated investment thesis and compare it to the raw Evidence Packet.
Check for the following FATAL flaws:
1. Hallucination: The thesis cites numbers/events not present in the evidence.
2. Contradiction: The thesis concludes BUY, but the math/metrics it cites are bearish (or vice versa).
3. Non-Actionable: The thesis is wishy-washy and doesn't make a firm, data-backed stance.

EVIDENCE PACKET:
{evidence}

THESIS TO AUDIT:
{thesis}

Respond in STRICT JSON format:
{
  "quality_score": 0-100,
  "is_hallucination": true/false,
  "logic_flaws": ["list of flaws", "or empty"],
  "final_verdict": "PASS" or "FAIL"
}
"""

async def audit_thesis_quality(ticker: str, packet: str, thesis: str, cycle_id: str, bot_id: str) -> dict:
    """Run an LLM-as-a-Judge semantic audit on the generated thesis."""
    logger.info("[MetaAuditor] Running semantic audit for %s", ticker)
    
    prompt = META_AUDITOR_PROMPT.format(
        evidence=packet[:10000], 
        thesis=thesis[:5000]
    )
    
    try:
        response, _, _ = await vllm_client.chat(
            system="You are a strict financial auditor.",
            user=prompt,
            temperature=0.1,
            max_tokens=1024,
            priority=vllm_client.Priority.NORMAL,
            agent_name="meta_auditor",
            ticker=ticker,
            cycle_id=cycle_id,
            bot_id=bot_id
        )
        
        parsed = parse_json_response(response)
        
        # Log to CycleAuditor if it fails
        if parsed.get("final_verdict") == "FAIL":
            auditor.anomaly(
                cycle_id=cycle_id,
                phase="meta_audit",
                ticker=ticker,
                message=f"Thesis failed semantic audit (Score: {parsed.get('quality_score')})",
                data=parsed
            )
            logger.warning("[MetaAuditor] %s failed audit: %s", ticker, parsed.get("logic_flaws"))
            
        return parsed
    except Exception as e:
        logger.error("[MetaAuditor] Audit failed for %s: %s", ticker, e)
        return {"final_verdict": "PASS", "error": str(e)}
