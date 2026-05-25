import asyncio
import logging
import os
import sys
import httpx

# Ensure local imports work
local_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(local_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

SHARED_CODE = os.environ.get(
    "SHARED_CODEBASE_PATH",
    os.path.join(parent_dir, "..", "trading-client"),
)
if os.path.isdir(SHARED_CODE) and SHARED_CODE not in sys.path:
    sys.path.append(SHARED_CODE)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_tmo_thesis")

async def main():
    from app.services.boot_service import BootService
    from app.cognition.evidence.packet_builder import build_evidence_packet
    from app.cognition.debate.thesis_agent import generate_thesis

    # 1. Boot the application services
    logger.info("Booting application services...")
    await BootService.startup()

    # 2. Build evidence packet for TMO
    ticker = "TMO"
    cycle_id = "test-cycle-12345"
    bot_id = "lazy-trader-v4"
    logger.info(f"Building evidence packet for {ticker}...")
    packet = await build_evidence_packet(ticker)

    logger.info(f"Evidence packet built: {len(packet.claims)} claims, {len(packet.structured_facts)} facts")

    # 3. Generate thesis
    logger.info(f"Generating thesis for {ticker}...")
    from app.services.prism_agent_caller import call_prism_agent
    from app.config.config_cognition import LLM_TEMPERATURES
    from app.services.vllm_client import Priority
    from app.utils.text_utils import parse_json_response, strip_think_tags

    claims_text = "\n".join(
        [
            f"- [{c.provenance.source_table}] {c.subject_entity_id} {c.predicate} {c.object_value} (conf: {c.confidence:.2f})"
            for c in packet.claims[:20]
        ]
    )
    if not claims_text:
        claims_text = "No explicit claims available."

    meta = []
    if packet.contradictions:
        meta.append("Contradictions: " + "; ".join([c.description for c in packet.contradictions]))
    if packet.missing_fields:
        meta.append("Missing Critical Data: " + "; ".join(packet.missing_fields))
    context_meta = "\n".join(meta) if meta else "No known contradictions or missing data."

    from app.cognition.debate.thesis_agent import SYSTEM_PROMPT_TEMPLATE, SYNTHESIS_SYSTEM_PROMPT
    from app.db.constitution import format_constitution_for_prompt
    from app.cognition.debate.action_gate import get_allowed_actions_str

    allowed_actions = get_allowed_actions_str(False)
    active_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        constitution_rules=format_constitution_for_prompt(),
        allowed_actions=allowed_actions,
    )

    user_prompt = """## Entity: TMO
## Bias: neutral

## Structured Facts:
{}

## Available Claims from Evidence:
{}

## Missing Data:
{}

Construct your case based ONLY on the data above. Cite specific values with [source:value] format.""".format(
        str(packet.structured_facts or {}),
        claims_text,
        context_meta
    )

    try:
        from app.services.vllm_client import llm
        model = llm._resolve_model("thesis_agent")
        provider = llm.resolve_provider_for_model(model)
        
        payload, url, headers = llm.prism_client.get_chat_payload_and_url(
            model=model,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=768,
            temperature=0.5,
            system_prompt=active_prompt,
            agent_name="CUSTOM_BULLISH_DEBATER",
            ticker=ticker,
            cycle_id=cycle_id,
            enable_thinking=False,
            tools=None,
            agentic_mode=False,
            provider=provider,
        )
        payload["autoApprove"] = True
        payload["skipConversation"] = False
        
        logger.info(f"Posting directly to Prism: {url}")
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(url, json=payload, headers=headers)
            logger.info(f"Prism Status: {r.status_code}")
            response_json = r.json()
            import json
            logger.info(f"Raw Prism Response JSON:\n{json.dumps(response_json, indent=2)}")
    except Exception as e:
        logger.error(f"Failed to generate thesis: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
