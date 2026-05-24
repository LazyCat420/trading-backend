"""
Post-Cycle Learning -- Extract reusable lessons from trading decisions.

Runs after each ticker's decision in decision_engine.py.
Uses a single vLLM call to extract a lesson, then writes to TradingMemory.

When to learn (not everything is worth learning):
  - High-conviction decisions (confidence >= 85) -- worth remembering the pattern
  - Low-conviction decisions (confidence <= 35) -- worth remembering the confusion
  - Escalated decisions (Config C -> D debate) -- worth remembering the conflict
  - Everything in between is skipped (noise)
"""

import logging

logger = logging.getLogger(__name__)

LEARN_SYSTEM = """You extract ONE concise trading lesson from an analysis result.
Rules:
- Max 400 chars
- Start with [TICKER] [TIER] (e.g., [NVDA] [HIGH] or [AAPL] [LOW])
- Include the specific indicator values that drove the decision
- If nothing is worth remembering, respond with exactly: SKIP
Examples of good lessons:
  [NVDA] [HIGH] RSI < 35 + institutional buying > $2B = strong BUY signal (worked 3x)
  [XOM] [LOW] Oil z-score > 1.5 preceded 2 consecutive earnings beats
  [AAPL] [MEDIUM] Congressional sells within 30d of earnings = reliable warning"""


async def maybe_learn(
    ticker: str,
    result: dict,
    escalated: bool = False,
) -> None:
    """Extract and store a lesson if the decision was noteworthy.

    Safe to call on every ticker -- does nothing for mid-range confidence.
    Wrapped in try/except so failures never affect the main pipeline.
    """
    try:
        conf = result.get("confidence", 50)
        tier = "HIGH" if conf >= 70 else "MEDIUM" if conf >= 40 else "LOW"

        action = result.get("action", "HOLD")
        rationale = result.get("rationale", "")

        # Truncate rationale to keep prompt small
        if len(rationale) > 500:
            rationale = rationale[:500] + "..."

        from app.services.vllm_client import llm, Priority
        from app.services.prism_agent_caller import call_prism_agent

        prompt = (
            f"Ticker: {ticker}\n"
            f"Action: {action}\n"
            f"Confidence: {conf}\n"
            f"Rationale: {rationale}\n"
            f"Escalated: {escalated}\n\n"
            f"Extract ONE reusable lesson or respond SKIP."
        )

        response, tokens, elapsed = await call_prism_agent(
            agent_id="CUSTOM_POST_CYCLE_LEARNER_AGENT",
            user_message=prompt,
            fallback_system_prompt=LEARN_SYSTEM,
            fallback_agent_name="post_cycle_learner",
            temperature=0.0,
            max_tokens=450,
            ticker=ticker,
            priority=Priority.LOW,
        )

        cleaned = response.strip()
        if "SKIP" in cleaned.upper() or len(cleaned) < 10:
            logger.debug("[LEARN] %s: LLM returned SKIP", ticker)
            return

        # Truncate to 400 chars
        lesson = cleaned[:400]

        from app.cognition.trading_memory import trading_memory

        add_result = trading_memory.add("market", lesson)
        if not add_result.get("success"):
            logger.warning(
                "[LEARN] Memory full, could not store lesson for %s: %s",
                ticker,
                add_result.get("message"),
            )

            # Try consolidation if memory is full
            consolidate_result = await trading_memory.consolidate("market")
            if consolidate_result.get("success"):
                # Retry after consolidation
                retry = trading_memory.add("market", lesson)
                if retry.get("success"):
                    logger.info(
                        "[PIPELINE] [LEARN] Stored after consolidation: %s", lesson[:80]
                    )
        else:
            logger.info(
                "[LEARN] %s: Stored lesson (%d tokens): %s",
                ticker,
                tokens,
                lesson[:80],
            )

        # ── Dual-write to lesson_store (same store evolve.py reads) ──
        try:
            from app.cognition.lesson_store import add_lesson as _add_evolution_lesson
            from datetime import datetime, timezone

            _add_evolution_lesson(
                text=lesson,
                metadata={
                    "session_id": f"live_{datetime.now(timezone.utc).strftime('%b%d').lower()}",
                    "round": 0,
                    "score": conf,
                    "status": action,
                    "source": "live_trade",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
        except Exception as ls_err:
            logger.debug(
                "[LEARN] lesson_store dual-write failed (non-fatal): %s", ls_err
            )

        # ── Living Graph: dual-write lesson as Claim node ──
        # Fire-and-forget — graph failure never affects pipeline
        try:
            from app.cognition.ontology.graph_mutations import create_claim

            create_claim(
                ticker=ticker,
                text=lesson,
                cycle_id=result.get("timestamp", "unknown"),
                confidence=conf / 100.0,
            )
        except Exception as graph_err:
            logger.debug("[LEARN] Graph claim write failed (non-fatal): %s", graph_err)

    except Exception as e:
        # Never crash the pipeline over a learning failure
        logger.warning("[PIPELINE] [LEARN] Failed for %s (non-fatal): %s", ticker, e)
