import asyncio
import time
import logging
import json

from app.config.config_tickers import ALT_ASSET_TICKERS
from app.config.config_tickers import classify_asset as _classify_asset
from app.agents.base_agent import run_agent

logger = logging.getLogger(__name__)

# Sequential timeout (seconds). Prevents one slow agent from blocking the entire sequence.
AGENT_TIMEOUT_SECONDS = 90.0


async def _run_with_timeout(
    coro, agent_name: str, timeout: float = AGENT_TIMEOUT_SECONDS
):
    """Run a coroutine with a per-agent timeout.

    Returns the result on success, or a TimeoutError/Exception on failure.
    Never raises — always returns a value so asyncio.gather can continue.
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning("[AGENTS] %s TIMED OUT after %.0fs", agent_name, timeout)
        return TimeoutError(f"{agent_name} timed out after {timeout}s")
    except asyncio.CancelledError:
        logger.warning("[AGENTS] %s was CANCELLED — treating as failure", agent_name)
        return RuntimeError(f"{agent_name} was cancelled")
    except Exception as e:
        logger.warning("[AGENTS] %s CRASHED: %s", agent_name, e)
        return e


def _get_resp(r) -> str:
    if isinstance(r, dict):
        resp = r.get("response", "")
        if isinstance(resp, (dict, list)):
            return json.dumps(resp)
        return str(resp)
    return str(r)


class PipelineAbortError(Exception):
    """Raised when a critical upstream specialist agent fails and recovery is impossible."""
    pass

async def run_specialist_agents(
    ticker: str,
    cycle_id: str,
    bot_id: str,
) -> dict:
    """Run specialist agents in sequential role-based order with capsule compression.

    Each agent's output is compressed into an AgentCapsule (~150 tokens)
    before being passed to the next agent. Full raw data is stored in the
    cycle_context DB table and accessible via get_cycle_context tool.

    For crypto/commodity tickers: skips fundamental agent (no P/E ratios).
    Comparative agent runs for all tickers to provide peer context.

    Returns dict with keys: agent results + '_capsules' list for downstream use.
    """
    from app.agents.context_compressor import generate_capsule, write_capsule_to_db
    from app.agents.capsule import format_capsule_stack
    from app.agents.base_agent import run_agent

    is_alt = ticker.upper() in ALT_ASSET_TICKERS
    asset_type = _classify_asset(ticker)
    # Role-Based Orchestration (Planner -> Retriever -> Verifier -> Synthesizer)
    agent_count = 4
    logger.info(
        "[AGENTS] Running %d sequential role-based agents for %s%s (with capsule compression)...",
        agent_count,
        ticker,
        f" ({asset_type})" if is_alt else "",
    )
    start = time.monotonic()
    
    results = {}
    capsules = []
    
    from app.agents.planner_agent import run_planner
    from app.agents.retriever_agent import run_retriever
    from app.agents.verifier_agent import run_verifier

    # Fetch ontology context once — shared by planner as baseline knowledge
    ontology_context = ""
    try:
        from app.graph.graph_queries import build_relationship_map
        rel_map = await build_relationship_map(ticker)
        ontology_context = rel_map.get("ontology_context", "") if isinstance(rel_map, dict) else ""
    except Exception as ont_err:
        logger.warning("[AGENTS] Failed to load ontology for planner: %s", ont_err)
    
    try:
        # 1. Planner
        planner = await _run_with_timeout(
            run_planner(ticker, cycle_id, bot_id, ontology_context),
            "planner"
        )
        if isinstance(planner, Exception):
            planner_rec = await _try_recover_agent("planner", planner, ticker, cycle_id, bot_id)
            if planner_rec.get("response", "").startswith("Agent failed:"):
                raise PipelineAbortError(f"Planner failed: {planner}")
            planner = planner_rec
        results["planner"] = planner
        
        planner_capsule = await generate_capsule(planner, "planner", cycle_id, ticker)
        if planner_capsule:
            await write_capsule_to_db(planner_capsule, _get_resp(planner))
            capsules.append(planner_capsule)
        
        # 2. Retriever
        capsule_context = format_capsule_stack(capsules)
        retriever = await _run_with_timeout(
            run_retriever(ticker, cycle_id, bot_id, capsule_context),
            "retriever"
        )
        if isinstance(retriever, Exception):
            retriever_rec = await _try_recover_agent("retriever", retriever, ticker, cycle_id, bot_id)
            if retriever_rec.get("response", "").startswith("Agent failed:"):
                raise PipelineAbortError(f"Retriever failed: {retriever}")
            retriever = retriever_rec
        results["retriever"] = retriever
        
        retriever_capsule = await generate_capsule(retriever, "retriever", cycle_id, ticker)
        if retriever_capsule:
            await write_capsule_to_db(retriever_capsule, _get_resp(retriever))
            capsules.append(retriever_capsule)
        
        # 3. Verifier
        capsule_context = format_capsule_stack(capsules)
        verifier = await _run_with_timeout(
            run_verifier(ticker, cycle_id, bot_id, capsule_context),
            "verifier"
        )
        if isinstance(verifier, Exception):
            verifier_rec = await _try_recover_agent("verifier", verifier, ticker, cycle_id, bot_id)
            if verifier_rec.get("response", "").startswith("Agent failed:"):
                raise PipelineAbortError(f"Verifier failed: {verifier}")
            verifier = verifier_rec
        results["verifier"] = verifier
        
        verifier_capsule = await generate_capsule(verifier, "verifier", cycle_id, ticker)
        if verifier_capsule:
            await write_capsule_to_db(verifier_capsule, _get_resp(verifier))
            capsules.append(verifier_capsule)
        
        # 4. Synthesizer
        capsules = [c for c in capsules if c]
        capsule_context = format_capsule_stack(capsules)
        
        if not capsules:
            logger.warning("[AGENTS] No capsules available for Synthesizer, forcing HOLD.")
            synthesizer = {
                "agent": "synthesizer",
                "ticker": ticker,
                "response": json.dumps({"signal": "HOLD", "confidence": 10, "rationale": "Pipeline override: No upstream data capsules available."}),
                "tokens_used": 0,
                "execution_ms": 0
            }
        else:
            synthesizer = await _run_with_timeout(
            run_agent(
                agent_name="synthesizer",
                ticker=ticker,
                cycle_id=cycle_id,
                bot_id=bot_id,
                system_prompt=(
                    "You are the Synthesizer agent. Build the final recommendation from verified evidence. "
                    "MANDATORY: You MUST call get_cycle_context at least once to expand the retriever's "
                    "raw findings before forming your recommendation. Do NOT rely solely on capsule summaries. "
                    "You are ENCOURAGED to dynamically execute quantitative strategies by calling specific "
                    "strategy execution tools (e.g., execute_momentum_strategy, execute_value_strategy) "
                    "to test your hypothesis rather than relying solely on your baked-in logic. "
                    "ANTI-DEGRADATION RULE: You MUST preserve hard quantitative facts (e.g., P/E ratios, specific percentage changes) from the upstream reports. Do not round them away. "
                    "CONTRADICTION RULE: If upstream reports conflict (e.g., one bullish, one bearish), you MUST explicitly surface and explain the contradiction. Do NOT average them out into a neutral summary. "
                    "Respond in JSON with signal, confidence, and rationale fields."
                ),
                user_prompt=f"Synthesize the verified data for {ticker} into a final recommendation:\n{capsule_context}",
                enable_tools=True
            ),
            "synthesizer"
        )
        if isinstance(synthesizer, Exception):
            synth_rec = await _try_recover_agent("synthesizer", synthesizer, ticker, cycle_id, bot_id)
            if synth_rec.get("response", "").startswith("Agent failed:"):
                raise PipelineAbortError(f"Synthesizer failed: {synthesizer}")
            synthesizer = synth_rec
        results["synthesizer"] = synthesizer
        
        synthesizer_capsule = await generate_capsule(synthesizer, "synthesizer", cycle_id, ticker)
        if synthesizer_capsule:
            await write_capsule_to_db(synthesizer_capsule, _get_resp(synthesizer))
            capsules.append(synthesizer_capsule)
        
    except PipelineAbortError:
        raise
    except Exception as seq_err:
        logger.error(f"[AGENTS] Sequential execution failed: {seq_err}")


    # Attach capsules to results for downstream consumption
    results["_capsules"] = capsules

    elapsed = time.monotonic() - start
    total_tokens = sum(
        r.get("tokens_used", 0) for k, r in results.items()
        if k != "_capsules" and isinstance(r, dict)
    )
    capsule_tokens = sum(c.tokens_estimated for c in capsules if c)
    logger.info(
        "[AGENTS] %d agents done in %.1fs (%s tokens, capsule stack ~%d tokens)",
        agent_count,
        elapsed,
        f"{total_tokens:,}",
        capsule_tokens,
    )
    return results


def format_agent_summaries(agent_results: dict) -> str:
    """Format agent results into a structured text block for the RLM context.

    If capsules are available (from run_specialist_agents), uses compressed
    capsule stack (~400-600 tokens) instead of full raw responses (~2-4K tokens).
    Falls back to legacy full-text format if no capsules are present.
    """
    from app.agents.capsule import format_capsule_stack

    # Prefer capsules if available
    capsules = agent_results.get("_capsules")
    if capsules and isinstance(capsules, list) and len(capsules) > 0:
        stack = format_capsule_stack(capsules, max_tokens=600)
        logger.info(
            "[AGENTS] Using capsule stack for RLM (%d capsules, ~%d chars)",
            len(capsules), len(stack),
        )
        return stack

    # Legacy fallback: full raw responses
    sections = []
    for name, result in agent_results.items():
        if name.startswith("_"):
            continue  # Skip internal keys like _capsules
        resp = _get_resp(result)
        tokens = result.get("tokens_used", 0) if isinstance(result, dict) else 0
        sections.append(f"## {name.upper()} AGENT ({tokens} tokens)\n{resp}")
    return "\n\n".join(sections)

async def _try_recover_agent(k: str, v: Exception, ticker: str, cycle_id: str, bot_id: str) -> dict:
    """Attempt fallback recovery for a failed agent."""
    logger.warning("[PIPELINE] [AGENTS] %s FAILED: %s", k, v)

    # ── CORAL Recovery: try to reroute to a fallback agent ──
    fallback_result = None
    try:
        from app.recovery.registry import agent_registry

        fallback_name = agent_registry.find_fallback(f"{k}_agent")
        if fallback_name and fallback_name != f"{k}_agent":
            logger.info(
                "[PIPELINE] [RECOVERY] Rerouting %s → %s for %s",
                k,
                fallback_name,
                ticker,
            )
            # Mark original agent as degraded for this cycle
            agent_registry.mark_degraded(f"{k}_agent")

            # Run the fallback agent (best-effort, don't block on failure)
            try:
                fallback_result = await run_agent(
                    agent_name=f"{fallback_name}_fallback_for_{k}",
                    ticker=ticker,
                    system_prompt=f"You are acting as a fallback for the {k} agent which failed. Provide a conservative {k} analysis.",
                    user_prompt=f"Please provide a conservative {k} analysis for {ticker}.",
                    cycle_id=cycle_id,
                    bot_id=bot_id,
                )
                if fallback_result:
                    fallback_result["fallback_for"] = k
                    fallback_result["fallback_agent"] = fallback_name
                    logger.info(
                        "[PIPELINE] [RECOVERY] Fallback %s succeeded for %s",
                        fallback_name,
                        ticker,
                    )
            except Exception as fb_err:
                logger.warning(
                    "[PIPELINE] [RECOVERY] Fallback %s also failed: %s",
                    fallback_name,
                    fb_err,
                )
    except ImportError:
        pass  # Registry not available yet
    except Exception as reg_err:
        logger.debug(
            "[PIPELINE] [RECOVERY] Registry lookup failed: %s", reg_err
        )

    if fallback_result:
        return fallback_result
    else:
        return {
            "agent": k,
            "ticker": ticker,
            "response": f"Agent failed: {v}",
            "tokens_used": 0,
            "execution_ms": 0,
        }
