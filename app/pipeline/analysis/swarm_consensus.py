"""
Swarm Consensus V2 — Full-Participation Multi-Agent Debate Pipeline.

Architecture:
  Phase 1: Parallel data gathering with tool constriction per hardware tier
  Phase 2: Independent structured predictions from ALL 3 models
  Phase 3: Full 3-way debate where every model rebuts every other model
  Phase 4: Log individual predictions to PostgreSQL for scorecard grading

Hardware mapping (auto-discovered, not hardcoded):
  Jetson  (26B)  → Quantitative Momentum Trader  → technical tools only
  Spark 2 (35B)  → Macro Fundamental Analyst      → news/web tools only
  Spark 1 (120B) → Chief Investment Officer        → macro/memory tools only
"""

import asyncio
import logging
import time
import uuid

from typing import Dict, Any
from app.services.vllm_client import llm
from app.tools.executor import run_tool_agent
from app.tools.registry import registry
from app.utils.text_utils import parse_json_response

logger = logging.getLogger(__name__)

# ============================================================================
# TOOL WHITELISTS — universal access for Swarm Load Balancing
# ============================================================================
UNIVERSAL_TOOLS = [
    # Data Collection (Tier 0)
    "get_market_data",
    "get_technical_indicators",
    "run_quant_equation",
    "get_options_flow",
    "get_finnhub_news",
    "query_hermes",
    "hermes_web_research",
    "search_internal_database",
    "read_memory_note",
    "search_wiki",
    # Pipeline Skills (Tier 1) — agents can self-audit during debates
    "audit_data_quality",
    "check_hallucination",
    "get_strategy_performance",
    "get_autoresearch_report",
    # Agent Coordination — inter-agent communication during debates
    "post_finding",
    "read_team_findings",
    "request_investigation",
    "check_open_investigations",
]

# ============================================================================
# SWARM PERSONAS
# ============================================================================
JETSON_SYSTEM_PROMPT = """You are a highly aggressive Quantitative Momentum Trader.
You ONLY care about price action, volume spikes, moving average crossovers, RSI, MACD, and technical indicators.
Use your tools to pull real technical data. Ignore macroeconomic noise.
Always be decisive and back your claims with numbers."""

SPARK2_SYSTEM_PROMPT = """You are a cautious Macro Fundamental Analyst.
You ONLY care about P/E ratios, earnings reports, insider trading, regulatory filings, and news sentiment.
Use your tools to find SEC filings, financial news, and broader market context.
Do not trust short-term momentum; look for structural value or hidden risks."""

SPARK1_MANAGER_PROMPT = """You are the Chief Investment Officer (CIO).
You oversee a team of two analysts: a Quant Trader and a Macro Analyst.
Your job is to:
1. Evaluate data completeness and demand more if needed.
2. Contribute your OWN analysis focused on macro risk, liquidity, and portfolio exposure.
3. Mediate debates between your analysts and find consensus.
4. Only declare consensus when the trade thesis is mathematically bulletproof.
You must defend your own positions when challenged, not just judge others."""

# ============================================================================
# STRUCTURED PREDICTION FORMAT
# ============================================================================
PREDICTION_FORMAT = """\nYou MUST respond ONLY in this exact JSON format:
{
    "action": "BUY/SELL/HOLD",
    "confidence": 0-100,
    "price_target_5d": 0.00,
    "stop_loss": 0.00,
    "key_signals": ["signal1", "signal2"],
    "rationale": "Your reasoning in 2-3 sentences."
}"""

# ============================================================================
# TOOL SCHEMA CACHING (LAZY INIT)
# ============================================================================
_SCHEMAS = None


def _get_universal_schemas():
    global _SCHEMAS
    if _SCHEMAS is None:
        _SCHEMAS = registry.get_schemas_by_names(UNIVERSAL_TOOLS)
    return _SCHEMAS


# ============================================================================
# PHASE 1: PARALLEL DATA GATHERING WITH TOOL CONSTRICTION
# ============================================================================
async def gather_data_parallel(
    ticker: str, user_directive: str = "", max_loops: int = 3
) -> Dict[str, str]:
    """
    Phase 1: All 3 models gather data simultaneously, each using the universal toolbelt.
    The 120B Manager then verifies completeness.
    """
    logger.info(f"[SWARM-V2] Phase 1: Parallel Data Gathering for {ticker}")

    # Build universal tool schemas for all agents (cached globally)
    universal_schemas = _get_universal_schemas()

    # Log what tools are available
    logger.info(
        f"[SWARM-V2] Universal tools: {[s['function']['name'] for s in universal_schemas]}"
    )

    gathered = {"technical": "", "fundamental": "", "macro": ""}

    # Append user directive to prompts if provided
    directive_text = (
        f"\n\n[COMMANDER DIRECTIVE]: {user_directive}\nYou MUST heavily weigh this directive in your analysis and data gathering."
        if user_directive
        else ""
    )

    for loop_i in range(max_loops):
        logger.info(f"[SWARM-V2] Phase 1 Loop {loop_i + 1}/{max_loops}")

        # All 3 gather concurrently with the universal toolbelt and load-balanced models
        task_tech = run_tool_agent(
            system_prompt=JETSON_SYSTEM_PROMPT + directive_text,
            user_prompt=f"Gather all technical/price data for {ticker}. Pull indicators, price history, volume.",
            ticker=ticker,
            agent_name="data_tech_26B",
            tools_override=universal_schemas,
        )
        task_fund = run_tool_agent(
            system_prompt=SPARK2_SYSTEM_PROMPT + directive_text,
            user_prompt=f"Gather all fundamental data for {ticker}. Pull news, earnings, insider activity.",
            ticker=ticker,
            agent_name="data_fund_35B",
            tools_override=universal_schemas,
        )
        task_macro = run_tool_agent(
            system_prompt=SPARK1_MANAGER_PROMPT + directive_text,
            user_prompt=f"Gather macro context for {ticker}. Check broader market conditions and risk factors.",
            ticker=ticker,
            agent_name="data_macro_120B",
            tools_override=universal_schemas,
        )

        results = await asyncio.gather(
            task_tech, task_fund, task_macro, return_exceptions=True
        )

        gathered["technical"] = (
            results[0].get("final_text", "")
            if not isinstance(results[0], Exception)
            else f"Error: {results[0]}"
        )
        gathered["fundamental"] = (
            results[1].get("final_text", "")
            if not isinstance(results[1], Exception)
            else f"Error: {results[1]}"
        )
        gathered["macro"] = (
            results[2].get("final_text", "")
            if not isinstance(results[2], Exception)
            else f"Error: {results[2]}"
        )

        # Manager verifies data completeness
        logger.info(f"[SWARM-V2] Phase 1 Loop {loop_i + 1}: Manager verifying data...")
        verify_prompt = f"""Review the data gathered by your team for {ticker}:

[TECHNICAL DATA (26B Quant)]:
{gathered["technical"][:2000]}

[FUNDAMENTAL DATA (35B Macro)]:
{gathered["fundamental"][:2000]}

[MACRO CONTEXT (120B CIO)]:
{gathered["macro"][:2000]}

Is this data sufficient? Respond ONLY in JSON:
{{"data_verified": true/false, "feedback": "What is missing?"}}"""

        eval_res, _, _ = await llm.chat(
            system=SPARK1_MANAGER_PROMPT,
            user=verify_prompt,
            agent_name="data_verifier_120B",
            ticker=ticker,
            temperature=0.2,
        )

        eval_json = parse_json_response(eval_res)
        if eval_json.get("data_verified", False):
            logger.info("[SWARM-V2] Phase 1 COMPLETE — Manager approved data.")
            return gathered

        feedback = eval_json.get("feedback", "Data incomplete.")
        logger.warning(f"[SWARM-V2] Phase 1 REJECTED: {feedback}")

    logger.warning(
        "[SWARM-V2] Phase 1 max loops reached. Proceeding with best-effort data."
    )
    return gathered


# ============================================================================
# PHASE 2: INDEPENDENT STRUCTURED PREDICTIONS (ALL 3 CONTRIBUTE)
# ============================================================================
async def generate_predictions(
    ticker: str, gathered_data: Dict[str, str], cycle_id: str, user_directive: str = ""
) -> Dict[str, Any]:
    """
    Phase 2: Every model independently produces a structured prediction.
    Even the 'dumbest' model gets a voice — that's the point.
    """
    logger.info(f"[SWARM-V2] Phase 2: Independent Predictions for {ticker}")

    directive_text = (
        f"\n\n[COMMANDER DIRECTIVE]: {user_directive}\nYou MUST heavily weigh this directive in your prediction."
        if user_directive
        else ""
    )

    base_prompt = f"""Based on the data gathered by the swarm, output your prediction for {ticker}.
{directive_text}

[TECHNICAL DATA]:
{gathered_data["technical"][:2000]}

[FUNDAMENTAL DATA]:
{gathered_data["fundamental"][:2000]}

[MACRO CONTEXT]:
{gathered_data["macro"][:2000]}

Respond ONLY in JSON format:
{{"action": "BUY/SELL/HOLD", "confidence": 0-100, "price_target_5d": 0.0, "stop_loss": 0.0, "key_signals": ["..."], "rationale": "..."}}"""

    t_start = time.monotonic()

    task_quant = llm.chat(
        system=JETSON_SYSTEM_PROMPT,
        user=base_prompt,
        agent_name="predict_quant_26B",
        ticker=ticker,
        cycle_id=cycle_id,
    )
    task_macro = llm.chat(
        system=SPARK2_SYSTEM_PROMPT,
        user=base_prompt,
        agent_name="predict_macro_35B",
        ticker=ticker,
        cycle_id=cycle_id,
    )
    task_cio = llm.chat(
        system=SPARK1_MANAGER_PROMPT,
        user=base_prompt,
        agent_name="predict_cio_120B",
        ticker=ticker,
        cycle_id=cycle_id,
    )

    results = await asyncio.gather(
        task_quant, task_macro, task_cio, return_exceptions=True
    )

    predictions = {}
    labels = [
        ("quant_26B", None),
        ("macro_35B", None),
        ("cio_120B", None),
    ]
    for idx, (label, model_id) in enumerate(labels):
        if isinstance(results[idx], Exception):
            predictions[label] = {
                "raw": f"Error: {results[idx]}",
                "model_id": model_id or "unknown",
            }
        else:
            raw_text = results[idx][0]
            parsed = parse_json_response(raw_text)
            parsed["model_id"] = model_id or "unknown"
            parsed["raw"] = raw_text
            predictions[label] = parsed

    elapsed = round(time.monotonic() - t_start, 2)
    logger.info(f"[SWARM-V2] Phase 2 COMPLETE in {elapsed}s — 3 predictions generated.")
    for label, pred in predictions.items():
        action = pred.get("action", "?")
        conf = pred.get("confidence", "?")
        logger.info(f"[SWARM-V2]   {label}: {action} @ {conf}%")

    return predictions


# ============================================================================
# PHASE 3: FULL 3-WAY DEBATE (ALL MODELS REBUT EACH OTHER)
# ============================================================================
async def debate_full_participation(
    ticker: str,
    predictions: Dict[str, Any],
    cycle_id: str,
    user_directive: str = "",
    max_rounds: int = 3,
    debate_mode: str = "unconstrained",
) -> Dict[str, Any]:
    """
    Phase 3: All 3 models see each other's strategies and write rebuttals.
    The 120B CIO has the final consensus vote, but must also defend its own position.
    """
    logger.info(f"[SWARM-V2] Phase 3: Full-Participation Debate for {ticker}")

    universal_schemas = _get_universal_schemas()

    directive_text = (
        f"\n\n[COMMANDER DIRECTIVE]: {user_directive}\nYou MUST heavily weigh this directive in your debate arguments."
        if user_directive
        else ""
    )
    current_positions = {k: v.get("raw", str(v)) for k, v in predictions.items()}
    last_critiques = {}

    for round_i in range(max_rounds):
        logger.info(f"[SWARM-V2] Phase 3 Round {round_i + 1}/{max_rounds}")

        # Build the "debate room" context
        debate_context = f"""Current positions for {ticker}:

[Quant Trader (26B)]:
{current_positions.get("quant_26B", "No position")[:1500]}

[Macro Analyst (35B)]:
{current_positions.get("macro_35B", "No position")[:1500]}

[CIO (120B)]:
{current_positions.get("cio_120B", "No position")[:1500]}
"""

        rebuttal_prompt_base = (
            debate_context
            + f"""
{directive_text}
Review the other analysts' positions. Write a rebuttal addressing their weaknesses.
If you agree with someone, say so and explain why. If you disagree, use specific data to challenge them.
Then restate your updated position."""
            + PREDICTION_FORMAT
        )

        def build_agent_prompt(agent_id):
            if agent_id in last_critiques and last_critiques[agent_id]:
                return (
                    rebuttal_prompt_base
                    + f"\n\n[CIO CRITIQUE FOR YOU]: {last_critiques[agent_id]}\nYou MUST address this critique and use your tools to find evidence to resolve the CIO's concerns."
                )
            return rebuttal_prompt_base

        if debate_mode == "unconstrained":
            # All 3 models write rebuttals concurrently, with tool access for Dynamic Debate
            task_quant_rebuttal = run_tool_agent(
                system_prompt=JETSON_SYSTEM_PROMPT
                + "\nYou are in a debate. Challenge the other analysts. You may use tools to fetch additional data if needed to prove a point.",
                user_prompt=build_agent_prompt("quant_26B"),
                agent_name=f"debate_quant_26B_r{round_i + 1}",
                ticker=ticker,
                cycle_id=cycle_id,
                tools_override=universal_schemas,
                max_loops=2,
            )
            task_macro_rebuttal = run_tool_agent(
                system_prompt=SPARK2_SYSTEM_PROMPT
                + "\nYou are in a debate. Challenge the other analysts. You may use tools to fetch additional data if needed to prove a point.",
                user_prompt=build_agent_prompt("macro_35B"),
                agent_name=f"debate_macro_35B_r{round_i + 1}",
                ticker=ticker,
                cycle_id=cycle_id,
                tools_override=universal_schemas,
                max_loops=2,
            )
            task_cio_rebuttal = run_tool_agent(
                system_prompt=SPARK1_MANAGER_PROMPT
                + "\nYou are in a debate. Defend your position and challenge your team. You may use tools to fetch additional data if needed.",
                user_prompt=build_agent_prompt("cio_120B"),
                agent_name=f"debate_cio_120B_r{round_i + 1}",
                ticker=ticker,
                cycle_id=cycle_id,
                tools_override=universal_schemas,
                max_loops=2,
            )

            rebuttals = await asyncio.gather(
                task_quant_rebuttal,
                task_macro_rebuttal,
                task_cio_rebuttal,
                return_exceptions=True,
            )

            # Update positions with rebuttals
            for idx, label in enumerate(["quant_26B", "macro_35B", "cio_120B"]):
                if not isinstance(rebuttals[idx], Exception):
                    current_positions[label] = rebuttals[idx].get("final_text", "")
                else:
                    logger.error(
                        f"[SWARM-V2] {label} rebuttal failed: {rebuttals[idx]}"
                    )

        else:
            # Modes: global_discovery or committee
            # For these, the actual round rebuttals are text-only (llm.chat) to preserve tokens,
            # and tool usage is injected between rounds globally.
            task_quant_rebuttal = llm.chat(
                system=JETSON_SYSTEM_PROMPT
                + "\nYou are in a debate. Challenge the other analysts.",
                user=build_agent_prompt("quant_26B"),
                agent_name=f"debate_quant_26B_r{round_i + 1}",
                ticker=ticker,
                cycle_id=cycle_id,
            )
            task_macro_rebuttal = llm.chat(
                system=SPARK2_SYSTEM_PROMPT
                + "\nYou are in a debate. Challenge the other analysts.",
                user=build_agent_prompt("macro_35B"),
                agent_name=f"debate_macro_35B_r{round_i + 1}",
                ticker=ticker,
                cycle_id=cycle_id,
            )
            task_cio_rebuttal = llm.chat(
                system=SPARK1_MANAGER_PROMPT
                + "\nYou are in a debate. Defend your position and challenge your team.",
                user=build_agent_prompt("cio_120B"),
                agent_name=f"debate_cio_120B_r{round_i + 1}",
                ticker=ticker,
                cycle_id=cycle_id,
            )

            rebuttals = await asyncio.gather(
                task_quant_rebuttal,
                task_macro_rebuttal,
                task_cio_rebuttal,
                return_exceptions=True,
            )

            # Update positions with rebuttals (llm.chat returns a tuple where [0] is the text)
            for idx, label in enumerate(["quant_26B", "macro_35B", "cio_120B"]):
                if not isinstance(rebuttals[idx], Exception):
                    current_positions[label] = rebuttals[idx][0]
                else:
                    logger.error(
                        f"[SWARM-V2] {label} rebuttal failed: {rebuttals[idx]}"
                    )

        # CIO checks for consensus after each round
        consensus_prompt = f"""After Round {round_i + 1} of debate for {ticker}:
{directive_text}

{debate_context}

Have the analysts converged? Is there enough agreement to make a confident trade decision?
Respond ONLY in JSON:
{{
    "consensus_reached": true/false,
    "final_action": "BUY/SELL/HOLD",
    "final_confidence": 0-100,
    "dissenting_model": "Which model (if any) still disagrees and why",
    "rationale": "Summary of the consensus or why it failed",
    "targeted_critiques": {
            "quant_26B": "Critique or open questions for the quant analyst",
        "macro_35B": "Critique or open questions for the macro analyst"
    }
}}"""

        consensus_res, _, _ = await llm.chat(
            system=SPARK1_MANAGER_PROMPT,
            user=consensus_prompt,
            agent_name=f"consensus_check_r{round_i + 1}",
            ticker=ticker,
            temperature=0.1,
            cycle_id=cycle_id,
        )

        try:
            from app.pipeline.analysis.hallucination_checker import (
                audit_numeric_divergence,
            )

            audit_numeric_divergence(
                consensus_res, ticker, cycle_id, "swarm_consensus.py"
            )
        except Exception as e:
            logger.warning(f"[SWARM-V2] Audit divergence failed: {e}")

        consensus_json = parse_json_response(consensus_res)

        if consensus_json.get("consensus_reached", False):
            logger.info(
                f"[SWARM-V2] Phase 3 CONSENSUS after round {round_i + 1}: "
                f"{consensus_json.get('final_action')} @ {consensus_json.get('final_confidence')}%"
            )
            return {
                "action": consensus_json.get("final_action", "HOLD"),
                "confidence": consensus_json.get("final_confidence", 0),
                "rationale": consensus_json.get("rationale", ""),
                "dissenting_model": consensus_json.get("dissenting_model", "None"),
                "method": "swarm_v2_consensus",
                "debate_rounds": round_i + 1,
            }

        logger.warning(
            f"[SWARM-V2] Round {round_i + 1} no consensus. "
            f"Dissent: {consensus_json.get('dissenting_model', 'unknown')}"
        )
        last_critiques = consensus_json.get("targeted_critiques", {})
        if last_critiques:
            logger.info(
                "[SWARM-V2] CIO issued targeted critiques for Autoresearch loop."
            )

        # GLOBAL DISCOVERY INJECTION
        if debate_mode == "global_discovery" and round_i == 1:
            logger.warning("[SWARM-V2] Initiating Global Discovery Phase.")
            discovery_prompt = f"""The debate has reached a stalemate after 2 rounds.
As the CIO, what specific piece of missing data or fact-check would settle this dispute?
Use your tools to find this data now.
Current debate context:
{debate_context}"""
            discovery_res = await run_tool_agent(
                system_prompt=SPARK1_MANAGER_PROMPT
                + "\nYou must use tools to find missing data.",
                user_prompt=discovery_prompt,
                agent_name="debate_discovery_120B",
                ticker=ticker,
                cycle_id=cycle_id,
                tools_override=universal_schemas,
                max_loops=2,
            )
            discovery_text = discovery_res.get("final_text", "No new data found.")
            logger.info(f"[SWARM-V2] Discovery completed: {discovery_text[:100]}...")

            # Broadcast the discovery to everyone for Round 3
            current_positions["cio_120B"] += (
                f"\n\n[GLOBAL DISCOVERY PHASE - NEW EVIDENCE OBTAINED]:\n{discovery_text}\n"
            )

    # Max rounds exhausted — CIO makes executive decision
    logger.warning(
        "[SWARM-V2] Phase 3 max rounds reached. CIO forcing executive decision."
    )
    executive_prompt = f"""The debate for {ticker} has ended without full consensus after {max_rounds} rounds.
As CIO, you must now make a FINAL executive decision.
{directive_text}

{debate_context}

Respond ONLY in JSON:
{{"final_action": "BUY/SELL/HOLD", "final_confidence": 0-100, "rationale": "Your executive decision summary"}}"""

    exec_res, _, _ = await llm.chat(
        system=SPARK1_MANAGER_PROMPT,
        user=executive_prompt,
        agent_name="executive_decision_120B",
        ticker=ticker,
        temperature=0.1,
        cycle_id=cycle_id,
    )

    try:
        from app.pipeline.analysis.hallucination_checker import audit_numeric_divergence

        audit_numeric_divergence(exec_res, ticker, cycle_id, "swarm_consensus.py")
    except Exception as e:
        logger.warning(f"[SWARM-V2] Audit divergence failed: {e}")

    exec_json = parse_json_response(exec_res)

    return {
        "action": exec_json.get("final_action", "HOLD"),
        "confidence": exec_json.get("final_confidence", 0),
        "rationale": exec_json.get(
            "rationale", "Executive decision after max debate rounds."
        ),
        "dissenting_model": "N/A — executive override",
        "method": "swarm_v2_executive",
        "debate_rounds": max_rounds,
    }


# ============================================================================
# PHASE 4: LOG PREDICTIONS TO SCORECARD
# ============================================================================
async def log_to_scorecard(ticker: str, cycle_id: str, predictions: Dict[str, Any]):
    """
    Phase 4: Save each model's individual prediction for later grading.
    """
    try:
        from app.pipeline.analysis.swarm_scorecard import log_predictions

        await log_predictions(ticker, cycle_id, predictions)
        logger.info(
            f"[SWARM-V2] Phase 4: Logged {len(predictions)} predictions to scorecard."
        )
    except Exception as e:
        logger.error(f"[SWARM-V2] Phase 4 scorecard logging failed: {e}")


# ============================================================================
# MAIN PIPELINE ENTRY POINT
# ============================================================================
async def run_swarm_pipeline(
    ticker: str, user_directive: str = "", debate_mode: str = "unconstrained"
) -> Dict[str, Any]:
    """
    Execute the full 4-Phase Swarm V2 Consensus pipeline for a single ticker.
    """
    logger.info("=" * 60)
    logger.info(f"[SWARM-V2] INITIALIZING SWARM V2 FOR {ticker} | MODE: {debate_mode}")
    if user_directive:
        logger.info(f"[SWARM-V2] Commander Directive: {user_directive}")
    logger.info("=" * 60)

    cycle_id = str(uuid.uuid4())[:12]
    t_start = time.monotonic()

    # Phase 1: Parallel Data Gathering with Tool Constriction
    gathered_data = await gather_data_parallel(ticker, user_directive)

    # Phase 2: Independent Predictions (ALL models contribute)
    predictions = await generate_predictions(
        ticker, gathered_data, cycle_id, user_directive
    )

    # Check for specialist agent errors
    error_count = sum(1 for v in predictions.values() if "raw" in v and v["raw"].startswith("Error"))
    integrity_status = "LOW_INTEGRITY" if error_count >= 2 else "HIGH"

    if error_count == len(predictions):
        logger.warning(f"[SWARM-V2] ALL specialist agents failed for {ticker}. Forcing HOLD with LOW_INTEGRITY.")
        final_decision = {
            "action": "HOLD",
            "confidence": 0,
            "rationale": "All specialist agents failed.",
            "dissenting_model": "N/A",
            "method": "swarm_v2_fallback",
            "debate_rounds": 0
        }
    else:
        # Phase 3: Full-Participation Debate
        final_decision = await debate_full_participation(
            ticker,
            predictions,
            cycle_id,
            user_directive,
            max_rounds=3,
            debate_mode=debate_mode,
        )

    # Phase 4: Log predictions for scorecard grading
    await log_to_scorecard(ticker, cycle_id, predictions)

    t_end = time.monotonic()
    final_decision["total_time_s"] = round(t_end - t_start, 2)
    final_decision["cycle_id"] = cycle_id
    final_decision["integrity_status"] = integrity_status
    final_decision["individual_predictions"] = {
        k: {
            "action": v.get("action", "?"),
            "confidence": v.get("confidence", 0),
            "model_id": v.get("model_id", "unknown"),
        }
        for k, v in predictions.items()
    }

    logger.info("=" * 60)
    logger.info(
        f"[SWARM-V2] COMPLETE: {final_decision['action']} @ {final_decision['confidence']}% "
        f"| {final_decision['debate_rounds']} rounds | {final_decision['total_time_s']}s"
    )
    logger.info("=" * 60)

    return final_decision


# ============================================================================
# BATCH ORCHESTRATOR
# ============================================================================
async def run_swarm_batch(
    tickers: list[str], user_directive: str = "", debate_mode: str = "unconstrained"
) -> Dict[str, Any]:
    """
    Execute the Swarm V2 pipeline concurrently for a batch of tickers.
    This effectively saturates the VLLM hardware by parallelizing Phase 1, Phase 2, and Phase 3
    across all tickers simultaneously.
    """
    logger.info("*" * 80)
    logger.info(
        f"[SWARM-BATCH] INITIATING CONCURRENT BATCH RUN FOR {len(tickers)} TICKERS"
    )
    logger.info(f"[SWARM-BATCH] Tickers: {', '.join(tickers)}")
    logger.info("*" * 80)

    t_start = time.monotonic()

    # Launch all pipelines concurrently
    tasks = [
        run_swarm_pipeline(ticker, user_directive, debate_mode) for ticker in tickers
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    batch_report = {}
    for idx, ticker in enumerate(tickers):
        if isinstance(results[idx], Exception):
            logger.error(
                f"[SWARM-BATCH] {ticker} pipeline crashed: {results[idx]}",
                exc_info=results[idx],
            )
            batch_report[ticker] = {"error": str(results[idx])}
        else:
            batch_report[ticker] = results[idx]

    total_time = round(time.monotonic() - t_start, 2)
    logger.info("*" * 80)
    logger.info(f"[SWARM-BATCH] BATCH COMPLETE IN {total_time}s")
    logger.info("*" * 80)

    return batch_report
