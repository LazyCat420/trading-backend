"""
Agent Loop — The core autonomous execution engine for agents.
Supports multi-turn tool dispatch, self-healing, and budget governance.
"""

import asyncio
import logging
import json
from typing import Any

from app.services.vllm_client import llm, Priority
from app.tools.registry import registry
from app.agents.agent_budget import AgentBudget
from app.recovery.engine import RecoveryEngine
from app.recovery.failure_types import (
    RecoveryAction as FTypeRecoveryAction,
    FailureEvent,
    FailureType,
)

logger = logging.getLogger(__name__)

# Initialize a global recovery engine for the loop
recovery_engine = RecoveryEngine()


class AgentYielded(Exception):
    def __init__(self, partial_result: dict):
        self.partial_result = partial_result
        super().__init__(
            f"Agent yielded after {partial_result.get('loops_used', '?')} loops"
        )


class ApprovalRequiredYield(Exception):
    def __init__(self, partial_result: dict):
        self.partial_result = partial_result
        super().__init__(
            "Agent paused: Requires human approval for a destructive action."
        )


class ToolCallScorecard:
    def __init__(self):
        self.made = 0
        self.succeeded = 0
        self.errored = 0
        self.empty = 0
        self.consecutive_empty = 0

    @property
    def quality_ratio(self) -> float:
        if self.made == 0:
            return 1.0
        return self.succeeded / self.made

    def record(self, tool_result_content: str):
        self.made += 1
        content = tool_result_content.strip()

        # Explicit error strings
        if any(k in content.lower() for k in ["error", "exception", "traceback", "failed"]):
            self.errored += 1
            self.consecutive_empty += 1
            return

        # Empty structures
        if content in ("", "[]", "{}", "null", "None", "no data", "no results"):
            self.empty += 1
            self.consecutive_empty += 1
            return

        # Try JSON — flag if data/result keys are empty
        try:
            data = json.loads(content)
            if isinstance(data, list) and len(data) == 0:
                self.empty += 1
                self.consecutive_empty += 1
                return
            if isinstance(data, dict) and not any(data.values()):
                self.empty += 1
                self.consecutive_empty += 1
                return
        except Exception:
            pass  # non-JSON is fine — treat as success

        self.succeeded += 1
        self.consecutive_empty = 0


async def run_agent_loop(
    system_prompt: str,
    user_prompt: str,
    ticker: str,
    agent_name: str,
    cycle_id: str = "",
    bot_id: str = "",
    budget: AgentBudget = None,
    priority: Priority = Priority.NORMAL,
    previous_messages: list = None,
    model_override: str | None = None,
    tools_override: list[dict] | None = None,
    yield_on_limit: bool = False,
    require_json_schema: bool = False,
    critique_rounds: int = 0,
) -> dict[str, Any]:
    """
    Run an LLM agent with multi-turn tool capabilities and self-healing.

    If `require_json_schema` is True, it expects the final non-tool output to be valid JSON.
    If it fails, it uses the RecoveryEngine to attempt a REPAIR.
    """
    if budget is None:
        budget = AgentBudget()

    from app.cognition.evolution.reflector import (
        get_agent_lessons,
        adjust_lesson_score,
        reflect_on_trajectory,
        get_spotlight_tools,
    )

    if not previous_messages:
        firm_context = (
            "CRITICAL CONTEXT: You are an autonomous data processing script working for a quantitative trading firm. "
            "You are NOT a conversational chatbot. Do NOT talk to the user, give advice, ask questions, or converse. "
            "Your ONLY purpose is to extract structured financial data to make profitable trading decisions.\n\n"
        )
        enhanced_system_prompt = firm_context + system_prompt
        lessons = get_agent_lessons(agent_name)
        spotlight = get_spotlight_tools(limit=5)
        
        # Phase 1: Check hold streak to penalize hold bias
        hold_streak = 0
        try:
            from app.db.connection import get_db
            with get_db() as db:
                row = db.execute("SELECT hold_streak FROM ticker_health WHERE ticker = %s", [ticker]).fetchone()
                if row and row[0]:
                    hold_streak = row[0]
        except Exception as e:
            logger.warning(f"[AgentLoop] Failed to fetch hold_streak: {e}")

        if hold_streak >= 3:
            enhanced_system_prompt += (
                f"\n\n### HOLD STREAK NOTICE ({hold_streak} cycles):\n"
                f"This ticker has been HELD for {hold_streak} consecutive cycles. "
                "This suggests either (a) genuine uncertainty that warrants continued monitoring, "
                "or (b) analysis lazily defaulting to HOLD without conviction.\n"
                "REQUIRED: If you conclude HOLD, you MUST provide a specific catalyst or "
                "condition that would change your decision (e.g., 'HOLD until earnings on June 3' "
                "or 'HOLD unless RSI drops below 30'). Generic 'wait and see' is NOT acceptable.\n"
                "If the data clearly supports BUY or SELL, act on it — do not let the hold streak "
                "make you overly cautious."
            )

        if lessons:
            lesson_text = "\n".join([f"- {l}" for l in lessons])
            enhanced_system_prompt += f"\n\n### PAST LESSONS LEARNED (Follow these to maximize success):\n{lesson_text}"
            
        if spotlight:
            spotlight_text = ", ".join(spotlight)
            enhanced_system_prompt += (
                "\n\n### REQUIRED TOOL CHECK:\n"
                f"The following tools have NOT been used recently: [{spotlight_text}].\n"
                "Before writing your final answer, you MUST ask yourself: "
                "Does my current analysis have a gap that one of these tools would fill? "
                "If yes, call it now. If no, briefly state why it's not relevant."
            )
            
        enhanced_system_prompt += "\n\n### WORKING MEMORY RULE:\nAfter every tool call, before taking your next action, you MUST output a compressed structured memory object summarizing the evidence. Include: evidence type, freshness, confidence, contradiction flags, decision relevance, and source reference."
        
        enhanced_system_prompt += "\n\n### TOOL USE RULE:\nBefore calling any tool, you MUST briefly state your rationale for calling it (why you need it) and your planned next action after receiving the data."
            
        messages = [{"role": "system", "content": enhanced_system_prompt}]
    else:
        spotlight = []  # We only have spotlight tools on the first message
        messages = previous_messages.copy()

    if user_prompt:
        messages.append({"role": "user", "content": user_prompt})

    active_tools = tools_override if tools_override is not None else registry.schemas

    # Brain-Action Split: select only needed tools when pool is large.
    # SKIP when tools_override was explicitly provided — the caller (e.g.
    # run_split_agent_loop) already performed selection.  Re-selecting
    # wastes an LLM call and can over-prune the already-filtered set.
    if tools_override is None and active_tools and len(active_tools) > 5:
        try:
            from app.agents.tool_selector import select_tools_for_task
            task_desc = f"{system_prompt[:500]}\n\nTask: {user_prompt[:1500]}"
            active_tools = await select_tools_for_task(
                task_description=task_desc,
                available_tool_schemas=active_tools,
                agent_name=f"{agent_name}_selector",
                ticker=ticker,
                cycle_id=cycle_id,
                priority=priority,
                max_tools=5,
            )
            logger.info(
                "[AgentLoop] Tool selection: %d tools selected for %s → %s",
                len(active_tools),
                agent_name,
                [t["function"]["name"] for t in active_tools],
            )
        except Exception as sel_err:
            logger.warning(
                "[AgentLoop] Tool selection failed for %s, using full pool: %s",
                agent_name, sel_err,
            )

    # Phase 0.5: Tool Optimization (Highlight & Pruning)
    try:
        from app.services.tool_optimizer import optimize_agent_tools
        if messages and messages[0].get("role") == "system":
            opt_tools, opt_prompt = await optimize_agent_tools(agent_name, active_tools, messages[0]["content"])
            active_tools = opt_tools
            messages[0]["content"] = opt_prompt
        else:
            opt_tools, _ = await optimize_agent_tools(agent_name, active_tools, "")
            active_tools = opt_tools
    except Exception as opt_err:
        logger.warning("[AgentLoop] Tool optimization failed for %s: %s", agent_name, opt_err)

    final_content = ""
    hit_limit_with_pending_tools = False
    scorecard = ToolCallScorecard()
    stop_reason = "success"

    from app.agents.context_compressor import compress_history, summarize_tool_result
    from app.config.context_budget import get_context_budget as _get_ctx_budget

    # Aggregate tool result budget: limits TOTAL tool result tokens across
    # all tool calls in this agent run.  Without this, 5 tool calls each
    # at 20% of effective = 100% of context consumed by tool results alone.
    _ctx_budget = _get_ctx_budget()
    _aggregate_tool_budget = _ctx_budget.tool_result_budget * 2  # Allow up to 2x single-tool budget total
    _aggregate_tool_used = 0

    while budget.consume_turn():
        # ── Pipeline stop check ──────────────────────────────────────
        # Ensures the agent loop halts immediately when the user stops
        # the pipeline, rather than continuing until budget exhausts.
        try:
            from app.pipeline.orchestration.cycle_control import cycle_control
            if cycle_control.is_stopped:
                logger.info(
                    "[AgentLoop] Pipeline stopped — aborting %s mid-loop",
                    agent_name,
                )
                raise asyncio.CancelledError("Pipeline stopped during agent loop")
        except ImportError:
            pass

        # Compress history if it gets too large (threshold is model-aware via context_budget)
        messages = await compress_history(messages)

        try:
            result = await llm.chat_with_tools(
                messages=messages,
                tools=active_tools,
                agent_name=agent_name,
                ticker=ticker,
                cycle_id=cycle_id,
                bot_id=bot_id,
                priority=priority,
                max_tokens=2048,
                model_override=model_override,
            )
        except Exception as e:
            logger.error(f"[AgentLoop] chat_with_tools failed: {e}")
            err_msg = str(e) or type(e).__name__
            final_content = f"Error during agent execution: {err_msg}"
            stop_reason = "blocked"
            break

        content = result.get("text", "")
        tool_calls = result.get("tool_calls")
        endpoint_name = result.get("endpoint_name", "")
        model_name = result.get("model_name", "")

        # Consume tokens and check budget exhaustion
        if not budget.consume_tokens(result.get("total_tokens", 0)):
            logger.warning(
                f"[AgentLoop] Budget exhausted for {agent_name}. Terminating."
            )
            hit_limit_with_pending_tools = True if tool_calls else False
            stop_reason = "exhausted"
            break

        # ── Context telemetry ──
        try:
            from app.monitoring.context_telemetry import log_context_usage

            sys_chars = len(messages[0].get("content", "")) if messages else 0
            hist_chars = sum(
                len(m.get("content", ""))
                for m in messages[1:]
                if m.get("role") in ("user", "assistant")
            )
            tool_chars = sum(
                len(m.get("content", ""))
                for m in messages
                if m.get("role") == "tool"
            )
            total_chars = sum(len(m.get("content", "")) for m in messages)

            log_context_usage(
                cycle_id=cycle_id,
                agent_name=agent_name,
                system_prompt_chars=sys_chars,
                history_chars=hist_chars,
                tool_result_chars=tool_chars,
                total_prompt_chars=total_chars,
                notes=f"turn={budget.current_turns}",
            )
        except Exception:
            pass  # Telemetry should never break the loop

        final_content = content

        assistant_msg = {"role": "assistant"}
        if content:
            assistant_msg["content"] = content
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if not tool_calls:
            # Trigger critique phase if rounds remain
            if critique_rounds > 0:
                critique_rounds -= 1
                logger.info(
                    f"[AgentLoop] Agent '{agent_name}' triggered CRITIQUE round. {critique_rounds} remaining."
                )
                critique_msg = (
                    "CRITIQUE PHASE: Review your previous findings. Consider alternative perspectives, missing context, or conflicting data. "
                    "If you need to verify anything or fill gaps, use your tools to dig deeper now. "
                    "If you are 100% confident you have everything required, provide your final summarized conclusion."
                )
                messages.append({"role": "user", "content": critique_msg})
                continue

            # Reached a terminal answer. If JSON schema is required, validate it here.
            if require_json_schema and content:
                from app.utils.text_utils import parse_json_response

                try:
                    parsed = parse_json_response(content)
                except Exception as parse_err:
                    logger.warning(
                        f"[AgentLoop] Agent '{agent_name}' JSON parsing failed: {parse_err}. "
                        "Treating as empty to trigger recovery/repair."
                    )
                    parsed = {}
                if not parsed or (isinstance(parsed, dict) and not parsed):
                    # Attempt self-healing REPAIR
                    fail_event = FailureEvent(
                        failure_type=FailureType.DEGRADED,
                        agent_name=agent_name,
                        step_name="json_parse",
                        ticker=ticker,
                        cycle_id=cycle_id,
                        exception_type="JSONDecodeError",
                        exception_msg="Agent produced invalid JSON.",
                    )
                    recovery_res = recovery_engine.handle(fail_event)

                    if recovery_res.action == FTypeRecoveryAction.REPAIR:
                        logger.warning(
                            f"[AgentLoop] Agent '{agent_name}' produced invalid JSON. RecoveryEngine triggered REPAIR."
                        )
                        repair_msg = (
                            "Your previous output was not valid JSON. "
                            "Please correct your response and ensure it exactly matches the requested JSON format."
                        )
                        messages.append({"role": "user", "content": repair_msg})
                        continue  # Loop back and let the LLM try again on the next turn
                    else:
                        logger.error(
                            "[AgentLoop] RecoveryEngine declined REPAIR for invalid JSON. Yielding."
                        )
                        stop_reason = "invalid_output"
                        break

            logger.info(
                f"[AgentLoop] Agent '{agent_name}' finished successfully after {budget.current_turns} turns."
            )
            break

        # Log tool calls
        for tc in tool_calls:
            logger.info(
                f"[AgentLoop] Turn {budget.current_turns}: {agent_name} requested tool -> {tc.get('function', {}).get('name')}"
            )

        # Execute tool calls
        requires_approval = False
        approval_details = None

        for tc in tool_calls:
            import time
            start_time = time.monotonic()
            
            func_data = tc.get('function')
            if not func_data:
                tc['function'] = {}
                func_data = tc['function']

            tool_name = func_data.get('name')
            tool_args = func_data.get('arguments', '')
            
            # Intercept invalid / None / empty tool names to shield Prism Gateway
            is_invalid_tool = False
            if not tool_name or tool_name == "None" or tool_name.strip() == "":
                is_invalid_tool = True
                func_data['name'] = "unknown_tool"
                tool_name = "unknown_tool"

            if is_invalid_tool:
                tool_res = {
                    "role": "tool",
                    "name": "unknown_tool",
                    "tool_call_id": tc.get("id", "") or "unknown_id",
                    "content": json.dumps({"error": "Invalid tool name provided. Please choose a valid tool from the schemas."})
                }
            else:
                try:
                    tool_res = await asyncio.wait_for(
                        registry.execute_tool_call(
                            tc, agent_name=agent_name, ticker=ticker, cycle_id=cycle_id
                        ),
                        timeout=45.0
                    )
                except asyncio.TimeoutError:
                    logger.error(f"[AgentLoop] Tool execution timed out after 45s for {tool_name}")
                    tool_res = {
                        "role": "tool",
                        "name": tool_name,
                        "tool_call_id": tc.get("id", "") or "unknown_id",
                        "content": json.dumps({"error": f"Tool '{tool_name}' timed out after 45 seconds."})
                    }

            latency_ms = int((time.monotonic() - start_time) * 1000)
            
            messages.append(tool_res)
            scorecard.record(tool_res.get("content", ""))

            # ── Inline tool result compression (with aggregate budget) ──
            # If the tool result is oversized, truncate it NOW before
            # it inflates the context window on the next LLM call.
            # The per-tool budget shrinks as aggregate usage grows.
            raw_content = tool_res.get("content", "")
            _remaining_budget = max(512, _aggregate_tool_budget - _aggregate_tool_used)
            compressed = summarize_tool_result(
                raw_content,
                tool_name=tool_name or "unknown",
                budget_tokens=min(_ctx_budget.tool_result_budget, _remaining_budget),
            )
            _result_tokens = len(compressed) // 4  # Rough token estimate
            _aggregate_tool_used += _result_tokens
            if _aggregate_tool_used > _aggregate_tool_budget:
                logger.info(
                    "[AgentLoop] Aggregate tool budget exceeded (%d/%d tokens) for %s/%s",
                    _aggregate_tool_used, _aggregate_tool_budget, agent_name, ticker,
                )
            if len(compressed) < len(raw_content):
                # Replace the content in the already-appended message
                messages[-1] = {**messages[-1], "content": compressed}
            
            # Extract the rationale from the LLM's text output leading up to the tool call
            rationale = content.strip()[:1000] if content else "No rationale provided"
            
            # Emit trace
            try:
                import uuid
                from app.db.connection import get_db
                
                service_source = tool_res.get("service_source", "trading-service")
                with get_db() as db:
                    db.execute(
                        """INSERT INTO agent_traces 
                           (id, run_id, agent_name, task_type, goal, planned_next_action, 
                            tool_name, tool_args, tool_result_summary, why_tool_was_called, 
                            tokens_before, tokens_after, latency_ms, did_tool_change_decision, 
                            loop_step, stop_reason, endpoint_name, model_name, service_source)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                        [
                            str(uuid.uuid4()), cycle_id, agent_name, "analysis", "execute_task", "evaluate_evidence",
                            tool_name, str(tool_args), tool_res.get("content", "")[:1000], rationale,
                            budget.current_tokens, budget.current_tokens, latency_ms, False, budget.current_turns, stop_reason,
                            endpoint_name, model_name, service_source
                        ]
                    )
            except Exception as e:
                logger.error(f"[AgentLoop] Failed to insert trace: {e}")

            # Check for approval signal
            try:
                res_data = json.loads(tool_res.get("content", "{}"))
                if res_data.get("requires_approval"):
                    requires_approval = True
                    approval_details = {
                        "tool_name": tool_res.get("name"),
                        "command": res_data.get("pending_command"),
                        "reason": res_data.get("error"),
                    }
            except Exception:
                pass

        # Check consecutive empty/error threshold to abort runaway agents
        if scorecard.consecutive_empty >= 3:
            logger.warning(
                f"[AgentLoop] Agent '{agent_name}' reached consecutive empty/error threshold ({scorecard.consecutive_empty}). Aborting loop to protect Jetson model server."
            )
            stop_reason = "error_threshold"
            break

        if requires_approval:
            logger.warning(
                f"[AgentLoop] Tool requires approval. Yielding loop for agent {agent_name}"
            )
            try:
                import uuid
                from app.db.connection import get_db

                with get_db() as db:
                    db.execute(
                        "INSERT INTO pending_approvals (id, agent_name, command, reason, status) VALUES (%s, %s, %s, %s, %s)",
                        [
                            str(uuid.uuid4()),
                            agent_name,
                            approval_details["command"],
                            approval_details["reason"],
                            "pending",
                        ],
                    )
            except Exception as e:
                logger.error(f"[AgentLoop] Failed to write to pending_approvals: {e}")

            base_result = {
                "final_text": "Agent loop paused: Waiting for human approval of a destructive action.",
                "token_usage": budget.current_tokens,
                "execution_ms": 0,
                "chat_history": messages,
                "loops_used": budget.current_turns,
                "yielded": True,
                "cost_usd": budget.current_usd,
                "requires_approval": True,
                "stop_reason": "blocked",
            }
            raise ApprovalRequiredYield(base_result)

    else:
        # Loop exhausted due to budget
        logger.warning(
            f"[AgentLoop] Agent '{agent_name}' hit turn limit with pending tools."
        )
        if budget.is_exhausted():
            logger.warning(
                "[agent_loop] %s hit turn budget on %s — turns_used=%d, tool_calls=%d, quality=%.0f%%",
                agent_name, ticker, budget.current_turns, scorecard.made,
                scorecard.quality_ratio * 100,
            )
        hit_limit_with_pending_tools = True

    base_result = {
        "final_text": final_content,
        "token_usage": budget.current_tokens,
        "execution_ms": 0,  # Time tracking can be added if needed around llm call
        "chat_history": messages,
        "loops_used": budget.current_turns,
        "yielded": hit_limit_with_pending_tools,
        "cost_usd": budget.current_usd,
        "stop_reason": stop_reason,
    }



    success = not hit_limit_with_pending_tools

    # Trigger Autoresearch reflection on success AND failure for organic learning
    # We always reflect to capture what worked (success) or what failed
    _reflect_task = asyncio.create_task(
        reflect_on_trajectory(
            agent_name=agent_name,
            ticker=ticker,
            cycle_id=cycle_id,
            loops_used=budget.current_turns,
            yielded=hit_limit_with_pending_tools,
            chat_history=messages,
            success=success,
            spotlight_tools=spotlight,
        )
    )
    # Prevent "Task exception was never retrieved" warnings
    _reflect_task.add_done_callback(
        lambda t: (
            logger.warning(
                "[AgentLoop] reflect_on_trajectory failed for %s/%s: %s",
                agent_name, ticker, t.exception(),
            )
            if t.exception()
            else None
        )
    )

    # Adjust success score for any recently applied lessons
    adjust_lesson_score(agent_name, success)

    # Record tool optimization stats in a background task
    try:
        from app.services.tool_optimizer import record_tool_optimization_usage
        # Extract used tools
        used_tools = []
        for msg in messages:
            if msg.get("role") == "tool" and msg.get("name"):
                used_tools.append(msg["name"])
            elif "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    name = tc.get("function", {}).get("name")
                    if name:
                        used_tools.append(name)
        asyncio.create_task(
            record_tool_optimization_usage(
                agent_name=agent_name,
                offered_tools=active_tools,
                used_tool_names=used_tools,
            )
        )
    except Exception as rec_err:
        logger.warning("[AgentLoop] Failed to record tool optimization usage: %s", rec_err)


    if hit_limit_with_pending_tools and yield_on_limit:
        raise AgentYielded(base_result)

    try:
        from app.db.connection import get_db
        import uuid

        with get_db() as db:
            db.execute(
                """INSERT INTO agent_loop_stats 
                   (id, cycle_id, agent_name, ticker, loops_used, token_usage, cost_usd, yielded)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                [
                    str(uuid.uuid4()),
                    cycle_id,
                    agent_name,
                    ticker,
                    budget.current_turns,
                    budget.current_tokens,
                    budget.current_usd,
                    hit_limit_with_pending_tools,
                ],
            )
    except Exception as e:
        logger.error(f"[AgentLoop] Failed to save stats: {e}")

    return base_result


async def run_split_agent_loop(
    system_prompt: str,
    user_prompt: str,
    ticker: str,
    agent_name: str,
    cycle_id: str = "",
    bot_id: str = "",
    budget: AgentBudget = None,
    priority: Priority = Priority.NORMAL,
    previous_messages: list = None,
    model_override: str | None = None,
    tools_override: list[dict] | None = None,
    yield_on_limit: bool = False,
    require_json_schema: bool = False,
    critique_rounds: int = 0,
    max_selector_tools: int = 5,
) -> dict[str, Any]:
    """Brain-Action Split Agent Loop.

    Wraps the standard run_agent_loop with a two-phase approach:
      1. SELECTOR PHASE: A lightweight LLM call picks which tools are needed
         from the full pool (no JSON schemas sent — just names/descriptions).
      2. EXECUTOR PHASE: The real agent loop runs with ONLY the selected
         tool schemas, dramatically reducing context window usage.

    Falls back to the standard monolithic loop if selection fails.

    Args:
        Same as run_agent_loop, plus:
        max_selector_tools: Maximum number of tools the selector can pick.

    Returns:
        Same dict as run_agent_loop.
    """
    from app.agents.tool_selector import select_tools_for_task

    # Determine the full tool pool
    if tools_override is not None:
        full_pool = tools_override
    else:
        full_pool = registry.schemas

    # Skip the selector if the pool is already small
    if len(full_pool) <= max_selector_tools:
        logger.debug(
            "[SplitLoop] Pool size %d <= %d, skipping selector for %s",
            len(full_pool),
            max_selector_tools,
            agent_name,
        )
        return await run_agent_loop(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            ticker=ticker,
            agent_name=agent_name,
            cycle_id=cycle_id,
            bot_id=bot_id,
            budget=budget,
            priority=priority,
            previous_messages=previous_messages,
            model_override=model_override,
            tools_override=full_pool if tools_override is not None else None,
            yield_on_limit=yield_on_limit,
            require_json_schema=require_json_schema,
            critique_rounds=critique_rounds,
        )

    # Phase 1: Tool Selection (lightweight, no schemas in context)
    # Build the task description from system + user prompts
    task_desc = f"{system_prompt[:500]}\n\nTask: {user_prompt[:1500]}"

    selected_schemas = await select_tools_for_task(
        task_description=task_desc,
        available_tool_schemas=full_pool,
        agent_name=f"{agent_name}_selector",
        ticker=ticker,
        cycle_id=cycle_id,
        priority=priority,
        max_tools=max_selector_tools,
    )

    selected_names = [s["function"]["name"] for s in selected_schemas]
    full_names = [s["function"]["name"] for s in full_pool]

    logger.info(
        "[SplitLoop] Tool selection complete for %s/%s: %d/%d tools selected → %s",
        agent_name,
        ticker,
        len(selected_schemas),
        len(full_pool),
        selected_names,
    )

    # Phase 2: Run the real agent loop with the reduced tool set
    return await run_agent_loop(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        ticker=ticker,
        agent_name=agent_name,
        cycle_id=cycle_id,
        bot_id=bot_id,
        budget=budget,
        priority=priority,
        previous_messages=previous_messages,
        model_override=model_override,
        tools_override=selected_schemas,
        yield_on_limit=yield_on_limit,
        require_json_schema=require_json_schema,
        critique_rounds=critique_rounds,
    )

