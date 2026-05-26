import os
import sys
import asyncio
import json

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from app.services.vllm_client import llm, Priority
from app.tools.registry import registry
from app.agents.tool_whitelists import get_agent_tools, AGENT_TOOL_WHITELISTS

async def run_experiment():
    print("====================================================")
    print("TOOL SELECTION EFFICIENCY & ACCURACY A/B TEST")
    print("====================================================\n")

    # Ensure roles are discovered
    await llm.discover_roles()
    print(f"Default model in client: {llm.model}")

    # Use the 'retriever' agent's tools as our pool for Scenario A (Control)
    # The retriever whitelist has 16 key tools used for fetching data
    retriever_tool_names = AGENT_TOOL_WHITELISTS.get("retriever", [])
    control_tools = registry.get_schemas_by_names(retriever_tool_names)
    
    print(f"Tool pool count for test: {len(control_tools)} tools.")
    print("Tools in pool:")
    for t in control_tools:
        print(f" - {t['function']['name']}")
    print("")

    # Task to perform
    task = "Verify AAPL stock's recent performance. Retrieve its 14-day RSI (relative strength index) indicator, find its current market price, and check the latest news headlines."
    print(f"Test Task: \"{task}\"\n")

    # Define system prompt for the agent
    system_prompt = (
        "You are a helpful analyst assistant. Your task is to query the necessary data "
        "using your tools to answer the user's request. You must select and call the appropriate "
        "tools in the first turn to gather the evidence."
    )

    # ----------------------------------------------------
    # SCENARIO A: CONTROL (Monolithic Tool Presentation)
    # ----------------------------------------------------
    print("--- Running Scenario A: Control (Monolithic) ---")
    control_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task}
    ]

    t0_control = asyncio.get_event_loop().time()
    try:
        control_res = await llm.chat_with_tools(
            messages=control_messages,
            tools=control_tools,
            agent_name="retriever",
            ticker="AAPL",
            cycle_id="test-efficiency-control",
            priority=Priority.HIGH
        )
        t_elapsed_control = asyncio.get_event_loop().time() - t0_control
        control_tokens = control_res.get("total_tokens", 0)
        control_text = control_res.get("text", "")
        control_calls = control_res.get("tool_calls", [])
        
        print("Scenario A Results:")
        print(f" - Latency: {t_elapsed_control:.2f} seconds")
        print(f" - Total Tokens reported: {control_tokens}")
        print(f" - Tool Calls made: {[c.get('function', {}).get('name') for c in control_calls]}")
    except Exception as e:
        print(f"Scenario A failed: {e}")
        control_tokens = 0
        control_calls = []
        t_elapsed_control = 0

    print("\n----------------------------------------------------")
    # ----------------------------------------------------
    # SCENARIO B: VARIANT (Split-Agent Tool Selection)
    # ----------------------------------------------------
    print("--- Running Scenario B: Variant (Split Agent) ---")
    
    # Step 1: Routing/Selection Agent
    # We pass only names and descriptions as plain text, avoiding verbose JSON schemas
    tools_list_text = "\n".join([
        f"- {t['function']['name']}: {t['function']['description']}" 
        for t in control_tools
    ])
    
    selector_system_prompt = (
        "You are an expert Tool Routing Agent. Given a user request and a text list of available tools, "
        "your job is to select the exact tools (maximum 3) needed to answer the request.\n"
        "You must output a JSON object containing a list of the selected tool names. "
        "Do not explain your reasoning, output ONLY the JSON object. Example format:\n"
        '{"selected_tools": ["tool_a", "tool_b"]}'
    )
    
    selector_user_prompt = f"User Request: {task}\n\nAvailable Tools:\n{tools_list_text}"
    
    selector_messages = [
        {"role": "system", "content": selector_system_prompt},
        {"role": "user", "content": selector_user_prompt}
    ]

    t0_select = asyncio.get_event_loop().time()
    try:
        select_res = await llm.chat_with_tools(
            messages=selector_messages,
            tools=None, # No tools passed in payload!
            agent_name="tool_selector",
            ticker="AAPL",
            cycle_id="test-efficiency-select",
            priority=Priority.HIGH
        )
        t_elapsed_select = asyncio.get_event_loop().time() - t0_select
        select_tokens = select_res.get("total_tokens", 0)
        select_text = select_res.get("text", "").strip()
        
        print("Routing Agent Raw Output:")
        print(select_text)
        
        # Parse selected tools
        import re
        json_match = re.search(r"\{.*\}", select_text, re.DOTALL)
        if json_match:
            parsed_data = json.loads(json_match.group(0))
            selected_tool_names = parsed_data.get("selected_tools", [])
        else:
            selected_tool_names = []
            
        print(f"Selected Tools: {selected_tool_names}")
    except Exception as e:
        print(f"Scenario B Step 1 failed: {e}")
        selected_tool_names = []
        select_tokens = 0
        t_elapsed_select = 0

    # Step 2: Action Agent Execution with Subset of Tools
    if selected_tool_names:
        # Retrieve full JSON schemas ONLY for the selected tools
        variant_tools = registry.get_schemas_by_names(selected_tool_names)
        print(f"\nVariant Tool schemas generated for: {[t['function']['name'] for t in variant_tools]}")
        
        action_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task}
        ]
        
        t0_action = asyncio.get_event_loop().time()
        try:
            action_res = await llm.chat_with_tools(
                messages=action_messages,
                tools=variant_tools, # Only subset of tools!
                agent_name="action_executor",
                ticker="AAPL",
                cycle_id="test-efficiency-action",
                priority=Priority.HIGH
            )
            t_elapsed_action = asyncio.get_event_loop().time() - t0_action
            action_tokens = action_res.get("total_tokens", 0)
            action_calls = action_res.get("tool_calls", [])
            
            print("Action Agent Results:")
            print(f" - Latency: {t_elapsed_action:.2f} seconds")
            print(f" - Total Tokens reported: {action_tokens}")
            print(f" - Tool Calls made: {[c.get('function', {}).get('name') for c in action_calls]}")
        except Exception as e:
            print(f"Scenario B Step 2 failed: {e}")
            action_tokens = 0
            action_calls = []
            t_elapsed_action = 0
    else:
        print("No tools selected, skipping Action Agent execution.")
        action_tokens = 0
        action_calls = []
        t_elapsed_action = 0

    # ----------------------------------------------------
    # FINAL METRIC COMPARISON
    # ----------------------------------------------------
    print("\n====================================================")
    print("FINAL COMPARISON RESULTS")
    print("====================================================")
    
    total_variant_tokens = select_tokens + action_tokens
    total_variant_latency = t_elapsed_select + t_elapsed_action
    
    print(f"{'Metric':<25} | {'Scenario A (Control)':<25} | {'Scenario B (Variant)':<25}")
    print("-" * 80)
    print(f"{'Input/Output Tokens':<25} | {control_tokens:<25} | {total_variant_tokens:<25}")
    print(f"  - Selection step: {'N/A':<23} | {select_tokens:<25}")
    print(f"  - Action step: {'N/A':<23} | {action_tokens:<25}")
    print(f"{'Execution Latency':<25} | {t_elapsed_control:<22.2f}s | {total_variant_latency:<22.2f}s")
    print(f"  - Selection step: {'N/A':<23} | {t_elapsed_select:<22.2f}s")
    print(f"  - Action step: {'N/A':<23} | {t_elapsed_action:<22.2f}s")
    
    control_calls_set = set([c.get('function', {}).get('name') for c in control_calls])
    action_calls_set = set([c.get('function', {}).get('name') for c in action_calls])
    
    print("-" * 80)
    print(f"Control Tool Calls:  {list(control_calls_set)}")
    print(f"Variant Tool Calls:  {list(action_calls_set)}")
    
    if control_calls_set == action_calls_set:
        print("\n✅ SUCCESS: Both scenarios produced identical tool calling actions!")
    else:
        missing_in_variant = control_calls_set - action_calls_set
        extra_in_variant = action_calls_set - control_calls_set
        if not missing_in_variant:
            print("\n✅ SUCCESS: Variant successfully executed all tools required by Control (plus some extra).")
        else:
            print(f"\n⚠️ WARNING: Tool call mismatch. Control called {control_calls_set}, Variant called {action_calls_set}")

    if control_tokens > 0 and total_variant_tokens > 0:
        token_savings = control_tokens - total_variant_tokens
        saving_pct = (token_savings / control_tokens) * 100
        print(f"Token Savings: {token_savings} tokens ({saving_pct:.2f}% reduction)")
    print("====================================================")

if __name__ == "__main__":
    asyncio.run(run_experiment())
