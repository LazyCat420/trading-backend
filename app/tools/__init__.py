from app.tools.registry import registry, PermissionLevel
from app.tools.finance_tools import (
    get_market_data,
    get_finnhub_news,
    get_technical_indicators,
)
from app.tools.wiki_tools import write_memory_note, read_memory_note, search_wiki
from app.tools.web_tools import search_web, web_search, scrape_url, query_hermes
from app.tools.quant_tools import execute_momentum_strategy, execute_value_strategy

from app.tools.db_tools import search_internal_database, update_youtube_channel_handle
from app.tools.system_tools import run_local_command
from app.tools.browser_tools import (
    browser_navigate,
    run_playwright_script,
)
from app.tools.youtube_tools import (
    youtube_test_channel,
)

# Phase 2: Pipeline Skills as Tools
from app.tools.pipeline_tools import (
    audit_decision_quality,
    check_hallucination,
)

# Phase 3: Agent Coordination Tools
from app.tools.coordination_tools import (
    post_finding,
    read_team_findings,
    request_investigation,
    check_open_investigations,
)

# Phase 4: Persistent Profile Memory
from app.tools.profile_tools import (
    read_profile_tool,
    update_preference_tool,
    add_agent_note_tool,
)

# Phase 5: Trading Tools
from app.tools.trading_tools import buy_stock, sell_stock

# Phase 8: Portfolio Awareness & Benchmarking Tools
from app.tools.portfolio_tools import (
    get_portfolio_state_tool,
    get_position_pnl_tool,
    get_performance_metrics_tool,
    propose_constitution_amendment_tool,
)

# Phase 7: Schedule Management Tools
from app.tools.schedule_tools import (
    create_or_update_schedule,
    list_active_schedules,
)

# Phase 9: Order Trigger Tools
from app.tools.trigger_tools import (
    set_price_trigger,
    list_active_triggers,
    cancel_price_trigger,
)

# Phase 6: Deterministic Financial Calculators
from app.tools.calculator_tools import (
    calculate_position_size,
    calculate_stop_loss,
    calculate_risk_reward,
    calculate_portfolio_allocation,
)

# Phase 10: Capsule Context Expansion Tools
from app.tools.context_tools import get_cycle_context, get_cycle_context_all

# Phase 5: Sandboxed Python Execution (Quant Scripts)
from app.tools.script_sandbox import execute_quant_script, execute_python

# Phase 6: Prism Agent Harness (Onion Layer)
from app.tools.prism_agent_harness import run_prism_agent

# Phase 11: LLM-Steered Data Collection
from app.tools.collection_request_tool import request_data_collection

# Phase 12: Precision RAG Tools (Hybrid Context Model)
from app.tools.precision_rag import (
    query_financial_metrics,
    query_technical_indicator,
    search_database_facts,
)

# Phase 13: Agentic Risk Assessment Tools (Composite)
from app.tools.risk_tools import (
    get_market_regime_tool,
    query_brain_graph_tool,
    assess_risk_environment_tool,
)

__all__ = [
    "registry",
    "PermissionLevel",
    "get_market_data",
    "get_finnhub_news",
    "get_technical_indicators",
    "write_memory_note",
    "read_memory_note",
    "search_wiki",
    "search_web",
    "web_search",
    "scrape_url",
    "query_hermes",
    "search_internal_database",
    "update_youtube_channel_handle",
    "run_local_command",
    "browser_navigate",
    "run_playwright_script",
    "youtube_test_channel",
    # Phase 2: Pipeline Tools
    "audit_decision_quality",
    "check_hallucination",
    # Phase 3: Coordination Tools
    "post_finding",
    "read_team_findings",
    "request_investigation",
    "check_open_investigations",
    # Phase 4: Profile Memory Tools
    "read_profile_tool",
    "update_preference_tool",
    "add_agent_note_tool",
    # Phase 5: Trading Tools
    "buy_stock",
    "sell_stock",
    # Phase 5b: Sandboxed Quant Execution
    "execute_quant_script",
    "execute_python",
    "execute_momentum_strategy",
    "execute_value_strategy",
    # Phase 6: Deterministic Financial Calculators
    "calculate_position_size",
    "calculate_stop_loss",
    "calculate_risk_reward",
    "calculate_portfolio_allocation",
    # Phase 6b: Prism Agent Harness
    "run_prism_agent",
    # Phase 7: Schedule Management Tools
    "create_or_update_schedule",
    "list_active_schedules",
    # Phase 8: Portfolio Awareness & Benchmarking Tools
    "get_portfolio_state_tool",
    "get_position_pnl_tool",
    "get_performance_metrics_tool",
    "propose_constitution_amendment_tool",
    # Phase 9: Order Trigger Tools
    "set_price_trigger",
    "list_active_triggers",
    "cancel_price_trigger",
    # Phase 10: Capsule Context Tools
    "get_cycle_context",
    "get_cycle_context_all",
    # Phase 11: LLM-Steered Data Collection
    "request_data_collection",
    # Phase 12: Precision RAG Tools
    "query_financial_metrics",
    "query_technical_indicator",
    "search_database_facts",
    # Phase 13: Agentic Risk Assessment Tools
    "get_market_regime_tool",
    "query_brain_graph_tool",
    "assess_risk_environment_tool",
]
