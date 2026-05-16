"""
Target Mapping Engine — Maps AutoResearch failure categories to actual source files.

When the AutoResearch report flags an issue (e.g. "reddit data missing"), this module
resolves the exact Python file, class, or prompt template that needs to be fixed.
The resolved content is then passed to the Evolution Debate Council.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # vllm-trading-bot/


# ── Static registry: maps known agent/source names → file paths ──
# Keys are lowercase identifiers that appear in AutoResearch JSON blobs.
# Values are relative paths from the project root.

SCRAPER_MAP: dict[str, str] = {
    # Data collectors (from app/collectors/)
    "price_history": "app/collectors/yfinance_collector.py",
    "yfinance": "app/collectors/yfinance_collector.py",
    "finnhub": "app/collectors/finnhub_collector.py",
    "reddit": "app/collectors/reddit_collector.py",
    "reddit_posts": "app/collectors/reddit_collector.py",
    "youtube": "app/collectors/youtube_collector.py",
    "youtube_transcripts": "app/collectors/youtube_collector.py",
    "news": "app/collectors/news_collector.py",
    "news_articles": "app/collectors/news_collector.py",
    "fundamentals": "app/collectors/yfinance_collector.py",
    "technicals": "app/collectors/yfinance_collector.py",
    "sec": "app/collectors/sec_collector.py",
    "sec_13f": "app/services/sec_13f_service.py",
    "congress": "app/collectors/congress_collector.py",
    "congress_trades": "app/collectors/congress_collector.py",
    "fred": "app/collectors/fred_collector.py",
    "macro_indicators": "app/collectors/fred_collector.py",
    "coingecko": "app/collectors/coingecko_collector.py",
    "commodity": "app/collectors/commodity_collector.py",
    "options": "app/collectors/options_collector.py",
    "earnings": "app/collectors/earnings_collector.py",
    "insider": "app/collectors/insider_collector.py",
    "fmp": "app/collectors/fmp_collector.py",
    "gdelt": "app/collectors/gdelt_collector.py",
    "eia": "app/collectors/eia_collector.py",
}

PROMPT_MAP: dict[str, str] = {
    # Debate / analysis agent prompts
    "debate": "app/cognition/debate/debate_coordinator.py",
    "debate_coordinator": "app/cognition/debate/debate_coordinator.py",
    "debate_judge": "app/cognition/debate/debate_judge.py",
    "thesis_agent": "app/cognition/debate/thesis_agent.py",
    "specialized_agents": "app/cognition/debate/specialized_agents.py",
    # Decision / trading prompts
    "decision_engine": "app/services/pipeline_service.py",
    "trading_phase": "app/services/pipeline_service.py",
    # RLM
    "rlm": "app/services/rlm_prompts.py",
    "rlm_wrapper": "app/services/rlm_wrapper.py",
    # Evolution prompts
    "evolve_designer": "app/agents/prompts/evolve_designer.md",
    "evolve_analyzer": "app/agents/prompts/evolve_analyzer.md",
    # Data curation
    "hermes_research": "app/services/web_search.py",
    "data_janitor": "app/services/data_flag_service.py",
    # Memory / RAG
    "memory_briefing": "app/services/memory/briefing.py",
    # Evaluation
    "judge_agent": "app/cognition/evaluation/judge_agent.py",
    "strategy_auditor": "app/cognition/evaluation/strategy_auditor.py",
}

STRATEGY_MAP: dict[str, str] = {
    "strategy_candidate": "scripts/strategy_candidate.py",
    "strategy": "scripts/strategy_candidate.py",
}

OPTIMIZER_MAP: dict[str, str] = {
    # Expanded evolution scope — optimize pipeline subsystems
    "collection_optimizer": "app/pipeline/data_phase.py",
    "collection_scheduler": "app/pipeline/collection_scheduler.py",
    "memory_optimizer": "app/services/memory/working_memory.py",
    "processing_optimizer": "app/services/pipeline_service.py",
    "summarization_optimizer": "app/services/summarization.py",
    "janitor_optimizer": "app/services/data_flag_service.py",
    "debate_optimizer": "app/cognition/evolution/debate.py",
}


def resolve_target(target_type: str, target_name: str) -> dict:
    """
    Resolve a target_type + target_name into the actual file path and its contents.

    Returns:
        {
            "file_path": str,           # absolute path
            "relative_path": str,       # relative to project root
            "content": str,             # file contents (truncated to 8000 chars for LLM context)
            "exists": bool,
            "target_type": str,
            "target_name": str,
        }
    """
    target_lower = target_name.lower().replace("_prompt", "").replace("_scraper", "")

    # Try the appropriate map based on target_type
    if target_type == "scraper":
        rel_path = SCRAPER_MAP.get(target_lower)
    elif target_type == "prompt":
        rel_path = PROMPT_MAP.get(target_lower)
    elif target_type == "strategy":
        rel_path = STRATEGY_MAP.get(target_lower, "strategy_candidate.py")
    elif target_type in (
        "optimizer",
        "collection_optimizer",
        "memory_optimizer",
        "processing_optimizer",
        "summarization_optimizer",
        "janitor_optimizer",
        "debate_optimizer",
    ):
        rel_path = OPTIMIZER_MAP.get(target_lower) or OPTIMIZER_MAP.get(target_type)
    else:
        # Try all maps
        rel_path = (
            SCRAPER_MAP.get(target_lower)
            or PROMPT_MAP.get(target_lower)
            or STRATEGY_MAP.get(target_lower)
            or OPTIMIZER_MAP.get(target_lower)
        )

    if not rel_path:
        logger.warning(
            "[TARGET-MAP] No mapping found for %s/%s", target_type, target_name
        )
        return {
            "file_path": None,
            "relative_path": None,
            "content": None,
            "exists": False,
            "target_type": target_type,
            "target_name": target_name,
        }

    abs_path = PROJECT_ROOT / rel_path
    exists = abs_path.exists()
    content = None

    if exists:
        try:
            raw = abs_path.read_text(encoding="utf-8")
            # Cap at 8000 chars to fit within LLM context windows
            content = raw[:8000]
            if len(raw) > 8000:
                content += f"\n\n... [TRUNCATED — full file is {len(raw)} chars] ..."
        except Exception as e:
            logger.error("[TARGET-MAP] Failed to read %s: %s", abs_path, e)
            content = f"# Error reading file: {e}"
    else:
        logger.warning("[TARGET-MAP] File does not exist: %s", abs_path)

    return {
        "file_path": str(abs_path),
        "relative_path": rel_path,
        "content": content,
        "exists": exists,
        "target_type": target_type,
        "target_name": target_name,
    }


def list_available_targets() -> dict:
    """Return all known targets grouped by type (for the UI)."""
    return {
        "scrapers": list(SCRAPER_MAP.keys()),
        "prompts": list(PROMPT_MAP.keys()),
        "strategies": list(STRATEGY_MAP.keys()),
        "optimizers": list(OPTIMIZER_MAP.keys()),
    }
