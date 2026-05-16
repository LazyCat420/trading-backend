"""
Benchmark Agent — Post-cycle strategy evaluator.

Analyzes the bot's trading performance over recent cycles using the
get_performance_metrics tool, compares it against the active Trading
Constitution, and proposes amendments via the propose_constitution_amendment
tool if the performance justifies a change.
"""

import json
import logging

from app.services.vllm_client import llm, Priority
from app.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the Benchmark Agent, the quantitative strategy evaluator for an autonomous trading bot.
Your objective is to evaluate the bot's historical performance metrics and determine if its Trading Constitution (its hardcoded rules) needs to be amended.

AVAILABLE DATA:
You have access to tools that can pull recent performance metrics (win rate, average profit, average loss, open position count).
You must use these tools to fetch data before making any decisions.

AMENDMENT RULES:
- If win rate is high (>60%) and the bot has excess cash, consider proposing an increase to 'max_positions' or 'max_sector_pct'.
- If the win rate is low (<40%) or average loss exceeds average profit significantly, consider proposing a decrease to 'max_positions', tightening the 'rsi_threshold', or reducing 'max_holding_days'.
- Do NOT propose an amendment if performance is stable or there is not enough closed trade data (e.g., < 3 trades).
- Be conservative. Minor tweaks are better than drastic changes.

AVAILABLE CONSTITUTION PARAMETERS YOU CAN AMEND:
1. max_positions: Maximum number of concurrent open positions (bounds: 4 to 20).
2. max_sector_pct: Maximum percentage of positions in a single sector (bounds: 15 to 60).
3. rsi_threshold: RSI level to trigger a SELL (bounds: 50 to 90).
4. pe_multiplier: P/E multiplier vs sector average to trigger SELL (bounds: 1.0 to 3.0).
5. max_holding_days: Maximum days to hold a position without thesis confirmation (bounds: 3 to 60).
6. min_pct / max_pct: Position sizing percentages.
7. rsi_max: Max RSI allowed for a BUY (bounds: 40 to 80).

If you believe an amendment is necessary, respond with a JSON object:
{"status": "amend", "parameter": "<name>", "old_value": <current>, "new_value": <proposed>, "rationale": "..."}

If no amendment is needed, return:
{"status": "no_change", "rationale": "Performance is acceptable."}
"""


async def run_benchmark_agent(cycle_id: str, days_back: int = 30) -> dict:
    """Run the benchmark agent to evaluate performance and potentially propose amendments.

    Returns a dict with the outcome of the agent's evaluation.
    """
    logger.info(
        "[BENCHMARK] Starting post-cycle performance evaluation (days_back=%d)",
        days_back,
    )

    # We fetch the current constitution so the agent knows what the rules are
    from app.pipeline.trading_constitution import format_constitution_for_prompt

    constitution_block = format_constitution_for_prompt()

    # Fetch live performance metrics to inject as context
    perf_context = ""
    try:
        from app.trading.paper_trader import get_portfolio

        portfolio = get_portfolio(settings.BOT_ID)
        perf_context = (
            f"\nCURRENT PORTFOLIO:\n"
            f"  Cash: ${portfolio.get('cash', 0):,.2f}\n"
            f"  Positions: {portfolio.get('position_count', 0)}\n"
            f"  Total PnL: ${portfolio.get('total_pnl', 0):,.2f}\n"
        )
    except Exception as e:
        logger.debug("[BENCHMARK] Failed to fetch portfolio: %s", e)

    # ── Fetch real trade performance from decision_outcomes (ground truth) ──
    trade_perf = ""
    try:
        from app.db.connection import get_db

        with get_db() as db:
            # Win rate and PnL stats from resolved outcomes
            stats_row = db.execute(
                """
                SELECT
                    COUNT(*) as total,
                    COUNT(CASE WHEN outcome = 'WIN' THEN 1 END) as wins,
                    COUNT(CASE WHEN outcome = 'LOSS' THEN 1 END) as losses,
                    COUNT(CASE WHEN outcome = 'FLAT' THEN 1 END) as flats,
                    COALESCE(AVG(CASE WHEN outcome = 'WIN' THEN pnl_pct END), 0) as avg_win,
                    COALESCE(AVG(CASE WHEN outcome = 'LOSS' THEN pnl_pct END), 0) as avg_loss,
                    COALESCE(AVG(pnl_pct), 0) as avg_pnl
                FROM decision_outcomes
                WHERE resolved_at IS NOT NULL
                  AND outcome != 'CANCELED'
                  AND resolved_at > CURRENT_TIMESTAMP - INTERVAL '%s days'
                """ % days_back,
            ).fetchone()

            if stats_row and stats_row[0] > 0:
                total, wins, losses, flats = stats_row[0], stats_row[1], stats_row[2], stats_row[3]
                win_rate = (wins / total * 100) if total > 0 else 0
                trade_perf = (
                    f"\nTRADE PERFORMANCE (last {days_back} days):\n"
                    f"  Total closed trades: {total}\n"
                    f"  Win rate: {win_rate:.1f}% ({wins}W / {losses}L / {flats}F)\n"
                    f"  Avg win: +{stats_row[4]:.2f}%\n"
                    f"  Avg loss: {stats_row[5]:.2f}%\n"
                    f"  Overall avg PnL: {stats_row[6]:.2f}%\n"
                )

            # Recent lot closures for context
            closures = db.execute(
                """
                SELECT ticker, realized_pnl, holding_days, closed_at
                FROM lot_closures
                WHERE closed_at > CURRENT_TIMESTAMP - INTERVAL '%s days'
                ORDER BY closed_at DESC LIMIT 10
                """ % days_back,
            ).fetchall()

            if closures:
                trade_perf += "\n  RECENT CLOSED POSITIONS:\n"
                for c in closures:
                    trade_perf += f"    {c[0]}: ${c[1]:+,.2f} PnL, held {c[2] or '?'} days\n"

    except Exception as e:
        logger.debug("[BENCHMARK] Failed to fetch trade performance: %s", e)

    user_prompt = (
        f"Cycle {cycle_id} has completed.\n\n"
        f"CURRENT CONSTITUTION:\n"
        f"{constitution_block or 'No active constitution rules.'}\n\n"
        f"{perf_context}\n"
        f"{trade_perf}\n"
        f"Please evaluate our performance over the last {days_back} days. "
        f"Analyze the data and determine if any constitution amendments are needed."
    )

    try:
        response, tokens, elapsed_ms = await llm.chat(
            system=SYSTEM_PROMPT,
            user=user_prompt,
            temperature=0.2,
            max_tokens=2048,
            priority=Priority.LOW,
            agent_name="benchmark_agent",
            cycle_id=cycle_id,
        )

        logger.info("[BENCHMARK] Agent responded (%d tokens, %dms)", tokens, elapsed_ms)

        # If the LLM returned raw JSON
        if isinstance(response, str) and response.strip().startswith("{"):
            try:
                parsed = json.loads(response)
                return parsed
            except json.JSONDecodeError:
                pass

        return {"status": "completed", "llm_output": response}

    except Exception as e:
        logger.error("[BENCHMARK] Agent failed: %s", e)
        return {"status": "error", "message": str(e)}
