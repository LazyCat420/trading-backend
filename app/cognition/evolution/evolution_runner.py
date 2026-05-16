"""
Evolution Runner — Generates and validates novel trading strategies.

Pulls lessons from previous evolution runs, generates Python strategy code,
tests it in the sandbox executor, and writes metrics back to evolution_nodes.
Promotes successful strategies into the generated_agent_prompts pool.
"""

import logging
import uuid
import json
import asyncio
from datetime import datetime, timezone

from app.db.connection import get_db
from app.trading.sandbox_executor import run_sandboxed_backtest
from app.services.vllm_client import llm, Priority
from app.utils.text_utils import parse_json_response

logger = logging.getLogger(__name__)

EVOLUTION_SYSTEM_PROMPT = """You are an expert quantitative strategist and Python developer.
Generate a pandas-based trading strategy. It MUST define a function `generate_signals(df)` that returns a pandas Series of signals (-1.0 for short, 0.0 for hold, 1.0 for long) where the index matches the input DataFrame.

The input `df` contains OHLCV columns: ['open', 'high', 'low', 'close', 'volume'].
You may use pandas and numpy. Do not use forbidden libraries or attempt network access.

Respond ONLY with valid JSON in this format:
{"motivation": "Brief explanation of the strategy", "code": "def generate_signals(df):\n    ..."}"""

async def generate_strategy_candidates(session_id: str, num_candidates: int = 3):
    """Generate novel strategy code using the LLM and store as pending."""
    try:
        with get_db() as db:
            lessons = db.execute(
                "SELECT status, score FROM evolution_lessons ORDER BY round DESC LIMIT 5"
            ).fetchall()
        
        lesson_text = "\n".join([f"- {l[0]} (score: {l[1]})" for l in lessons]) if lessons else "No previous lessons."
        user_prompt = f"Design a novel technical trading strategy.\nPrevious outcomes:\n{lesson_text}\n"

        for _ in range(num_candidates):
            response, _, _ = await llm.chat(
                system=EVOLUTION_SYSTEM_PROMPT,
                user=user_prompt,
                temperature=0.8,
                priority=Priority.LOW,
                agent_name="evo_generator"
            )
            parsed = parse_json_response(response)
            if "code" in parsed:
                node_id = str(uuid.uuid4())
                with get_db() as db:
                    db.execute(
                        """
                        INSERT INTO evolution_nodes (id, session_id, round, motivation, code, status, timestamp)
                        VALUES (%s, %s, %s, %s, %s, 'pending', %s)
                        """,
                        [node_id, session_id, 1, parsed.get("motivation", ""), parsed["code"], datetime.now(timezone.utc).isoformat()]
                    )
                logger.info(f"[EvoRunner] Generated new strategy candidate {node_id[:8]}")
    except Exception as e:
        logger.error(f"[EvoRunner] Failed to generate candidates: {e}")

async def evaluate_pending_strategies(data_path: str):
    """Run sandboxed backtests on pending strategy nodes."""
    try:
        with get_db() as db:
            pending = db.execute(
                "SELECT id, session_id, round, code FROM evolution_nodes WHERE status = 'pending'"
            ).fetchall()

        for row in pending:
            node_id, session_id, current_round, code = row
            
            # Execute in sandbox
            result = run_sandboxed_backtest(code, data_path)
            
            status = "failed"
            score = 0.0
            metrics_json = json.dumps({})
            
            if result.status == "SUCCESS" and result.metrics:
                metrics_json = json.dumps(result.metrics)
                sharpe = result.metrics.get("sharpe", 0.0)
                win_rate = result.metrics.get("win_rate", 0.0)
                
                if sharpe > 1.0 and win_rate > 0.5:
                    status = "success"
                    score = sharpe
                else:
                    status = "rejected"
                    score = sharpe
                    
            elif result.status == "TIMEOUT":
                status = "timeout"
            else:
                status = "error"
                
            with get_db() as db:
                db.execute(
                    """
                    UPDATE evolution_nodes 
                    SET status = %s, score = %s, metrics = %s
                    WHERE id = %s
                    """,
                    [status, score, metrics_json, node_id]
                )
                
                db.execute(
                    """
                    INSERT INTO evolution_lessons (id, session_id, round, score, status)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    [str(uuid.uuid4()), session_id, current_round, score, status]
                )
                
                if status == "success":
                    logger.info(f"[EvoRunner] Strategy {node_id[:8]} SUCCESS. Promoting to active lenses.")
                    
                    from app.agents.meta_agent import generate_prompt
                    
                    # Fetch debate insights to inform the meta agent
                    debate_insights = "No recent debates"
                    try:
                        debates = db.execute(
                            "SELECT context FROM debate_history ORDER BY id DESC LIMIT 3"
                        ).fetchall()
                        if debates:
                            debate_insights = "\\n".join([d[0][:200] for d in debates])
                    except Exception:
                        pass
                    
                    meta_lens = await generate_prompt(
                        winning_patterns=f"Strategy code that worked:\\n{code}\\nMetrics:\\n{metrics_json}",
                        losing_patterns="",
                        debate_insights=debate_insights,
                        cycle_id=session_id
                    )
                    
                    if meta_lens and meta_lens.get("system_prompt"):
                        db.execute(
                            """
                            INSERT INTO generated_agent_prompts 
                            (id, name, lens_type, system_prompt, prompt_hash, active)
                            VALUES (%s, %s, %s, %s, %s, TRUE)
                            ON CONFLICT DO NOTHING
                            """,
                            [
                                str(uuid.uuid4()), 
                                meta_lens["name"], 
                                meta_lens["lens_type"], 
                                meta_lens["system_prompt"], 
                                node_id
                            ]
                        )

    except Exception as e:
        logger.error(f"[EvoRunner] Failed to evaluate strategies: {e}")

async def run_evolution_loop(data_path: str):
    """Main orchestration loop for evolution runner."""
    session_id = f"evo_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}"
    logger.info(f"[EvoRunner] Starting evolution loop {session_id}")
    
    await generate_strategy_candidates(session_id, num_candidates=2)
    await evaluate_pending_strategies(data_path)
    
    logger.info(f"[EvoRunner] Evolution loop {session_id} complete.")
