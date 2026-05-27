"""
Meta-Agent Judge — Self-improving prompt lifecycle manager.

Runs at the end of every autoresearch cycle to evaluate, fire, promote,
and generate prompt variants for the Trade Execution Agent.

Lifecycle:
  1. EVALUATE: Check all 'active' prompt_templates with >= MIN_EVAL_TRADES trades
  2. BENCH:    Deactivate prompts with win_rate < BENCH_THRESHOLD
  3. PROMOTE:  Promote 'candidate' prompts with win_rate > PROMOTE_THRESHOLD
  4. GENERATE: If any sector has < MIN_ACTIVE prompts, generate a new variant
               by mutating the best-performing prompt in that sector

Configuration (from config.py):
  - META_AGENT_ENABLED: bool   — gate for the entire meta-agent
  - META_AGENT_INTERVAL_HOURS  — minimum hours between runs
  - MAX_ACTIVE_GENERATED_PROMPTS — cap on total active generated prompts

Usage:
    from app.agents.meta_agent_judge import run_meta_agent_judge
    result = await run_meta_agent_judge(cycle_id="cycle-123")
"""

import hashlib
import logging
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

MIN_EVAL_TRADES = 20       # Need N trades before evaluating a prompt
BENCH_THRESHOLD = 0.40     # Bench prompts below 40% win rate
PROMOTE_THRESHOLD = 0.55   # Promote candidates above 55% win rate
MIN_ACTIVE_PER_SECTOR = 2  # Generate if fewer than this many active per sector


async def run_meta_agent_judge(cycle_id: str) -> dict:
    """Run the meta-agent judge lifecycle.

    Returns a summary dict with counts of evaluated, benched, promoted,
    and generated prompts.
    """
    from app.config import settings

    if not settings.META_AGENT_ENABLED:
        logger.debug("[META_JUDGE] Meta-agent is disabled (META_AGENT_ENABLED=False)")
        return {"status": "disabled"}

    logger.info("[META_JUDGE] Running meta-agent judge for cycle %s", cycle_id)

    try:
        from app.db.connection import get_db
    except ImportError:
        logger.error("[META_JUDGE] Cannot import get_db")
        return {"status": "error", "reason": "no_db"}

    summary = {
        "status": "ok",
        "cycle_id": cycle_id,
        "evaluated": 0,
        "benched": [],
        "promoted": [],
        "generated": [],
    }

    try:
        with get_db() as db:
            # ── Step 1: EVALUATE ──
            # Find all prompts with enough trades to evaluate
            rows = db.execute("""
                SELECT id, sector, action_type, status, total_trades,
                       wins, losses, win_rate, avg_pnl_pct, generation
                FROM prompt_templates
                WHERE total_trades >= %s
                ORDER BY sector, win_rate DESC
            """, [MIN_EVAL_TRADES]).fetchall()

            summary["evaluated"] = len(rows)
            if not rows:
                logger.info("[META_JUDGE] No prompts have enough trades (%d) to evaluate", MIN_EVAL_TRADES)
                return summary

            # ── Step 2: BENCH underperformers ──
            for row in rows:
                pt_id, sector, action_type, status, total_trades, wins, losses, win_rate, avg_pnl, gen = row

                if status == "active" and win_rate < BENCH_THRESHOLD:
                    db.execute("""
                        UPDATE prompt_templates
                        SET status = 'benched', benched_at = %s
                        WHERE id = %s
                    """, [datetime.now(timezone.utc), pt_id])
                    summary["benched"].append({
                        "id": pt_id, "sector": sector,
                        "win_rate": win_rate, "trades": total_trades,
                    })
                    logger.warning(
                        "[META_JUDGE] BENCHED prompt %s (sector=%s, win_rate=%.1f%%, trades=%d)",
                        pt_id[:12], sector, win_rate * 100, total_trades,
                    )

            # ── Step 3: PROMOTE top candidates ──
            for row in rows:
                pt_id, sector, action_type, status, total_trades, wins, losses, win_rate, avg_pnl, gen = row

                if status == "candidate" and win_rate >= PROMOTE_THRESHOLD:
                    db.execute("""
                        UPDATE prompt_templates
                        SET status = 'active', promoted_at = %s
                        WHERE id = %s
                    """, [datetime.now(timezone.utc), pt_id])
                    summary["promoted"].append({
                        "id": pt_id, "sector": sector,
                        "win_rate": win_rate, "trades": total_trades,
                    })
                    logger.info(
                        "[META_JUDGE] PROMOTED prompt %s (sector=%s, win_rate=%.1f%%, trades=%d)",
                        pt_id[:12], sector, win_rate * 100, total_trades,
                    )

            # ── Step 4: GENERATE new variants for under-served sectors ──
            # Check total active generated prompts
            total_active = db.execute("""
                SELECT COUNT(*) FROM prompt_templates
                WHERE status = 'active' AND created_by = 'meta_judge'
            """).fetchone()[0]

            if total_active >= settings.MAX_ACTIVE_GENERATED_PROMPTS:
                logger.info(
                    "[META_JUDGE] Active generated prompt cap reached (%d/%d), skipping generation",
                    total_active, settings.MAX_ACTIVE_GENERATED_PROMPTS,
                )
            else:
                # Find sectors with too few active prompts
                sector_counts = db.execute("""
                    SELECT sector, COUNT(*) as active_count
                    FROM prompt_templates
                    WHERE status IN ('active', 'candidate')
                    GROUP BY sector
                """).fetchall()

                sector_map = {row[0]: row[1] for row in sector_counts}
                known_sectors = ["technology", "energy", "healthcare", "financial services", "consumer"]

                for sector in known_sectors:
                    count = sector_map.get(sector, 0)
                    if count < MIN_ACTIVE_PER_SECTOR:
                        new_prompt = await _generate_variant(db, sector, cycle_id)
                        if new_prompt:
                            summary["generated"].append(new_prompt)

    except Exception as e:
        logger.error("[META_JUDGE] Failed: %s", e, exc_info=True)
        summary["status"] = "error"
        summary["error"] = str(e)

    logger.info(
        "[META_JUDGE] Done: evaluated=%d, benched=%d, promoted=%d, generated=%d",
        summary["evaluated"],
        len(summary["benched"]),
        len(summary["promoted"]),
        len(summary["generated"]),
    )

    return summary


async def _generate_variant(db, sector: str, cycle_id: str) -> dict | None:
    """Generate a new prompt variant for the given sector by mutating the best.

    Uses the LLM to generate a variation of the best-performing prompt in the
    sector, or creates a new one from the hardcoded sector guidance if none exist.

    Returns the new prompt metadata dict, or None if generation fails.
    """
    # Find the best existing prompt in this sector
    best = db.execute("""
        SELECT id, system_prompt, win_rate, generation
        FROM prompt_templates
        WHERE sector = %s AND status IN ('active', 'benched')
        ORDER BY win_rate DESC, total_trades DESC
        LIMIT 1
    """, [sector]).fetchone()

    parent_id = None
    parent_prompt = None
    parent_gen = 0

    if best:
        parent_id = best[0]
        parent_prompt = best[1]
        parent_gen = best[3] or 0

    # Generate a mutated prompt
    try:
        from app.agents.base_agent import run_agent

        mutation_system = """You are a prompt engineering expert specializing in stock trading analysis.
Your job is to create an IMPROVED system prompt for a trading execution agent.

Rules:
1. The new prompt must be functionally similar but with different analytical emphasis
2. Keep the JSON output schema IDENTICAL to the original
3. The prompt should be 100-200 words
4. Focus on making the analysis sharper for the specific sector
5. Add one new analytical lens the original didn't have (e.g., supply chain analysis, insider activity, options flow)
6. Output ONLY the new system prompt text, nothing else"""

        if parent_prompt:
            user_msg = (
                f"Sector: {sector}\n"
                f"Current best prompt (win_rate={best[2]*100:.1f}%):\n"
                f"---\n{parent_prompt[:2000]}\n---\n"
                f"Create an improved variant with a different analytical emphasis."
            )
        else:
            from app.agents.trade_execution_agent import _get_sector_guidance
            guidance = _get_sector_guidance(sector)
            user_msg = (
                f"Sector: {sector}\n"
                f"Sector context: {guidance}\n"
                f"Create a trading execution agent system prompt for BUY decisions in this sector."
            )

        result = await run_agent(
            agent_name="meta_judge_generator",
            ticker="META_JUDGE",
            cycle_id=cycle_id,
            bot_id="meta-judge",
            system_prompt=mutation_system,
            user_prompt=user_msg,
            max_tokens=1024,
            temperature=0.7,  # Higher temp for creative variation
        )

        new_prompt_text = result.get("response", "").strip()
        if not new_prompt_text or len(new_prompt_text) < 50:
            logger.warning("[META_JUDGE] Generated prompt too short for sector %s, skipping", sector)
            return None

        # Store in DB
        new_id = f"pt-{uuid.uuid4().hex[:12]}"
        db.execute("""
            INSERT INTO prompt_templates
            (id, sector, action_type, system_prompt, status, parent_id, generation, created_by)
            VALUES (%s, %s, 'BUY', %s, 'candidate', %s, %s, 'meta_judge')
        """, [new_id, sector, new_prompt_text, parent_id, parent_gen + 1])

        logger.info(
            "[META_JUDGE] GENERATED new prompt %s for sector=%s (gen=%d, parent=%s)",
            new_id, sector, parent_gen + 1, parent_id[:12] if parent_id else "none",
        )

        return {
            "id": new_id,
            "sector": sector,
            "generation": parent_gen + 1,
            "parent_id": parent_id,
        }

    except Exception as gen_err:
        logger.warning("[META_JUDGE] Prompt generation failed for sector %s: %s", sector, gen_err)
        return None


def record_prompt_outcome(prompt_template_id: str, is_win: bool, pnl_pct: float) -> None:
    """Record a trade outcome against a prompt template.

    Called after a trade is closed to update the win/loss/pnl stats
    of the prompt that was used to size the trade.

    Args:
        prompt_template_id: ID of the prompt_template that was active during the trade.
        is_win: Whether the trade was profitable.
        pnl_pct: Percentage P&L of the trade.
    """
    try:
        from app.db.connection import get_db
        with get_db() as db:
            db.execute("""
                UPDATE prompt_templates
                SET total_trades = total_trades + 1,
                    wins = wins + CASE WHEN %s THEN 1 ELSE 0 END,
                    losses = losses + CASE WHEN %s THEN 0 ELSE 1 END,
                    win_rate = CASE
                        WHEN total_trades + 1 > 0
                        THEN (wins + CASE WHEN %s THEN 1 ELSE 0 END)::DOUBLE PRECISION / (total_trades + 1)
                        ELSE 0.0
                    END,
                    avg_pnl_pct = CASE
                        WHEN total_trades + 1 > 0
                        THEN (avg_pnl_pct * total_trades + %s) / (total_trades + 1)
                        ELSE %s
                    END
                WHERE id = %s
            """, [is_win, is_win, is_win, pnl_pct, pnl_pct, prompt_template_id])
    except Exception as e:
        logger.warning("[META_JUDGE] Failed to record outcome for prompt %s: %s", prompt_template_id, e)
