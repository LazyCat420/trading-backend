"""
Re-Analysis Worker — multi-angle analysis through different lenses.

Finds data records that haven't been fully analyzed (analysis_count < max_analyses),
selects the next unused lens, and produces a strategy_candidate for each pass.

Each lens provides a unique analytical perspective on the same underlying data,
producing diverse trading signals that feed into the debate/decision engine.

Architecture:
    1. Query for under-analyzed records (news, reddit, youtube)
    2. For each record, pick the next unused lens
    3. Build context from existing DB data (via context_builder)
    4. Run the lens via base_agent.run_agent() (all LLM through vllm_client)
    5. Write output to strategy_candidates table
    6. Increment analysis_count on the source record

Usage:
    from app.pipeline.data.reanalysis_worker import run_reanalysis_pass

    # Run one pass of multi-angle re-analysis:
    results = await run_reanalysis_pass(cycle_id="cycle-abc123")
"""

import logging
import time
import uuid
from datetime import datetime, timezone

from app.config import settings
from app.config.config_lenses import get_unused_lenses
from app.db.connection import get_db
from app.pipeline.data.scraper_queue import enqueue_request

logger = logging.getLogger(__name__)


def _get_underanalyzed_tickers(limit: int = 20) -> list[dict]:
    """Find tickers with data that hasn't been fully analyzed.

    Queries news, reddit, and youtube tables for records where
    analysis_count < max_analyses and returns unique tickers
    with their available data types.
    """
    with get_db() as db:
        tickers: dict[str, dict] = {}

        # Check each data source for under-analyzed records
        sources = [
            ("news_articles", "news"),
            ("reddit_posts", "reddit"),
            ("youtube_transcripts", "youtube"),
        ]

        for table, data_type in sources:
            try:
                rows = db.execute(
                    f"""
                    SELECT DISTINCT ticker, COUNT(*) as pending_count
                    FROM {table}
                    WHERE analysis_count < max_analyses
                      AND ticker IS NOT NULL
                      AND ticker != ''
                    GROUP BY ticker
                    ORDER BY pending_count DESC
                    LIMIT %s
                    """,
                    [limit],
                ).fetchall()

                for row in rows:
                    ticker = row[0]
                    if ticker not in tickers:
                        tickers[ticker] = {"ticker": ticker, "data_types": []}
                    tickers[ticker]["data_types"].append(data_type)

            except Exception as e:
                # Table might not have analysis_count column yet (pre-migration)
                logger.debug("[REANALYSIS] Skipping %s: %s", table, e)

        return list(tickers.values())[:limit]


async def _analyze_with_lens(
    ticker: str,
    lens: dict,
    cycle_id: str,
    bot_id: str = "",
) -> dict | None:
    """Run a single lens analysis for a ticker.

    Builds context, calls the LLM via base_agent, and writes
    the result to strategy_candidates.

    Returns the strategy candidate dict, or None on failure.
    """
    from app.agents.base_agent import run_agent
    from app.pipeline.analysis.context_builder import build_context_blob
    from app.utils.text_utils import parse_json_response, hash_prompt

    lens_name = lens["name"]
    system_prompt = lens["system_prompt"]
    prompt_hash = hash_prompt(system_prompt)

    start = time.monotonic()

    try:
        # Check if prerequisite data exists
        with get_db() as db:
            for table in lens.get("required_tables", []):
                try:
                    count = db.execute(
                        f"SELECT COUNT(*) FROM {table} WHERE ticker = %s",
                        [ticker],
                    ).fetchone()[0]
                    if count == 0:
                        # Missing data — enqueue JIT request
                        enqueue_request(
                            ticker=ticker,
                            data_type=table,
                            priority=settings.SCRAPER_JIT_PRIORITY,
                            requested_by_lens=lens_name,
                        )
                        logger.info(
                            "[REANALYSIS] %s: missing %s data for lens %s, enqueued JIT request",
                            ticker,
                            table,
                            lens_name,
                        )
                except Exception:
                    pass  # Table might not support ticker filtering (e.g., market_regime)

            # Build context blob for this ticker
            context = await build_context_blob(ticker)

            if not context or len(context) < 100:
                logger.warning(
                    "[REANALYSIS] %s: insufficient context (%d chars), skipping",
                    ticker,
                    len(context) if context else 0,
                )
                return None

            # Run the lens via base_agent (Rule 2 compliant)
            user_prompt = (
                f"Analyze {ticker} using the data provided below. "
                f"Focus on the analytical framework described in your system prompt.\n\n"
                f"{context}"
            )

            target_hardware = lens.get("target_hardware", "spark")
            endpoint_override = "dgx_spark" if target_hardware == "spark" else "jetson"

            result = await run_agent(
                agent_name=f"lens_{lens_name}",
                ticker=ticker,
                cycle_id=cycle_id,
                bot_id=bot_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=lens.get("max_tokens", 512),
                temperature=0.3,
                endpoint_override=endpoint_override,
            )

            # Parse the LLM response
            parsed = parse_json_response(result.get("response", ""))
            signal = parsed.get("signal", "HOLD").upper()
            if signal not in ("BUY", "SELL", "HOLD"):
                signal = "HOLD"
            confidence = int(parsed.get("confidence", 0))
            rationale = parsed.get("rationale", result.get("response", "")[:300])

            elapsed_ms = int((time.monotonic() - start) * 1000)

            # Write to strategy_candidates
            candidate_id = str(uuid.uuid4())
            db.execute(
                """
                INSERT INTO strategy_candidates
                (id, cycle_id, ticker, lens_name, system_prompt_hash,
                 summary, signal, confidence_score, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    candidate_id,
                    cycle_id,
                    ticker,
                    lens_name,
                    prompt_hash,
                    rationale,
                    signal,
                    confidence,
                    datetime.now(timezone.utc),
                ],
            )

            logger.info(
                "[REANALYSIS] %s via %s: %s @ %d%% (%dms, %d tokens)",
                ticker,
                lens_name,
                signal,
                confidence,
                elapsed_ms,
                result.get("tokens_used", 0),
            )

            return {
                "candidate_id": candidate_id,
                "ticker": ticker,
                "lens": lens_name,
                "signal": signal,
                "confidence": confidence,
                "rationale": rationale,
                "tokens_used": result.get("tokens_used", 0),
                "elapsed_ms": elapsed_ms,
            }

    except Exception as e:
        logger.error("[REANALYSIS] %s via %s failed: %s", ticker, lens_name, e)
        return None


def _increment_analysis_count(ticker: str) -> None:
    """Increment analysis_count on all data records for this ticker.

    Called after a lens pass completes so the next pass
    picks a different lens.
    """
    with get_db() as db:
        tables = ["news_articles", "reddit_posts", "youtube_transcripts"]

        for table in tables:
            try:
                db.execute(
                    f"""
                    UPDATE {table}
                    SET analysis_count = COALESCE(analysis_count, 0) + 1
                    WHERE ticker = %s
                      AND analysis_count < max_analyses
                    """,
                    [ticker],
                )
            except Exception:
                pass  # Column might not exist yet (pre-migration)


async def run_reanalysis_pass(
    cycle_id: str = "",
    bot_id: str = "",
    max_tickers: int = 10,
    emit=None,
) -> list[dict]:
    """Run one pass of multi-angle re-analysis.

    Steps:
        1. Find under-analyzed tickers
        2. For each, select the next unused lens
        3. Run analysis through the lens
        4. Increment analysis count

    Args:
        cycle_id: Current cycle ID for audit trail
        bot_id: Bot ID
        max_tickers: Max tickers to analyze per pass
        emit: Optional event callback

    Returns:
        List of strategy candidate dicts
    """
    if not settings.REANALYSIS_ENABLED:
        logger.debug("[REANALYSIS] Disabled by config (REANALYSIS_ENABLED=False)")
        return []

    underanalyzed = _get_underanalyzed_tickers(limit=max_tickers)

    if not underanalyzed:
        logger.debug("[REANALYSIS] No under-analyzed tickers found")
        return []

    logger.info(
        "[REANALYSIS] Found %d tickers for re-analysis",
        len(underanalyzed),
    )

    if emit:
        emit(
            "analyzing",
            "reanalysis_start",
            f"Re-analysis: {len(underanalyzed)} tickers queued",
            status="running",
        )

    results = []
    for ticker_info in underanalyzed:
        ticker = ticker_info["ticker"]

        # Get unused lenses for this ticker
        unused_lenses = get_unused_lenses(ticker)
        if not unused_lenses:
            logger.debug("[REANALYSIS] %s: all lenses already applied", ticker)
            continue

        # Pick the next lens (first unused)
        lens = unused_lenses[0]

        candidate = await _analyze_with_lens(
            ticker=ticker,
            lens=lens,
            cycle_id=cycle_id,
            bot_id=bot_id,
        )

        if candidate:
            results.append(candidate)
            _increment_analysis_count(ticker)

    if emit:
        emit(
            "analyzing",
            "reanalysis_done",
            f"Re-analysis complete: {len(results)} candidates generated",
            status="ok",
            data={"candidates": len(results), "tickers": len(underanalyzed)},
        )

    logger.info(
        "[REANALYSIS] Pass complete: %d/%d tickers produced candidates",
        len(results),
        len(underanalyzed),
    )

    return results
