import logging
import json
from typing import List, Dict

from app.services.memory.episodic_memory import episodic_memory_store

logger = logging.getLogger(__name__)


class CycleCloser:
    """
    Runs at the end of every cycle.
    Formats the cycle's context into a concise summary and
    stores it in the Episodic Memory layer.
    """

    @classmethod
    async def close_cycle(
        cls,
        cycle_id: str,
        tickers: List[str],
        mode: str,
        summary: dict,
        results: List[Dict],
    ):
        """Called by PipelineService when the cycle finishes."""
        if not results:
            return

        results_by_ticker = {r.get("ticker"): r for r in results if r.get("ticker")}

        for ticker in tickers:
            if ticker not in results_by_ticker:
                continue

            try:
                decision = results_by_ticker[ticker]
                action = decision.get("action", "HOLD")
                confidence = decision.get("confidence", 0)
                rationale = decision.get("rationale", "")

                # Truncate rationale if it's too long
                if len(rationale) > 200:
                    rationale = rationale[:197] + "..."

                episode_text = f"Analyzed in '{mode}' mode. Agent decided {action} with {confidence}% confidence. Rationale: {rationale}"

                # Check outcome: Did it execute?
                trade_executed = decision.get("trade_executed")
                trade_skipped = decision.get("trade_skipped")

                if trade_executed:
                    episode_text += f" | Trade EXECUTED: {action} {trade_executed.get('fill_qty', '')} shares @ {trade_executed.get('fill_price', '')}"
                elif trade_skipped:
                    episode_text += (
                        f" | Trade SKIPPED: {trade_skipped.get('reason', 'Unknown')}"
                    )

                agents = ["analyst", "trader"] if trade_executed else ["analyst"]

                from app.utils.text_utils import sanitize_surrogates
                episode_text = sanitize_surrogates(episode_text)
                rationale = sanitize_surrogates(rationale)

                episodic_memory_store.write_episode(
                    cycle_id=cycle_id,
                    ticker=ticker,
                    summary=episode_text,
                    key_decisions=json.dumps([action]),
                    outcome="neutral",  # Update retrospectively in future
                    outcome_score=0.0,
                    agents_involved=json.dumps(agents),
                )

                # Also store a semantic memory capturing the core insight
                # Semantic memory is for durable facts. We'll store the rationale as a 'thesis_insight'
                from app.services.memory.semantic_memory import semantic_memory_store
                semantic_memory_store.write_semantic(
                    ticker=ticker,
                    mem_type="thesis_insight",
                    content=f"On {action} decision: {rationale}",
                    confidence=float(confidence) / 100.0 if confidence else 0.5,
                    source_agent="analyst",
                )
            except Exception as e:
                logger.error(
                    f"[CYCLE_CLOSER] Failed to write memory for {ticker}: {e}"
                )


cycle_closer = CycleCloser()
