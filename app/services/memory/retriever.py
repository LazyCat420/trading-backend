import logging
from typing import Any, Dict, List
from datetime import datetime, timezone

from app.services.memory.repository import MemoryRepository

logger = logging.getLogger(__name__)

# Config
MIN_CONFIDENCE = 0.20
MAX_AGE_DAYS_FOR_DECAY = 180
MAX_RETURNED_MEMORIES = 10
MAX_BRIEF_CHARS = 3000


def _is_stale(memory: Dict[str, Any]) -> bool:
    """Determine if a memory is considered stale."""
    # Using last_validated_at or updated_at
    date_str = memory.get("last_validated_at") or memory.get("updated_at")
    if date_str:
        try:
            # handle 'Z' missing issues or whatever ISO formats
            if date_str.endswith("Z"):
                date_str = date_str[:-1] + "+00:00"
            dt = datetime.fromisoformat(date_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)

            now = datetime.now(timezone.utc)
            days_old = (now - dt).days
            if days_old > MAX_AGE_DAYS_FOR_DECAY:
                return True
        except ValueError:
            pass
    return False


def fetch_candidate_memories(
    ticker: str, sector: str | None = None
) -> List[Dict[str, Any]]:
    """Fetch active canonical memories for a specific ticker OR its sector."""
    return MemoryRepository.fetch_candidate_memories(ticker, sector)


def score_memory(
    memory: Dict[str, Any],
    query_ticker: str,
    query_sector: str | None,
    query_tags: list[str] | None,
) -> float:
    """Calculate a relevance score for the canonical memory."""
    score = 0.0

    m_ticker = memory.get("ticker")
    m_sector = memory.get("sector")

    # 1. Exact ticker match (highest priority)
    if m_ticker and m_ticker.upper() == query_ticker.upper():
        score += 10.0

    # 2. Sector match
    if m_sector and query_sector and m_sector.upper() == query_sector.upper():
        score += 5.0
        if not m_ticker:
            score += 2.0  # Sector-wide rule boost

    # 3. Tags overlap
    if query_tags:
        m_tags = set(t.lower() for t in memory.get("tags", []))
        q_tags = set(t.lower() for t in query_tags)
        overlap = len(m_tags.intersection(q_tags))
        score += overlap * 0.5

    # 4. Confidence multiplier
    conf = float(memory.get("confidence_score") or 0.0)
    score += conf * 5.0

    # 5. Status bonus / penalty
    status = memory.get("status")
    if status == "tentative":
        score -= 2.0
    elif status == "active":
        score += 1.0

    # 6. Recency penalty (older = lower score)
    date_str = memory.get("last_validated_at") or memory.get("updated_at")
    if date_str:
        try:
            if date_str.endswith("Z"):
                date_str = date_str[:-1] + "+00:00"
            dt = datetime.fromisoformat(date_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            days_old = (datetime.now(timezone.utc) - dt).days
            if days_old > 0:
                # scale penalty linearly up to 3.0 points across MAX_AGE_DAYS_FOR_DECAY
                penalty = min(3.0, (days_old / MAX_AGE_DAYS_FOR_DECAY) * 3.0)
                score -= penalty
        except ValueError:
            pass

    return score


class MemoryRetriever:
    @classmethod
    def retrieve(
        cls, ticker: str, sector: str | None = None, tags: list[str] | None = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves, filters, and ranks the relevant canonical memories for a given ticker.
        Satisfies the RetrieveResult contract format.
        # TODO: fallback missing - no vector search implemented yet (currently uses SQL keyword/tag match)
        """
        raw_memories = fetch_candidate_memories(ticker, sector)
        candidates = []

        for m in raw_memories:
            conf = float(m.get("confidence_score") or 0.0)
            status = m.get("status", "active")

            # Filtering out bad memories
            if status == "deprecated":
                continue
            if conf < MIN_CONFIDENCE:
                continue
            if _is_stale(m):
                continue

            score = score_memory(
                m, query_ticker=ticker, query_sector=sector, query_tags=tags
            )

            # Determine "reason"
            m_ticker = m.get("ticker")
            m_sector = m.get("sector")
            reason = "General context"
            if m_ticker and m_ticker.upper() == ticker.upper():
                reason = "Exact ticker match"
            elif m_sector and sector and m_sector.upper() == sector.upper():
                reason = "Sector-wide pattern match"

            candidates.append(
                {
                    "memory_id": m["id"],
                    "summary": m.get("summary", ""),
                    "type": m.get("type", "unknown"),
                    "ticker": m_ticker,
                    "score": score,
                    "confidence_score": conf,
                    "status": status,
                    "reason": reason,
                }
            )

        # Sort by score DESC
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # Enforce max returned count
        return candidates[:MAX_RETURNED_MEMORIES]

    @classmethod
    def build_memory_brief(
        cls, retrieval_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Renders the retrieved candidates into the Memory Brief Result contract payload.
        Enforces char count limits.
        """
        if not retrieval_results:
            return {"brief_text": "", "source_memory_ids": [], "char_count": 0}

        brief_lines = [
            "========================================",
            "CANONICAL MEMORY BRIEF",
            "========================================",
        ]

        used_ids = []
        current_char_count = sum(len(line) + 1 for line in brief_lines)

        for res in retrieval_results:
            m_id = res["memory_id"]
            summary = res["summary"]
            m_type = res["type"].upper()
            reason = res["reason"]
            conf = res["confidence_score"]

            entry = f"[{m_type} | Conf: {conf:.2f} | {reason}] {summary}"

            if current_char_count + len(entry) + 1 > MAX_BRIEF_CHARS:
                brief_lines.append("... (memory truncated due to size limits) ...")
                break

            brief_lines.append(entry)
            used_ids.append(m_id)
            current_char_count += len(entry) + 1

        brief_lines.append("========================================")
        brief_text = "\n".join(brief_lines)

        return {
            "brief_text": brief_text,
            "source_memory_ids": used_ids,
            "char_count": len(brief_text),
        }
