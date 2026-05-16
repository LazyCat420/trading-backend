import json
from typing import Any, Dict, List
from app.db.connection import get_db


class MemoryRepository:
    """Shared database repository for accessing memory tables."""

    @staticmethod
    def _parse_tags(tags_val: Any) -> list[str]:
        if not tags_val:
            return []
        if isinstance(tags_val, str):
            try:
                return json.loads(tags_val)
            except json.JSONDecodeError:
                return []
        if isinstance(tags_val, list):
            return tags_val
        return []

    @classmethod
    def get_memories_by_ticker(
        cls, ticker: str, active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """Fetches memory rules pertinent to a specific ticker."""
        with get_db() as db:
            query = """
                SELECT id, type, ticker, sector, summary, tags, 
                       confidence_score, evidence_count, status, 
                       last_used_at, last_validated_at, created_at, updated_at
                FROM canonical_memories 
                WHERE ticker = %s
            """
            params = [ticker]
            if active_only:
                query += " AND status = 'active'"

            cursor = db.execute(query, params)
            cols = [desc[0] for desc in cursor.description]

            memories = []
            for row in cursor.fetchall():
                record = dict(zip(cols, row))
                record["tags"] = cls._parse_tags(record.get("tags"))
                memories.append(record)

            return memories

    @classmethod
    def fetch_candidate_memories(
        cls, ticker: str, sector: str | None = None
    ) -> List[Dict[str, Any]]:
        """Fetch active canonical memories for a specific ticker OR its sector."""
        with get_db() as db:
            query = """
                SELECT * FROM canonical_memories 
                WHERE status != 'deprecated'
                  AND (
                      ticker = %s 
                      OR (ticker IS NULL AND sector = %s)
                      OR (ticker IS NULL AND sector IS NULL)
                  )
            """
            cursor = db.execute(query, [ticker, sector])
            columns = [desc[0] for desc in cursor.description]

            results = []
            for row in cursor.fetchall():
                record = dict(zip(columns, row))
                record["tags"] = cls._parse_tags(record.get("tags"))
                results.append(record)

            return results
