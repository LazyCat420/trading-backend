"""
Entity Resolver - Maps raw text to a canonical graph entity
"""

import uuid
import logging
import re
from typing import Optional
from app.db.connection import get_db
from app.cognition.graph.models import ResolvedEntity
from app.cognition.ontology.schema import NodeType

logger = logging.getLogger(__name__)

NAMESPACE_COGNITION = uuid.UUID("00000000-0000-0000-0000-000000000000")


def _generate_entity_id(entity_type: str, canonical_name: str) -> str:
    return str(uuid.uuid5(NAMESPACE_COGNITION, f"{entity_type}:{canonical_name}"))


def resolve_entity(
    raw_text: str, entity_type_hint: Optional[str] = None
) -> ResolvedEntity:
    """
    Resolution strategy:
    1. Exact alias match
    2. Ticker match
    3. Fuzzy alias
    4. New entity fallback
    """
    with get_db() as db:
        text_clean = raw_text.strip()

        # Ensure schema since we check aliases
        from app.cognition.graph.storage import _ensure_schema

        _ensure_schema()

        # 1. Exact alias
        alias_row = db.execute(
            """
            SELECT a.entity_id, a.entity_type, e.canonical_name 
            FROM cognition.ontology_aliases a
            JOIN cognition.ontology_entities e ON e.id = a.entity_id
            WHERE a.alias = %s
        """,
            [text_clean],
        ).fetchone()

        if alias_row:
            return ResolvedEntity(
                entity_id=alias_row[0],
                entity_type=alias_row[1],
                canonical_name=alias_row[2],
                confidence=1.0,
                aliases_matched=[text_clean],
            )

        # 2. Ticker match
        if not entity_type_hint or entity_type_hint in [
            NodeType.COMPANY,
            NodeType.ASSET,
        ]:
            ticker_candidate = text_clean.upper()
            if re.match(r"^[A-Z]{1,5}$", ticker_candidate):
                # check company_registry
                try:
                    cr_row = db.execute(
                        "SELECT symbol, company_name FROM company_registry WHERE symbol = %s",
                        [ticker_candidate],
                    ).fetchone()
                    if cr_row:
                        canonical_name = cr_row[1] or ticker_candidate
                        entity_type = NodeType.COMPANY
                        ent_id = _generate_entity_id(entity_type, canonical_name)
                        return ResolvedEntity(
                            entity_id=ent_id,
                            entity_type=entity_type,
                            canonical_name=canonical_name,
                            confidence=0.9,
                            aliases_matched=[ticker_candidate],
                        )
                except Exception:
                    pass  # Table might not exist in tests

                # check ticker_metadata
                try:
                    tm_row = db.execute(
                        "SELECT ticker, name FROM ticker_metadata WHERE ticker = %s",
                        [ticker_candidate],
                    ).fetchone()
                    if tm_row:
                        canonical_name = tm_row[1] or ticker_candidate
                        entity_type = NodeType.COMPANY
                        ent_id = _generate_entity_id(entity_type, canonical_name)
                        return ResolvedEntity(
                            entity_id=ent_id,
                            entity_type=entity_type,
                            canonical_name=canonical_name,
                            confidence=0.9,
                            aliases_matched=[ticker_candidate],
                        )
                except Exception:
                    pass

        # 3. Fuzzy alias match
        fuzzy_row = db.execute(
            """
            SELECT a.entity_id, a.entity_type, e.canonical_name 
            FROM cognition.ontology_aliases a
            JOIN cognition.ontology_entities e ON e.id = a.entity_id
            WHERE lower(a.alias) = %s
        """,
            [text_clean.lower()],
        ).fetchone()
        if fuzzy_row:
            return ResolvedEntity(
                entity_id=fuzzy_row[0],
                entity_type=fuzzy_row[1],
                canonical_name=fuzzy_row[2],
                confidence=0.8,
                aliases_matched=[text_clean],
            )

        # 4. Fallback: Unresolved so we propose a new entity mapping
        final_type = entity_type_hint or NodeType.ASSET
        norm_name = text_clean
        if norm_name.upper() in ["GOOG", "GOOGL"]:
            norm_name = "Alphabet Inc."
            final_type = NodeType.COMPANY
        elif norm_name.lower() in ["meta", "meta platforms"]:
            norm_name = "Meta Platforms Inc."
            final_type = NodeType.COMPANY
        elif norm_name.lower() in ["nvidia", "nvda"]:
            norm_name = "NVIDIA Corp"
            final_type = NodeType.COMPANY

        ent_id = _generate_entity_id(final_type, norm_name)

        return ResolvedEntity(
            entity_id=ent_id,
            entity_type=final_type,
            canonical_name=norm_name,
            confidence=0.5,
            aliases_matched=[],
        )
