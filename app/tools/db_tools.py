"""
Database Tools -- Exposes internal vector search capabilities to LLM agents.
"""

from typing import Dict, Any
from app.tools.registry import registry
from app.db.vector_store import vector_store


@registry.register(
    name="search_internal_database",
    description="Perform semantic search across all previously scraped news, reddit, and youtube transcripts in the internal database to find specific information.",
    tier=1,
    source="internal_db",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The exact topic or question to search for (e.g. 'debt restructuring details' or 'CEO resignation reasons').",
            },
            "ticker": {
                "type": "string",
                "description": "Optional ticker to restrict the search to a specific stock. Leave empty for macro-economic queries.",
            },
        },
        "required": ["query"],
    },
)
async def search_internal_database(query: str, ticker: str = None) -> Dict[str, Any]:
    """Search internal vector database for relevant snippets."""
    try:
        from app.services.embedding_service import embedder

        # Embed query with BAAI instruction prefix for better retrieval
        query_vec = embedder.embed_text(
            query, prefix="Represent this sentence for searching relevant passages: "
        )

        # Use existing search_cosine from vector_store
        results = vector_store.search_cosine(
            query_embedding=query_vec, ticker=ticker, top_k=5
        )

        if not results:
            return {
                "status": "no_results",
                "message": "No relevant snippets found in internal database.",
            }

        formatted_results = []
        for r in results:
            source = r.get("source_table", "unknown")
            snippet = r.get("content_preview", "")
            score = r.get("score", 0)
            formatted_results.append(f"[{source}] (Relevance: {score:.2f}) {snippet}")

        return {"status": "success", "results": formatted_results}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@registry.register(
    name="update_youtube_channel_handle",
    description="Update the handle for a broken YouTube channel in the PostgreSQL database.",
    tier=1,
    source="internal_db",
    parameters={
        "type": "object",
        "properties": {
            "old_handle": {
                "type": "string",
                "description": "The current, broken handle in the database (e.g. 'Bloomberg' or 'FundstratTomLee').",
            },
            "new_handle": {
                "type": "string",
                "description": "The new, verified working handle (e.g. 'markets' or 'Fundstrat_Direct').",
            },
        },
        "required": ["old_handle", "new_handle"],
    },
)
async def update_youtube_channel_handle(
    old_handle: str, new_handle: str
) -> Dict[str, Any]:
    """Update a broken YouTube channel handle in the database."""
    try:
        from app.db.connection import get_db
        import logging

        logger = logging.getLogger(__name__)
        logger.info(
            f"[DBTools] Updating YouTube channel from {old_handle} to {new_handle}"
        )

        with get_db() as db:
            db.execute(
                "UPDATE youtube_channels SET channel_handle=%s WHERE channel_handle=%s",
                (new_handle, old_handle),
            )
            # PooledCursor auto-commits, but we can check if any rows were affected
            affected = db._cursor.rowcount

        if affected > 0:
            return {
                "status": "success",
                "message": f"Successfully updated handle to {new_handle} ({affected} rows affected).",
            }
        else:
            return {
                "status": "error",
                "message": f"Handle '{old_handle}' not found in the database.",
            }

    except Exception as e:
        return {"status": "error", "message": str(e)}
