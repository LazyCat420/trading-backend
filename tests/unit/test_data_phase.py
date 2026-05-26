import pytest
import asyncio
from unittest.mock import patch, MagicMock

from app.pipeline.data.data_phase import run

@pytest.mark.asyncio
async def test_data_phase_concurrency_order():
    """
    Ensure that the background curation task is scheduled 
    BEFORE the global collection scraper begins. This allows the LLM 
    processing to run concurrently with scraping from the beginning.

    Note: The janitor background task is deliberately disabled in data_phase.py
    (smart_janitor handles filtering/extraction instead).
    """
    # We will track the order of calls
    execution_order = []

    async def mock_global_collection(*args, **kwargs):
        execution_order.append("run_global_collection")
        await asyncio.sleep(0.01)

    # We need to mock create_task to see when the background tasks are started
    original_create_task = asyncio.create_task

    def mock_create_task(coro, *args, **kwargs):
        coro_name = coro.__name__ if hasattr(coro, "__name__") else str(coro)
        if "_run_curation_bg" in coro_name:
            execution_order.append("curation_task_created")
        elif "_run_janitor_bg" in coro_name:
            execution_order.append("janitor_task_created")
        return original_create_task(coro, *args, **kwargs)

    with patch("app.pipeline.data.data_global_collection.run_global_collection", new=mock_global_collection), \
         patch("app.pipeline.data.data_ticker_discovery.run_ticker_discovery_and_gates", return_value=[]), \
         patch("app.graph.sector_collector.collect_metadata"), \
         patch("app.pipeline.data.data_phase.should_collect", return_value=False), \
         patch("app.pipeline.data.data_perticker_collection.run_perticker_collection"), \
         patch("app.processors.deduplicator.deduplicate_news", return_value=0), \
         patch("app.processors.summarizer.summarize_unsummarized", return_value={}), \
         patch("app.processors.consensus_engine.run_consensus_engine", return_value={}), \
         patch("app.pipeline.database_curator.run_data_curation", return_value={}), \
         patch("app.pipeline.data.data_janitor.run_data_janitor", return_value={}), \
         patch("asyncio.create_task", side_effect=mock_create_task) as mock_ct:

        await run(tickers=["AAPL", "MSFT"], max_tickers=10)

    # Verify that the curation task was created before the global collection
    assert "curation_task_created" in execution_order
    assert "run_global_collection" in execution_order
    
    curation_idx = execution_order.index("curation_task_created")
    global_idx = execution_order.index("run_global_collection")
    
    assert curation_idx < global_idx, "Curation task should start before global collection"
    
    # Janitor task is intentionally disabled — verify it was NOT created
    assert "janitor_task_created" not in execution_order, (
        "Janitor background task should NOT be created (disabled in data_phase.py)"
    )

