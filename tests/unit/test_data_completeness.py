import pytest
from unittest.mock import patch, MagicMock

# We need to mock the external components before importing data_completeness
# because it does some heavy lifting on import if not careful.

@pytest.mark.asyncio
@patch("app.pipeline.data.collection_scheduler.record_collection")
@patch("app.pipeline.data.collection_scheduler.should_collect")
@patch("app.collectors.youtube_collector.collect_for_ticker")
@patch("app.collectors.reddit_collector.collect_for_ticker")
@patch("app.collectors.news_collector.collect_for_ticker")
@patch("app.collectors.yfinance_collector.collect_fundamentals")
@patch("app.collectors.yfinance_collector.collect_price_history")
@patch("app.processors.technical_processor.compute_technicals")
@patch("app.pipeline.data.data_completeness.get_db")
async def test_data_completeness_record_collection(
    mock_get_db,
    mock_compute_tech,
    mock_price_collect,
    mock_fund_collect,
    mock_news_collect,
    mock_reddit_collect,
    mock_yt_collect,
    mock_should_collect,
    mock_record_collection
):
    from app.pipeline.data.data_completeness import check_and_fill
    
    # Setup mocks
    # should_collect returns True to allow JIT fetch
    mock_should_collect.return_value = True
    
    # Mocks return 5 items each
    mock_news_collect.return_value = 5
    mock_reddit_collect.return_value = 5
    mock_yt_collect.return_value = {"stored": 5}
    mock_fund_collect.return_value = 5
    mock_price_collect.return_value = 50
    mock_compute_tech.return_value = 50
    
    # Mock DB counts to 0 to trigger everything
    mock_db = MagicMock()
    mock_get_db.return_value.__enter__.return_value = mock_db
    
    # DB calls:
    # 1. tech count
    # 2. fund count
    # 3. news count/stale
    # 4. reddit count/stale
    # 5. congress count
    # 6. yt count/stale
    mock_db.execute.return_value.fetchone.return_value = (0, True)  # 0 rows, stale=True
    
    # Execute the check_and_fill function for a ticker
    report = await check_and_fill("TEST", enqueue_only=False)
    
    # Assert that record_collection was called for the expected sources
    assert mock_record_collection.call_count >= 4
    
    # Collect all arguments passed to record_collection
    called_sources = []
    for call in mock_record_collection.call_args_list:
        args, kwargs = call
        source, ticker = args[0], args[1]
        rows = kwargs.get("rows", 0)
        called_sources.append((source, ticker, rows))
        
    assert ("news_finnhub", "TEST", 5) in called_sources
    assert ("news_yfinance", "TEST", 5) in called_sources
    assert ("reddit", "TEST", 5) in called_sources
    assert ("youtube", "TEST", 5) in called_sources
    
    from app.pipeline.data.data_completeness import check_data_sufficiency
    sufficiency_result = check_data_sufficiency(report)
    assert sufficiency_result["sufficient"] is True
