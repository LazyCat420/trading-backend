import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import datetime
import pandas as pd

from app.collectors.yfinance_collector import (
    collect_price_history,
    collect_fundamentals,
    collect_financials,
    collect_balance_sheet,
    collect_news,
    collect_all,
)

@pytest.fixture
def mock_db():
    with patch("app.collectors.yfinance_collector.get_db") as mock_get_db_yf:
        with patch("app.collectors.news_collector.get_db") as mock_get_db_news:
            db = MagicMock()
            mock_get_db_yf.return_value.__enter__.return_value = db
            mock_get_db_news.return_value.__enter__.return_value = db
            yield db

@pytest.mark.asyncio
@patch("app.collectors.yfinance_collector.yf.Ticker")
async def test_collect_price_history_success(mock_ticker, mock_db):
    
    # Mock DataFrame
    df = pd.DataFrame({
        "Open": [100.0, 101.0],
        "High": [105.0, 106.0],
        "Low": [95.0, 96.0],
        "Close": [102.0, 103.0],
        "Volume": [1000, 2000]
    }, index=pd.to_datetime(["2023-01-01", "2023-01-02"]))
    
    mock_ticker_inst = MagicMock()
    mock_ticker_inst.history.return_value = df
    mock_ticker.return_value = mock_ticker_inst
    
    count = await collect_price_history("AAPL")
    
    assert count == 2
    mock_db.executemany.assert_called_once()

@pytest.mark.asyncio
@patch("app.collectors.yfinance_collector.yf.Ticker")
async def test_collect_price_history_empty(mock_ticker):
    mock_ticker_inst = MagicMock()
    mock_ticker_inst.history.return_value = pd.DataFrame()
    mock_ticker.return_value = mock_ticker_inst
    
    count = await collect_price_history("AAPL")
    assert count == 0

@pytest.mark.asyncio
@patch("app.collectors.yfinance_collector.yf.Ticker")
async def test_collect_fundamentals_success(mock_ticker, mock_db):
    mock_ticker_inst = MagicMock()
    mock_ticker_inst.info = {"symbol": "AAPL", "marketCap": 1000000}
    mock_ticker.return_value = mock_ticker_inst
    
    result = await collect_fundamentals("AAPL")
    
    assert result is True
    mock_db.execute.assert_called_once()

@pytest.mark.asyncio
@patch("app.collectors.yfinance_collector.yf.Ticker")
async def test_collect_fundamentals_missing_data(mock_ticker):
    mock_ticker_inst = MagicMock()
    mock_ticker_inst.info = {}
    mock_ticker.return_value = mock_ticker_inst
    
    result = await collect_fundamentals("AAPL")
    assert result is False

@pytest.mark.asyncio
@patch("app.collectors.news_collector._scrape_article_body_via_service", new_callable=AsyncMock)
@patch("app.collectors.yfinance_collector.yf.Ticker")
async def test_collect_news_success(mock_ticker, mock_scrape, mock_db):
    
    # Mock bad publishers
    mock_db.execute.return_value.fetchall.return_value = []
    mock_scrape.return_value = "A" * 200
    
    mock_ticker_inst = MagicMock()
    mock_ticker_inst.news = [
        {"content": {"title": "Test 1", "canonicalUrl": {"url": "http://1"}, "provider": {"displayName": "Provider 1"}, "pubDate": "2023-10-01T12:00:00Z"}},
        {"content": {"title": "Test 2", "clickThroughUrl": {"url": "http://2"}}},
    ]
    mock_ticker.return_value = mock_ticker_inst
    
    count = await collect_news("AAPL")
    
    assert count == 2
    mock_db.execute.assert_called()

@pytest.mark.asyncio
@patch("app.collectors.news_collector._scrape_article_body_via_service", new_callable=AsyncMock)
@patch("app.collectors.yfinance_collector.yf.Ticker")
async def test_collect_news_bad_publisher(mock_ticker, mock_scrape, mock_db):
    
    # Mock bad publishers (win_rate < 0.1, total_items >= 5)
    mock_db.execute.return_value.fetchall.return_value = [("Bad Provider", 0.0, 10)]
    mock_scrape.return_value = "A" * 200
    
    mock_ticker_inst = MagicMock()
    mock_ticker_inst.news = [
        {"content": {"title": "Test 1", "canonicalUrl": {"url": "http://1"}, "provider": {"displayName": "Bad Provider"}}},
        {"content": {"title": "Test 2", "canonicalUrl": {"url": "http://2"}, "provider": {"displayName": "Good Provider"}}},
    ]
    mock_ticker.return_value = mock_ticker_inst
    
    count = await collect_news("AAPL")
    
    # Should only insert the good provider
    assert count == 1

@pytest.mark.asyncio
@patch("app.collectors.yfinance_collector.collect_balance_sheet")
@patch("app.collectors.yfinance_collector.collect_financials")
@patch("app.collectors.yfinance_collector.collect_fundamentals")
@patch("app.collectors.yfinance_collector.collect_price_history")
async def test_collect_all(mock_price, mock_fundamentals, mock_financials, mock_balance):
    mock_price.return_value = 10
    mock_fundamentals.return_value = True
    mock_financials.return_value = 4
    mock_balance.return_value = 2
    
    result = await collect_all("AAPL")
    
    assert result["ticker"] == "AAPL"
    assert result["price_rows"] == 10
    assert result["fundamentals"] is True
    assert result["financial_rows"] == 4
    assert result["balance_rows"] == 2
