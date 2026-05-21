import pytest
from unittest.mock import patch, MagicMock
import datetime

from app.collectors.finnhub_collector import (
    collect_news,
    collect_analyst_targets,
    collect_earnings_calendar,
    collect_recommendation_trends,
    collect_all,
)


@pytest.fixture
def mock_db():
    with patch("app.collectors.finnhub_collector.get_db") as mock_get_db:
        db = MagicMock()
        mock_get_db.return_value.__enter__.return_value = db
        yield db


@pytest.fixture
def mock_client():
    client = MagicMock()
    return client


@pytest.mark.asyncio
@patch("app.collectors.news_collector.collect_finnhub_news")
async def test_collect_news_success(mock_collect_finnhub, mock_db):
    mock_collect_finnhub.return_value = 1
    
    count = await collect_news("AAPL", days_back=7)
    
    assert count == 1
    mock_collect_finnhub.assert_called_once_with("AAPL", days=7)

@pytest.mark.asyncio
@patch("app.collectors.news_collector.collect_finnhub_news")
async def test_collect_news_api_error(mock_collect_finnhub, mock_db):
    mock_collect_finnhub.side_effect = Exception("API rate limit")
    
    # Should handle error gracefully and return 0
    count = await collect_news("AAPL")
    assert count == 0


@pytest.mark.asyncio
@patch("app.collectors.finnhub_collector._get_client")
async def test_collect_analyst_targets_success(mock_get_client, mock_client):
    mock_get_client.return_value = mock_client
    mock_client.price_target.return_value = {"targetHigh": 200, "targetLow": 100, "targetMean": 150}
    
    result = await collect_analyst_targets("AAPL")
    assert result is True


@pytest.mark.asyncio
@patch("app.collectors.finnhub_collector._get_client")
async def test_collect_analyst_targets_no_data(mock_get_client, mock_client):
    mock_get_client.return_value = mock_client
    mock_client.price_target.return_value = {}  # Missing targetHigh
    
    result = await collect_analyst_targets("AAPL")
    assert result is False


@pytest.mark.asyncio
@patch("app.collectors.finnhub_collector._get_client")
async def test_collect_earnings_calendar(mock_get_client, mock_client):
    mock_get_client.return_value = mock_client
    mock_client.earnings_calendar.return_value = {"earningsCalendar": [{"date": "2023-11-01"}]}
    
    events = await collect_earnings_calendar("AAPL")
    assert len(events) == 1
    assert events[0]["date"] == "2023-11-01"


@pytest.mark.asyncio
@patch("app.collectors.finnhub_collector._get_client")
async def test_collect_recommendation_trends(mock_get_client, mock_client):
    mock_get_client.return_value = mock_client
    mock_client.recommendation_trends.return_value = [{"buy": 10, "hold": 5, "sell": 1}]
    
    trends = await collect_recommendation_trends("AAPL")
    assert len(trends) == 1
    assert trends[0]["buy"] == 10


@pytest.mark.asyncio
@patch("app.collectors.finnhub_collector.collect_recommendation_trends")
@patch("app.collectors.finnhub_collector.collect_earnings_calendar")
@patch("app.collectors.finnhub_collector.collect_analyst_targets")
@patch("app.collectors.finnhub_collector.collect_news")
async def test_collect_all(mock_news, mock_targets, mock_earnings, mock_trends):
    mock_news.return_value = 5
    mock_targets.return_value = True
    mock_earnings.return_value = [1, 2]
    mock_trends.return_value = [1]
    
    result = await collect_all("AAPL")
    assert result["ticker"] == "AAPL"
    assert result["news_articles"] == 5
    assert result["analyst_targets"] is True
    assert result["earnings_events"] == 2
    assert result["recommendation_snapshots"] == 1
