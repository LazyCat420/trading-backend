import pytest
from unittest.mock import patch, AsyncMock
import httpx

from app.services.scraper_client import scraper_client, ScraperServiceClient

@pytest.fixture
def client():
    return ScraperServiceClient("http://test-scraper:8001")

@pytest.mark.asyncio
async def test_scrape_success(client):
    mock_request = httpx.Request("POST", "http://test-scraper:8001/scrape")
    mock_response = httpx.Response(200, json={"success": True, "content": "mocked article text"}, request=mock_request)
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        
        result = await client.scrape("http://example.com", engine="playwright")
        
        assert result["content"] == "mocked article text"
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "http://test-scraper:8001/scrape"
        assert kwargs["json"]["url"] == "http://example.com"
        assert kwargs["json"]["engine"] == "playwright"

@pytest.mark.asyncio
async def test_scrape_failure(client):
    mock_request = httpx.Request("POST", "http://test-scraper:8001/scrape")
    mock_response = httpx.Response(200, json={"success": False, "error": "timeout"}, request=mock_request)
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        result = await client.scrape("http://example.com")
        assert result is None

@pytest.mark.asyncio
async def test_collect_success(client):
    mock_request = httpx.Request("POST", "http://test-scraper:8001/collect")
    mock_response = httpx.Response(200, json={
        "source": "reddit",
        "count": 2,
        "items": [{"id": "1"}, {"id": "2"}]
    }, request=mock_request)
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        
        result = await client.collect("reddit", {"subreddits": ["wallstreetbets"]})
        
        assert len(result) == 2
        assert result[0]["id"] == "1"
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "http://test-scraper:8001/collect"
        assert kwargs["json"]["source"] == "reddit"
        assert kwargs["json"]["subreddits"] == ["wallstreetbets"]

@pytest.mark.asyncio
async def test_collect_http_error(client):
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.HTTPStatusError("500", request=httpx.Request("POST", ""), response=httpx.Response(500))
        
        result = await client.collect("reddit", {"subreddits": ["wallstreetbets"]})
        assert result == []
