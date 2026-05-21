import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from app.collectors.adaptive_scraper import (
    extract_domain,
    validate_script,
    get_script,
    save_script,
    report_success,
    report_failure,
    run_adaptive,
)


def test_extract_domain():
    assert extract_domain("https://www.example.com/article") == "example.com"
    assert extract_domain("http://sub.domain.co.uk/path") == "sub.domain.co.uk"
    assert extract_domain("example.com") == "example.com"
    assert extract_domain("") == ""


@pytest.mark.parametrize("script, expected", [
    ("(() => { return document.querySelector('article').innerText; })();", True),
    ("const fetchedData = 'hello'; return fetchedData;", True),
    ("const prefetchData = {};", True),
    ("fetch('http://evil.com')", False),
    ("window.fetch('http://evil.com')", False),
    ("const req = new XMLHttpRequest();", False),
    ("eval('alert(1)')", False),
    ("new Function('a', 'b', 'return a + b')", False),
    ("process.env.SECRET", False),
    ("require('fs')", False),
    ("import('path')", False),
    ("new WebSocket('ws://test')", False),
])
def test_validate_script(script, expected):
    assert validate_script(script) is expected


@pytest.mark.parametrize("db_result, expected_script", [
    (None, None),
    (("test script", "active"), "test script"),
    (("test script", "failed"), None),
])
@patch("app.collectors.adaptive_scraper.get_db")
def test_get_script_states(mock_get_db, db_result, expected_script):
    mock_db = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = db_result
    mock_db.execute.return_value = mock_cursor
    
    # Context manager setup
    mock_get_db.return_value.__enter__.return_value = mock_db
    
    assert get_script("example.com") == expected_script


@patch("app.collectors.adaptive_scraper.get_db")
def test_report_success(mock_get_db):
    mock_db = MagicMock()
    mock_get_db.return_value.__enter__.return_value = mock_db
    
    report_success("example.com")
    
    # Check that update was called correctly
    mock_db.execute.assert_called_once()
    query, params = mock_db.execute.call_args[0]
    assert "UPDATE scraper_scripts" in query
    assert "success_count = success_count + 1" in query
    assert "example.com" in params


@patch("app.collectors.adaptive_scraper.get_db")
def test_report_failure(mock_get_db):
    mock_db = MagicMock()
    # Mock finding the row to check fail_count after update
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (5,) # Mock returned fail count
    mock_db.execute.return_value = mock_cursor
    
    mock_get_db.return_value.__enter__.return_value = mock_db
    
    report_failure("example.com")
    
    assert mock_db.execute.call_count == 2  # UPDATE RETURNING, UPDATE status
    
    # Check that status was updated to failed because fail_count >= 5
    last_query, last_params = mock_db.execute.call_args_list[-1][0]
    assert "UPDATE scraper_scripts SET status = 'failed'" in last_query
    assert "example.com" in last_params


@patch("app.services.scraper_client.scraper_client.scrape", new_callable=AsyncMock)
@patch("app.collectors.adaptive_scraper.generate_script")
@patch("app.collectors.adaptive_scraper.get_script")
@pytest.mark.asyncio
async def test_run_adaptive_flow_existing_script(mock_get_script, mock_generate, mock_scrape):
    mock_get_script.return_value = "return 'hello';"
    long_text = "hello " * 20
    mock_scrape.return_value = {"success": True, "content": long_text, "screenshot_b64": None}
    
    with patch("app.collectors.adaptive_scraper.report_success") as mock_rep_success:
        result = await run_adaptive("http://example.com/article")
        assert result == long_text
        mock_rep_success.assert_called_once_with("example.com")
        mock_generate.assert_not_called()


@patch("app.collectors.adaptive_scraper.save_script")
@patch("app.services.scraper_client.scraper_client.scrape", new_callable=AsyncMock)
@patch("app.collectors.adaptive_scraper.generate_script")
@patch("app.collectors.adaptive_scraper.get_script")
@pytest.mark.asyncio
async def test_run_adaptive_flow_generate(mock_get_script, mock_generate, mock_scrape, mock_save):
    mock_get_script.return_value = None
    mock_generate.return_value = "return 'new text';"
    
    long_text = "new text " * 20
    # First scrape_url for screenshot, second for running the script
    mock_scrape.side_effect = [
        {"success": True, "content": "", "screenshot_b64": "b64string"},
        {"success": True, "content": long_text, "screenshot_b64": None}
    ]
    
    with patch("app.collectors.adaptive_scraper.report_success") as mock_rep_success:
        result = await run_adaptive("http://example.com/article")
        assert result == long_text
        mock_generate.assert_called_once_with("example.com", "b64string", previous_script=None)
        mock_save.assert_called_once_with("example.com", "return 'new text';")
        mock_rep_success.assert_called_once_with("example.com")


@patch("app.services.scraper_client.scraper_client.scrape", new_callable=AsyncMock)
@patch("app.collectors.adaptive_scraper.generate_script")
@patch("app.collectors.adaptive_scraper.get_script")
@pytest.mark.asyncio
async def test_run_adaptive_flow_empty_response(mock_get_script, mock_generate, mock_scrape):
    mock_get_script.return_value = None
    mock_generate.return_value = None  # LLM returns None
    
    mock_scrape.return_value = {"success": True, "content": "", "screenshot_b64": "b64string"}
    
    result = await run_adaptive("http://example.com/article")
    assert result is None


@patch("app.collectors.adaptive_scraper.save_script")
@patch("app.services.scraper_client.scraper_client.scrape", new_callable=AsyncMock)
@patch("app.collectors.adaptive_scraper.generate_script")
@patch("app.collectors.adaptive_scraper.get_script")
@pytest.mark.asyncio
async def test_run_adaptive_flow_retry_loop(mock_get_script, mock_generate, mock_scrape, mock_save):
    mock_get_script.return_value = None
    
    # LLM returns invalid script twice, then valid script
    mock_generate.side_effect = [
        "fetch('bad')",
        "eval('bad')",
        "return 'good text';"
    ]
    
    long_text = "good text " * 20
    mock_scrape.side_effect = [
        {"success": True, "content": "", "screenshot_b64": "b64string"}, # Screenshot
        {"success": True, "content": long_text, "screenshot_b64": None}  # Final success run
    ]
    
    with patch("app.collectors.adaptive_scraper.report_success") as mock_rep_success:
        result = await run_adaptive("http://example.com/article")
        assert result == long_text
        assert mock_generate.call_count == 3
        mock_save.assert_called_once_with("example.com", "return 'good text';")
        mock_rep_success.assert_called_once_with("example.com")

@patch("httpx.AsyncClient.post")
@pytest.mark.asyncio
async def test_generate_script_strips_markdown(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "```javascript\nreturn 'test';\n```"}}]
    }
    mock_post.return_value = mock_response
    
    from app.collectors.adaptive_scraper import generate_script
    script = await generate_script("example.com", "fakeb64")
    assert script == "return 'test';"
    
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "```\nreturn 'test2';\n```"}}]
    }
    script2 = await generate_script("example.com", "fakeb64")
    assert script2 == "return 'test2';"
