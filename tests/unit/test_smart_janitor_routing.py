import pytest
from unittest.mock import MagicMock, patch
import json
import datetime

from app.processors.smart_janitor import run_smart_janitor_for_article, run_smart_janitor_for_reddit

@pytest.mark.asyncio
async def test_smart_janitor_article_routing(monkeypatch, mock_db):
    """Verify that a news article gets correctly cloned to another ticker and discarded from the original ticker."""
    from contextlib import contextmanager
    
    @contextmanager
    def mock_get_db():
        yield mock_db
        
    monkeypatch.setattr("app.processors.smart_janitor.get_db", mock_get_db)
    
    # 1. Mock DB select for article
    # Columns: title, publisher, summary, url, published_at, ticker
    def mock_execute(query, *args, **kwargs):
        cursor = MagicMock()
        if "SELECT title" in query or "SELECT id" in query:
            cursor.fetchone.return_value = (
                "Adobe Reports Earnings", "Seeking Alpha", "Adobe reported strong earnings...", "http://url", datetime.datetime.now(), "HIMS"
            )
        elif "SELECT 1 FROM news_articles" in query:
            cursor.fetchone.return_value = None
        else:
            cursor.fetchone.return_value = None
        return cursor
        
    mock_db.execute.side_effect = mock_execute
    
    # 2. Mock call_prism_agent to return ADBE signal
    mock_prism_response = """
    {
      "decision": "keep",
      "actual_tickers": ["ADBE"],
      "category": "transient",
      "impact": "bullish",
      "suggested_theme": "Earnings Beat",
      "relevance_label": "Major Strategic Catalyst",
      "bullet_points": ["Adobe reported strong beat"],
      "justification": "Good fundamental growth."
    }
    """
    
    async def mock_call_prism(*args, **kwargs):
        return mock_prism_response, None, None
        
    monkeypatch.setattr("app.processors.smart_janitor.call_prism_agent", mock_call_prism)
    
    # 3. Mock company registry lookup/validate
    mock_registry = MagicMock()
    mock_registry.is_known.side_effect = lambda t: t == "ADBE"
    monkeypatch.setattr("app.processors.ticker_extractor.get_registry", lambda: mock_registry)
    
    # Mock is_banned to prevent real DB access from watchlist module
    monkeypatch.setattr("app.trading.watchlist.is_banned", lambda t: False)
    
    # Mock _get_article_id to prevent news_collector DB access
    monkeypatch.setattr("app.collectors.news_collector._get_article_id", lambda title, ticker: f"janitor_{ticker}_{hash(title)}")
    
    # Run smart janitor
    result = await run_smart_janitor_for_article("art_123")
    assert result is True
    
    # 4. Verify DB calls
    # Should have updated the original article to discard/re-route
    # Should have inserted the cloned article for ADBE
    db_calls = mock_db.execute.call_args_list
    
    inserts = [c for c in db_calls if "INSERT INTO news_articles" in c[0][0]]
    assert len(inserts) == 1
    insert_args = inserts[0][0][1]
    assert insert_args[1] == "ADBE" # Ticker should be ADBE
    
    updates = [c for c in db_calls if "UPDATE news_articles" in c[0][0]]
    assert len(updates) == 1
    update_args = updates[0][0][1]
    updated_draft = json.loads(update_args[0])
    assert updated_draft["decision"] == "discard"
    assert "Re-routed to: ADBE" in updated_draft["justification"]


@pytest.mark.asyncio
async def test_smart_janitor_reddit_routing(monkeypatch, mock_db):
    """Verify that a Reddit post gets cloned to another ticker and discarded from original ticker."""
    from contextlib import contextmanager
    
    @contextmanager
    def mock_get_db():
        yield mock_db
        
    monkeypatch.setattr("app.processors.smart_janitor.get_db", mock_get_db)
    
    # Columns in SELECT: title, subreddit, body, created_utc, ticker, score, upvote_ratio, comment_count, flair, sentiment_score, award_count, comment_velocity
    def mock_execute(query, *args, **kwargs):
        cursor = MagicMock()
        if "SELECT title" in query or "SELECT id" in query:
            cursor.fetchone.return_value = (
                "Adobe DD post", "stocks", "Adobe is a buy...", datetime.datetime.now(), "HIMS",
                100, 0.95, 20, "DD", 0.8, 2, 1.5
            )
        elif "SELECT 1 FROM reddit_posts" in query:
            cursor.fetchone.return_value = None
        else:
            cursor.fetchone.return_value = None
        return cursor
        
    mock_db.execute.side_effect = mock_execute
    
    mock_prism_response = """
    {
      "decision": "keep",
      "actual_tickers": ["ADBE"],
      "category": "transient",
      "impact": "bullish",
      "suggested_theme": "Earnings DD",
      "relevance_label": "Major Strategic Catalyst",
      "bullet_points": ["Adobe looking solid"],
      "justification": "Fundamentals look good."
    }
    """
    
    async def mock_call_prism(*args, **kwargs):
        return mock_prism_response, None, None
        
    monkeypatch.setattr("app.processors.smart_janitor.call_prism_agent", mock_call_prism)
    
    mock_registry = MagicMock()
    mock_registry.is_known.side_effect = lambda t: t == "ADBE"
    monkeypatch.setattr("app.processors.ticker_extractor.get_registry", lambda: mock_registry)
    
    # Mock is_banned to prevent real DB access from watchlist module
    monkeypatch.setattr("app.trading.watchlist.is_banned", lambda t: False)
    
    # Run smart janitor
    result = await run_smart_janitor_for_reddit("post_123_HIMS")
    assert result is True
    
    db_calls = mock_db.execute.call_args_list
    
    inserts = [c for c in db_calls if "INSERT INTO reddit_posts" in c[0][0]]
    assert len(inserts) == 1
    insert_args = inserts[0][0][1]
    assert insert_args[0] == "post_123_ADBE" # New ID
    assert insert_args[1] == "ADBE" # Ticker
    
    updates = [c for c in db_calls if "UPDATE reddit_posts" in c[0][0]]
    assert len(updates) == 1
    update_args = updates[0][0][1]
    updated_draft = json.loads(update_args[0])
    assert updated_draft["decision"] == "discard"
    assert "Re-routed to: ADBE" in updated_draft["justification"]
