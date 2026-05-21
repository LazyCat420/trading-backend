import pytest
from unittest.mock import patch, AsyncMock

@pytest.mark.asyncio
async def test_flash_briefing_generation(real_db):
    """Verify that a flash briefing is generated and saved in the real database."""
    from contextlib import contextmanager

    @contextmanager
    def fake_get_db():
        yield real_db

    import datetime
    now_dt = datetime.datetime.now(datetime.UTC)
    
    # Insert dummy news articles into the real DB
    real_db.execute(
        "INSERT INTO news_articles (id, ticker, title, publisher, url, published_at, summary) "
        "VALUES ('news-1', 'AAPL', 'News Title 1', 'Publisher A', 'http://example.com/1', %s, 'Summary 1')",
        [now_dt]
    )
    real_db.execute(
        "INSERT INTO news_articles (id, ticker, title, publisher, url, published_at, summary) "
        "VALUES ('news-2', 'MSFT', 'News Title 2', 'Publisher B', 'http://example.com/2', %s, 'Summary 2')",
        [now_dt]
    )
    real_db.execute(
        "INSERT INTO news_articles (id, ticker, title, publisher, url, published_at, summary) "
        "VALUES ('news-3', 'TSLA', 'News Title 3', 'Publisher C', 'http://example.com/3', %s, 'Summary 3')",
        [now_dt]
    )

    with patch("app.services.flash_briefing.get_db", fake_get_db), \
         patch("app.services.flash_briefing.llm.chat", AsyncMock(return_value=("Flash briefing content", 100, 100))):

        from app.services.flash_briefing import generate_flash_briefing
        await generate_flash_briefing()

    # Verify insert query in the real DB
    briefings = real_db.execute("SELECT report_content, source_urls, article_count FROM flash_briefings").fetchall()
    assert len(briefings) == 1
    b = briefings[0]
    assert b[0] == "Flash briefing content"
    assert "http://example.com/1" in b[1]
    assert b[2] == 3 # article_count

@pytest.mark.asyncio
async def test_morning_briefing_generation(real_db):
    """Verify that a morning briefing is generated and saved in the real database."""
    from contextlib import contextmanager

    @contextmanager
    def fake_get_db():
        yield real_db

    with patch("app.pipeline.analysis.morning_briefing.get_db", new=fake_get_db), \
         patch("app.pipeline.analysis.morning_briefing.get_current_state", return_value={"positions": [{"ticker": "AAPL"}]}), \
         patch("app.pipeline.analysis.morning_briefing.get_active", return_value=[{"ticker": "TSLA"}]), \
         patch("app.pipeline.analysis.morning_briefing.get_thesis") as mock_thesis, \
         patch("app.pipeline.analysis.morning_briefing.llm.chat", new_callable=AsyncMock) as mock_llm:
         
        from dataclasses import dataclass
        import datetime
        @dataclass
        class FakeThesis:
            verdict: str
            confidence: int
            updated_at: datetime.datetime
            summary: str

        mock_thesis.return_value = FakeThesis("BUY", 80, datetime.datetime.now(datetime.UTC), "Mock")
        mock_llm.return_value = ("Morning briefing content", 100, 100)

        from app.pipeline.analysis.morning_briefing import generate_morning_briefing
        await generate_morning_briefing()

        # Verify insert query in the real DB
        briefings = real_db.execute("SELECT report_content, tickers_evaluated FROM morning_briefings").fetchall()
        assert len(briefings) == 1
        b = briefings[0]
        assert b[0] == "Morning briefing content"
        import json
        tickers = json.loads(b[1])
        assert sorted(tickers) == sorted(["AAPL", "TSLA"])
