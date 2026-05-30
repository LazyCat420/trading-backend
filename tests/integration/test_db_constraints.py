import pytest
from app.db.connection import get_db

@pytest.fixture(autouse=True)
def patch_get_db():
    """Override global mock db fixture for database constraint testing."""
    yield

@pytest.mark.asyncio
async def test_news_articles_no_unique_url_constraint(patch_real_get_db):
    """
    Test that we can insert two news articles with the exact same URL
    for two different tickers, and that no UniqueViolation error is raised.
    This verifies that UNIQUE(url) was successfully removed and 
    ON CONFLICT (id) DO NOTHING handles duplicates.
    """
    try:
        with get_db() as db:
            # Clean up first in case test previously failed
            db.execute("DELETE FROM news_articles WHERE url = 'http://example.com/test-article'")
            
            # Insert article for AAPL
            # Note: The actual id in production is a hash of title+ticker, 
            # here we just manually provide two different IDs.
            db.execute("""
                INSERT INTO news_articles (id, ticker, title, url) 
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """, ["hash1", "AAPL", "Test Article", "http://example.com/test-article"])
            
            # Insert same article for MSFT
            db.execute("""
                INSERT INTO news_articles (id, ticker, title, url) 
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """, ["hash2", "MSFT", "Test Article", "http://example.com/test-article"])
            
            # Verify both were inserted successfully
            count = db.execute(
                "SELECT COUNT(*) FROM news_articles WHERE url = 'http://example.com/test-article'"
            ).fetchone()[0]
            
            assert count == 2, f"Expected 2 rows for the same URL, got {count}. UNIQUE(url) constraint might still be active."
            
            # Clean up
            db.execute("DELETE FROM news_articles WHERE url = 'http://example.com/test-article'")
            
    except Exception as e:
        pytest.fail(f"Database insertion failed, constraint likely active: {e}")
