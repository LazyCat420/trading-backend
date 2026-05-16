import asyncio
from app.db.connection import get_db

def _sync_check_content(ticker: str) -> bool:
    """
    Synchronous DB check for existing content.
    Returns True if any news, reddit, or youtube content exists for the ticker.
    """
    with get_db() as db:
        # Check news
        db.execute("SELECT 1 FROM news_articles WHERE ticker = %s LIMIT 1", (ticker,))
        if db.fetchone():
            return True
        
        # Check reddit
        db.execute("SELECT 1 FROM reddit_posts WHERE ticker = %s LIMIT 1", (ticker,))
        if db.fetchone():
            return True
            
        # Check youtube
        db.execute("SELECT 1 FROM youtube_transcripts WHERE ticker = %s LIMIT 1", (ticker,))
        if db.fetchone():
            return True
            
    return False

async def check_content(ticker: str) -> bool:
    """
    Checks if there is any content for the ticker in the database.
    Returns a boolean indicating if content was found.
    """
    return await asyncio.to_thread(_sync_check_content, ticker)
