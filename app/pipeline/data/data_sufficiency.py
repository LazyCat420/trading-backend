import logging
from app.db.connection import get_db

logger = logging.getLogger(__name__)

def check_data_sufficiency(ticker: str, hours: int = 48, threshold: int = 5) -> bool:
    """
    Checks if we have enough high-quality, summarized data for this ticker
    in the past `hours` hours to make a decision without further scraping.
    """
    try:
        with get_db() as db:
            # Count accepted, summarized news articles
            news_count = db.execute(
                """
                SELECT COUNT(*) FROM news_articles
                WHERE ticker = %s
                  AND quality_status = 'accepted'
                  AND summarized_at IS NOT NULL
                  AND published_at >= NOW() - INTERVAL '%s hours'
                """,
                [ticker, hours],
            ).fetchone()[0]
            
            # We could also count reddit/youtube, but let's just sum them
            reddit_count = db.execute(
                """
                SELECT COUNT(*) FROM reddit_posts
                WHERE ticker = %s
                  AND quality_status = 'accepted'
                  AND summarized_at IS NOT NULL
                  AND created_utc >= NOW() - INTERVAL '%s hours'
                """,
                [ticker, hours],
            ).fetchone()[0]
            
            total_quality_datapoints = news_count + reddit_count
            
            if total_quality_datapoints >= threshold:
                logger.info(f"[PIPELINE]   [Sufficiency] {ticker} has {total_quality_datapoints} high-quality recent datapoints. SUFFICIENT.")
                return True
                
            return False
    except Exception as e:
        logger.error(f"[PIPELINE]   [Sufficiency] Error checking sufficiency for {ticker}: {e}")
        return False
