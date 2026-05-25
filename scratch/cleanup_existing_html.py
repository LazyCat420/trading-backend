import sys
import logging
from app.db.connection import get_db
from app.utils.text_utils import is_html, clean_html

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_migration():
    logger.info("Starting one-time news summary HTML sanitization migration...")
    
    with get_db() as db:
        rows = db.execute("""
            SELECT id, ticker, publisher, title, summary 
            FROM news_articles
            WHERE (summary LIKE '<!DOCTYPE%' OR summary LIKE '%<html%' OR summary LIKE '%<script%' OR summary LIKE '%<p>%')
        """).fetchall()
        
    total = len(rows)
    logger.info(f"Found {total} candidate articles with HTML markers.")
    
    if total == 0:
        logger.info("No HTML-marked summaries found. Exiting.")
        return
        
    cleaned_count = 0
    with get_db() as db:
        for idx, (uid, ticker, publisher, title, summary) in enumerate(rows):
            if summary and is_html(summary):
                clean_text = clean_html(summary)
                if clean_text and clean_text != summary:
                    db.execute(
                        "UPDATE news_articles SET summary = %s WHERE id = %s",
                        [clean_text, uid]
                    )
                    cleaned_count += 1
            
            if (idx + 1) % 50 == 0 or (idx + 1) == total:
                logger.info(f"Processed {idx + 1}/{total} articles... Cleaned {cleaned_count} so far.")
                
    logger.info(f"Migration completed successfully! Cleaned {cleaned_count} summaries in total.")

if __name__ == "__main__":
    run_migration()
