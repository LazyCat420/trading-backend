"""
Sentiment Processor — aggregate news/social signals.

Pure Python. No LLM calls. No hallucinations.
"""

from app.db.connection import get_db


def get_signals(ticker: str, max_headlines: int = 10) -> str:
    """
    Get pre-formatted sentiment signals for the LLM.
    Reads from news_articles table.
    """
    with get_db() as db:
        lines = [f"=== SENTIMENT ANALYSIS: {ticker} ==="]

        # ── Recent headlines ──
        articles = db.execute(
            """
            SELECT title, publisher, published_at, source
            FROM news_articles
            WHERE ticker = %s AND (quality_status != 'discarded' OR quality_status IS NULL)
            ORDER BY published_at DESC
            LIMIT %s
        """,
            [ticker, max_headlines],
        ).fetchall()

        if articles:
            lines.append(f"\nRecent Headlines ({len(articles)} most recent):")
            for a in articles:
                ts = str(a[2])[:16] if a[2] else "unknown"
                lines.append(f"  [{ts}] {a[1]}: {a[0]}")
        else:
            lines.append("No recent news articles found.")

        # ── News volume (articles per day, last 7 days) ──
        volume = db.execute(
            """
            SELECT CAST(published_at AS DATE) as day, COUNT(*) as cnt
            FROM news_articles
            WHERE ticker = %s
              AND published_at >= CURRENT_TIMESTAMP - INTERVAL '7 days'
              AND (quality_status != 'discarded' OR quality_status IS NULL)
            GROUP BY day
            ORDER BY day DESC
        """,
            [ticker],
        ).fetchall()

        if volume:
            total = sum(v[1] for v in volume)
            avg = total / len(volume)
            latest_count = volume[0][1] if volume else 0

            label = (
                "SPIKE"
                if latest_count > avg * 2
                else "QUIET"
                if latest_count < avg * 0.5
                else "NORMAL"
            )
            lines.append("\nNews Volume (7-day):")
            lines.append(f"  Total articles: {total}")
            lines.append(f"  Daily average: {avg:.0f}")
            lines.append(f"  Latest day: {latest_count} ({label})")

        # ── Reddit posts (if any) ──
        # NOTE: Using space-padded matching to avoid false positives while matching both title and body mentions.
        reddit = db.execute(
            """
            SELECT title, subreddit, score, comment_count, sentiment_score
            FROM reddit_posts
            WHERE ticker = %s 
               OR (title LIKE %s OR title LIKE %s OR title LIKE %s OR title LIKE %s)
               OR (body LIKE %s OR body LIKE %s OR body LIKE %s OR body LIKE %s)
            ORDER BY created_utc DESC
            LIMIT 5
        """,
            [
                ticker,
                f"% {ticker} %",
                f"%${ticker}%",
                f"{ticker} %",
                f"% {ticker}",
                f"% {ticker} %",
                f"%${ticker}%",
                f"{ticker} %",
                f"% {ticker}",
            ],
        ).fetchall()

        if reddit:
            lines.append(f"\nReddit Mentions ({len(reddit)} recent):")
            for r in reddit:
                score_label = f"↑{r[2]}" if r[2] else ""
                comments = f"{r[3]} comments" if r[3] else ""
                lines.append(f"  r/{r[1]}: {r[0]} {score_label} {comments}")

        # ── YouTube analysis (if any) ──
        # NOTE: query-level exact matching on comma lists has limitations,
        # but these explicit formatting rules prevent the worst false positives.
        youtube = db.execute(
            """
            SELECT title, channel, COALESCE(summary, '') AS summary
            FROM youtube_transcripts
            WHERE ticker = %s 
               OR tickers_mentioned = %s
               OR tickers_mentioned LIKE %s 
               OR tickers_mentioned LIKE %s
               OR tickers_mentioned LIKE %s
               OR tickers_mentioned LIKE %s
               OR tickers_mentioned LIKE %s
               OR tickers_mentioned LIKE %s
            ORDER BY published_at DESC
            LIMIT 3
        """,
            [
                ticker,
                ticker,
                f'%"{ticker}"%',
                f"{ticker},%",
                f"%, {ticker},%",
                f"%,{ticker},%",
                f"%, {ticker}",
                f"%,{ticker}",
            ],
        ).fetchall()

        if youtube:
            lines.append(f"\nYouTube Analysis ({len(youtube)} recent):")
            for yt in youtube:
                summary_text = yt[2][:200] if yt[2] else "No summary"
                lines.append(f"  [{yt[1]}] {yt[0]}")
                lines.append(f"    {summary_text}")

        return "\n".join(lines)
