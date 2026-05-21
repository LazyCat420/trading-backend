"""
News Deduplicator — Cleans up duplicate news articles in the database.

Removes articles with duplicate titles to improve LLM context quality.
"""

import re
import datetime
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.db.connection import get_db

def _score_article(title: str, summary: str) -> int:
    """Score an article based on content density and quantitative data."""
    text = f"{title} {summary}"
    score = len(text)
    # Give bonus points for numerical data (digits)
    score += len(re.findall(r"\d", text)) * 10
    # Bonus for money/percentages
    score += len(re.findall(r"[$%]", text)) * 50
    return score

def deduplicate_news(ticker: str | None = None) -> tuple[int, list[str]]:
    """
    Remove duplicate news articles from the database.
    Uses TF-IDF semantic clustering and a QualityScorer.
    Updates source_trust leaderboard based on winners.
    Returns a tuple: (number of duplicates removed, list of tickers with high redundancy).
    """
    with get_db() as db:
        if ticker:
            ticker = ticker.upper()
            before = db.execute("SELECT COUNT(*) FROM news_articles WHERE ticker = %s", [ticker]).fetchone()[0]
        else:
            before = db.execute("SELECT COUNT(*) FROM news_articles").fetchone()[0]

        # 1. Exact title dedup first (fastest, cheapest)
        if ticker:
            db.execute("""
                DELETE FROM news_articles 
                WHERE ticker = %s AND id NOT IN (
                    SELECT id FROM (
                        SELECT id,
                        ROW_NUMBER() OVER(PARTITION BY ticker, LOWER(TRIM(title)) ORDER BY published_at DESC) as rn
                        FROM news_articles
                        WHERE ticker = %s
                    ) t WHERE t.rn = 1
                )
            """, [ticker, ticker])
        else:
            db.execute("""
                DELETE FROM news_articles 
                WHERE id NOT IN (
                    SELECT id FROM (
                        SELECT id,
                        ROW_NUMBER() OVER(PARTITION BY ticker, LOWER(TRIM(title)) ORDER BY published_at DESC) as rn
                        FROM news_articles
                    ) t WHERE t.rn = 1
                )
            """)

        # 2. Semantic Clustering & Quality Assessment
        # We only want to cluster recent articles to avoid cross-year mismatches, but let's just group by ticker.
        # To avoid massive memory, let's fetch articles from the last 14 days.
        fourteen_days_ago = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=14)
        
        if ticker:
            rows = db.execute("""
                SELECT id, ticker, title, summary, publisher 
                FROM news_articles 
                WHERE published_at >= %s AND ticker = %s
            """, [fourteen_days_ago, ticker]).fetchall()
        else:
            rows = db.execute("""
                SELECT id, ticker, title, summary, publisher 
                FROM news_articles 
                WHERE published_at >= %s
            """, [fourteen_days_ago]).fetchall()

        if not rows:
            if ticker:
                after = db.execute("SELECT COUNT(*) FROM news_articles WHERE ticker = %s", [ticker]).fetchone()[0]
            else:
                after = db.execute("SELECT COUNT(*) FROM news_articles").fetchone()[0]
            return before - after, []

        ticker_groups = defaultdict(list)
        for r in rows:
            ticker_groups[r[1]].append({
                "id": r[0],
                "ticker": r[1],
                "title": r[2] or "",
                "summary": r[3] or "",
                "publisher": r[4] or "unknown",
                "score": _score_article(r[2] or "", r[3] or "")
            })

        to_delete = []
        publisher_stats = defaultdict(lambda: {"total": 0, "wins": 0})
        high_redundancy_tickers = set()

        for ticker_key, articles in ticker_groups.items():
            if len(articles) < 2:
                for a in articles:
                    publisher_stats[a["publisher"]]["total"] += 1
                    publisher_stats[a["publisher"]]["wins"] += 1
                continue

            # Prepare texts for TF-IDF
            texts = [f"{a['title']} {a['summary']}" for a in articles]
            vectorizer = TfidfVectorizer(stop_words='english')
            try:
                tfidf_matrix = vectorizer.fit_transform(texts)
                sim_matrix = cosine_similarity(tfidf_matrix)
            except ValueError:
                # E.g. empty vocab
                for a in articles:
                    publisher_stats[a["publisher"]]["total"] += 1
                    publisher_stats[a["publisher"]]["wins"] += 1
                continue

            processed_indices = set()
            for i in range(len(articles)):
                if i in processed_indices:
                    continue
                
                # Find cluster
                cluster_indices = [i]
                for j in range(i + 1, len(articles)):
                    if j not in processed_indices and sim_matrix[i, j] >= 0.55: # Semantic similarity threshold
                        cluster_indices.append(j)

                # Track total seen for all publishers in this cluster
                for idx in cluster_indices:
                    pub = articles[idx]["publisher"]
                    publisher_stats[pub]["total"] += 1
                    processed_indices.add(idx)

                if len(cluster_indices) > 1:
                    if len(cluster_indices) >= 5:
                        high_redundancy_tickers.add(ticker_key)
                        
                    # Pick winner
                    best_idx = max(cluster_indices, key=lambda idx: articles[idx]["score"])
                    pub_winner = articles[best_idx]["publisher"]
                    publisher_stats[pub_winner]["wins"] += 1
                    
                    # Mark losers for deletion
                    for idx in cluster_indices:
                        if idx != best_idx:
                            to_delete.append(articles[idx]["id"])
                else:
                    # Lone article is its own winner
                    pub = articles[i]["publisher"]
                    publisher_stats[pub]["wins"] += 1

        # 3. Delete losers
        if to_delete:
            chunk_size = 100
            for i in range(0, len(to_delete), chunk_size):
                chunk = to_delete[i:i+chunk_size]
                db.execute("DELETE FROM news_articles WHERE id = ANY(%s)", [chunk])

        # 4. Update Source Reputation Leaderboard
        for pub, stats in publisher_stats.items():
            db.execute("""
                INSERT INTO source_trust (source_type, source_name, total_items, quality_wins, win_rate)
                VALUES ('publisher', %s, %s, %s, %s)
                ON CONFLICT (source_type, source_name) 
                DO UPDATE SET 
                total_items = source_trust.total_items + EXCLUDED.total_items,
                quality_wins = source_trust.quality_wins + EXCLUDED.quality_wins,
                win_rate = CASE 
                    WHEN (source_trust.total_items + EXCLUDED.total_items) > 0 
                    THEN (source_trust.quality_wins + EXCLUDED.quality_wins)::FLOAT / (source_trust.total_items + EXCLUDED.total_items)
                    ELSE 0.0 END,
                last_updated = CURRENT_TIMESTAMP
            """, [pub, stats["total"], stats["wins"], stats["wins"] / max(1, stats["total"])])

        if ticker:
            after = db.execute("SELECT COUNT(*) FROM news_articles WHERE ticker = %s", [ticker]).fetchone()[0]
        else:
            after = db.execute("SELECT COUNT(*) FROM news_articles").fetchone()[0]
        removed = before - after

        if removed > 0:
            print(f"[dedup] Removed {removed} duplicate semantic/exact news articles. Tracked {len(publisher_stats)} sources.")
            if high_redundancy_tickers:
                print(f"[dedup] Flagged {len(high_redundancy_tickers)} tickers for Deep Dive due to high redundancy.")

        return removed, list(high_redundancy_tickers)
