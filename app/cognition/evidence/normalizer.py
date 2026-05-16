"""
Document normalizer for evidence fusion.
Transforms raw database rows and API responses into a unified NormalizedDocument format.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime, timezone


@dataclass
class NormalizedDocument:
    source_ref: str
    source_type: str
    content: str
    metadata: Dict[str, Any]
    timestamp: datetime
    author: Optional[str] = None


# Scrape artifact detection — imported from shared utilities
from app.utils.text_utils import is_scrape_artifact


def normalize_news(row: tuple, columns: list[str]) -> Optional[NormalizedDocument]:
    row_dict = dict(zip(columns, row))
    summary = row_dict.get("best_summary") or row_dict.get("summary") or ""

    if is_scrape_artifact(summary):
        return None

    return NormalizedDocument(
        source_ref=f"news_articles_{row_dict.get('id', 'unknown')}",
        source_type="news",
        content=f"Title: {row_dict.get('title')}\nSummary: {summary}",
        metadata={"publisher": row_dict.get("publisher"), "url": row_dict.get("url")},
        timestamp=row_dict.get("published_at", datetime.now(timezone.utc)),
        author=row_dict.get("publisher"),
    )


def normalize_reddit(row: tuple, columns: list[str]) -> NormalizedDocument:
    row_dict = dict(zip(columns, row))
    content = row_dict.get("content") or row_dict.get("body") or ""

    return NormalizedDocument(
        source_ref=f"reddit_posts_{row_dict.get('id', 'unknown')}",
        source_type="reddit",
        content=f"Title: {row_dict.get('title')}\nBody: {content}",
        metadata={
            "subreddit": row_dict.get("subreddit"),
            "score": row_dict.get("score"),
            "comment_count": row_dict.get("comment_count"),
        },
        timestamp=row_dict.get("created_utc", datetime.now(timezone.utc)),
        author=row_dict.get("author"),
    )


def normalize_youtube(row: tuple, columns: list[str]) -> NormalizedDocument:
    row_dict = dict(zip(columns, row))
    content = (
        row_dict.get("content")
        or row_dict.get("summary")
        or row_dict.get("raw_transcript")
        or ""
    )

    return NormalizedDocument(
        source_ref=f"youtube_transcripts_{row_dict.get('video_id', 'unknown')}",
        source_type="youtube",
        content=f"Title: {row_dict.get('title')}\nTranscript/Summary: {content}",
        metadata={
            "channel": row_dict.get("channel"),
            "tickers_mentioned": row_dict.get("tickers_mentioned"),
        },
        timestamp=row_dict.get("published_at", datetime.now(timezone.utc)),
        author=row_dict.get("channel"),
    )


def normalize_structured_row(
    table_name: str, fact_type: str, pk_id: str, value: Any, timestamp: datetime
) -> NormalizedDocument:
    return NormalizedDocument(
        source_ref=f"{table_name}_{pk_id}",
        source_type="structured",
        content=str(value),
        metadata={"fact_type": fact_type},
        timestamp=timestamp,
    )
