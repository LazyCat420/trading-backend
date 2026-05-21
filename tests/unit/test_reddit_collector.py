import pytest
from app.collectors.reddit_collector import _is_quality_post, _is_relevant_to_ticker

def test_is_quality_post_deleted():
    post = {"selftext": "[deleted]", "score": 100, "num_comments": 50}
    assert _is_quality_post(post) is False

    post = {"selftext": "[removed]", "score": 100, "num_comments": 50}
    assert _is_quality_post(post) is False

def test_is_quality_post_nsfw():
    post = {"selftext": "Good post", "score": 100, "num_comments": 50, "over_18": True}
    assert _is_quality_post(post) is False

def test_is_quality_post_low_engagement():
    post = {"selftext": "Long post with good content but no engagement", "score": 2, "num_comments": 1}
    assert _is_quality_post(post) is False

def test_is_quality_post_title_only_low_score():
    post = {"selftext": "Short", "score": 40, "num_comments": 10}
    assert _is_quality_post(post) is False

def test_is_quality_post_title_only_high_score():
    post = {"selftext": "Short", "score": 60, "num_comments": 10}
    assert _is_quality_post(post) is True

def test_is_quality_post_valid():
    post = {"selftext": "A very long detailed DD post about a company that is more than fifty characters long.", "score": 10, "num_comments": 5}
    assert _is_quality_post(post) is True

def test_is_relevant_to_ticker_financial_sub():
    post = {
        "subreddit": "wallstreetbets",
        "title": "Thoughts on NVDA",
        "selftext": "I think it will go up."
    }
    # Ticker in title counts as 2 mentions
    assert _is_relevant_to_ticker(post, "NVDA") is True

    post2 = {
        "subreddit": "stocks",
        "title": "Good stock",
        "selftext": "I bought NVDA yesterday."
    }
    # Ticker in body counts as 1 mention
    assert _is_relevant_to_ticker(post2, "NVDA") is True

def test_is_relevant_to_ticker_non_financial_sub():
    post = {
        "subreddit": "gaming",
        "title": "New graphics card",
        "selftext": "I bought an NVDA card."
    }
    # For non-financial sub, it must have $TICKER or ticker in title. This has neither.
    assert _is_relevant_to_ticker(post, "NVDA") is False

    post_with_dollar = {
        "subreddit": "gaming",
        "title": "New graphics card",
        "selftext": "I bought $NVDA calls."
    }
    assert _is_relevant_to_ticker(post_with_dollar, "NVDA") is True

    post_title_ticker = {
        "subreddit": "gaming",
        "title": "NVDA releases new GPU",
        "selftext": "Looks good."
    }
    assert _is_relevant_to_ticker(post_title_ticker, "NVDA") is True
