import pytest
import datetime
from unittest.mock import patch, MagicMock, AsyncMock

from app.collectors.news_api_rotator import NewsApiRotator, ProviderConfig, NewsArticle
from app.collectors.news_collector import collect_finnhub_news

@pytest.fixture
def mock_db():
    with patch("app.collectors.news_collector.get_db") as mock_get_db:
        db = MagicMock()
        mock_get_db.return_value.__enter__.return_value = db
        yield db

@pytest.mark.asyncio
@patch("app.collectors.news_api_rotator._persist_articles")
async def test_news_api_rotator_429_fallback(mock_persist, mock_db):
    """
    Test that if one provider fails (e.g. 429 Too Many Requests returning []),
    the rotator continues to the next provider and still successfully inserts data.
    """
    providers = [
        ProviderConfig("marketaux", "key1", daily_limit=10),
        ProviderConfig("newsapi", "key2", daily_limit=10)
    ]
    
    rotator = NewsApiRotator(providers=providers, tickers=["AAPL"])
    
    # We will mock _fetch_from_provider to return [] for the first and data for the second
    call_count = 0
    
    async def mock_fetch_from_provider(provider, query):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Simulate 429 Too Many Requests -> Returns empty list
            return []
        else:
            # Second provider succeeds
            return [
                NewsArticle(
                    title="Apple announces new product",
                    url="https://example.com/apple",
                    summary="Apple is doing great.",
                    source=provider.name,
                    published_at=datetime.datetime.now(datetime.UTC),
                    tickers=["AAPL"]
                )
            ]
            
    mock_persist.return_value = 1
            
    with patch.object(rotator, "_fetch_from_provider", side_effect=mock_fetch_from_provider):
        async with rotator:
            count = await rotator.fetch_news(query="AAPL", persist=True)
            
    # We expect 1 article to be persisted
    assert count == 1
    mock_persist.assert_called_once()
    persisted_articles = mock_persist.call_args[0][0]
    assert len(persisted_articles) == 1
    assert persisted_articles[0].title == "Apple announces new product"


@pytest.mark.asyncio
@patch.dict("os.environ", {"FINNHUB_API_KEY": "fake_key"})
async def test_collect_finnhub_news_jaccard_dedup(mock_db):
    """
    Test the Jaccard similarity threshold for identical articles in Finnhub.
    """
    # Summaries must be >= 150 chars to pass the STRICT QUALITY GATE
    long_summary_1 = "Apple reported record quarterly earnings, beating analyst expectations across all segments. " \
                      "Revenue came in above consensus estimates driven by strong iPhone and Services growth. " \
                      "The company also announced a massive stock buyback program."
    long_summary_2 = "Apple once again delivered a beat on quarterly earnings with revenue surpassing expectations. " \
                      "All segments contributed positively. The stock jumped in after-hours trading on " \
                      "the back of strong guidance for the next quarter ahead."
    long_summary_3 = "Tim Cook took the stage at WWDC to unveil a new set of iPhone features including advanced AI. " \
                      "The announcement was met with enthusiasm from developers and analysts alike. " \
                      "Apple Intelligence was the highlight of the keynote event."

    # Mock Finnhub client
    with patch("finnhub.Client") as mock_finnhub_class:
        mock_client = MagicMock()
        mock_finnhub_class.return_value = mock_client
        
        # We will return 3 articles:
        # 1. Base article
        # 2. Duplicate article (>60% similarity)
        # 3. Unique article (<60% similarity)
        mock_client.company_news.return_value = [
            {
                "headline": "Apple reports strong quarterly earnings and revenue beat",
                "summary": long_summary_1,
                "url": "http://example.com/1",
                "source": "Yahoo",
                "datetime": 1700000000
            },
            {
                # Duplicate: many shared words "Apple", "reports", "strong", "quarterly", "earnings", "revenue"
                "headline": "Apple reports strong quarterly earnings beating revenue expectations",
                "summary": long_summary_2,
                "url": "http://example.com/2",
                "source": "Finnhub",
                "datetime": 1700000001
            },
            {
                # Unique
                "headline": "Tim Cook announces new iPhone features at WWDC",
                "summary": long_summary_3,
                "url": "http://example.com/3",
                "source": "Bloomberg",
                "datetime": 1700000002
            }
        ]
        
        # Also mock the trust DB check to not drop these sources
        mock_db.execute.return_value.fetchall.return_value = []
        
        count = await collect_finnhub_news("AAPL", days=7)
        
        # Expecting 2 articles to be inserted (the first and the third)
        assert count == 2
        # Verify db execute was called twice with INSERT
        insert_calls = [c for c in mock_db.execute.call_args_list if "INSERT INTO news_articles" in c[0][0]]
        assert len(insert_calls) == 2
        
        # First call should be for WWDC (since it's newest by timestamp, it gets processed first due to sort!)
        # Wait, the sort puts newest first:
        # 1700000002 -> WWDC
        # 1700000001 -> Duplicate
        # 1700000000 -> Base
        # So WWDC is processed first, then Duplicate is processed. Duplicate and WWDC are not similar.
        # Then Base is processed. Base and Duplicate are similar! Since Duplicate was processed first, Base is skipped!
        # Either way, only 2 articles should make it through.
        inserted_titles = [c[0][1][2] for c in insert_calls] # Index 2 is the title
        assert any("WWDC" in title for title in inserted_titles)

