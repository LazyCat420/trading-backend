import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from app.collectors.youtube_collector import _is_channel_blocked, collect_channel


@pytest.fixture
def mock_db():
    with patch("app.collectors.youtube_collector.get_db") as mock_get_db:
        db = MagicMock()
        mock_get_db.return_value.__enter__.return_value = db
        yield db


def test_is_channel_blocked_true(mock_db):
    mock_db.execute.return_value.fetchone.return_value = (1,)
    assert _is_channel_blocked("bad_channel") is True


def test_is_channel_blocked_false(mock_db):
    mock_db.execute.return_value.fetchone.return_value = None
    assert _is_channel_blocked("good_channel") is False


@pytest.mark.asyncio
@patch("app.collectors.youtube_collector._is_channel_blocked", return_value=True)
async def test_collect_channel_blocked(mock_is_blocked, mock_db):
    stats = await collect_channel("bad_channel")
    assert stats["videos_found"] == 0


@pytest.mark.asyncio
@patch("app.collectors.youtube_collector._is_channel_blocked", return_value=False)
@patch("app.services.scraper_client.scraper_client.collect", new_callable=AsyncMock)
@patch("app.collectors.youtube_collector._process_video")
async def test_collect_channel_success(mock_process, mock_collect, mock_is_blocked, mock_db):
    mock_collect.return_value = [
        {"id": "video1", "title": "NVDA Earnings"},
        {"id": "video2", "title": "AAPL Review"}
    ]
    mock_process.side_effect = ["stored", "skipped_old"]

    stats = await collect_channel("good_channel")

    assert stats["videos_found"] == 2
    assert stats["stored"] == 1
    assert stats["skipped_old"] == 1
