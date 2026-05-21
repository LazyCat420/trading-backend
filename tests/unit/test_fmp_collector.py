import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import datetime

from app.collectors.fmp_collector import (
    collect_congress_trades,
    collect_price_history,
    collect_fundamentals,
    collect_financials,
    collect_balance_sheet,
    collect_all,
)

@pytest.fixture
def mock_db():
    with patch("app.collectors.fmp_collector.get_db") as mock_get_db:
        db = MagicMock()
        mock_get_db.return_value.__enter__.return_value = db
        yield db

@pytest.mark.asyncio
@patch("app.collectors.fmp_collector._get_key", return_value="fake_key")
@patch("app.collectors.fmp_collector.httpx.AsyncClient")
async def test_collect_congress_trades_success(mock_client_class, mock_get_key, mock_db):
    
    mock_client = MagicMock()
    mock_client.get = AsyncMock()
    mock_client_class.return_value.__aenter__.return_value = mock_client
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [
        {"ticker": "AAPL", "senator": "John Doe", "transaction_date": "2023-10-01", "type": "Purchase", "amount": "$1,000 - $15,000"},
        {"ticker": "MSFT", "representative": "Jane Smith", "transaction_date": "2023-10-02", "type": "Sale", "amount": "$15,001 - $50,000"}
    ]
    mock_client.get.return_value = mock_resp
    
    # Test with filter
    count = await collect_congress_trades("AAPL")
    
    assert count == 1
    mock_db.execute.assert_called_once()
    assert "AAPL" in mock_db.execute.call_args[0][1]

@pytest.mark.asyncio
@patch("app.collectors.fmp_collector._get_key", return_value="fake_key")
@patch("app.collectors.fmp_collector.httpx.AsyncClient")
async def test_collect_congress_trades_403(mock_client_class, mock_get_key, mock_db):
    mock_client = MagicMock()
    mock_client.get = AsyncMock()
    mock_client_class.return_value.__aenter__.return_value = mock_client
    
    mock_resp = MagicMock()
    mock_resp.status_code = 403
    mock_client.get.return_value = mock_resp
    
    count = await collect_congress_trades("AAPL")
    assert count == 0

@pytest.mark.asyncio
@patch("app.collectors.fmp_collector._get_key", return_value="fake_key")
@patch("app.services.request_utils.SmartClient")
async def test_collect_price_history_success(mock_smart_client, mock_get_key, mock_db):
    
    mock_client = MagicMock()
    mock_client.get = AsyncMock()
    mock_smart_client.return_value.__aenter__.return_value = mock_client
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    # Provide a date that is clearly within the default 365 days cutoff
    today_str = datetime.date.today().isoformat()
    mock_resp.json.return_value = {
        "historical": [
            {"date": today_str, "open": 100, "high": 105, "low": 95, "close": 102, "volume": 1000}
        ]
    }
    mock_client.get.return_value = mock_resp
    
    count = await collect_price_history("AAPL")
    
    assert count == 1
    mock_db.execute.assert_called_once()

@pytest.mark.asyncio
@patch("app.collectors.fmp_collector._get_key", return_value="fake_key")
@patch("app.services.request_utils.SmartClient")
async def test_collect_fundamentals_success(mock_smart_client, mock_get_key, mock_db):
    
    mock_client = MagicMock()
    mock_client.get = AsyncMock()
    mock_smart_client.return_value.__aenter__.return_value = mock_client
    
    mock_prof_resp = MagicMock()
    mock_prof_resp.status_code = 200
    mock_prof_resp.json.return_value = [{"mktCap": 1000000, "beta": 1.2}]
    
    mock_metrics_resp = MagicMock()
    mock_metrics_resp.status_code = 200
    mock_metrics_resp.json.return_value = [{"peRatioTTM": 15.0}]
    
    mock_client.get.side_effect = [mock_prof_resp, mock_metrics_resp]
    
    result = await collect_fundamentals("AAPL")
    
    assert result is True
    mock_db.execute.assert_called_once()

@pytest.mark.asyncio
@patch("app.collectors.fmp_collector._get_key", return_value="fake_key")
@patch("app.services.request_utils.SmartClient")
async def test_collect_financials_success(mock_smart_client, mock_get_key, mock_db):
    
    mock_client = MagicMock()
    mock_client.get = AsyncMock()
    mock_smart_client.return_value.__aenter__.return_value = mock_client
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [
        {"date": "2023-10-01", "revenue": 1000, "grossProfit": 500, "operatingIncome": 200, "netIncome": 100, "eps": 1.5}
    ]
    mock_client.get.return_value = mock_resp
    
    count = await collect_financials("AAPL")
    
    assert count == 1
    mock_db.execute.assert_called_once()

@pytest.mark.asyncio
@patch("app.collectors.fmp_collector._get_key", return_value="fake_key")
@patch("app.services.request_utils.SmartClient")
async def test_collect_balance_sheet_success(mock_smart_client, mock_get_key, mock_db):
    
    mock_client = MagicMock()
    mock_client.get = AsyncMock()
    mock_smart_client.return_value.__aenter__.return_value = mock_client
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [
        {"date": "2023-10-01", "totalAssets": 1000, "totalLiabilities": 500, "totalStockholdersEquity": 500, "cashAndCashEquivalents": 100, "totalDebt": 200}
    ]
    mock_client.get.return_value = mock_resp
    
    count = await collect_balance_sheet("AAPL")
    
    assert count == 1
    mock_db.execute.assert_called_once()
