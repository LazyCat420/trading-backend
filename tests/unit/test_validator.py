import pytest
from unittest.mock import patch, MagicMock
from app.validation.models import ValidationResult, ValidationStatus, QuarantineReason
from app.validation.validator import validate_ticker

@pytest.mark.asyncio
@patch("app.validation.validator.check_yfinance")
@patch("app.validation.validator.check_finviz")
@patch("app.validation.validator.check_content")
@patch("app.validation.validator.check_sufficiency")
@patch("app.validation.validator.check_wikipedia")
async def test_validate_ticker_valid(mock_check_wikipedia, mock_check_sufficiency, mock_check_content, mock_check_finviz, mock_check_yfinance):
    # Setup mocks for a valid ticker
    mock_check_yfinance.return_value = (True, None)
    mock_check_finviz.return_value = (True, None)
    mock_check_content.return_value = True
    
    # Sufficiency logic says it's valid, no need to escalate
    mock_check_sufficiency.return_value = (ValidationStatus.VALID, None, False)
    
    result = await validate_ticker("AAPL")
    
    assert isinstance(result, ValidationResult)
    assert result.ticker == "AAPL"
    assert result.status == ValidationStatus.VALID
    assert result.reason is None
    assert result.yfinance_pass is True
    assert result.finviz_pass is True
    assert result.content_pass is True
    assert result.wikipedia_pass is False
    
    mock_check_wikipedia.assert_not_called()

@pytest.mark.asyncio
@patch("app.validation.validator.check_yfinance")
@patch("app.validation.validator.check_finviz")
@patch("app.validation.validator.check_content")
@patch("app.validation.validator.check_sufficiency")
@patch("app.validation.validator.check_wikipedia")
async def test_validate_ticker_quarantine_no_escalate(mock_check_wikipedia, mock_check_sufficiency, mock_check_content, mock_check_finviz, mock_check_yfinance):
    # Setup mocks for a rate limited ticker (pending)
    mock_check_yfinance.return_value = (False, QuarantineReason.RATE_LIMIT_EXCEEDED)
    mock_check_finviz.return_value = (False, QuarantineReason.RATE_LIMIT_EXCEEDED)
    mock_check_content.return_value = False
    
    mock_check_sufficiency.return_value = (ValidationStatus.PENDING, QuarantineReason.RATE_LIMIT_EXCEEDED, False)
    
    result = await validate_ticker("TEST")
    
    assert result.status == ValidationStatus.PENDING
    assert result.reason == QuarantineReason.RATE_LIMIT_EXCEEDED
    mock_check_wikipedia.assert_not_called()

@pytest.mark.asyncio
@patch("app.validation.validator.check_yfinance")
@patch("app.validation.validator.check_finviz")
@patch("app.validation.validator.check_content")
@patch("app.validation.validator.check_sufficiency")
@patch("app.validation.validator.check_wikipedia")
async def test_validate_ticker_escalate_to_wikipedia_pass(mock_check_wikipedia, mock_check_sufficiency, mock_check_content, mock_check_finviz, mock_check_yfinance):
    # Setup mocks to trigger Wikipedia escalation
    mock_check_yfinance.return_value = (False, QuarantineReason.DELISTED)
    mock_check_finviz.return_value = (False, QuarantineReason.DELISTED)
    mock_check_content.return_value = False
    
    # Needs escalation
    mock_check_sufficiency.return_value = (ValidationStatus.QUARANTINE, QuarantineReason.DELISTED, True)
    
    # Wikipedia confirms it's valid
    mock_check_wikipedia.return_value = True
    
    result = await validate_ticker("FAKE")
    
    assert result.status == ValidationStatus.VALID
    assert result.reason is None
    assert result.wikipedia_pass is True
    mock_check_wikipedia.assert_called_once_with("FAKE")

@pytest.mark.asyncio
@patch("app.validation.validator.check_yfinance")
@patch("app.validation.validator.check_finviz")
@patch("app.validation.validator.check_content")
@patch("app.validation.validator.check_sufficiency")
@patch("app.validation.validator.check_wikipedia")
async def test_validate_ticker_escalate_to_wikipedia_fail(mock_check_wikipedia, mock_check_sufficiency, mock_check_content, mock_check_finviz, mock_check_yfinance):
    # Setup mocks to trigger Wikipedia escalation
    mock_check_yfinance.return_value = (False, QuarantineReason.DELISTED)
    mock_check_finviz.return_value = (False, QuarantineReason.DELISTED)
    mock_check_content.return_value = False
    
    # Needs escalation
    mock_check_sufficiency.return_value = (ValidationStatus.QUARANTINE, QuarantineReason.DELISTED, True)
    
    # Wikipedia also fails
    mock_check_wikipedia.return_value = False
    
    result = await validate_ticker("BADTICKER")
    
    assert result.status == ValidationStatus.QUARANTINE
    assert result.reason == QuarantineReason.DELISTED
    assert result.wikipedia_pass is False
    mock_check_wikipedia.assert_called_once_with("BADTICKER")
