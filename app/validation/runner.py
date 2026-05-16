import logging
from app.validation.persistence import (
    get_pending_retries,
    save_validation_result,
    increment_rate_limit_and_check
)
from app.validation.validator import validate_ticker
from app.validation.models import ValidationResult, ValidationStatus, QuarantineReason

logger = logging.getLogger(__name__)

def run_validation_batch(batch_size: int = 50):
    """Run validation on a batch of pending tickers."""
    tickers = get_pending_retries()
    
    # limit batch size
    tickers_to_process = tickers[:batch_size]
    logger.info(f"Running validation batch for {len(tickers_to_process)} tickers")
    
    for ticker in tickers_to_process:
        try:
            result = validate_ticker(ticker)
            
            # Handle rate limiting logic
            if result.status == ValidationStatus.PENDING and result.reason == QuarantineReason.RATE_LIMIT_EXCEEDED:
                should_quarantine = increment_rate_limit_and_check(ticker)
                if should_quarantine:
                    logger.warning(f"Ticker {ticker} exceeded rate limit retries, quarantining.")
                    result = ValidationResult(
                        ticker=ticker,
                        status=ValidationStatus.QUARANTINE,
                        reason=QuarantineReason.RATE_LIMIT_EXCEEDED,
                        details="Exceeded 5 rate limit retries."
                    )
                else:
                    # We've already incremented, so just continue to avoid saving PENDING normally unless needed
                    # Wait, if we use increment_rate_limit_and_check, we don't need save_validation_result to increment it again.
                    # Let's adjust this: if we didn't quarantine, we just log and skip saving since we already incremented.
                    logger.info(f"Ticker {ticker} rate limited. Retrying later.")
                    continue
            
            save_validation_result(result)
        except Exception as e:
            logger.error(f"Error validating {ticker}: {e}")
