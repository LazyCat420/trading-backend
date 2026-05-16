from app.validation.models import ValidationResult, ValidationStatus, QuarantineReason
from app.validation.checks.check_yfinance import check_yfinance
from app.validation.checks.check_finviz import check_finviz
from app.validation.checks.check_content import check_content
from app.validation.checks.check_sufficiency import check_sufficiency
from app.validation.checks.check_wikipedia import check_wikipedia

async def validate_ticker(ticker: str) -> ValidationResult:
    """
    Orchestrates the validation checks in order.
    Returns a ValidationResult object. Zero DB writes.
    """
    # 1. YFinance Check
    yfinance_pass, yfinance_reason = await check_yfinance(ticker)
    
    # 2. Finviz Check
    finviz_pass, finviz_reason = await check_finviz(ticker)
    
    # 3. DB Content Check
    content_pass = await check_content(ticker)
    
    # 4. Sufficiency Logic
    status, reason, should_escalate = check_sufficiency(
        yfinance_pass=yfinance_pass,
        yfinance_reason=yfinance_reason,
        finviz_pass=finviz_pass,
        finviz_reason=finviz_reason,
        content_pass=content_pass
    )
    
    wikipedia_pass = False
    
    # 5. Wikipedia Fallback Check
    if should_escalate:
        wikipedia_pass = await check_wikipedia(ticker)
        if wikipedia_pass:
            status = ValidationStatus.VALID
            reason = None
        else:
            status = ValidationStatus.QUARANTINE
            reason = QuarantineReason.DELISTED
            
    return ValidationResult(
        ticker=ticker,
        status=status,
        reason=reason,
        yfinance_pass=yfinance_pass,
        finviz_pass=finviz_pass,
        content_pass=content_pass,
        wikipedia_pass=wikipedia_pass
    )
