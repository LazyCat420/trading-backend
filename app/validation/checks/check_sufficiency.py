from app.validation.models import ValidationStatus, QuarantineReason
from typing import Tuple, Optional

def check_sufficiency(
    yfinance_pass: bool,
    yfinance_reason: Optional[QuarantineReason],
    finviz_pass: bool,
    finviz_reason: Optional[QuarantineReason],
    content_pass: bool
) -> Tuple[ValidationStatus, Optional[QuarantineReason], bool]:
    """
    Pure logic evaluator taking the results of the previous checks.
    Returns a tuple of (status, reason, should_escalate).
    """
    # 1. If rate limited return PENDING
    if yfinance_reason == QuarantineReason.RATE_LIMIT_EXCEEDED or finviz_reason == QuarantineReason.RATE_LIMIT_EXCEEDED:
        return ValidationStatus.PENDING, QuarantineReason.RATE_LIMIT_EXCEEDED, False
        
    # 2. If yfinance+finviz pass return VALID
    if yfinance_pass and finviz_pass:
        return ValidationStatus.VALID, None, False
        
    # 3. If either passes + content exists return VALID
    if (yfinance_pass or finviz_pass) and content_pass:
        return ValidationStatus.VALID, None, False
        
    # 4. Otherwise escalate to Wikipedia check
    return ValidationStatus.QUARANTINE, QuarantineReason.NO_DATA, True
