import uuid
import logging
from app.db.connection import get_db
from app.schemas.alerts import FundAlert

logger = logging.getLogger(__name__)

def record_fund_alert(
    alert_type: str,
    entity_name: str,
    detail: str,
    severity: str,
    ticker: str | None = None,
    llm_summary: str | None = None
) -> dict:
    """
    Creates and records a fund alert into the database using strict Pydantic validation.
    """
    try:
        # Validate data
        alert = FundAlert(
            id=str(uuid.uuid4()),
            alert_type=alert_type,
            entity_name=entity_name,
            detail=detail,
            severity=severity,
            ticker=ticker,
            llm_summary=llm_summary
        )
        
        with get_db() as db:
            db.execute(
                """
                INSERT INTO fund_alerts (
                    id, created_at, alert_type, ticker, entity_name, 
                    detail, severity, llm_summary, is_read
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    alert.id,
                    alert.created_at,
                    alert.alert_type,
                    alert.ticker,
                    alert.entity_name,
                    alert.detail,
                    alert.severity,
                    alert.llm_summary,
                    alert.is_read
                ]
            )
        
        logger.info("[alert_service] Recorded %s alert (%s) for entity %s", severity, alert_type, entity_name)
        return alert.model_dump()
        
    except Exception as e:
        logger.error("[alert_service] Failed to record fund alert: %s", e)
        return {"error": str(e)}
