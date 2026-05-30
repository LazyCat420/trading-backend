import logging
import httpx
import asyncio
import os

logger = logging.getLogger(__name__)

WEBHOOK_URL = os.getenv("WEBHOOK_ALERT_URL", "")

async def send_critical_alert_async(message: str, details: dict = None):
    if not WEBHOOK_URL:
        return
        
    payload = {
        "content": f"🚨 **CRITICAL PIPELINE ALERT** 🚨\n\n**Message:** {message}\n"
    }
    
    if details:
        # Format details into code block
        details_str = "\n".join([f"- **{k}**: {v}" for k, v in details.items()])
        payload["content"] += f"\n**Details:**\n{details_str}"

    try:
        async with httpx.AsyncClient() as client:
            await client.post(WEBHOOK_URL, json=payload, timeout=5.0)
    except Exception as e:
        logger.error("[WebhookAlerter] Failed to send alert: %s", e)

def trigger_alert(message: str, details: dict = None):
    """Fire and forget wrapper to trigger webhook from synchronous code."""
    if not WEBHOOK_URL:
        return
        
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(send_critical_alert_async(message, details))
    except RuntimeError:
        # No running loop, run it synchronously if needed, but typically we have one
        pass
