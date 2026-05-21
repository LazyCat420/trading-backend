import logging
import httpx
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

class ScraperServiceClient:
    """Client to interact with the scraper-service endpoints."""
    
    def __init__(self, base_url: str | None = None):
        # Default to settings.SCRAPER_SERVICE_URL if not provided
        self.base_url = base_url or getattr(settings, "SCRAPER_SERVICE_URL", "http://scraper-service:8001")
        
    async def scrape(self, url: str, engine: str = "http", options: dict | None = None) -> dict | None:
        """
        Calls POST /scrape on scraper-service.
        Returns the parsed JSON response dict or None on failure.
        """
        payload = {
            "url": url,
            "engine": engine,
            "options": options or {}
        }
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await client.post(f"{self.base_url}/scrape", json=payload)
                r.raise_for_status()
                data = r.json()
                
                if data.get("success"):
                    return data
                else:
                    logger.warning(f"[scraper_client] Scrape failed for {url}: {data.get('error')}")
                    return None
        except httpx.HTTPError as e:
            logger.error(f"[scraper_client] HTTP error when scraping {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"[scraper_client] Unexpected error scraping {url}: {e}")
            return None
            
    async def collect(self, source: str, req_data: dict) -> list[dict[str, Any]]:
        """
        Calls POST /collect on scraper-service.
        Returns a list of items collected.
        """
        payload = {"source": source, **req_data}
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await client.post(f"{self.base_url}/collect", json=payload)
                r.raise_for_status()
                data = r.json()
                
                if "error" in data and data["error"]:
                    logger.warning(f"[scraper_client] Collect failed for {source}: {data['error']}")
                
                return data.get("items", [])
        except httpx.HTTPError as e:
            logger.error(f"[scraper_client] HTTP error when collecting from {source}: {e}")
            return []
        except Exception as e:
            logger.error(f"[scraper_client] Unexpected error collecting from {source}: {e}")
            return []

scraper_client = ScraperServiceClient()
