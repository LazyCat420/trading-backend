# Scraper Service Migration Plan

## PHASE 0 — ORIENTATION
**Objective:** Port over all existing scraping and raw data collection logic from the `trading-service` to the `scraper-service`. The `trading-service` will strictly act as an orchestrator/client, handling only the data necessary to tell `scraper-service` what to do, and processing the parsed results.
**Constraints:** No business/domain logic in `scraper-service`. `scraper-service` is already built; we only modify `trading-service`.

## PHASE 1 & 2 — PRE-FLIGHT AND PLAN
**Blast Radius:** 
- `app/collectors/*` (specifically `reddit_collector.py`, `news_collector.py`, `youtube_collector.py`, `adaptive_scraper.py`, `news_playwright.py`, `youtube_playwright.py`, `vision_scraper.py`).
- `app/services/scraper_client.py` (New file).

<architecture_plan>
1. **Remove Heavy Scrapers:** Delete Playwright and Crawl4AI code from `trading-service` (`news_playwright.py`, `youtube_playwright.py`, `crawl4ai_config.py`).
2. **Abstract Scraping to Client:** Introduce a lightweight `ScraperServiceClient` inside `trading-service` that wraps `httpx` to POST `/scrape` and `/collect` endpoints on `scraper-service`.
3. **Refactor API Collectors:** `reddit_collector.py`, `news_collector.py`, and `youtube_collector.py` will no longer make direct external API calls. Instead, they will call `ScraperServiceClient.collect(source="reddit", ...)` and store the returned standardized JSON in the local PostgreSQL DB.
4. **Testing Infrastructure:** Set up mock tests using `@pytest.mark.asyncio` and `respx` (or `pytest-httpx`) to simulate `scraper-service` responses without needing the external service to be running.
</architecture_plan>

<data_and_interfaces>
```python
# app/services/scraper_client.py
import httpx
from typing import Any

class ScraperServiceClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
    
    async def scrape(self, url: str, engine: str = "http", options: dict | None = None) -> str | None:
        """Calls POST /scrape on scraper-service."""
        pass
        
    async def collect(self, source: str, req_data: dict) -> list[dict[str, Any]]:
        """Calls POST /collect on scraper-service."""
        pass
```
</data_and_interfaces>

<implementation>
**Step 1:** Create `app/services/scraper_client.py`.
**Step 2:** Refactor `reddit_collector.py` to use `ScraperServiceClient.collect(source="reddit", subreddits=...)` and insert the returned items.
**Step 3:** Refactor `news_collector.py` and `youtube_collector.py` using the same pattern.
**Step 4:** Refactor `adaptive_scraper.py` and other web scrapers to use `ScraperServiceClient.scrape(...)` with the `playwright` or `vision` engines.
**Step 5:** Delete deprecated heavy engine files (e.g., `news_playwright.py`, `crawl4ai_config.py`) from `trading-service`.
</implementation>

## PHASE 3 & 4 — IMPLEMENTATION & MOCK TESTS
To ensure the `trading-service` correctly handles the `scraper-service` data without real network IO:

- [ ] Create `tests/test_scraper_client.py`: Mock httpx responses for `/scrape` and `/collect`.
- [ ] Create `tests/test_reddit_collector.py`: Use mocked `ScraperServiceClient` and verify it writes the expected rows to the local test database.
- [ ] Create `tests/test_adaptive_scraper.py`: Verify fallback logic and JS execution requests form properly.

## PHASE 6 & 7 — POST-CHECKLIST & REGRESSION
- [ ] All direct Playwright/Crawl4AI dependencies removed from `trading-service/requirements.txt`.
- [ ] Ensure `trading-service` handles 503/429 gracefully from `scraper-service`.
- [ ] Add regression test for timeout handling.

---

### Questions for the User
1. Would you like me to start by creating the `ScraperServiceClient` and its associated mock tests first?
2. Should I handle the `reddit_collector.py` refactor as the immediate next step after the client is built?
3. Are there any specific environment variables you'd like to use for the `scraper-service` URL besides `SCRAPER_SERVICE_URL`?
