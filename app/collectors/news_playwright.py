"""
Playwright-based news article scraper.

Fallback for when httpx/cloudscraper get 403'd by Cloudflare/bot protection.
Same stealth pattern as youtube_playwright.py:
  - Automation flag removal
  - Human-like behavior (mouse jitter, scroll)
  - Randomized viewport
  - Extract article text from rendered DOM

Run standalone: python -m app.collectors.news_playwright "https://url"
"""

import logging

logger = logging.getLogger(__name__)


import random
import re
import sys


def scrape_article_sync(
    url: str, headless: bool = True, timeout_ms: int = 20000, max_chars: int = 3000
) -> str | None:
    """Scrape article body using Playwright stealth browser.

    Returns plain text article body or None on failure.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.info("[news-pw] playwright not installed")
        return None

    article_text = None

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=headless,
                args=["--disable-blink-features=AutomationControlled"],
            )
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                viewport={
                    "width": 1280 + random.randint(0, 100),
                    "height": 900 + random.randint(0, 50),
                },
                locale="en-US",
            )
            page = context.new_page()

            # Block heavy resources to speed up loading
            page.route(
                "**/*.{png,jpg,jpeg,gif,svg,mp4,webm,woff,woff2}",
                lambda route: route.abort(),
            )

            logger.info(f"[news-pw] Loading {url}")
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)

            # Stealth: human-like behavior
            page.mouse.move(random.randint(200, 800), random.randint(200, 600))
            page.wait_for_timeout(800 + random.randint(0, 500))
            page.evaluate("window.scrollBy(0, 400)")
            page.wait_for_timeout(1000 + random.randint(0, 500))

            # Dismiss cookie banners / modals
            for dismiss_sel in [
                "button:has-text('Accept')",
                "button:has-text('Accept all')",
                "button:has-text('I agree')",
                "button:has-text('Continue')",
                "[aria-label='Close']",
            ]:
                try:
                    btn = page.locator(dismiss_sel)
                    if btn.count() > 0:
                        btn.first.click(timeout=1500)
                        page.wait_for_timeout(500)
                        break
                except Exception:
                    continue

            # Wait a bit more for content to render
            page.wait_for_timeout(1000)

            # Extract article text using multiple strategies
            article_text = page.evaluate("""
                () => {
                    // Strategy 1: <article> tag
                    const article = document.querySelector('article');
                    if (article) {
                        const text = article.innerText;
                        if (text && text.length > 200) return text;
                    }
                    
                    // Strategy 2: Common article content selectors
                    const selectors = [
                        '.article-body', '.article-content', '.article__body',
                        '.post-content', '.entry-content', '.story-body',
                        '[data-testid="article-body"]',
                        '.caas-body', // Yahoo Finance
                        '#article-body', '#story-content',
                        '.paywall', '.premium-content',
                        'main article', 'main .content',
                    ];
                    for (const sel of selectors) {
                        const el = document.querySelector(sel);
                        if (el) {
                            const text = el.innerText;
                            if (text && text.length > 200) return text;
                        }
                    }
                    
                    // Strategy 3: Find the largest <p> cluster
                    const paragraphs = Array.from(document.querySelectorAll('p'));
                    if (paragraphs.length > 3) {
                        const text = paragraphs.map(p => p.innerText.trim()).filter(t => t.length > 30).join('\\n');
                        if (text.length > 200) return text;
                    }
                    
                    // Strategy 4: main tag
                    const main = document.querySelector('main');
                    if (main) return main.innerText;
                    
                    return null;
                }
            """)

            browser.close()

    except Exception as e:
        logger.info(f"[news-pw] Error scraping {url}: {e}")
        return None

    if article_text:
        # Clean up
        article_text = re.sub(r"\n{3,}", "\n\n", article_text).strip()
        if len(article_text) > max_chars:
            article_text = article_text[:max_chars]
        if len(article_text) > 100:
            logger.info(f"[news-pw] Scraped {len(article_text)} chars from {url}")
            return article_text

    logger.info(f"[news-pw] No article text found at {url}")
    return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.info("Usage: python -m app.collectors.news_playwright <url>")
        sys.exit(1)
    text = scrape_article_sync(sys.argv[1])
    if text:
        logger.info(f"\n{'=' * 60}\n{text[:500]}")
