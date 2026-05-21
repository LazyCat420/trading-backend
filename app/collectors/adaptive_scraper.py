"""
Adaptive LLM Scraper -- Uses Vision LLM to generate custom JS scrapers for websites.
"""

import logging
import re
from urllib.parse import urlparse
import httpx

from app.db.connection import get_db
from app.config import settings

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """
You are an expert JavaScript scraper.
Analyze the provided screenshot of a news article.
Write a single JavaScript expression (an IIFE or just a block of code returning a string) that extracts the main body text of the article.
RULES:
1. DOM-only queries. NO network calls (fetch, XMLHttpRequest). NO websockets.
2. Target the article body specifically.
3. Strip navigation, ads, and footers from your output.
4. Return ONLY raw JavaScript code. Do NOT wrap it in markdown blockticks (```javascript).
5. The code must evaluate to a string.
"""

# Blocklist of JS patterns to prevent malicious/unwanted behavior
BLOCKLIST = [
    "fetch(",
    "XMLHttpRequest",
    "WebSocket",
    "eval(",
    "Function(",
    "process.env",
    "child_process",
    "fs.",
    "import(",
    "require("
]

def extract_domain(url: str) -> str:
    """Pulls clean domain from any URL."""
    try:
        if not url.startswith("http"):
            parsed = urlparse("http://" + url)
        else:
            parsed = urlparse(url)
            
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return ""

def validate_script(script: str) -> bool:
    """Blocklist check against forbidden JS patterns."""
    if not script:
        return False
        
    for blocked in BLOCKLIST:
        if blocked == "fetch(":
            if re.search(r'\bfetch\s*\(', script) or re.search(r'window\.fetch', script):
                logger.warning("[adaptive] Script validation failed: contains forbidden pattern 'fetch'")
                return False
        elif blocked in script:
            logger.warning(f"[adaptive] Script validation failed: contains forbidden pattern '{blocked}'")
            return False
    return True

def get_script(domain: str) -> str | None:
    """Looks up active script from DB, returns None if missing or failed."""
    if not domain:
        return None
        
    try:
        with get_db() as db:
            result = db.execute(
                "SELECT script, status FROM scraper_scripts WHERE domain = %s",
                [domain]
            ).fetchone()
            
            if result:
                script, status = result
                if status == 'active':
                    return script
    except Exception as e:
        logger.error(f"[adaptive] DB error getting script for {domain}: {e}")
        
    return None

def save_script(domain: str, script: str):
    """Inserts or updates DB record."""
    if not domain or not script:
        return
        
    try:
        with get_db() as db:
            db.execute(
                """
                INSERT INTO scraper_scripts (domain, script, script_type, status)
                VALUES (%s, %s, 'js_expression', 'active')
                ON CONFLICT (domain) DO UPDATE SET 
                    script = EXCLUDED.script,
                    status = 'active',
                    fail_count = 0
                """,
                [domain, script]
            )
    except Exception as e:
        logger.error(f"[adaptive] DB error saving script for {domain}: {e}")

def report_success(domain: str):
    """Increments success_count, updates last_success."""
    if not domain:
        return
        
    try:
        with get_db() as db:
            db.execute(
                """
                UPDATE scraper_scripts 
                SET success_count = success_count + 1, 
                    last_success = CURRENT_TIMESTAMP
                WHERE domain = %s
                """,
                [domain]
            )
    except Exception as e:
        logger.error(f"[adaptive] DB error reporting success for {domain}: {e}")

def report_failure(domain: str):
    """Increments fail_count; if fail_count >= 5 sets status='failed'."""
    if not domain:
        return
        
    try:
        with get_db() as db:
            result = db.execute(
                """
                UPDATE scraper_scripts 
                SET fail_count = fail_count + 1,
                    last_used = CURRENT_TIMESTAMP
                WHERE domain = %s
                RETURNING fail_count
                """,
                [domain]
            ).fetchone()
            
            if result and result[0] >= 5:
                db.execute(
                    "UPDATE scraper_scripts SET status = 'failed' WHERE domain = %s",
                    [domain]
                )
                logger.warning(f"[adaptive] Script for {domain} marked as failed (>= 5 failures)")
                
    except Exception as e:
        logger.error(f"[adaptive] DB error reporting failure for {domain}: {e}")

async def generate_script(domain: str, screenshot_b64: str, previous_script: str | None = None) -> str | None:
    """Sends screenshot to vision LLM with strict prompt, returns raw JS string."""
    if not screenshot_b64:
        return None
        
    content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
        }
    ]
    
    prompt = PROMPT_TEMPLATE
    if previous_script:
        prompt += f"\n\nNOTE: The previous script failed. Here is what you tried last time:\n```javascript\n{previous_script}\n```\nPlease fix it to correctly extract the article text."
        
    content.append({
        "type": "text",
        "text": prompt
    })
    
    payload = {
        "model": settings.ACTIVE_MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.1,
        "max_tokens": 1024,
    }
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            logger.info(f"[adaptive] Requesting JS script generation for {domain}...")
            r = await client.post(
                f"{settings.JETSON_VLLM_URL}/v1/chat/completions",
                json=payload,
            )
            r.raise_for_status()
            data = r.json()

            script = data["choices"][0]["message"]["content"]
            
            script = re.sub(r'^```(?:javascript)?\n|```$', '', script.strip(), flags=re.MULTILINE)
            
            return script.strip()
    except Exception as e:
        logger.error(f"[adaptive] Script generation error for {domain}: {e}")
        return None

async def _execute_script_and_evaluate(url: str, domain: str, script: str) -> str | None:
    """Helper to run the script via crawl4ai and report success/failure."""
    from app.services.scraper_client import scraper_client

    result = await scraper_client.scrape(url, engine="crawl4ai", options={"js_code": script})
    if result and result.get("success") and result.get("content") and len(result["content"]) > 50:
        report_success(domain)
        return result["content"]
    else:
        report_failure(domain)
        return None

async def run_adaptive(url: str) -> str | None:
    """Full orchestration function for the adaptive scraper."""
    domain = extract_domain(url)
    if not domain:
        return None
        
    script = get_script(domain)
    
    if script:
        if validate_script(script):
            logger.info(f"[adaptive] Found active script for {domain}, attempting scrape...")
            text = await _execute_script_and_evaluate(url, domain, script)
            if text:
                return text
            else:
                logger.info(f"[adaptive] Saved script for {domain} failed. Will attempt to regenerate.")
        else:
            logger.warning(f"[adaptive] Existing script for {domain} failed validation. Will regenerate.")
            
    logger.info(f"[adaptive] Taking screenshot for new scraper for {domain}...")
    from app.services.scraper_client import scraper_client
    
    screenshot_result = await scraper_client.scrape(url, engine="crawl4ai", options={"screenshot": True, "fast": False})
    if not screenshot_result:
        logger.error(f"[adaptive] Failed to capture screenshot for {domain}")
        return None
        
    screenshot_b64 = screenshot_result.get("screenshot_b64")
    
    if not screenshot_b64:
        logger.error(f"[adaptive] Failed to capture screenshot for {domain}")
        return None
        
    last_script = script
    for attempt in range(5):
        logger.info(f"[adaptive] Generating new scraper for {domain} (Attempt {attempt+1}/5)...")
        new_script = await generate_script(domain, screenshot_b64, previous_script=last_script)
        
        if not new_script:
            logger.error(f"[adaptive] Vision LLM returned no script for {domain}")
            break
            
        if not validate_script(new_script):
            logger.warning(f"[adaptive] Vision LLM generated an invalid script for {domain}. Retrying.")
            last_script = new_script
            continue
            
        save_script(domain, new_script)
        
        text = await _execute_script_and_evaluate(url, domain, new_script)
        if text:
            return text
            
        last_script = new_script
        
    return None
