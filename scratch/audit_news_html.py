import re
import json
import sys
from bs4 import BeautifulSoup
from app.db.connection import get_db

def is_html(text: str) -> bool:
    if not text:
        return False
    return bool(re.search(r"<!DOCTYPE html|<html|<body|<div|<p>|<script|<span", text, re.IGNORECASE))

def _extract_seeking_alpha_ssr(html: str) -> str | None:
    match = re.search(r"window\.SSR_DATA\s*=\s*(\{.*?\});?\s*</script>", html, re.DOTALL)
    if not match:
        match = re.search(r"window\.SSR_DATA\s*=\s*(\{.*?\}),?\s*\n", html, re.DOTALL)
    if not match:
        return None
    try:
        data_str = match.group(1)
        data = json.loads(data_str)
        article = data.get("article", {}).get("response", {}).get("data", {}).get("attributes", {})
        content_html = article.get("content")
        if content_html:
            soup = BeautifulSoup(content_html, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            insights = article.get("quickInsights", [])
            if insights:
                insights_text = []
                for ins in sorted(insights, key=lambda x: x.get("order", 0)):
                    q = ins.get("question", "")
                    a = ins.get("answer", "")
                    if q and a:
                        insights_text.append(f"Q: {q}\nA: {a}")
                if insights_text:
                    text = text + "\n\nQuick Insights:\n" + "\n".join(insights_text)
            return text.strip()
    except Exception as e:
        print(f"Seeking Alpha SSR extract error: {e}")
    return None

def clean_html(html: str) -> str:
    if not html:
        return ""
    
    # 1. Seeking Alpha SSR extraction
    if "seekingalpha" in html.lower() or "ssr_data" in html.lower():
        sa_text = _extract_seeking_alpha_ssr(html)
        if sa_text:
            return sa_text

    # 2. General BeautifulSoup parsing
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 50:
            return text
    except Exception as e:
        print(f"BS4 parsing error: {e}")

    # 3. Regex fallback
    cleaned = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<script[^>]*>.*?</script>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<svg[^>]*>.*?</svg>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<!--.*?-->", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def run_audit():
    with get_db() as db:
        rows = db.execute("SELECT id, ticker, publisher, title, summary FROM news_articles").fetchall()
        
    html_count = 0
    sa_count = 0
    total = len(rows)
    
    print(f"Total articles in DB: {total}")
    
    sa_raw = None
    sa_cleaned = None
    other_raw = None
    other_cleaned = None
    
    for row_id, ticker, publisher, title, summary in rows:
        if summary and is_html(summary):
            html_count += 1
            is_sa = "seekingalpha" in summary.lower() or "seeking alpha" in publisher.lower() or "seekingalpha" in title.lower()
            if is_sa:
                sa_count += 1
                if not sa_raw:
                    sa_raw = summary
                    sa_cleaned = clean_html(summary)
            else:
                if not other_raw:
                    other_raw = summary
                    other_cleaned = clean_html(summary)
                
    print(f"Articles containing HTML: {html_count} ({html_count/total*100:.1f}%)")
    print(f"Seeking Alpha HTML articles: {sa_count}")
    
    if sa_raw:
        print("\n=== SEEKING ALPHA RAW HTML (TRUNCATED 1000 CHARS) ===")
        print(sa_raw[:1000])
        print("\n=== SEEKING ALPHA CLEANED TEXT (TRUNCATED 1000 CHARS) ===")
        print(sa_cleaned[:1000])
        
    if other_raw:
        print("\n=== OTHER RAW HTML (TRUNCATED 1000 CHARS) ===")
        print(other_raw[:1000])
        print("\n=== OTHER CLEANED TEXT (TRUNCATED 1000 CHARS) ===")
        print(other_cleaned[:1000])

if __name__ == "__main__":
    run_audit()
