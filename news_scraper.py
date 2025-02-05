from langchain.document_loaders import PlaywrightURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_extraction_chain
import os
import requests
import json
from datetime import datetime

def extract_urls_from_searxng(results):
    """Extract URLs from SearxNG results"""
    urls = []
    try:
        # Handle string results
        if isinstance(results, str):
            # Look for http:// or https:// patterns
            import re
            urls.extend(re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', results))
        
        # Handle list results
        elif isinstance(results, list):
            for result in results:
                if isinstance(result, dict) and 'url' in result:
                    urls.append(result['url'])
                elif isinstance(result, str):
                    urls.extend(re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', result))
                    
        return list(set(urls))  # Remove duplicates
    except Exception as e:
        print(f"Error extracting URLs: {e}")
        return []

async def scrape_news_content(urls, max_urls=5):
    """Scrape content from URLs using Playwright"""
    try:
        # Limit number of URLs to process
        urls = urls[:max_urls]
        
        # Initialize Playwright loader
        loader = PlaywrightURLLoader(
            urls=urls,
            remove_selectors=["nav", "header", "footer", "aside"],
            continue_on_failure=True
        )
        
        # Load and parse documents
        documents = await loader.aload()
        
        # Split text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        return texts
    except Exception as e:
        print(f"Error scraping content: {e}")
        return []

def analyze_with_ollama(content):
    """Analyze scraped content with Ollama"""
    try:
        prompt = {
            "model": os.getenv('OLLAMA_MODEL'),  # Provide default model
            "messages": [{
                "role": "user",
                "content": f"""Analyze this financial news content and extract key information:
                {content}
                
                Provide analysis in JSON format with these fields:
                {{
                    "summary": "Brief summary of the news",
                    "companies_mentioned": ["List of company names and tickers"],
                    "sentiment": "POSITIVE/NEGATIVE/NEUTRAL",
                    "key_points": ["List of main points"],
                    "market_impact": "Potential impact on market"
                }}
                
                Ensure the response is valid JSON format.
                """
            }],
            "stream": False
        }

        response = requests.post(
            f"{os.getenv('OLLAMA_URL', 'http://10.0.0.29:11434').rstrip('/v1')}/api/chat",
            json=prompt,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'message' in result and 'content' in result['message']:
                content = result['message']['content']
                # Ensure content is a string before calling strip
                if isinstance(content, str):
                    # Clean the content to ensure it's valid JSON
                    content = content.strip()
                    if content.startswith('```json'):
                        content = content[7:]
                    if content.endswith('```'):
                        content = content[:-3]
                    content = content.strip()
                    
                    # Parse the JSON content
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON content: {e}")
                        print(f"Raw content: {content}")
                        return {
                            "summary": content,
                            "companies_mentioned": [],
                            "sentiment": "NEUTRAL",
                            "key_points": [],
                            "market_impact": "Unknown"
                        }
        return None
    except Exception as e:
        print(f"Error analyzing with Ollama: {e}")
        return None

async def process_news(query):
    """Main function to process news"""
    from searxng import search_market_news
    
    # Get search results
    results = search_market_news(query)
    
    # Extract URLs
    urls = extract_urls_from_searxng(results)
    if not urls:
        print("No URLs found in results")
        return []
        
    # Scrape content
    documents = await scrape_news_content(urls)
    
    # Analyze each document
    analyses = []
    for doc in documents:
        analysis = analyze_with_ollama(doc.page_content)
        if analysis:
            try:
                # Analysis is already a dict now
                analysis['url'] = doc.metadata.get('source')
                analysis['timestamp'] = datetime.now().isoformat()
                analyses.append(analysis)
            except Exception as e:
                print(f"Error processing analysis: {e}")
                
    return analyses

# Usage example:
if __name__ == "__main__":
    import asyncio
    
    async def main():
        analyses = await process_news("AAPL stock news today")
        print(json.dumps(analyses, indent=2))
    
    asyncio.run(main()) 