from langchain_community.utilities import SearxSearchWrapper
import os
import requests
import json
from dotenv import load_dotenv
import pprint

# Load environment variables from .env file
load_dotenv()

def get_optimal_result_count(query):
    """Get optimal number of results (5-10) based on query complexity"""
    try:
        prompt = {
            "model": os.getenv('OLLAMA_MODEL'),
            "messages": [{
                "role": "user",
                "content": f"""Given this search query: "{query}"
Determine how many search results (between 5 and 10) would be optimal to analyze.
Consider:
- Broader topics need more results
- Specific company queries might need fewer results
- Market analysis usually benefits from multiple perspectives

Respond with ONLY a number between 5 and 10."""
            }],
            "stream": False
        }

        response = requests.post(
            'http://10.0.0.29:11434/api/chat',
            json=prompt,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()['message']['content'].strip()
            try:
                k = int(result)
                return max(1, min(1, k))  # Ensure result is between 1 and 2
            except ValueError:
                return 1  # Default if parsing fails
    except Exception as e:
        print(f"Error getting result count: {e}")
        return 1  # Default fallback

def search_market_news(query="What is the latest stock market news?"):
    """Get structured market news results including URLs from SearxNG"""
    try:
        k = get_optimal_result_count(query)
        
        # Initialize SearxNG wrapper with base parameters
        searx_host = os.getenv('SEARXNG_URL')
        if not searx_host:
            raise ValueError("SEARXNG_URL environment variable is not set")
            
        search = SearxSearchWrapper(searx_host=searx_host)
        
        # Get results using the results() method instead of direct search
        results = search.results(
            query,
            num_results=k,
            engines=["news", "finance"],
            time_range="day",
            language="en"
        )
        
        if results:
            print(f"\nSearch results ({k} requested) for: {query}")
            processed_results = []
            
            for result in results:
                processed_result = {
                    'title': result.get('title', 'No title'),
                    'snippet': result.get('snippet', ''),
                    'url': result.get('link') or result.get('url', ''),
                    'source': result.get('source', 'Unknown source'),
                    'engines': result.get('engines', []),
                    'score': result.get('score', 0),
                    'publishedDate': result.get('publishedDate', result.get('published_date', 'No date'))
                }
                
                processed_results.append(processed_result)
                
                # Use pprint for cleaner output
                print("\nResult details:")
                pprint.pprint(processed_result)
                print("-" * 50)
            
            return processed_results
        
        print(f"No results found for query: {query}")
        return None
        
    except Exception as e:
        print(f"Error in SearxNG search: {e}")
        return None

if __name__ == "__main__":
    # Test the search function
    results = search_market_news("stock market news")
    if results:
        print("\nExtracted URLs:")
        urls = [result['url'] for result in results if result.get('url')]
        for url in urls:
            print(f"- {url}")
