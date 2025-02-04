from langchain_community.utilities import SearxSearchWrapper
import os
import pprint

def search_market_news(query="What is the latest stock market news?"):
    search = SearxSearchWrapper(searx_host=os.getenv('SEARXNG_URL'), k=1)
    results = search.results(
        "Large Language Model prompt",
        num_results=5,
        categories="science",
        time_range="year",
    )
    pprint.pp(results)


if __name__ == "__main__":
    
    # Use SearxNG to search for market news after simulation
    search_market_news("stock market news")
