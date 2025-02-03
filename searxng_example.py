from langchain_community.utilities import SearxSearchWrapper
import os

def search_market_news(query="What is the latest stock market news?"):
    search = SearxSearchWrapper(searx_host=os.getenv('SEARXNG_URL'), k=1)
    result = search.run(query)
    print("Search results for:", query)
    print(result)


if __name__ == "__main__":
    
    # Use SearxNG to search for market news after simulation
    search_market_news("stock market news")
