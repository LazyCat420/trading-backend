import json
from mongo_client import get_database
from datetime import datetime
import os

# Define JSON directory
JSON_DIR = "json"

def import_trades_to_mongodb():
    client = get_database()
    if not client:
        return
    
    try:
        # Get database and collections
        database = client['trading']
        trades_collection = database['trades']
        
        # Read trades from JSON file
        trades_file = os.path.join(JSON_DIR, "trades.json")
        with open(trades_file, 'r', encoding='utf-8') as file:
            trades = json.load(file)
        
        # Convert timestamp strings to datetime objects and add metadata
        for trade in trades:
            if 'timestamp' in trade:
                trade['timestamp'] = datetime.fromisoformat(trade['timestamp'])
            trade['imported_at'] = datetime.now()
        
        # Insert all trades
        if trades:
            result = trades_collection.insert_many(trades)
            print(f"Successfully imported {len(result.inserted_ids)} trades")
        
            # Create indexes for common queries
            trades_collection.create_index('timestamp')
            trades_collection.create_index('ticker')
            trades_collection.create_index([('ticker', 1), ('timestamp', 1)])
            trades_collection.create_index('status')
            trades_collection.create_index('action')
            trades_collection.create_index('imported_at')
        else:
            print("No trades to import")
        
    except Exception as e:
        print(f"An error occurred importing trades: {e}")
    finally:
        client.close()

def import_watchlist_to_mongodb():
    client = get_database()
    if not client:
        return
    
    try:
        database = client['trading']
        watchlist_collection = database['watchlist']
        
        # Read watchlist from JSON file
        watchlist_file = os.path.join(JSON_DIR, "watchlist.json")
        with open(watchlist_file, 'r', encoding='utf-8') as file:
            watchlist_data = json.load(file)
        
        # Convert timestamps and add metadata
        for ticker, data in watchlist_data.get('performance', {}).items():
            if 'added_date' in data:
                data['added_date'] = datetime.fromisoformat(data['added_date'])
            if 'last_updated' in data:
                data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        watchlist_data['imported_at'] = datetime.now()
        
        # Insert watchlist data
        result = watchlist_collection.insert_one(watchlist_data)
        print(f"Successfully imported watchlist data")
        
        # Create indexes
        watchlist_collection.create_index('last_updated')
        watchlist_collection.create_index('watchlist')
        watchlist_collection.create_index('imported_at')
        
    except Exception as e:
        print(f"An error occurred importing watchlist: {e}")
    finally:
        client.close()

def import_news_to_mongodb():
    client = get_database()
    if not client:
        return
    
    try:
        database = client['trading']
        news_collection = database['news']
        
        # Read news from JSON file
        news_file = os.path.join(JSON_DIR, "news.json")
        with open(news_file, 'r', encoding='utf-8') as file:
            news_data = json.load(file)
        
        # Convert timestamps and add metadata
        for article in news_data:
            if 'timestamp' in article:
                article['timestamp'] = datetime.fromisoformat(article['timestamp'])
            article['imported_at'] = datetime.now()
        
        # Insert news data
        if news_data:
            result = news_collection.insert_many(news_data)
            print(f"Successfully imported {len(result.inserted_ids)} news articles")
        
            # Create indexes for news collection
            news_collection.create_index('timestamp')
            news_collection.create_index('ticker')
            news_collection.create_index([('ticker', 1), ('timestamp', 1)])
            news_collection.create_index('imported_at')
            news_collection.create_index('sentiment')
            news_collection.create_index('category')
            news_collection.create_index('impact_score')
        else:
            print("No news to import")
        
    except Exception as e:
        print(f"An error occurred importing news: {e}")
    finally:
        client.close()

def import_portfolio_to_mongodb():
    client = get_database()
    if not client:
        return
    
    try:
        database = client['trading']
        portfolio_collection = database['portfolio']
        
        # Read portfolio from JSON file
        portfolio_file = os.path.join(JSON_DIR, "portfolio.json")
        with open(portfolio_file, 'r', encoding='utf-8') as file:
            portfolio_data = json.load(file)
        
        # Add metadata
        if 'timestamp' in portfolio_data:
            portfolio_data['timestamp'] = datetime.fromisoformat(portfolio_data['timestamp'])
        portfolio_data['imported_at'] = datetime.now()
        
        # Insert portfolio data
        result = portfolio_collection.insert_one(portfolio_data)
        print(f"Successfully imported portfolio data")
        
        # Create indexes
        portfolio_collection.create_index('timestamp')
        portfolio_collection.create_index('imported_at')
        
    except Exception as e:
        print(f"An error occurred importing portfolio: {e}")
    finally:
        client.close()

def import_summary_to_mongodb():
    client = get_database()
    if not client:
        return
    
    try:
        database = client['trading']
        summary_collection = database['summary']
        
        # Read summary from JSON file
        summary_file = os.path.join(JSON_DIR, "summary.json")
        with open(summary_file, 'r', encoding='utf-8') as file:
            summary_data = json.load(file)
        
        # Convert timestamps and add metadata
        for entry in summary_data:
            if 'timestamp' in entry:
                entry['timestamp'] = datetime.fromisoformat(entry['timestamp'])
        
        # Insert summary data
        if summary_data:
            result = summary_collection.insert_many(summary_data)
            print(f"Successfully imported {len(result.inserted_ids)} summary entries")
        
            # Create indexes
            summary_collection.create_index('timestamp')
            summary_collection.create_index('imported_at')
        else:
            print("No summary data to import")
        
    except Exception as e:
        print(f"An error occurred importing summary: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    print("\nImporting trades...")
    import_trades_to_mongodb()
    print("\nImporting watchlist...")
    import_watchlist_to_mongodb()
    print("\nImporting news...")
    import_news_to_mongodb()
    print("\nImporting portfolio...")
    import_portfolio_to_mongodb()
    print("\nImporting summary...")
    import_summary_to_mongodb() 