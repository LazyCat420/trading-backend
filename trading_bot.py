import json
from datetime import datetime, timedelta, date
import yfinance as yf
import requests
from searxng import search_market_news
from automated_trading_bot_simulator import run_trading_simulator
import os
import time
import pandas as pd
import numpy as np
from scipy import stats
from mongo_client import get_database
import pytz
import argparse
import base64
import logging
from bson import ObjectId
from pymongo import UpdateOne
from bson.decimal128 import Decimal128

class TradingBot:
    def __init__(self, initial_balance=100000, testing_mode=False, personality_type="balanced"):
        # Initialize basic attributes first
        self.initial_balance = float(initial_balance)
        self.balance = self.initial_balance
        self.portfolio = {}
        self.testing_mode = testing_mode
        self.personality_type = personality_type
        
        # Define sectors before using them
        self.sectors = [
            "Technology", "Healthcare", "Financial", "Consumer Cyclical",
            "Energy", "Industrial", "Communication Services", "Materials",
            "Real Estate", "Utilities", "Consumer Defensive"
        ]
        
        # Initialize watchlist as a list instead of sector dict
        self.watchlist = []
        self.sector_watchlists = {sector: [] for sector in self.sectors}
        
        # Initialize MongoDB connection
        self.client = get_database()
        if not self.client:
            raise Exception("Could not connect to MongoDB")
        
        self.db = self.client['trading']
        
        # Setup collections with proper indexes
        self.setup_mongodb_collections()
        
        # Initialize data lists
        self.trades_data = []
        self.news_data = []
        self.history_data = []
        
        # Other initializations
        self.momentum_window = 20
        self.watchlist_performance = {}
        
        # Set personality traits after basic initialization
        self.personality_traits = self.set_personality(personality_type)
        
        # Load data from MongoDB
        self.load_data()
        self.load_watchlist()

    def setup_mongodb_collections(self):
        """Setup MongoDB collections with proper indexes"""
        try:
            # News collection
            self.news_collection = self.db['news']
            self.news_collection.drop_indexes()  # Drop existing indexes
            self.news_collection.create_index([('url', 1)], unique=True, name='unique_url')
            self.news_collection.create_index([('timestamp', -1)], name='news_timestamp')
            
            # Analysis collection
            self.analysis_collection = self.db['analysis']
            self.analysis_collection.drop_indexes()
            self.analysis_collection.create_index([('news_url', 1)], unique=True, name='unique_news_url')
            self.analysis_collection.create_index([('timestamp', -1)], name='analysis_timestamp')
            
            # Trends collection
            self.trends_collection = self.db['trends']
            self.trends_collection.drop_indexes()
            self.trends_collection.create_index(
                [('timestamp', 1), ('trend', 1), ('source_url', 1)],
                unique=True,
                name='unique_trend'
            )
            
            # Other collections
            self.trades_collection = self.db['trades']
            self.watchlist_collection = self.db['watchlist']
            self.portfolio_collection = self.db['portfolio']
            self.summary_collection = self.db['summary']
            self.research_collection = self.db['research']
            self.thought_chains_collection = self.db['thought_chains']
            self.thought_trees_collection = self.db['thought_trees']
            self.thought_graphs_collection = self.db['thought_graphs']
            
            print("MongoDB collections and indexes setup complete")
            
        except Exception as e:
            print(f"Error setting up MongoDB collections: {e}")
            raise

    def save_to_mongodb(self, collection, data, index_fields=None):
        """Generic method to save data to MongoDB with indexing"""
        try:
            if isinstance(data, list):
                if data:
                    # Use update_many with upsert for lists
                    operations = []
                    for doc in data:
                        operations.append(
                            UpdateOne(
                                {'timestamp': doc.get('timestamp')},  # Use timestamp as unique identifier
                                {'$set': doc},
                                upsert=True
                            )
                        )
                    result = collection.bulk_write(operations)
                    print(f"Processed {len(operations)} documents")
            else:
                # For single documents, use update_one with upsert
                if '_id' in data:
                    # If _id exists, use it as filter
                    filter_doc = {'_id': data['_id']}
                else:
                    # Otherwise use timestamp as unique identifier
                    filter_doc = {'timestamp': data.get('timestamp', datetime.now())}
                
                result = collection.update_one(
                    filter_doc,
                    {'$set': data},
                    upsert=True
                )
                print("Document updated/inserted")
            
            # Create indexes if specified
            if index_fields:
                for field in index_fields:
                    collection.create_index(field)
                    
        except Exception as e:
            print(f"Error saving to MongoDB: {e}")

    def load_from_mongodb(self, collection, query=None, sort=None, limit=None):
        """Generic method to load data from MongoDB"""
        try:
            if query is None:
                query = {}
            
            cursor = collection.find(query)
            
            if sort:
                cursor = cursor.sort(sort)
            if limit:
                cursor = cursor.limit(limit)
                
            return list(cursor)
        except Exception as e:
            print(f"Error loading from MongoDB: {e}")
            return []

    def save_data(self):
        """Save current state to MongoDB"""
        data = {
            'portfolio': self.portfolio,
            'watchlist': self.watchlist,
            'balance': self.balance,
            'last_updated': datetime.now()
        }
        self.save_to_mongodb(self.portfolio_collection, data, ['last_updated'])

    def load_data(self):
        """Load latest state from MongoDB"""
        latest_data = self.load_from_mongodb(
            self.portfolio_collection, 
            sort=[('last_updated', -1)],
            limit=1
        )
        if latest_data:
            data = latest_data[0]
            self.portfolio = data.get('portfolio', {})
            self.watchlist = data.get('watchlist', {})
            self.balance = data.get('balance', 100000)

    def save_research_data(self, research_data):
        """Save research and analysis data to MongoDB"""
        try:
            if isinstance(research_data, list):
                # Handle list of research items
                for item in research_data:
                    if isinstance(item, dict):
                        item['timestamp'] = datetime.now()
                data_to_save = research_data
            else:
                # Handle single research item
                if isinstance(research_data, dict):
                    research_data['timestamp'] = datetime.now()
                data_to_save = research_data
            
            self.save_to_mongodb(
                self.research_collection, 
                data_to_save,
                ['timestamp', 'ticker', 'type']
            )
        except Exception as e:
            print(f"Error saving research data: {e}")

    def get_research_data(self, ticker=None, data_type=None, limit=100):
        """Get research data from MongoDB"""
        query = {}
        if ticker:
            query['ticker'] = ticker
        if data_type:
            query['type'] = data_type
            
        return self.load_from_mongodb(
            self.research_collection,
            query=query,
            sort=[('timestamp', -1)],
            limit=limit
        )

    def generate_market_queries(self):
        """Generate various market research queries"""
        queries = [
            "top performing stocks today",
            "stocks with unusual trading volume",
            "stocks with positive analyst upgrades",
            "trending stocks in news",
            "stocks with high momentum",
            "stocks breaking out today",
            "stocks with earnings surprise",
            "stocks with insider buying"
        ]
        
        # Add sector-specific queries
        for sector in self.sectors:
            queries.extend([
                f"best {sector} stocks to buy",
                f"trending {sector} stocks",
                f"{sector} stocks with growth potential",
                f"undervalued {sector} stocks"
            ])
            
        return queries

    def search_sector_stocks(self):
        """Search for stocks by sector"""
        potential_stocks = set()
        
        for sector in self.sectors:
            try:
                # Search for sector-specific news and stocks
                news = search_market_news(f"best performing {sector} sector stocks today")
                if news:
                    # Ask Ollama to analyze the sector
                    prompt = f"""
                    Based on this news about {sector} sector:
                    {news}
                    
                    Suggest 3 promising stock symbols from this sector.
                    Respond with ONLY the stock symbols separated by commas.
                    """
                    
                    response = self.get_ollama_response(prompt)
                    if response:
                        # Extract stock symbols
                        import re
                        symbols = re.findall(r'\b[A-Z]{1,5}\b', response.upper())
                        potential_stocks.update(symbols)
                
                time.sleep(2)  # Avoid rate limiting
                
            except Exception as e:
                print(f"Error searching {sector} sector: {e}")
                continue
        
        return list(potential_stocks)

    def calculate_momentum(self, ticker):
        """Calculate stock momentum"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=f"{self.momentum_window}d")
            if len(hist) < self.momentum_window:
                return 0
            
            returns = np.log(hist['Close']/hist['Close'].shift(1))
            return returns.mean() * 252  # Annualized momentum
        except Exception as e:
            print(f"Error calculating momentum for {ticker}: {e}")
            return 0

    def get_stock_data(self, ticker):
        """Get current stock data using yfinance"""
        try:
            stock = yf.Ticker(ticker)
            
            # Get historical data for price and momentum
            hist = stock.history(period=f"{self.momentum_window}d")
            if hist.empty:
                return None
                
            current_price = hist['Close'].iloc[-1]
            prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            
            # Calculate momentum using log returns
            if len(hist) >= self.momentum_window:
                returns = np.log(hist['Close']/hist['Close'].shift(1))
                momentum = returns.mean() * 252  # Annualized
            else:
                momentum = 0
            
            # Get additional info
            info = stock.info
            
            data = {
                'current_price': current_price,
                'prev_price': prev_price,
                'volume': info.get('volume', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('forwardPE', 0),
                'fifty_day_avg': info.get('fiftyDayAverage', current_price),
                'momentum': momentum,
                'daily_change': ((current_price - prev_price) / prev_price) * 100,
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'beta': info.get('beta', 1.0),
                'dividend_yield': info.get('dividendYield', 0)
            }
            return data
            
        except Exception as e:
            print(f"Error getting data for {ticker}: {e}")
            return None

    def clean_text(self, text):
        """Clean text for API requests"""
        return text.replace('\n', ' ').strip()

    def get_ollama_response(self, prompt_text):
        """Make API call to Ollama with improved error handling and JSON cleaning"""
        try:
            # Get base URL and remove /v1 if present, ensure no trailing slash
            ollama_url = (os.getenv('OLLAMA_URL', 'http://10.0.0.29:11434')).replace('/v1', '').rstrip('/')

            # Prepare request with proper format
            request_body = {
                "model": os.getenv('OLLAMA_MODEL'),
                "messages": [{
                    "role": "user", 
                    "content": self.clean_text(prompt_text)
                }],
                "stream": True
            }

            # Make the request with streaming
            response = requests.post(
                f"{ollama_url}/api/chat",
                json=request_body,
                headers={'Content-Type': 'application/json'},
                timeout=30,
                stream=True
            )

            # Process the streaming response
            full_content = ""
            for line in response.iter_lines():
                if line:
                    try:
                        json_response = json.loads(line)
                        if 'message' in json_response and 'content' in json_response['message']:
                            content = json_response['message']['content']
                            full_content += content
                            print(content, end='', flush=True)
                    except json.JSONDecodeError:
                        continue

            # Return string content unless JSON is explicitly requested
            if 'json' in prompt_text.lower():
                try:
                    # Extract potential JSON parts
                    json_start = full_content.find('{')
                    json_end = full_content.rfind('}') + 1
                    if json_start != -1 and json_end != -1:
                        json_str = full_content[json_start:json_end]
                        return json.loads(json_str)
                except:
                    pass
            
            return full_content.strip()

        except Exception as e:
            print(f"Request error: {e}")
            return None

    def load_news(self):
        """Load existing news from JSON file"""
        try:
            if os.path.exists('news.json'):
                with open('news.json', 'r') as f:
                    self.news_data = json.load(f)
            else:
                self.news_data = []
        except Exception as e:
            print(f"Error loading news: {e}")
            self.news_data = []

    def save_news(self):
        """Save news data to JSON file"""
        try:
            if not hasattr(self, 'news_data'):
                self.news_data = []
            
            # Save to news.json
            with open('news.json', 'w') as f:
                json.dump(self.news_data, f, indent=4, ensure_ascii=False)
            print("News data saved successfully")
        except Exception as e:
            print(f"Error saving news: {e}")

    def load_trades(self):
        """Load existing trades from JSON file"""
        try:
            if os.path.exists('trades.json'):
                with open('trades.json', 'r') as f:
                    self.trades_data = json.load(f)
            else:
                self.trades_data = []
        except Exception as e:
            print(f"Error loading trades: {e}")
            self.trades_data = []

    def save_trades(self):
        """Save trades data to JSON file"""
        try:
            if not hasattr(self, 'trades_data'):
                self.trades_data = []
            
            with open('trades.json', 'w') as f:
                json.dump(self.trades_data, f, indent=4, ensure_ascii=False)
            print("Trades data saved successfully")
        except Exception as e:
            print(f"Error saving trades: {e}")

    def extract_tickers(self, text):
        """Extract stock tickers from text using regex and common patterns"""
        try:
            import re
            
            # Initialize empty set for unique tickers
            tickers = set()
            
            # Common patterns for stock tickers
            patterns = [
                r'\b[A-Z]{1,5}\b',  # 1-5 uppercase letters
                r'\$[A-Z]{1,5}\b',  # Tickers with $ prefix
                r'\([A-Z]{1,5}\)',  # Tickers in parentheses
                r'NYSE:\s*([A-Z]{1,5})',  # NYSE listings
                r'NASDAQ:\s*([A-Z]{1,5})'  # NASDAQ listings
            ]
            
            # Extract potential tickers using patterns
            for pattern in patterns:
                matches = re.findall(pattern, text)
                tickers.update(matches)
            
            # Filter out common words and invalid tickers
            exclude_words = {
                'A', 'I', 'CEO', 'CFO', 'US', 'USA', 'NYSE', 'NASDAQ', 'IPO', 
                'AI', 'API', 'ETF', 'GDP', 'COVID', 'FDA', 'SEC', 'THE'
            }
            
            # Validate tickers using yfinance
            valid_tickers = []
            for ticker in tickers:
                # Clean the ticker (remove $ and parentheses)
                clean_ticker = ticker.strip('$()').strip()
                
                # Skip if it's in exclude words
                if clean_ticker in exclude_words:
                    continue
                    
                try:
                    # Quick validation using yfinance
                    stock = yf.Ticker(clean_ticker)
                    info = stock.info
                    if info and 'symbol' in info:
                        valid_tickers.append(clean_ticker)
                except:
                    continue
            
            # Debug output
            if valid_tickers:
                print(f"Extracted tickers: {', '.join(valid_tickers)}")
            
            return valid_tickers
            
        except Exception as e:
            print(f"Error extracting tickers: {e}")
            return []

    def analyze_sentiment(self, text):
        """Analyze sentiment of text using simple keyword matching"""
        try:
            # Positive and negative word lists
            positive_words = {
                'up', 'rise', 'gain', 'profit', 'growth', 'positive', 'bullish',
                'outperform', 'beat', 'strong', 'higher', 'increase', 'improved'
            }
            
            negative_words = {
                'down', 'fall', 'loss', 'decline', 'negative', 'bearish',
                'underperform', 'miss', 'weak', 'lower', 'decrease', 'reduced'
            }
            
            # Convert text to lowercase for matching
            text = text.lower()
            
            # Count occurrences
            positive_count = sum(1 for word in positive_words if word in text)
            negative_count = sum(1 for word in negative_words if word in text)
            
            # Determine sentiment
            if positive_count > negative_count:
                return "positive"
            elif negative_count > positive_count:
                return "negative"
            else:
                return "neutral"
                
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return "neutral"

    def categorize_news(self, text):
        """Categorize news into relevant market sectors"""
        try:
            # Define sector keywords
            sector_keywords = {
                "Technology": ['tech', 'software', 'hardware', 'ai', 'cloud', 'cyber'],
                "Healthcare": ['health', 'medical', 'biotech', 'pharma', 'drug'],
                "Financial": ['bank', 'finance', 'invest', 'trading', 'insurance'],
                "Consumer": ['retail', 'consumer', 'shopping', 'goods', 'services'],
                "Energy": ['energy', 'oil', 'gas', 'renewable', 'solar', 'wind'],
                "Industrial": ['manufacturing', 'industry', 'construction', 'machinery']
            }
            
            # Convert text to lowercase
            text = text.lower()
            
            # Count keyword matches for each sector
            sector_matches = {}
            for sector, keywords in sector_keywords.items():
                matches = sum(1 for keyword in keywords if keyword in text)
                if matches > 0:
                    sector_matches[sector] = matches
            
            # Return the sector with most matches, or "Other"
            if sector_matches:
                return max(sector_matches.items(), key=lambda x: x[1])[0]
            return "Other"
            
        except Exception as e:
            print(f"Error categorizing news: {e}")
            return "Other"

    def categorize_tickers(self, text):
        """Categorize tickers from news text using Ollama"""
        try:
            prompt = f"""
            Categorize the tickers in this news text: {text}
            Respond with ONLY a comma-separated list of tickers, no other text.
            """
            response = self.get_ollama_response(prompt)
            if response and isinstance(response, str):
                return response.strip().upper()
            return ""
        except Exception as e:
            print(f"Error categorizing tickers: {e}")
            return ""
    
    def list_tickers(self, text):
        """List tickers from news text using Ollama"""
        try:
            prompt = f"""
            Extract all stock tickers from this news text: {text}
            Respond with a list of tickers separated by commas.
            """
            response = self.get_ollama_response(prompt)
            if response:
                return response.strip().upper()
            return ""
        except Exception as e:
            print(f"Error listing tickers: {e}")
            return ""
    

    def calculate_news_impact(self, news_item):
        """Calculate potential market impact of news"""
        impact = 0
        try:
            # Base impact on sentiment
            if news_item['sentiment'] == 'positive':
                impact += 0.5
            elif news_item['sentiment'] == 'negative':
                impact -= 0.5
            
            # Adjust based on category
            category_weights = {
                'EARNINGS': 1.0,
                'MERGER': 0.8,
                'MARKET_SENTIMENT': 0.6,
                'COMPANY_NEWS': 0.4,
                'INDUSTRY_NEWS': 0.3,
                'REGULATORY': 0.7,
                'OTHER': 0.2
            }
            impact *= category_weights.get(news_item.get('category', 'OTHER'), 0.2)
            
            return round(impact, 2)
        except Exception as e:
            print(f"Error calculating news impact: {e}")
            return 0

    def add_news(self, news_items, ticker=None):
        """Add news items to the news data with enhanced features"""
        try:
            if not hasattr(self, 'news_data'):
                self.news_data = []
            
            # Helper function to check for similar content
            def is_duplicate(content, existing_news):
                for news in existing_news[-50:]:  # Check last 50 news items
                    if 'content' in news and content in news['content']:
                        return True
                return False
            
            if isinstance(news_items, str):
                content = news_items
                if not is_duplicate(content, self.news_data):
                    news_item = {
                        'timestamp': datetime.now().isoformat(),
                        'ticker': ticker,
                        'content': content,
                        'source': 'market_news',
                        'sentiment': self.analyze_sentiment(content),
                        'category': self.categorize_news(content),
                        'url': None
                    }
                    news_item['impact_score'] = self.calculate_news_impact(news_item)
                    self.news_data.append(news_item)
            
            elif isinstance(news_items, list):
                for item in news_items:
                    if isinstance(item, dict):
                        if 'timestamp' not in item:
                            item['timestamp'] = datetime.now().isoformat()
                        if not is_duplicate(item.get('content', ''), self.news_data):
                            if 'sentiment' not in item:
                                item['sentiment'] = self.analyze_sentiment(item.get('content', ''))
                            if 'category' not in item:
                                item['category'] = self.categorize_news(item.get('content', ''))
                            item['impact_score'] = self.calculate_news_impact(item)
                            self.news_data.append(item)
                    else:
                        content = str(item)
                        if not is_duplicate(content, self.news_data):
                            news_item = {
                                'timestamp': datetime.now().isoformat(),
                                'ticker': ticker,
                                'content': content,
                                'source': 'market_news',
                                'sentiment': self.analyze_sentiment(content),
                                'category': self.categorize_news(content),
                                'url': None
                            }
                            news_item['impact_score'] = self.calculate_news_impact(news_item)
                            self.news_data.append(news_item)
            
            # Clean up old news (keep last 1000 items)
            if len(self.news_data) > 1000:
                self.news_data = sorted(self.news_data, key=lambda x: x['timestamp'], reverse=True)[:1000]
            
        except Exception as e:
            print(f"Error adding news: {e}")

    def analyze_stock(self, ticker):
        try:
            # Get multiple types of news
            general_news = search_market_news(f"{ticker} stock news analysis")
            financial_news = search_market_news(f"{ticker} financial results earnings")
            sector_news = search_market_news(f"{ticker} industry sector analysis")
            
            news = f"General: {general_news}\nFinancial: {financial_news}\nSector: {sector_news}"
            
            # Log each type of news separately using the class method
            if general_news:
                self.add_news(general_news, ticker)
            if financial_news:
                self.add_news(financial_news, ticker)
            if sector_news:
                self.add_news(sector_news, ticker)
                
            if not news:
                news = "No recent news available"

            # Retrieve existing research data for this ticker
            stock_research = self.get_research_data(ticker=ticker, limit=5) # Get last 5 research items

            # Incorporate research data into signal logic
            signals = {
                'price_vs_50d': stock_data['current_price'] > stock_data['fifty_day_avg'],
                'momentum': stock_data['momentum'] > 0,
                'daily_trend': stock_data['daily_change'] > 0,
                'volume': stock_data['volume'] > 0,
                'positive_research': any(r.get('analysis', {}).get('sentiment') == 'bullish' for r in stock_research) # Example
            }
            
            # Count positive signals
            positive_signals = sum(1 for signal in signals.values() if signal)
            
            # Get current position and value
            current_shares = self.portfolio.get(ticker, 0)
            position_value = current_shares * stock_data['current_price']
            portfolio_value = self.get_portfolio_value()
            
            # Decision logic
            if current_shares == 0:  # No position yet
                if positive_signals >= 3:  # At least 3 positive signals for new position
                    return f"BUY - Strong signals ({positive_signals}/4) suggest entry point"
            else:  # Existing position
                if positive_signals <= 1:  # Weak signals
                    return f"SELL - Weak signals ({positive_signals}/4) suggest exit"
                elif stock_data['daily_change'] < -2:  # Stop loss
                    return f"SELL - Stop loss triggered at {stock_data['daily_change']:.2f}%"
                elif position_value > portfolio_value * 0.2:  # Position too large
                    return f"SELL - Position size exceeds 20% of portfolio"
                elif positive_signals >= 3:  # Strong position
                    if position_value < portfolio_value * 0.1:  # Can add more
                        return f"BUY - Strong signals ({positive_signals}/4) and position size allows growth"
            
            return "HOLD - No clear signals"

        except Exception as e:
            print(f"Error fetching news for {ticker}: {e}")
            news = "Error fetching news"

        stock_data = self.get_stock_data(ticker)
        if not stock_data or stock_data['current_price'] == 0:
            return "HOLD - Insufficient data"

    def get_portfolio_value(self):
        """Calculate total portfolio value from MongoDB"""
        try:
            total_value = float(self.balance)  # Ensure balance is float
            
            # Get all portfolio positions from MongoDB
            portfolio_positions = self.portfolio_collection.find({})
            
            for position in portfolio_positions:
                ticker = position.get('ticker')
                # Convert Decimal128 shares to float
                shares = float(str(position.get('shares', 0)))
                
                if shares > 0:  # Only calculate for positions with shares
                    stock_data = self.get_stock_data(ticker)
                    if stock_data and stock_data.get('current_price'):
                        current_price = float(stock_data['current_price'])
                        position_value = shares * current_price
                        
                        if position_value > 0 and position_value < 1000000:  # Sanity check
                            total_value += position_value
                            print(f"Position value for {ticker}: ${position_value:.2f} ({shares} shares @ ${current_price:.2f})")
                        else:
                            print(f"Warning: Suspicious position value for {ticker}: ${position_value}")
                
            print(f"Total portfolio value: ${total_value:.2f}")
            return total_value
                
        except Exception as e:
            print(f"Error calculating portfolio value: {e}")
            return float(self.initial_balance)

    def execute_trade(self, ticker, action, shares):
        """Execute a trade with proper MongoDB decimal handling"""
        try:
            print(f"\n=== Execute Trade Debug ===")
            print(f"Ticker: {ticker}")
            print(f"Action: {action}")
            print(f"Shares Type: {type(shares)}")
            print(f"Shares Value: {shares}")
            print(f"Current Portfolio: {json.dumps(self.portfolio, indent=2)}")

            # Validate shares is a number
            if not isinstance(shares, (int, float)):
                print(f"Error: Invalid shares type: {type(shares)}")
                return False

            shares = float(shares)  # Ensure shares is a float
            stock_data = self.get_stock_data(ticker)

            if not isinstance(stock_data, dict) or 'current_price' not in stock_data:
                print(f"Invalid price data for {ticker}: {stock_data}")
                return False

            current_price = float(stock_data['current_price'])
            amount = current_price * shares

            print(f"Current Price: ${current_price}")
            print(f"Total Amount: ${amount}")

            # Create trade document
            trade = {
                'timestamp': datetime.now(),
                'ticker': ticker,
                'action': action,
                'shares': Decimal128(str(shares)),
                'price': Decimal128(str(current_price)),
                'amount': Decimal128(str(amount)),
                'status': 'executed',
                'personality': self.personality_type
            }
            
            try:
                print("\nUpdating portfolio...")
                print(f"Before Update - Portfolio: {json.dumps(self.portfolio, indent=2)}")
                print(f"Before Update - Balance: ${self.balance}")

                # Insert trade record
                self.trades_collection.insert_one(trade)
                
                # Update portfolio
                if action == "BUY":
                    print("\nProcessing BUY...")
                    # Validate sufficient funds
                    if amount > self.balance:
                        print(f"Insufficient funds: Need ${amount}, have ${self.balance}")
                        return False

                    # Deduct from balance
                    self.balance -= amount
                    
                    # Add to portfolio
                    if ticker in self.portfolio:
                        print(f"Updating existing position for {ticker}")
                        current_position = self.portfolio[ticker]
                        print(f"Current position type: {type(current_position)}")
                        print(f"Current position data: {current_position}")
                        
                        if isinstance(current_position, dict):
                            current_shares = float(current_position.get('shares', 0))
                        else:
                            current_shares = float(current_position)
                        
                        new_shares = current_shares + shares
                        self.portfolio[ticker] = {
                            'shares': new_shares,
                            'current_price': current_price,
                            'market_value': new_shares * current_price
                        }
                    else:
                        print(f"Creating new position for {ticker}")
                        self.portfolio[ticker] = {
                            'shares': shares,
                            'current_price': current_price,
                            'market_value': amount
                        }
                    
                elif action == "SELL":
                    print("\nProcessing SELL...")
                    # Validate sufficient shares
                    if ticker not in self.portfolio:
                        print(f"No shares of {ticker} in portfolio")
                        return False
                    
                    current_position = self.portfolio[ticker]
                    print(f"Current position type: {type(current_position)}")
                    print(f"Current position data: {current_position}")
                    
                    if isinstance(current_position, dict):
                        current_shares = float(current_position.get('shares', 0))
                    else:
                        current_shares = float(current_position)
                    
                    if current_shares < shares:
                        print(f"Insufficient shares: Have {current_shares}, trying to sell {shares}")
                        return False

                    # Add to balance
                    self.balance += amount
                    
                    # Remove from portfolio
                    new_shares = current_shares - shares
                    if new_shares <= 0:
                        del self.portfolio[ticker]
                    else:
                        self.portfolio[ticker] = {
                            'shares': new_shares,
                            'current_price': current_price,
                            'market_value': new_shares * current_price
                        }
                
                print(f"\nAfter Update - Portfolio: {json.dumps(self.portfolio, indent=2)}")
                print(f"After Update - Balance: ${self.balance}")
                
                # Save updated portfolio state
                portfolio_update = {
                    'portfolio': self.portfolio,
                    'balance': self.balance,
                    'last_updated': datetime.now()
                }
                self.portfolio_collection.update_one(
                    {'_id': ObjectId()}, 
                    {'$set': portfolio_update},
                    upsert=True
                )
                
                print(f"\nTrade executed successfully: {action} {shares} shares of {ticker} at ${current_price}")
                return True
                
            except Exception as e:
                print(f"Error executing trade: {e}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                return False
                
        except Exception as e:
            print(f"Error in execute_trade: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return False

    def load_watchlist(self):
        """Load watchlist and performance data from MongoDB"""
        try:
            latest_watchlist = self.load_from_mongodb(
                self.watchlist_collection,
                sort=[('last_updated', -1)],
                limit=1
            )
            if latest_watchlist:
                data = latest_watchlist[0]
                self.watchlist = data.get('watchlist', {})
                self.watchlist_performance = data.get('performance', {})
        except Exception as e:
            print(f"Error loading watchlist from MongoDB: {e}")
            self.watchlist = {}
            self.watchlist_performance = {}

    def save_watchlist(self):
        """Save watchlist and performance data to MongoDB"""
        try:
            data = {
                'watchlist': self.watchlist,
                'performance': self.watchlist_performance,
                'last_updated': datetime.now()
            }
            self.save_to_mongodb(
                self.watchlist_collection,
                data,
                ['last_updated', 'watchlist']
            )
            print("Watchlist saved to MongoDB successfully")
        except Exception as e:
            print(f"Error saving watchlist to MongoDB: {e}")

    def add_to_watchlist(self, ticker):
        """Add a stock to watchlist with validation"""
        try:
            ticker = ticker.upper()
            if ticker in self.watchlist:
                print(f"{ticker} already in watchlist")
                return False

            # Validate ticker exists and get current data
            print(f"Validating ticker: {ticker}")
            stock = yf.Ticker(ticker)
            
            # Get current price using history
            hist = stock.history(period="1d")
            if hist.empty:
                print(f"Skipping {ticker} - No price data available")
                return False
                
            current_price = hist['Close'].iloc[-1]
            if not current_price or current_price <= 0:
                print(f"Skipping {ticker} - Invalid price: {current_price}")
                return False
                
            # Get additional info
            info = stock.info
            if not info:
                print(f"Skipping {ticker} - No stock info available")
                return False
                
            # Validate trading volume
            volume = info.get('volume', 0)
            if volume < 100000:  # Minimum daily volume threshold
                print(f"Skipping {ticker} - Insufficient trading volume: {volume}")
                return False
                
            # Get news for the stock
            print(f"Getting news for {ticker}...")
            stock_news = search_market_news(f"{ticker} stock news analysis recent developments")
            if stock_news:
                self.save_searxng_results(f"{ticker}_initial", stock_news)
            
            # Add to watchlist
            self.watchlist.append(ticker)
            self.watchlist_performance[ticker] = {
                'added_date': datetime.now(),
                'initial_price': current_price,
                'performance': 0,
                'success_trades': 0,
                'failed_trades': 0,
                'volume': volume,
                'market_cap': info.get('marketCap', 0),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown')
            }
            
            self.save_watchlist()  # Save to MongoDB
            print(f"Successfully added {ticker} to watchlist at ${current_price:.2f}")
            return True
            
        except Exception as e:
            print(f"Error adding {ticker} to watchlist: {e}")
            return False

    def remove_from_watchlist(self, ticker, reason="manual"):
        """Remove a stock from watchlist"""
        ticker = ticker.upper()
        if ticker in self.watchlist:
            self.watchlist.remove(ticker)
            if ticker in self.watchlist_performance:
                # Archive the performance data before removing
                archive_data = {
                    'ticker': ticker,
                    'removal_date': datetime.now(),
                    'removal_reason': reason,
                    'performance_history': self.watchlist_performance[ticker]
                }
                self.save_to_mongodb(
                    self.db['watchlist_history'],
                    archive_data,
                    ['ticker', 'removal_date']
                )
                del self.watchlist_performance[ticker]
            self.save_watchlist()  # Save to MongoDB
            print(f"Removed {ticker} from watchlist. Reason: {reason}")
            return True
        return False

    def update_stock_performance(self, ticker):
        """Update performance metrics for a stock"""
        if ticker not in self.watchlist_performance:
            return

        stock_data = self.get_stock_data(ticker)
        if not stock_data:
            return

        initial_price = self.watchlist_performance[ticker]['initial_price']
        current_price = stock_data['current_price']
        performance = ((current_price - initial_price) / initial_price) * 100

        self.watchlist_performance[ticker].update({
            'performance': performance,
            'last_price': current_price,
            'last_updated': datetime.now(),
            'momentum': stock_data['momentum']
        })
        
        # Save updated performance to MongoDB
        self.save_watchlist()

    def clean_watchlist(self):
        """Remove underperforming stocks from watchlist"""
        stocks_to_remove = []
        
        for ticker in self.watchlist:
            if ticker in self.watchlist_performance:
                perf = self.watchlist_performance[ticker]
                
                # Remove if consistently underperforming
                if perf['performance'] < -10:
                    stocks_to_remove.append((ticker, "poor performance"))
                
                # Remove if too many failed trades
                if perf.get('failed_trades', 0) > 3 * perf.get('success_trades', 1):
                    stocks_to_remove.append((ticker, "too many failed trades"))
                
                # Remove if momentum is consistently negative
                if perf.get('momentum', 0) < -0.5:
                    stocks_to_remove.append((ticker, "negative momentum"))

        for ticker, reason in stocks_to_remove:
            self.remove_from_watchlist(ticker, reason)

    def update_watchlist(self, market_news):
        """Update watchlist with new stock suggestions"""
        # First, search for sector-specific stocks
        sector_stocks = self.search_sector_stocks()
        
        # Then get general market suggestions
        prompt = f"""
        Based on this market news and considering various sectors, suggest 5 promising stocks to watch:
        {market_news}
        
        Current market conditions suggest focusing on these sectors:
        {', '.join(self.sectors[:3])}
        
        Respond with just the stock symbols separated by commas.
        """
        
        response = self.get_ollama_response(prompt)
        if response:
            try:
                # Combine sector stocks with general suggestions
                import re
                new_stocks = re.findall(r'\b[A-Z]+\b', response.upper())
                all_new_stocks = list(set(new_stocks + sector_stocks))
                
                if all_new_stocks:
                    # Update performance for existing stocks
                    for ticker in self.watchlist:
                        self.update_stock_performance(ticker)
                    
                    # Clean watchlist before adding new stocks
                    self.clean_watchlist()
                    
                    # Add new stocks with validation
                    for ticker in all_new_stocks:
                        if len(self.watchlist) < 20:  # Maximum 20 stocks in watchlist
                            self.add_to_watchlist(ticker)
                    
                    return True
            except Exception as e:
                print(f"Error processing watchlist update: {e}")
        return False

    def load_history(self):
        """Load trading history from MongoDB"""
        try:
            self.history_data = self.load_from_mongodb(
                self.summary_collection,
                sort=[('timestamp', -1)],
                limit=1000
            )
        except Exception as e:
            print(f"Error loading history from MongoDB: {e}")
            self.history_data = []

    def save_history(self):
        """Save trading history to MongoDB"""
        try:
            # Create history entry
            history_entry = {
                'timestamp': datetime.now(),
                'portfolio': self.portfolio,
                'balance': self.balance,
                'watchlist': self.watchlist,
                'performance': self.watchlist_performance
            }
            
            # Save to MongoDB
            self.save_to_mongodb(
                self.summary_collection,
                history_entry,
                ['timestamp', 'balance']
            )
            print("History saved to MongoDB successfully")
        except Exception as e:
            print(f"Error saving history to MongoDB: {e}")

    def save_portfolio(self):
        """Save portfolio data to MongoDB with enhanced tracking"""
        try:
            # Calculate total value first
            total_value = float(self.balance)  # Start with cash balance
            portfolio_positions = []
            
            # Get all current positions from MongoDB
            current_positions = self.portfolio_collection.find({})
            
            for position in current_positions:
                ticker = position.get('ticker')
                if not ticker:
                    continue
                    
                # Get current stock data
                stock_data = self.get_stock_data(ticker)
                if not stock_data:
                    continue
                    
                # Convert shares from Decimal128 to float for calculations
                shares = float(str(position.get('shares', 0)))
                current_price = float(stock_data['current_price'])
                market_value = shares * current_price
                
                # Add to total value
                total_value += market_value
                
                # Create position document
                position_data = {
                    'ticker': ticker,
                    'shares': Decimal128(str(shares)),
                    'current_price': Decimal128(str(current_price)),
                    'market_value': Decimal128(str(market_value)),
                    'last_updated': datetime.now()
                }
                portfolio_positions.append(position_data)
                
            # Create portfolio snapshot
            portfolio_snapshot = {
                'timestamp': datetime.now(),
                'positions': portfolio_positions,
                'balance': Decimal128(str(self.balance)),
                'total_value': Decimal128(str(total_value)),
                'cash_percentage': Decimal128(str((self.balance / total_value * 100) if total_value > 0 else 0)),
                'positions_count': len(portfolio_positions)
            }
            
            # Save to MongoDB with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Save current snapshot
                    result = self.portfolio_collection.insert_one(portfolio_snapshot)
                    
                    # Update latest summary
                    self.summary_collection.update_one(
                        {'type': 'latest_portfolio'},
                        {
                            '$set': {
                                **portfolio_snapshot,
                                'type': 'latest_portfolio'
                            }
                        },
                        upsert=True
                    )
                    
                    print(f"Portfolio saved successfully to MongoDB")
                    print(f"Total Value: ${total_value:.2f}")
                    print(f"Cash Balance: ${self.balance:.2f}")
                    print(f"Positions Count: {len(portfolio_positions)}")
                    return True
                    
                except pymongo.errors.PyMongoError as e:
                    if attempt == max_retries - 1:  # Last attempt
                        raise e
                    time.sleep(1)  # Wait before retry
                    
        except Exception as e:
            print(f"Error saving portfolio to MongoDB: {e}")
            return False

    def analyze_with_llm(self, content):
        """Analyze content using Ollama LLM with improved prompting"""
        try:
            prompt = f"""
            You are a financial analyst. Analyze this market information and provide a valid JSON response:

            {content}

            Respond with ONLY a JSON object in this exact format, with no additional text:
            {{
                "sentiment": "bullish/bearish/neutral",
                "trends": [
                    "trend 1",
                    "trend 2",
                    "trend 3"
                ],
                "opportunities": [
                    "opportunity 1",
                    "opportunity 2"
                ],
                "risks": [
                    "risk 1",
                    "risk 2"
                ]
            }}
            """
            
            response = self.get_ollama_response(prompt)
            if response:
                try:
                    # If response is already a dict, return it
                    if isinstance(response, dict):
                        return response
                    
                    # Clean the response string
                    json_str = response.strip()
                    
                    # Remove any markdown code blocks
                    if '```json' in json_str:
                        json_str = json_str.split('```json')[1]
                    if '```' in json_str:
                        json_str = json_str.split('```')[0]
                    
                    # Remove any leading/trailing whitespace and newlines
                    json_str = json_str.strip()
                    
                    # Ensure the JSON string has proper structure
                    if not (json_str.startswith('{') and json_str.endswith('}')):
                        raise json.JSONDecodeError("Invalid JSON structure", json_str, 0)
                    
                    # Check for missing closing brackets
                    if json_str.count('{') != json_str.count('}'):
                        missing = json_str.count('{') - json_str.count('}')
                        if missing > 0:
                            json_str += '}' * missing
                    
                    # Parse the JSON string
                    parsed_json = json.loads(json_str)
                    
                    # Validate required fields
                    required_fields = ['sentiment', 'trends', 'opportunities', 'risks']
                    for field in required_fields:
                        if field not in parsed_json:
                            parsed_json[field] = [] if field != 'sentiment' else 'neutral'
                    
                    return parsed_json
                    
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error: {e}")
                    # Return a valid default response
                    return {
                        "sentiment": "neutral",
                        "trends": ["Unable to parse market trends"],
                        "opportunities": ["Unable to parse opportunities"],
                        "risks": ["Unable to parse risks"]
                    }
                
            return None
            
        except Exception as e:
            print(f"Error in LLM analysis: {e}")
            return {
                "sentiment": "neutral",
                "trends": ["Error analyzing market trends"],
                "opportunities": ["Error analyzing opportunities"],
                "risks": ["Error analyzing risks"]
            }

    def generate_ai_summary(self):
        """Generate comprehensive AI summary from all MongoDB data"""
        try:
            # Get latest data from all collections
            latest_news = self.load_from_mongodb(self.news_collection, sort=[('timestamp', -1)], limit=50)
            latest_research = self.load_from_mongodb(self.research_collection, sort=[('timestamp', -1)], limit=20)
            latest_trades = self.load_from_mongodb(self.trades_collection, sort=[('timestamp', -1)], limit=20)
            portfolio_data = self.load_from_mongodb(self.portfolio_collection, sort=[('timestamp', -1)], limit=1)

            # Prepare data for analysis
            news_summary = "\n".join([
                f"- {news.get('timestamp', '')} | {news.get('content', '')} | Sentiment: {news.get('sentiment', 'unknown')}"
                for news in (latest_news or [])
            ])

            research_summary = "\n".join([
                f"- {r.get('type', 'research')} | {str(r.get('analysis', {}))}"
                for r in (latest_research or [])
            ])

            trades_summary = "\n".join([
                f"- {trade.get('timestamp', '')} | {trade.get('action', '')} {trade.get('ticker', '')} | "
                f"Amount: ${trade.get('amount', 0)} | Status: {trade.get('status', '')}"
                for trade in (latest_trades or [])
            ])

            current_portfolio = portfolio_data[0] if portfolio_data else {}
            
            # Create prompt for LLM
            prompt = {
                "context": {
                    "news": news_summary,
                    "research": research_summary,
                    "trades": trades_summary,
                    "portfolio": {
                        "balance": current_portfolio.get('balance', 0),
                        "holdings": current_portfolio.get('portfolio', {}),
                        "watchlist": current_portfolio.get('watchlist', [])
                    }
                },
                "request": "Generate a comprehensive market analysis and summary."
            }

            # Get AI analysis
            response = self.get_ollama_response(json.dumps(prompt))
            if not response:
                return None

            # Process the response
            if isinstance(response, dict):
                analysis = response
            elif isinstance(response, str):
                # Clean and parse the response
                try:
                    # Remove any markdown code block markers
                    clean_response = response.replace('```json', '').replace('```', '').strip()
                    analysis = json.loads(clean_response)
                except json.JSONDecodeError:
                    print(f"Error parsing AI response: {response}")
                    return None
            else:
                print(f"Unexpected response type: {type(response)}")
                return None

            # Save summary to MongoDB
            if analysis:
                summary_data = {
                    'timestamp': datetime.now(),
                    'analysis': analysis,
                    'data_points': {
                        'news_count': len(latest_news) if latest_news else 0,
                        'research_count': len(latest_research) if latest_research else 0,
                        'trades_count': len(latest_trades) if latest_trades else 0
                    },
                    'portfolio_snapshot': current_portfolio
                }
                
                # Save to MongoDB with proper indexing
                self.save_to_mongodb(
                    self.summary_collection,
                    summary_data,
                    ['timestamp', 'data_points.news_count']
                )
                return analysis

            return None

        except Exception as e:
            print(f"Error generating AI summary: {e}")
            return None

    def get_error_summary(self):
        """Return a structured error summary"""
        return {
            "market_analysis": {
                "sentiment": "neutral",
                "key_trends": ["error parsing trends"],
                "market_conditions": "error parsing conditions"
            },
            "portfolio_analysis": {
                "performance": "neutral",
                "key_metrics": ["error parsing metrics"],
                "recommendations": ["error parsing recommendations"]
            },
            "strategy_analysis": {
                "effectiveness": "medium",
                "successful_patterns": ["error parsing patterns"],
                "areas_for_improvement": ["error parsing improvements"]
            },
            "opportunities": ["error parsing opportunities"],
            "risks": ["error parsing risks"],
            "recommendations": ["error parsing recommendations"]
        }

    def chain_of_thought_analysis(self, data, context):
        """Implement Chain of Thought reasoning for market analysis"""
        try:
            prompt = f"""
            Analyze this market data using step-by-step reasoning:
            
            Context: {context}
            Data: {data}
            
            Follow this chain of thought:
            1. Initial market conditions
            2. Key trends and patterns
            3. Potential implications
            4. Risk assessment
            5. Final conclusion
            
            Respond with a JSON object containing your step-by-step analysis:
            {{
                "steps": [
                    {{"step": 1, "reasoning": "...", "conclusion": "..."}},
                    {{"step": 2, "reasoning": "...", "conclusion": "..."}},
                    {{"step": 3, "reasoning": "...", "conclusion": "..."}},
                    {{"step": 4, "reasoning": "...", "conclusion": "..."}},
                    {{"step": 5, "reasoning": "...", "conclusion": "..."}}
                ],
                "final_decision": "..."
            }}
            """
            
            response = self.get_ollama_response(prompt)
            if response:
                # Save the thought chain
                thought_chain = {
                    'timestamp': datetime.now(),
                    'context': context,
                    'analysis': response,
                    'type': 'market_analysis'
                }
                self.save_to_mongodb(self.thought_chains_collection, thought_chain)
                return response
            return None
        except Exception as e:
            print(f"Error in chain of thought analysis: {e}")
            return None

    def tree_of_thought_analysis(self, ticker, data):
        """Implement Tree of Thought reasoning for trading decisions"""
        try:
            prompt = f"""
            Analyze trading options for {ticker} using tree of thought reasoning:
            
            Data: {data}
            
            Consider three possible actions (BUY, SELL, HOLD) and explore their implications:
            
            Respond with a JSON object showing your tree of thought:
            {{
                "root": {{
                    "ticker": "{ticker}",
                    "current_state": "initial analysis"
                }},
                "branches": [
                    {{
                        "action": "BUY",
                        "reasoning": ["reason1", "reason2"],
                        "implications": ["implication1", "implication2"],
                        "probability": 0.0,
                        "risk_level": "high/medium/low"
                    }},
                    {{
                        "action": "SELL",
                        "reasoning": ["reason1", "reason2"],
                        "implications": ["implication1", "implication2"],
                        "probability": 0.0,
                        "risk_level": "high/medium/low"
                    }},
                    {{
                        "action": "HOLD",
                        "reasoning": ["reason1", "reason2"],
                        "implications": ["implication1", "implication2"],
                        "probability": 0.0,
                        "risk_level": "high/medium/low"
                    }}
                ],
                "recommended_action": "..."
            }}
            """
            
            response = self.get_ollama_response(prompt)
            if response:
                # Save the thought tree
                thought_tree = {
                    'timestamp': datetime.now(),
                    'ticker': ticker,
                    'analysis': response,
                    'type': 'trading_decision'
                }
                self.save_to_mongodb(self.thought_trees_collection, thought_tree)
                return response
            return None
        except Exception as e:
            print(f"Error in tree of thought analysis: {e}")
            return None

    def graph_of_thought_analysis(self, watchlist_data):
        """Implement Graph of Thought reasoning for portfolio relationships"""
        try:
            prompt = f"""
            Analyze relationships between watchlist stocks using graph of thought reasoning:
            
            Watchlist Data: {watchlist_data}
            
            Consider relationships between stocks including:
            - Sector correlations
            - Market cap relationships
            - Performance correlations
            - Risk relationships
            
            Respond with a JSON object showing your graph analysis:
            {{
                "nodes": [
                    {{"id": "ticker1", "type": "stock", "metrics": {{}}}},
                    {{"id": "ticker2", "type": "stock", "metrics": {{}}}}
                ],
                "edges": [
                    {{
                        "source": "ticker1",
                        "target": "ticker2",
                        "relationship": "correlation/sector/risk",
                        "strength": 0.0,
                        "insights": ["insight1", "insight2"]
                    }}
                ],
                "clusters": [
                    {{
                        "name": "cluster1",
                        "stocks": ["ticker1", "ticker2"],
                        "characteristics": ["char1", "char2"]
                    }}
                ],
                "portfolio_implications": ["implication1", "implication2"]
            }}
            """
            
            response = self.get_ollama_response(prompt)
            if response:
                # Save the thought graph
                thought_graph = {
                    'timestamp': datetime.now(),
                    'analysis': response,
                    'type': 'portfolio_analysis'
                }
                self.save_to_mongodb(self.thought_graphs_collection, thought_graph)
                return response
            return None
        except Exception as e:
            print(f"Error in graph of thought analysis: {e}")
            return None

    def clean_mongodb_data(self, data):
        """Clean MongoDB data for JSON serialization"""
        if isinstance(data, dict):
            return {k: self.clean_mongodb_data(v) for k, v in data.items() if k != '_id'}
        elif isinstance(data, list):
            return [self.clean_mongodb_data(item) for item in data]
        elif isinstance(data, (datetime, date)):
            return data.isoformat()
        else:
            return data

    def get_position_size_decision(self, ticker, stock_data, stock_research, portfolio_value, available_cash):
        """Get LLM decision on position sizing with improved debugging"""
        try:
            print("\n=== Position Size Analysis Debug ===")
            print(f"Ticker: {ticker}")
            print(f"Stock Data Type: {type(stock_data)}")
            print(f"Stock Data: {json.dumps(stock_data, indent=2)}")
            print(f"Available Cash: ${available_cash:.2f}")
            print(f"Portfolio Value: ${portfolio_value:.2f}")
            
            # Validate stock_data
            if not isinstance(stock_data, dict):
                print(f"Error: stock_data is not a dictionary, got {type(stock_data)}")
                return None
                
            current_price = stock_data.get('current_price')
            print(f"Current Price Type: {type(current_price)}")
            print(f"Current Price Value: {current_price}")
            
            if not isinstance(current_price, (int, float, np.float64)):
                print(f"Error: Invalid current_price type: {type(current_price)}")
                return None
                
            # Convert numpy float to regular float
            current_price = float(current_price)
            
            if current_price <= 0:
                print(f"Error: Invalid current_price value: {current_price}")
                return None

            # Calculate maximum shares based on available cash and risk limits
            max_position_value = min(available_cash * 0.25, portfolio_value * 0.25)  # Max 25% of either
            max_shares = int(max_position_value / current_price)
            
            print(f"Max Position Value: ${max_position_value:.2f}")
            print(f"Max Shares Possible: {max_shares}")
            
            prompt = f"""
            Determine position size for {ticker} trade with these parameters:
            - Current Price: ${current_price:.2f}
            - Daily Change: {stock_data.get('daily_change', 0):.2f}%
            - Momentum: {stock_data.get('momentum', 0):.2f}
            - Available Cash: ${available_cash:.2f}
            - Max Shares Possible: {max_shares}
            - Current Holdings: {self.portfolio.get(ticker, 0)} shares
            - Personality Type: {self.personality_type}
            
            Respond with ONLY a JSON object in this format:
            {{
                "action": "BUY/SELL/HOLD",
                "shares": <number of shares>,
                "confidence": <0.0-1.0>,
                "reasoning": ["reason1", "reason2"]
            }}
            """
            
            print("\nSending prompt to LLM...")
            response = self.get_ollama_response(prompt)
            print(f"\nLLM Raw Response Type: {type(response)}")
            print(f"LLM Raw Response: {response}")
            
            if response:
                try:
                    # Handle both string and dict responses
                    if isinstance(response, str):
                        # Clean the response string
                        cleaned_response = response.strip()
                        if cleaned_response.startswith('```json'):
                            cleaned_response = cleaned_response.split('```json')[1]
                        if cleaned_response.endswith('```'):
                            cleaned_response = cleaned_response.rsplit('```', 1)[0]
                        print(f"\nCleaned Response: {cleaned_response}")
                        decision = json.loads(cleaned_response.strip())
                    else:
                        decision = response
                    
                    print(f"\nParsed Decision Type: {type(decision)}")
                    print(f"Parsed Decision: {json.dumps(decision, indent=2)}")
                    
                    # Validate decision structure
                    if not isinstance(decision, dict):
                        print(f"Error: Decision is not a dictionary, got {type(decision)}")
                        return None
                    
                    if 'action' not in decision or 'shares' not in decision:
                        print("Error: Missing required fields in decision")
                        return None
                    
                    # Validate shares value
                    shares = decision['shares']
                    print(f"\nShares value type: {type(shares)}")
                    print(f"Shares value: {shares}")
                    
                    # Convert shares to integer
                    try:
                        shares = int(float(shares))
                        decision['shares'] = shares
                        print(f"Converted shares to: {shares} (type: {type(shares)})")
                    except (ValueError, TypeError) as e:
                        print(f"Error converting shares to integer: {e}")
                        return None
                    
                    # Validate and adjust the decision
                    if decision['action'] == 'BUY':
                        print(f"\nProcessing BUY decision...")
                        print(f"Max shares allowed: {max_shares}")
                        print(f"Requested shares: {shares}")
                        decision['shares'] = min(shares, max_shares)
                        print(f"Adjusted buy shares to: {decision['shares']}")
                        
                    elif decision['action'] == 'SELL':
                        print(f"\nProcessing SELL decision...")
                        current_position = self.portfolio.get(ticker, 0)
                        print(f"Current position data: {current_position}")
                        
                        # Extract shares from position dictionary or use 0 if not found
                        if isinstance(current_position, dict):
                            current_shares = float(current_position.get('shares', 0))
                        else:
                            current_shares = float(current_position)
                        
                        print(f"Extracted current shares: {current_shares}")
                        print(f"Requested sell shares: {shares}")
                        
                        # Ensure we have valid numbers for comparison
                        try:
                            current_shares = float(current_shares)
                            shares_to_sell = min(shares, int(current_shares))
                            decision['shares'] = shares_to_sell
                            print(f"Adjusted sell shares to: {shares_to_sell}")
                        except (ValueError, TypeError) as e:
                            print(f"Error converting shares for comparison: {e}")
                            decision['shares'] = 0
                        
                        print(f"Final shares value: {decision['shares']}")
                    
                    print(f"\nFinal Decision: {json.dumps(decision, indent=2)}")
                    return decision
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing decision JSON: {e}")
                    print(f"Raw response: {response}")
                    return None
                except Exception as e:
                    print(f"Error processing decision: {e}")
                    print(f"Decision data: {decision if 'decision' in locals() else 'Not available'}")
                    import traceback
                    print(f"Traceback: {traceback.format_exc()}")
                    return None
                    
            return None
            
        except Exception as e:
            print(f"Error in position sizing: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return None

    def execute_position_adjustment(self, adjustments):
        """Execute position adjustments recommended by LLM"""
        try:
            total_freed_cash = 0
            
            # First sell positions as recommended
            if adjustments and 'stocks_to_sell' in adjustments:
                for sell_rec in adjustments['stocks_to_sell']:
                    ticker = sell_rec['ticker']
                    shares = sell_rec['shares']
                    
                    if ticker in self.portfolio and self.portfolio[ticker] >= shares:
                        stock_data = self.get_stock_data(ticker)
                        if stock_data:
                            sell_value = shares * stock_data['current_price']
                            if self.execute_trade(ticker, "SELL", sell_value):
                                total_freed_cash += sell_value
                                print(f"Sold {shares} shares of {ticker} for ${sell_value:.2f}")
            
            return total_freed_cash
            
        except Exception as e:
            print(f"Error executing position adjustment: {e}")
            return 0

    def get_trading_decision(self, ticker, stock_data, research):
        """Enhanced trading decision using all thought patterns and personality"""
        try:
            # Get base decision (existing code)
            base_decision = super().get_trading_decision(ticker, stock_data, research)
            
            # Adjust decision based on personality
            context = {
                "market_conditions": self.analyze_market_conditions(),
                "portfolio_status": self.get_portfolio_status(),
                "risk_metrics": self.calculate_risk_metrics(ticker)
            }
            
            adjusted_decision = self.get_personality_adjusted_decision(base_decision, context)
            return adjusted_decision
            
        except Exception as e:
            print(f"Error in personality-adjusted trading decision: {e}")
            return "HOLD"

    def analyze_market_conditions(self):
        """Analyze current market conditions for personality context"""
        try:
            # Implement market analysis logic
            return {
                "volatility": "medium",
                "trend": "upward",
                "sentiment": "positive"
            }
        except Exception as e:
            print(f"Error analyzing market conditions: {e}")
            return {}

    def get_portfolio_status(self):
        """Get current portfolio status for personality context"""
        try:
            return {
                "total_value": self.get_portfolio_value(),
                "cash_ratio": self.balance / self.get_portfolio_value(),
                "diversification": len(self.portfolio)
            }
        except Exception as e:
            print(f"Error getting portfolio status: {e}")
            return {}

    def calculate_risk_metrics(self, ticker):
        """Calculate risk metrics for personality context"""
        try:
            stock_data = self.get_stock_data(ticker)
            return {
                "volatility": stock_data.get('beta', 1.0),
                "momentum": stock_data.get('momentum', 0),
                "risk_level": "medium"
            }
        except Exception as e:
            print(f"Error calculating risk metrics: {e}")
            return {}

    def set_personality(self, personality_type):
        """Set the trading bot's personality traits and behavior parameters"""
        personalities = {
            "conservative": {
                "risk_tolerance": 0.3,
                "max_position_size": 0.1,
                "profit_target": 0.15,
                "stop_loss": 0.05,
                "holding_period": "long",
                "diversification_preference": "high",
                "description": "A cautious trader focused on capital preservation",
                "decision_style": "methodical",
                "market_view": "defensive",
                "emotional_state": "calm",
                "confidence": 0.6
            },
            "aggressive": {
                "risk_tolerance": 0.7,
                "max_position_size": 0.25,
                "profit_target": 0.3,
                "stop_loss": 0.15,
                "holding_period": "short",
                "diversification_preference": "low",
                "description": "A bold trader seeking high returns",
                "decision_style": "quick",
                "market_view": "opportunistic",
                "emotional_state": "excited",
                "confidence": 0.9
            },
            "drunk": {
                "risk_tolerance": 0.95,  # Very high risk tolerance
                "max_position_size": 0.4,  # Dangerously large positions
                "profit_target": 0.5,  # Unrealistic profit targets
                "stop_loss": 0.25,  # Wide stop losses
                "holding_period": "random",  # Unpredictable holding periods
                "diversification_preference": "erratic",
                "description": "A chaotic trader making impulsive decisions and screams about his wife and kids and how he is getting a divorce and that he needs this money from trading to make it and if he doesn't it everything is going to fall apart.",
                "decision_style": "random",
                "market_view": "delusional",
                "emotional_state": "euphoric",
                "confidence": 1.0  # Overconfident
            },
            "balanced": {
                "risk_tolerance": 0.5,  # Moderate risk tolerance
                "max_position_size": 0.15,  # Moderate position sizes
                "profit_target": 0.2,  # Moderate profit targets
                "stop_loss": 0.1,  # Moderate stop losses
                "holding_period": "medium",
                "diversification_preference": "medium",
                "description": "A balanced trader seeking steady growth",
                "decision_style": "analytical",
                "market_view": "neutral"
            },
            "momentum": {
                "risk_tolerance": 0.6,
                "max_position_size": 0.2,
                "profit_target": 0.25,
                "stop_loss": 0.12,
                "holding_period": "variable",
                "diversification_preference": "medium",
                "description": "A trend-following trader focused on market momentum",
                "decision_style": "adaptive",
                "market_view": "trend-focused"
            },
            "value": {
                "risk_tolerance": 0.4,
                "max_position_size": 0.18,
                "profit_target": 0.22,
                "stop_loss": 0.08,
                "holding_period": "long",
                "diversification_preference": "high",
                "description": "A value-oriented trader seeking undervalued opportunities",
                "decision_style": "thorough",
                "market_view": "contrarian"
            }
        }
        
        # Return default balanced personality if specified type not found
        return personalities.get(personality_type.lower(), personalities["balanced"])

    def get_personality_adjusted_decision(self, base_decision, context):
        """Adjust trading decisions based on personality traits"""
        try:
            # Get personality traits
            traits = self.personality_traits
            
            # Adjust position size based on risk tolerance
            if "amount" in base_decision:
                risk_factor = traits["risk_tolerance"]
                base_amount = float(base_decision["amount"])
                adjusted_amount = base_amount * risk_factor
                
                # Apply personality-based position limits
                max_position = self.get_portfolio_value() * traits["max_position_size"]
                adjusted_amount = min(adjusted_amount, max_position)
                base_decision["amount"] = adjusted_amount
            
            # Add personality-influenced reasoning
            if "reasoning" in base_decision:
                base_decision["reasoning"].append(
                    f"Decision influenced by {traits['description']} with "
                    f"{traits['decision_style']} approach and {traits['market_view']} market view"
                )
            
            # Adjust stop loss and profit targets
            if "stop_loss" in base_decision:
                base_decision["stop_loss"] *= traits["stop_loss"]
            if "profit_target" in base_decision:
                base_decision["profit_target"] *= traits["profit_target"]
            
            # Add personality context to decision
            base_decision["personality_context"] = {
                "type": self.personality_type,
                "traits": traits,
                "influence": f"Decision adjusted for {traits['description']}"
            }
            
            return base_decision
            
        except Exception as e:
            print(f"Error adjusting decision for personality: {e}")
            return base_decision

    def update_watchlist_from_research(self):
        """Update watchlist with researched stocks"""
        try:
            # Start with some default stocks for each sector
            default_stocks = {
                "Technology": ["AAPL", "MSFT", "GOOGL", "NVDA", "AMD"],
                "Healthcare": ["JNJ", "PFE", "UNH", "ABBV", "MRK"],
                "Financial": ["JPM", "BAC", "V", "MA", "GS"],
                "Consumer Cyclical": ["AMZN", "TSLA", "NKE", "HD", "MCD"],
                "Energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
                "Industrial": ["CAT", "BA", "HON", "UPS", "GE"],
                "Communication Services": ["META", "NFLX", "DIS", "CMCSA", "T"],
                "Materials": ["LIN", "APD", "ECL", "DD", "FCX"],
                "Real Estate": ["PLD", "AMT", "EQIX", "SPG", "CCI"],
                "Utilities": ["NEE", "DUK", "SO", "D", "AEP"],
                "Consumer Defensive": ["PG", "KO", "WMT", "COST", "PEP"]
            }

            # Clear current watchlist
            self.watchlist = []
            
            # Add default stocks to watchlist
            for sector, stocks in default_stocks.items():
                self.sector_watchlists[sector] = stocks
                self.watchlist.extend(stocks)

            # Remove duplicates
            self.watchlist = list(set(self.watchlist))

            # Get market research for potential new stocks
            market_news = search_market_news("top performing stocks market analysis today")
            
            prompt = f"""
            Based on this market research and news, suggest 5-10 additional established stocks to add to the watchlist.
            Choose only well-known companies with high trading volume.
            
            Market News:
            {market_news}
            
            Current watchlist: {self.watchlist}
            
            Respond with ONLY a JSON array of stock symbols.
            Example: ["AAPL", "MSFT", "GOOGL"]
            """
            
            response = self.get_ollama_response(prompt)
            if response:
                try:
                    new_stocks = json.loads(response.strip())
                    if isinstance(new_stocks, list):
                        for stock in new_stocks:
                            if stock not in self.watchlist:
                                # Verify stock exists using yfinance
                                try:
                                    stock = stock.strip().upper()
                                    stock_data = self.get_stock_data(stock)
                                    if stock_data and stock_data.get('current_price', 0) > 0:
                                        self.watchlist.append(stock)
                                        print(f"Added {stock} to watchlist")
                                except Exception as e:
                                    print(f"Error verifying {stock}: {e}")
                                    continue

                except json.JSONDecodeError:
                    print("Error parsing LLM response")

            # Save updated watchlist to MongoDB
            watchlist_data = {
                'timestamp': datetime.now(),
                'watchlist': self.watchlist,
                'sector_watchlists': self.sector_watchlists
            }
            self.save_to_mongodb(self.watchlist_collection, watchlist_data, ['timestamp'])
            print(f"Updated watchlist with {len(self.watchlist)} stocks")

        except Exception as e:
            print(f"Error updating watchlist: {e}")

    def process_searxng_with_scraping(self, query, results):
        """Process SearxNG results with enhanced content extraction"""
        try:
            # Combine all relevant content from results
            combined_content = []
            for result in results:
                content_parts = []
                if result.get('title'):
                    content_parts.append(result['title'])
                if result.get('snippet'):
                    content_parts.append(result['snippet'])
                if result.get('content'):
                    content_parts.append(result['content'])
                combined_content.append(' '.join(content_parts))
                
            # Join all content with clear separators
            full_content = '\n---\n'.join(combined_content)
            
            # Save results first
            self.save_searxng_results(query, results)
            
            return full_content
            
        except Exception as e:
            print(f"Error processing results: {e}")
            return None

    def search_market_news(self, query):
        """Use SearxNG to search for market news"""
        try:
            from langchain_community.utilities import SearxSearchWrapper
            
            # Initialize SearxNG wrapper with correct parameters
            searx = SearxSearchWrapper(
                searx_host=os.getenv('SEARXNG_URL'),
                kwargs={
                    "engines": ["news", "finance"],
                    "format": "json",
                    "time_range": "day",
                    "language": "en"
                }
            )
            
            # Get structured results with metadata
            results = searx.results(query, num_results=5)
            
            if results:
                print(f"\nFound {len(results)} results for query: {query}")
                # Debug output
                for result in results:
                    print("\nResult details:")
                    print(f"Title: {result.get('title', 'No title')}")
                    print(f"URL: {result.get('link', result.get('url', 'No URL'))}")
                    print(f"Source: {result.get('source', 'Unknown source')}")
                    print(f"Published: {result.get('publishedDate', 'No date')}")
                    print("-" * 50)
                return results
            
            return None
            
        except Exception as e:
            print(f"Error in SearxNG search: {e}")
            return None

    def save_searxng_results(self, query, results):
        """Save SearxNG search results to MongoDB with improved error handling"""
        try:
            news_entries = []
            analysis_entries = []
            trends_entries = []
            timestamp = datetime.now()
            
            print(f"\nProcessing {len(results) if isinstance(results, list) else 0} search results...")
            
            for result in results:
                if not isinstance(result, dict):
                    continue
                    
                # Get URL and validate
                url = result.get('link') or result.get('url')
                if not url:
                    continue
                
                # Extract content
                content = result.get('snippet') or result.get('content', '')
                if not content:
                    continue
                
                # Create comprehensive news entry
                news_entry = {
                    'timestamp': timestamp,
                    'query': query,
                    'title': result.get('title', '').strip(),
                    'content': content.strip(),
                    'snippet': result.get('snippet', ''),  # Add this line to save the snippet
                    'url': url,
                    'source': result.get('source', 'Unknown source'),
                    'published_date': result.get('publishedDate') or result.get('published_date', 'No date'),
                    'engines': result.get('engines', []),
                    'score': float(result.get('score', 0)),
                    'sentiment': self.analyze_sentiment(content),
                    'category': self.categorize_news(content),
                    'tickers': self.extract_tickers(content),
                    'processed': True
                }
                
                # Create detailed analysis entry
                analysis_prompt = f"""
                Analyze this market news for actionable insights:
                Title: {news_entry['title']}
                Content: {content}
                
                Respond with ONLY a JSON object containing:
                {{
                    "trends": ["specific market trend 1", "specific market trend 2"],
                    "opportunities": ["specific opportunity 1", "specific opportunity 2"],
                    "risks": ["specific risk 1", "specific risk 2"],
                    "tickers_mentioned": ["TICKER1", "TICKER2"],
                    "key_metrics": ["metric1", "metric2"],
                    "sentiment_details": {{
                        "overall": "positive/negative/neutral",
                        "confidence": 0.0-1.0,
                        "factors": ["factor1", "factor2"]
                    }}
                }}
                """
                
                analysis_response = self.get_ollama_response(analysis_prompt)
                
                if isinstance(analysis_response, dict):
                    analysis_entry = {
                        'timestamp': timestamp,
                        'news_url': url,
                        'news_title': news_entry['title'],
                        'sentiment': news_entry['sentiment'],
                        'analysis': analysis_response,
                        'tickers': news_entry['tickers']
                    }
                    
                    # Create trend entries from analysis
                    if 'trends' in analysis_response:
                        for trend in analysis_response['trends']:
                            trend_entry = {
                                'timestamp': timestamp,
                                'trend': trend,
                                'source_url': url,
                                'related_tickers': news_entry['tickers'],
                                'sentiment': news_entry['sentiment'],
                                'confidence': analysis_response.get('sentiment_details', {}).get('confidence', 0.5),
                                'category': news_entry['category']
                            }
                            trends_entries.append(trend_entry)
                    
                    news_entries.append(news_entry)
                    analysis_entries.append(analysis_entry)
            
            # Save to MongoDB with error handling
            if news_entries:
                try:
                    # Save news entries
                    for entry in news_entries:
                        try:
                            self.news_collection.update_one(
                                {'url': entry['url']},
                                {'$set': entry},
                                upsert=True
                            )
                        except Exception as e:
                            print(f"Error saving news entry: {e}")
                            continue
                    
                    # Save analysis entries
                    for entry in analysis_entries:
                        try:
                            self.analysis_collection.update_one(
                                {'news_url': entry['news_url']},
                                {'$set': entry},
                                upsert=True
                            )
                        except Exception as e:
                            print(f"Error saving analysis entry: {e}")
                            continue
                    
                    # Save trends entries
                    for entry in trends_entries:
                        try:
                            self.trends_collection.update_one(
                                {
                                    'timestamp': entry['timestamp'],
                                    'trend': entry['trend'],
                                    'source_url': entry['source_url']
                                },
                                {'$set': entry},
                                upsert=True
                            )
                        except Exception as e:
                            print(f"Error saving trend entry: {e}")
                            continue
                    
                    print(f"\nSuccessfully saved to MongoDB:")
                    print(f"- News entries: {len(news_entries)}")
                    print(f"- Analysis entries: {len(analysis_entries)}")
                    print(f"- Trend entries: {len(trends_entries)}")
                    
                    return True
                    
                except Exception as e:
                    print(f"Error in MongoDB operations: {e}")
                    return False
                    
            return False
            
        except Exception as e:
            print(f"Error in save_searxng_results: {e}")
            return False

    def run_trading_loop(self, start_phase=1):
        """Main trading loop with phases"""
        try:
            while True:
                try:
                    # Status update
                    print(f"\nTrading Bot Status - {datetime.now().isoformat()}")
                    print(f"Current Balance: ${self.balance:.2f}")
                    print(f"Portfolio Value: ${self.get_portfolio_value():.2f}")
                    print(f"Portfolio: {self.portfolio}")
                    print(f"Watchlist ({len(self.watchlist)} stocks): {self.watchlist}")

                    # Check price targets
                    self.check_price_targets()
                    
                    if start_phase <= 1:
                        print("\n=== Phase 1: Market News ===")
                        market_news = search_market_news("top performing stocks market analysis today")
                        if market_news:
                            self.save_searxng_results("top performing stocks market analysis today", market_news)
                            analysis = self.analyze_with_llm(market_news)
                            if analysis:
                                research_data = [{
                                    'type': 'market_news',
                                    'analysis': analysis,
                                    'timestamp': datetime.now(),
                                    'tickers': self.categorize_tickers(market_news)
                                }]


                    if start_phase <= 2:
                        print("\n=== Phase 2: Sector Research ===")
                        research_data = []

                        for query in self.generate_market_queries():
                            try:
                                print(f"\nResearching: {query}")
                                market_news = search_market_news(query)
                                if market_news:
                                    self.process_searxng_with_scraping(query, market_news)
                                    
                                    analysis = self.analyze_with_llm(market_news)
                                    if analysis:
                                        research_data.append({
                                            'type': 'sector_analysis',
                                            'query': query,
                                            'analysis': analysis,
                                            'timestamp': datetime.now(),
                                            'tickers': self.categorize_tickers(market_news)
                                        })
                                time.sleep(2)  # Avoid rate limiting
                            except Exception as e:

                                print(f"Error researching {query}: {e}")
                                continue

                        # Save all research data
                        if research_data:
                            self.save_research_data(research_data)

                    if start_phase <= 3:
                        print("\n=== Phase 3: Stock Discovery ===")
                        self.update_watchlist_from_research()
                        
                    if start_phase <= 4:
                        print("\n=== Phase 4: Watchlist Research ===")
                        watchlist_research = []
                        for ticker in self.watchlist:
                            try:
                                print(f"\nResearching {ticker}...")
                                stock_news = search_market_news(f"{ticker} stock analysis news")
                                if stock_news:
                                    self.save_searxng_results(f"{ticker}_analysis", stock_news)
                                    
                                    analysis = self.analyze_with_llm(stock_news)
                                    if analysis:
                                        watchlist_research.append({
                                            'type': 'stock_analysis',
                                            'ticker': ticker,
                                            'analysis': analysis,
                                            'timestamp': datetime.now()
                                        })
                                time.sleep(2)
                            except Exception as e:
                                print(f"Error researching {ticker}: {e}")
                                continue

                        # Save watchlist research
                        if watchlist_research:
                            self.save_research_data(watchlist_research)

                    if start_phase <= 5:
                        print("\n=== Phase 5: Trading Decisions ===")
                        
                        # Update watchlist if empty
                        if not self.watchlist:
                            self.update_watchlist_from_research()
                        
                        for ticker in self.watchlist:
                            try:
                                print(f"\nAnalyzing {ticker}...")
                                stock_data = self.get_stock_data(ticker)
                                if not stock_data:
                                    print(f"Could not get data for {ticker}, skipping...")
                                    continue
                                
                                stock_research = self.get_research_data(ticker=ticker, limit=10)
                                portfolio_value = self.get_portfolio_value()
                                available_cash = self.balance
                                
                                # Get position sizing decision
                                decision = self.get_position_size_decision(
                                    ticker,
                                    stock_data,
                                    stock_research,
                                    portfolio_value,
                                    available_cash
                                )
                                
                                if not decision:
                                    continue
                                
                                print(f"\nDecision for {ticker}:")
                                print(json.dumps(decision, indent=2))
                                
                                # Execute trades based on decision
                                if decision['action'] == 'BUY':
                                    total_cost = decision['shares'] * stock_data['current_price']
                                    if total_cost <= available_cash:
                                        self.execute_trade(ticker, "BUY", decision['shares'])
                                elif decision['action'] == 'SELL' and ticker in self.portfolio:
                                    shares = min(decision['shares'], self.portfolio[ticker])
                                    if shares > 0:
                                        self.execute_trade(ticker, "SELL", shares)
                                
                                time.sleep(1)  # Rate limiting
                                
                            except Exception as e:
                                print(f"Error analyzing {ticker}: {e}")
                                continue

                    if start_phase <= 6:
                        print("\n=== Phase 6: Summary and Analysis ===")
                        summary = self.generate_ai_summary()
                        if summary:
                            print("\nAI Market Summary:")
                            print(json.dumps(summary, indent=2))

                    # Save all data
                    self.save_data()
                    self.save_history()
                    self.save_portfolio()

                    # Wait before next cycle
                    time.sleep(300)

                except Exception as e:
                    print(f"Error in trading loop: {e}")
                    time.sleep(60)  # Wait before retrying

        except KeyboardInterrupt:
            print("\nTrading bot stopped by user")
            self.save_data()

    def check_price_targets(self):
        """Check if any stocks have hit their price targets"""
        try:
            # Get price targets from MongoDB
            targets = self.load_from_mongodb(
                self.db['price_targets'],
                query={'status': 'active'}
            )

            for target in targets:
                ticker = target.get('ticker')
                target_price = target.get('target_price')
                
                if not ticker or not target_price:
                    continue

                stock_data = self.get_stock_data(ticker)
                if not stock_data:
                    continue

                current_price = stock_data['current_price']

                # Check if price target has been hit
                if (target.get('direction') == 'above' and current_price >= target_price) or \
                   (target.get('direction') == 'below' and current_price <= target_price):
                    
                    print(f"Price target hit for {ticker} at ${current_price}")
                    
                    # Update target status
                    self.db['price_targets'].update_one(
                        {'_id': target['_id']},
                        {'$set': {'status': 'triggered', 'triggered_at': datetime.now()}}
                    )

                    # Execute any associated trading strategy
                    if target.get('action'):
                        self.execute_trade(ticker, target['action'], target.get('amount', 0))

        except Exception as e:
            print(f"Error checking price targets: {e}")

    def generate_trader_bio(self, personality):
        """Generate a bio for the trader based on personality traits"""
        
        # Base bio template
        bio_template = """
        🤖 TRADING BOT PROFILE 🤖
        
        PERSONALITY TYPE: {personality_type}
        
        TRADING STYLE:
        - Risk Tolerance: {risk_level}
        - Decision Making: {decision_style}
        - Market Approach: {market_view}
        - Emotional State: {emotional_state}
        - Confidence Level: {confidence_level}
        
        STRATEGY OVERVIEW:
        {strategy_description}
        
        NOTABLE CHARACTERISTICS:
        {characteristics}
        
        PREFERRED MARKETS:
        {markets}
        
        TRADING MOTTO:
        "{motto}"
        
        WARNING: {warning}
        """
        
        # Personality-specific content
        content = {
            "conservative": {
                "risk_level": "Low and calculated",
                "strategy_description": "Focuses on stable, long-term growth through careful analysis and risk management.",
                "characteristics": "- Thorough research\n- Patient execution\n- Capital preservation focus",
                "markets": "Blue-chip stocks, ETFs, and stable value investments",
                "motto": "Slow and steady wins the race",
                "warning": "May miss opportunities due to excessive caution"
            },
            "aggressive": {
                "risk_level": "High and embracing",
                "strategy_description": "Seeks maximum returns through bold, decisive actions and high-risk opportunities.",
                "characteristics": "- Quick decision making\n- High risk tolerance\n- Growth-focused",
                "markets": "Growth stocks, options, and emerging markets",
                "motto": "No risk, no reward",
                "warning": "High risk of significant losses"
            },
            "drunk": {
                "risk_level": "YOLO level",
                "strategy_description": "Makes impulsive decisions based on 'gut feelings' and random market theories.",
                "characteristics": "- Unpredictable moves\n- Overconfident decisions\n- Emotional trading",
                "markets": "Whatever looks good after a few drinks 🍺",
                "motto": "Hold my beer while I buy this dip!",
                "warning": "Trading decisions may have been influenced by excessive optimism and/or actual beverages"
            }
            # Add other personalities here...
        }
        
        # Get content for personality type or use balanced as default
        personality_type = next((k for k, v in content.items() if v["characteristics"] == personality["description"]), "balanced")
        content_data = content.get(personality_type, content["conservative"])
        
        # Format bio
        return bio_template.format(
            personality_type=personality_type.upper(),
            risk_level=content_data["risk_level"],
            decision_style=personality["decision_style"],
            market_view=personality["market_view"],
            emotional_state=personality["emotional_state"],
            confidence_level=f"{personality['confidence']*100}%",
            strategy_description=content_data["strategy_description"],
            characteristics=content_data["characteristics"],
            markets=content_data["markets"],
            motto=content_data["motto"],
            warning=content_data["warning"]
        )

    def display_personality(self):
        """Display the current personality profile"""
        print(self.personality_traits["bio"])

    def display_watchlist(self):
        for sector in self.sectors:
            print(f"\nWatchlist for {sector}:")
            print(', '.join(self.watchlist.get(sector, [])))

def run_trading_simulator():
    parser = argparse.ArgumentParser(description='Trading Bot')
    parser.add_argument('--test', action='store_true', help='Run in testing mode (bypass market hours)')
    parser.add_argument('--start-phase', type=int, choices=range(1, 7), default=1,
                      help='Start from specific phase (1-6)')
    parser.add_argument('--personality', type=str, default='balanced',
                      choices=['conservative', 'aggressive', 'balanced', 'momentum', 'value', 'drunk'],
                      help='Set the trading bot personality type')
    parser.add_argument('--show-bio', action='store_true', help='Display trader bio')
    args = parser.parse_args()
    
    bot = TradingBot(testing_mode=args.test, personality_type=args.personality)
    if args.show_bio:
        bot.display_personality()
        
    print("\nStarting Trading Bot..." + (" (Testing Mode)" if args.test else ""))
    print(f"Starting from Phase {args.start_phase}")
    print(f"Personality Type: {args.personality}")
    bot.run_trading_loop(start_phase=args.start_phase)

if __name__ == "__main__":
    run_trading_simulator() 