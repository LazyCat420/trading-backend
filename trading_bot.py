import json
from datetime import datetime, timedelta
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

class TradingBot:
    def __init__(self, initial_balance=100000, testing_mode=False):  # Added testing_mode parameter
        self.initial_balance = float(initial_balance)  # Store initial balance
        self.balance = self.initial_balance
        self.portfolio = {}
        self.watchlist = []
        self.testing_mode = testing_mode  # Add testing mode flag
        self.client = get_database()
        if not self.client:
            raise Exception("Could not connect to MongoDB")
        
        self.db = self.client['trading']
        self.trades_collection = self.db['trades']
        self.watchlist_collection = self.db['watchlist']
        self.news_collection = self.db['news']
        self.portfolio_collection = self.db['portfolio']
        self.summary_collection = self.db['summary']
        self.research_collection = self.db['research']
        
        # Initialize data lists
        self.trades_data = []
        self.news_data = []
        self.history_data = []
        
        # Other initializations
        self.momentum_window = 20
        self.watchlist_performance = {}
        self.sectors = [
            "Technology", "Healthcare", "Financial", "Consumer Cyclical",
            "Energy", "Industrial", "Communication Services", "Materials",
            "Real Estate", "Utilities", "Consumer Defensive"
        ]
        
        # Load all data from MongoDB
        self.load_data()
        self.load_watchlist()

    def save_to_mongodb(self, collection, data, index_fields=None):
        """Generic method to save data to MongoDB with indexing"""
        try:
            if isinstance(data, list):
                if data:
                    result = collection.insert_many(data)
                    print(f"Inserted {len(result.inserted_ids)} documents")
            else:
                result = collection.insert_one(data)
                print("Inserted 1 document")
            
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
            self.watchlist = data.get('watchlist', [])
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
                "model": os.getenv('OLLAMA_MODEL', 'llama2'),
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

            # Try to parse as JSON if possible
            if full_content:
                try:
                    if '{' in full_content and '}' in full_content:
                        # Extract JSON part
                        json_start = full_content.find('{')
                        json_end = full_content.rfind('}') + 1
                        json_str = full_content[json_start:json_end]
                        
                        # Clean common JSON formatting issues
                        json_str = json_str.replace('```json', '').replace('```', '')
                        json_str = json_str.replace('\n', ' ').replace('\r', '')
                        
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError as je:
                            print(f"Invalid JSON structure: {je}")
                            # Try to fix common JSON errors
                            if '"add": [' in json_str and json_str.count('[') != json_str.count(']'):
                                json_str = json_str.replace('}}', ']}')
                            if json_str.count('{') != json_str.count('}'):
                                missing = json_str.count('{') - json_str.count('}')
                                json_str = json_str + ('}' * missing)
                            # Try parsing again after fixes
                            try:
                                return json.loads(json_str)
                            except:
                                pass
                            
                            # If still invalid, return a safe default
                            return {"error": "Invalid JSON response"}
                except:
                    pass
                
            return full_content

        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
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

    def analyze_sentiment(self, text):
        """Analyze sentiment of news text using Ollama"""
        try:
            prompt = f"""
            Analyze the sentiment of this news. Respond with exactly one word: POSITIVE, NEGATIVE, or NEUTRAL.
            News: {text}
            """
            response = self.get_ollama_response(prompt)
            if response:
                return response.strip().lower()
            return "neutral"
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return "neutral"

    def categorize_news(self, text):
        """Categorize news using Ollama"""
        try:
            prompt = f"""
            Categorize this news responding exactly with the category name and nothing else: EARNINGS, MERGER, MARKET_SENTIMENT, COMPANY_NEWS, INDUSTRY_NEWS, REGULATORY, OTHER
            News: {text}.
            """
            response = self.get_ollama_response(prompt)
            if response:
                return response.strip().upper()
            return "OTHER"
        except Exception as e:
            print(f"Error categorizing news: {e}")
            return "OTHER"
    
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
        except Exception as e:
            print(f"Error fetching news for {ticker}: {e}")
            news = "Error fetching news"

        stock_data = self.get_stock_data(ticker)
        if not stock_data or stock_data['current_price'] == 0:
            return "HOLD - Insufficient data"

        # Calculate buy/sell signals
        signals = {
            'price_vs_50d': stock_data['current_price'] > stock_data['fifty_day_avg'],
            'momentum': stock_data['momentum'] > 0,
            'daily_trend': stock_data['daily_change'] > 0,
            'volume': stock_data['volume'] > 0
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

    def get_portfolio_value(self):
        """Calculate total portfolio value"""
        try:
            total_value = float(self.balance)  # Ensure balance is float
            
            # Calculate value of stock holdings
            for ticker, shares in self.portfolio.items():
                stock_data = self.get_stock_data(ticker)
                if stock_data and stock_data.get('current_price', 0) > 0:
                    position_value = float(shares) * float(stock_data['current_price'])
                    if position_value > 0 and position_value < 1000000:  # Sanity check
                        total_value += position_value
                    else:
                        print(f"Warning: Suspicious position value for {ticker}: ${position_value}")
            
            # Sanity check on total value
            if total_value <= 0 or total_value > 1000000:  # Assuming max reasonable value is $1M
                print(f"Warning: Suspicious total portfolio value: ${total_value}")
                # Return last valid value or initial balance
                return float(self.initial_balance)
                
            return total_value
            
        except Exception as e:
            print(f"Error calculating portfolio value: {e}")
            return float(self.initial_balance)  # Return initial balance as fallback

    def execute_trade(self, ticker, action, amount):
        """Execute a trade and save to MongoDB"""
        try:
            stock_data = self.get_stock_data(ticker)
            if not stock_data or stock_data.get('current_price', 0) <= 0:
                print(f"Invalid price data for {ticker}")
                return False

            # Validate and limit the trade amount
            portfolio_value = self.get_portfolio_value()
            max_trade_amount = min(portfolio_value * 0.2, 50000)  # Max 20% of portfolio or $50k
            amount = min(amount, max_trade_amount)
            
            if amount < 100:  # Minimum trade size
                print(f"Trade amount ${amount} too small for {ticker}")
                return False

            # Create trade document
            trade = {
                'timestamp': datetime.now(),
                'ticker': ticker,
                'action': action,
                'amount': amount,
                'status': 'pending'
            }
            
            # Save to MongoDB first
            result = self.trades_collection.insert_one(trade)
            if not result.inserted_id:
                raise Exception("Failed to save trade to MongoDB")
            
            # Update trade status
            self.trades_collection.update_one(
                {'_id': result.inserted_id},
                {'$set': {'status': 'executed'}}
            )
            
            print(f"Trade executed and saved: {ticker} {action} ${amount:.2f}")
            return True

        except Exception as e:
            print(f"Error executing trade: {e}")
            return False

    def log_trade(self, trade_data):
        """Log trade to JSON file"""
        try:
            if os.path.exists('trades.json'):
                with open('trades.json', 'r') as f:
                    trades = json.load(f)
            else:
                trades = []
            
            trades.append(trade_data)
            with open('trades.json', 'w') as f:
                json.dump(trades, f, indent=4)
        except Exception as e:
            print(f"Error logging trade: {e}")

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
                self.watchlist = data.get('watchlist', [])
                self.watchlist_performance = data.get('performance', {})
        except Exception as e:
            print(f"Error loading watchlist from MongoDB: {e}")
            self.watchlist = []
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
        """Save portfolio data to MongoDB"""
        try:
            portfolio_data = {
                'timestamp': datetime.now(),
                'portfolio': self.portfolio,
                'balance': self.balance,
                'total_value': self.get_portfolio_value()
            }
            self.save_to_mongodb(
                self.portfolio_collection,
                portfolio_data,
                ['timestamp', 'balance', 'total_value']
            )
            print("Portfolio saved to MongoDB successfully")
        except Exception as e:
            print(f"Error saving portfolio to MongoDB: {e}")

    def analyze_with_llm(self, content):
        """Analyze content using Ollama LLM"""
        try:
            prompt = f"""
            You are a trading bot's analysis engine. Analyze this market information and provide insights.
            
            Market Information:
            {content}
            
            Analyze the above information considering:
            1. Market sentiment
            2. Sector trends
            3. Trading opportunities
            4. Risk factors
            
            Respond ONLY with a valid JSON object using this exact format:
            {{
                "sentiment": "bullish/bearish/neutral",
                "trends": ["trend1", "trend2", "trend3"],
                "opportunities": ["opportunity1", "opportunity2"],
                "risks": ["risk1", "risk2"]
            }}
            """
            
            response = self.get_ollama_response(prompt)
            if response:
                try:
                    # Handle both string and dictionary responses
                    if isinstance(response, dict):
                        return response
                    elif isinstance(response, str):
                        # Clean the response string
                        json_str = response.strip()
                        if json_str.startswith('```json'):
                            json_str = json_str.split('```json')[1]
                        if json_str.endswith('```'):
                            json_str = json_str.rsplit('```', 1)[0]
                        
                        return json.loads(json_str.strip())
                    else:
                        print(f"Unexpected response type: {type(response)}")
                        return {
                            "sentiment": "neutral",
                            "trends": ["error parsing trends"],
                            "opportunities": ["error parsing opportunities"],
                            "risks": ["error parsing analysis"]
                        }
                except json.JSONDecodeError as e:
                    print(f"Error parsing LLM response: {e}")
                    print(f"Raw response: {response}")
                    return {
                        "sentiment": "neutral",
                        "trends": ["error parsing trends"],
                        "opportunities": ["error parsing opportunities"],
                        "risks": ["error parsing analysis"]
                    }
            return None
        except Exception as e:
            print(f"Error in LLM analysis: {e}")
            return None

    def generate_ai_summary(self):
        """Generate comprehensive AI summary from all MongoDB data"""
        try:
            # Get latest data from all collections
            latest_news = self.load_from_mongodb(self.news_collection, sort=[('timestamp', -1)], limit=50)
            latest_research = self.load_from_mongodb(self.research_collection, sort=[('timestamp', -1)], limit=20)
            latest_trades = self.load_from_mongodb(self.trades_collection, sort=[('timestamp', -1)], limit=20)
            portfolio_data = self.load_from_mongodb(self.portfolio_collection, sort=[('last_updated', -1)], limit=1)

            # Prepare data for analysis
            news_summary = "\n".join([
                f"- {news.get('timestamp').strftime('%Y-%m-%d %H:%M')} | {news.get('content', '')} | Sentiment: {news.get('sentiment', 'unknown')}"
                for news in latest_news if news.get('content')
            ])

            research_summary = "\n".join([
                f"- {r.get('type', 'research')} | {json.dumps(r.get('analysis', {}))}"
                for r in latest_research
            ])

            trades_summary = "\n".join([
                f"- {trade.get('timestamp').strftime('%Y-%m-%d %H:%M')} | {trade.get('action')} {trade.get('ticker')} | "
                f"Amount: ${trade.get('amount')} | Status: {trade.get('status')}"
                for trade in latest_trades
            ])

            current_portfolio = portfolio_data[0] if portfolio_data else {}

            # Create comprehensive prompt for LLM
            prompt = f"""
            You are a trading bot's analysis engine. Generate a comprehensive market summary based on the following data.
            Your response must be a SINGLE valid JSON object.
            
            Recent Market News:
            {news_summary}

            Research Analysis:
            {research_summary}

            Recent Trading Activity:
            {trades_summary}

            Current Portfolio Status:
            - Balance: ${current_portfolio.get('balance', 0):.2f}
            - Holdings: {json.dumps(current_portfolio.get('portfolio', {}))}
            - Watchlist: {json.dumps(current_portfolio.get('watchlist', []))}

            Analyze all the above data and provide a detailed summary.
            
            Format your entire response as a SINGLE JSON object like this:
            {{
                "market_analysis": {{
                    "sentiment": "bullish/bearish/neutral",
                    "key_trends": ["trend1", "trend2"],
                    "market_conditions": "description"
                }},
                "portfolio_analysis": {{
                    "performance": "good/neutral/poor",
                    "key_metrics": ["metric1", "metric2"],
                    "recommendations": ["recommendation1", "recommendation2"]
                }},
                "strategy_analysis": {{
                    "effectiveness": "high/medium/low",
                    "successful_patterns": ["pattern1", "pattern2"],
                    "areas_for_improvement": ["area1", "area2"]
                }},
                "opportunities": ["opportunity1", "opportunity2"],
                "risks": ["risk1", "risk2"],
                "recommendations": ["action1", "action2"]
            }}

            DO NOT include multiple JSON objects or separate them with semicolons.
            """

            # Get AI analysis
            response = self.get_ollama_response(prompt)
            if response:
                try:
                    # Clean the response to ensure it only contains the JSON part
                    json_str = response.strip()
                    if json_str.startswith('```json'):
                        json_str = json_str.split('```json')[1]
                    if json_str.endswith('```'):
                        json_str = json_str.rsplit('```', 1)[0]
                    
                    analysis = json.loads(json_str.strip())
                    
                    # Save summary to MongoDB
                    summary_data = {
                        'timestamp': datetime.now(),
                        'analysis': analysis,
                        'data_points': {
                            'news_count': len(latest_news),
                            'research_count': len(latest_research),
                            'trades_count': len(latest_trades)
                        },
                        'portfolio_snapshot': current_portfolio,
                        'raw_data': {
                            'news_summary': news_summary,
                            'research_summary': research_summary,
                            'trades_summary': trades_summary
                        }
                    }
                    
                    self.save_to_mongodb(
                        self.summary_collection, 
                        summary_data, 
                        ['timestamp', 'data_points.news_count', 'analysis.market_analysis.sentiment']
                    )
                    return analysis
                except json.JSONDecodeError as e:
                    print(f"Error parsing AI summary response: {e}")
                    print(f"Raw response: {response}")
                    return self.get_error_summary()
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

    def get_trading_decision(self, ticker, stock_data, research):
        """Get trading decision using LLM with enhanced context"""
        try:
            # Prepare research summary
            research_summary = "\n".join([
                f"- {r.get('type', 'research')}: {r.get('analysis', {}).get('summary', 'No summary')}"
                for r in research
            ])
            
            prompt = f"""
            Analyze this stock and provide a trading decision:
            
            Ticker: {ticker}
            Current Price: ${stock_data['current_price']}
            Daily Change: {stock_data['daily_change']}%
            Momentum: {stock_data['momentum']}
            
            Recent Research:
            {research_summary}
            
            Portfolio Context:
            - Current Position: {self.portfolio.get(ticker, 0)} shares
            - Portfolio Value: ${self.get_portfolio_value()}
            - Available Cash: ${self.balance}
            
            Respond with a JSON object in this exact format:
            {{
                "decision": "BUY/SELL/HOLD",
                "reasoning": ["reason1", "reason2", "reason3"],
                "confidence": "high/medium/low",
                "risk_assessment": "description of risks",
                "target_price": optional target price as number
            }}
            """
            
            response = self.get_ollama_response(prompt)
            if response:
                try:
                    # Clean and parse JSON response
                    json_str = response.strip()
                    if json_str.startswith('```json'):
                        json_str = json_str.split('```json')[1]
                    if json_str.endswith('```'):
                        json_str = json_str.rsplit('```', 1)[0]
                    
                    decision_data = json.loads(json_str.strip())
                    
                    # Save decision to MongoDB
                    decision_data.update({
                        'timestamp': datetime.now(),
                        'ticker': ticker,
                        'stock_data': stock_data,
                        'research_summary': research_summary
                    })
                    
                    self.save_to_mongodb(
                        self.db['trading_decisions'],
                        decision_data,
                        ['timestamp', 'ticker', 'decision']
                    )
                    
                    return decision_data['decision']
                except json.JSONDecodeError:
                    return "HOLD - Error parsing decision"
            return "HOLD - No response from LLM"
        except Exception as e:
            print(f"Error getting trading decision: {e}")
            return "HOLD - Error in analysis"

    def update_watchlist_from_research(self):
        """Update watchlist based on accumulated research"""
        try:
            # Get recent research data
            recent_research = self.get_research_data(data_type='sector_analysis', limit=50)
            print(f"\nAnalyzing {len(recent_research)} research entries for stock suggestions...")
            
            # Extract potential stock symbols from research
            potential_stocks = set()
            
            # Helper function to extract stock symbols
            def extract_symbols(text):
                # Common stock market indices to exclude
                indices = {'SPX', 'DJI', 'IXIC', 'RUT', 'VIX', 'NYSE', 'NASDAQ'}
                # Common words to exclude
                common_words = {'THE', 'AND', 'INC', 'CO', 'CORP', 'LTD', 'LLC', 'PLC'}
                # Find 1-5 letter uppercase words that might be tickers
                import re
                symbols = re.findall(r'\b[A-Z]{1,5}\b', text)
                # Filter out common indices and words
                return {s for s in symbols if s not in indices and s not in common_words}
            
            # Look through research data
            for r in recent_research:
                analysis = r.get('analysis', {})
                
                # Extract from opportunities and trends
                for item in analysis.get('opportunities', []) + analysis.get('trends', []):
                    potential_stocks.update(extract_symbols(item))
                
                # Extract from the query itself
                query = r.get('query', '')
                if 'stock' in query.lower():
                    potential_stocks.update(extract_symbols(query))
            
            print(f"Found potential stocks: {list(potential_stocks)}")
            
            # If we found stocks directly, try to add them
            stocks_added = 0
            for ticker in potential_stocks:
                if len(self.watchlist) < 20:  # Maximum 20 stocks
                    if self.add_to_watchlist(ticker):
                        print(f"Added {ticker} to watchlist based on research mention")
                        stocks_added += 1
            
            # If watchlist is still small, use LLM for additional suggestions
            if len(self.watchlist) < 5:
                # Get recent market news for context
                market_news = search_market_news("top performing stocks market analysis today")
                
                prompt = f"""
                Based on this market research and news, suggest 3-5 established stocks to add to the watchlist.
                Choose only well-known companies with high trading volume.
                
                Market News:
                {market_news}
                
                Current market conditions and trends suggest these sectors are promising:
                {self.sectors}
                
                Current watchlist: {self.watchlist}
                
                Respond with ONLY a JSON object containing stock symbols to add.
                Format example:
                {{
                    "add": ["AAPL", "MSFT", "GOOGL"]
                }}
                
                Important: Respond with ONLY the JSON object, no other text.
                """
                
                print("\nAsking LLM for stock suggestions...")
                response = self.get_ollama_response(prompt)
                if response:
                    try:
                        changes = json.loads(response)
                        suggested_stocks = changes.get('add', [])
                        if suggested_stocks:
                            print(f"LLM suggested stocks: {suggested_stocks}")
                            
                            # Add suggested stocks
                            for ticker in suggested_stocks:
                                if len(self.watchlist) < 20:
                                    if self.add_to_watchlist(ticker):
                                        print(f"Added {ticker} to watchlist based on LLM suggestion")
                                        stocks_added += 1
                        else:
                            print("No valid stock suggestions from LLM")
                                    
                    except json.JSONDecodeError as e:
                        print(f"Error parsing LLM response: {e}")
                        print(f"Raw response: {response}")
            
            print(f"\nWatchlist update complete. Added {stocks_added} new stocks.")
            return stocks_added > 0
                    
        except Exception as e:
            print(f"Error updating watchlist from research: {e}")
            return False

    def process_searxng_with_scraping(self, query, results):
        """Process SearxNG results without scraping"""
        try:
            # Only save the SearxNG results
            return self.save_searxng_results(query, results)
            
        except Exception as e:
            print(f"Error processing results: {e}")

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
        """Save SearxNG search results to MongoDB news collection"""
        try:
            news_entries = []
            timestamp = datetime.now()
            
            if isinstance(results, list):
                for result in results:
                    if isinstance(result, dict):
                        # Get URL from either 'link' or 'url' field
                        url = result.get('link') or result.get('url')
                        if not url:
                            continue
                            
                        # Process structured SearxNG result
                        entry = {
                            'timestamp': timestamp,
                            'query': query,
                            'title': result.get('title', ''),
                            'content': result.get('snippet', result.get('content', '')),
                            'url': url,
                            'source': result.get('source', 'searxng'),
                            'published_date': result.get('publishedDate', result.get('published_date')),
                            'engines': result.get('engines', []),
                            'score': result.get('score'),
                            'sentiment': self.analyze_sentiment(result.get('snippet', result.get('content', ''))),
                            'category': self.categorize_news(result.get('snippet', result.get('content', ''))),
                            'processed': True
                        }
                        news_entries.append(entry)
            
            if news_entries:
                # Save to MongoDB with enhanced indexes
                self.save_to_mongodb(
                    self.news_collection, 
                    news_entries,
                    ['timestamp', 'query', 'sentiment', 'category', 'url', 'source']
                )
                print(f"\nSaved {len(news_entries)} news entries to MongoDB:")
                for entry in news_entries:
                    print(f"\nSaved entry:")
                    print(f"Title: {entry['title'][:100]}...")
                    print(f"URL: {entry['url']}")
                    print(f"Source: {entry['source']}")
                    print(f"Published: {entry['published_date']}")
                return True
            return False
            
        except Exception as e:
            print(f"Error saving Searxng results: {e}")
            return False

    def run_trading_loop(self, start_phase=1):
        """Run the trading bot starting from specified phase
        
        Args:
            start_phase (int): Phase to start from (1-6)
            1: Market News
            2: Sector Research 
            3: Stock Discovery
            4: Watchlist Research
            5: Trading Decisions
            6: Summary and Analysis
        """
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
                                'timestamp': datetime.now()
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
                                        'timestamp': datetime.now()
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
                    for ticker in self.watchlist:
                        try:
                            print(f"\nAnalyzing {ticker}...")
                            self.update_stock_performance(ticker)
                            
                            stock_data = self.get_stock_data(ticker)
                            stock_research = self.get_research_data(ticker=ticker, limit=10)
                            
                            if not stock_data:
                                continue
                                
                            decision = self.get_trading_decision(ticker, stock_data, stock_research)
                            print(f"Decision for {ticker}: {decision}")
                            
                            if "BUY" in decision.upper():
                                portfolio_value = self.get_portfolio_value()
                                max_position = min(10000, portfolio_value * 0.1)
                                
                                if stock_data['momentum'] > 0:
                                    confidence = min(stock_data['momentum'], 0.5)
                                    position_size = max_position * (0.5 + confidence)
                                else:
                                    position_size = max_position * 0.5
                                    
                                amount = min(position_size, self.balance)
                                if amount >= 100:
                                    self.execute_trade(ticker, "BUY", amount)
                                    
                            elif "SELL" in decision.upper() and ticker in self.portfolio:
                                shares = self.portfolio[ticker]
                                position_value = shares * stock_data['current_price']
                                
                                if "stop loss" in decision.lower() or stock_data['momentum'] < -0.5:
                                    self.execute_trade(ticker, "SELL", position_value)
                                else:
                                    self.execute_trade(ticker, "SELL", position_value * 0.5)
                        
                        except Exception as e:
                            print(f"Error processing {ticker}: {e}")
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
                time.sleep(300)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trading Bot')
    parser.add_argument('--test', action='store_true', help='Run in testing mode (bypass market hours)')
    parser.add_argument('--start-phase', type=int, choices=range(1, 7), default=1,
                      help='Start from specific phase (1-6)')
    args = parser.parse_args()
    
    bot = TradingBot(testing_mode=args.test)
    print("Starting Trading Bot..." + (" (Testing Mode)" if args.test else ""))
    print(f"Starting from Phase {args.start_phase}")
    bot.run_trading_loop(start_phase=args.start_phase) 