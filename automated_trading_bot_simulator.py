import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def run_trading_simulator(ticker="AAPL", start_date="2010-01-01", end_date="2020-01-01",
                          short_window=50, long_window=200, starting_capital=10000):
    # Fetch historical stock data
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Calculate Simple Moving Averages (SMA) for technical indicators
    data['SMA50'] = data['Close'].rolling(window=short_window).mean()
    data['SMA200'] = data['Close'].rolling(window=long_window).mean()
    
    # Generate buy/sell signals: Buy (1) if SMA50 > SMA200, Sell (-1) if SMA50 < SMA200
    data['Signal'] = 0
    data.loc[data['SMA50'] > data['SMA200'], 'Signal'] = 1
    data.loc[data['SMA50'] < data['SMA200'], 'Signal'] = -1

    # To avoid look-ahead bias, shift the signal to create positions
    data['Position'] = data['Signal'].shift(1)

    # Calculate daily percentage returns
    data['Daily Return'] = data['Close'].pct_change()
    
    # Strategy Return: if in position, profit is daily return, else 0 (assuming fully invested when in position)
    # For short positions, the return is inverted
    data['Strategy Return'] = data['Position'] * data['Daily Return']

    # Calculate cumulative returns for both the market and the strategy
    data['Cumulative Market Return'] = (1 + data['Daily Return']).cumprod()
    data['Cumulative Strategy Return'] = (1 + data['Strategy Return']).cumprod()

    # Calculate equity curves based on starting capital
    data['Market Equity'] = starting_capital * data['Cumulative Market Return']
    data['Strategy Equity'] = starting_capital * data['Cumulative Strategy Return']

    # Performance summary at the end of the period
    total_strategy_return = data['Strategy Equity'].iloc[-1] - starting_capital
    total_market_return = data['Market Equity'].iloc[-1] - starting_capital

    print(f"Total Strategy Return: {total_strategy_return:.2f} ({total_strategy_return/starting_capital:.2%})")
    print(f"Total Market Return: {total_market_return:.2f} ({total_market_return/starting_capital:.2%})")

    # Plot equity curves
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Market Equity'], label="Market Equity", color="blue")
    plt.plot(data.index, data['Strategy Equity'], label="Strategy Equity", color="red")
    plt.title(f"Automated Trading Simulator for {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Equity Value")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_trading_simulator() 