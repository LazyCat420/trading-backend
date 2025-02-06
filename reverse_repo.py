import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from fredapi import Fred
import os
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.ERROR)

# Load environment variables from .env file
load_dotenv()

def get_reverse_repo_data():
    """Get historical reverse repo data using FRED API"""
    try:
        # Retrieve the API key from environment variables
        fred_api_key = os.getenv('YOUR_FRED_API_KEY')
        
        if not fred_api_key:
            raise ValueError("FRED API key not found in environment variables")
        
        # Initialize FRED API with your API key
        fred = Fred(api_key=fred_api_key)
        
        # Fetch reverse repo data
        repo_data = fred.get_series('RRPONTSYD')
        
        if repo_data.empty:
            raise ValueError("No reverse repo data returned")
        
        # Convert to DataFrame and rename the column
        repo_df = repo_data.to_frame(name='ReverseRepo')
        
        # Make DatetimeIndex timezone-naive
        repo_df.index = repo_df.index.tz_localize(None)
        
        return repo_df
        
    except Exception as e:
        logging.error(f"Error getting reverse repo data: {e}")
        return None

def get_market_data(start_date=None, end_date=None):
    """Get S&P 500 historical data"""
    try:
        spy = yf.Ticker("SPY")
        df = spy.history(start=start_date, end=end_date)
        if df.empty:
            raise ValueError("No market data returned")
        
        # Make DatetimeIndex timezone-naive
        df.index = df.index.tz_localize(None)
        
        return df[['Close']].rename(columns={'Close': 'SPY'})
    except Exception as e:
        logging.error(f"Error getting market data: {e}")
        return None

def get_mmf_repo_assets_data():
    """Get Money Market Fund Repo Assets data using FRED API"""
    try:
        fred_api_key = os.getenv('YOUR_FRED_API_KEY')
        if not fred_api_key:
            raise ValueError("FRED API key not found in environment variables")
        fred = Fred(api_key=fred_api_key)
        
        # Fetch Money Market Fund Repo Assets data (Level)
        mmf_repo_level_data = fred.get_series('BOGZ1FL632051103Q')
        if mmf_repo_level_data.empty:
            logging.warning("No Money Market Fund Repo Assets (Level) data returned")
            mmf_repo_level_df = pd.DataFrame(columns=['MMFRepoAssetsLevel']) # Return empty DataFrame
        else:
            mmf_repo_level_df = mmf_repo_level_data.to_frame(name='MMFRepoAssetsLevel')
            mmf_repo_level_df.index = mmf_repo_level_df.index.tz_localize(None)

        return mmf_repo_level_df

    except Exception as e:
        logging.error(f"Error getting Money Market Fund Repo Assets data: {e}")
        return pd.DataFrame(columns=['MMFRepoAssetsLevel']) # Return empty DataFrame in case of error


def get_total_repo_data():
    """Get Total Reverse Repo data using FRED API"""
    try:
        fred_api_key = os.getenv('YOUR_FRED_API_KEY')
        if not fred_api_key:
            raise ValueError("FRED API key not found in environment variables")
        fred = Fred(api_key=fred_api_key)

        # Fetch Total Reverse Repo data
        total_repo_data = fred.get_series('RRPTTLD')
        if total_repo_data.empty:
            logging.warning("No Total Reverse Repo data returned")
            total_repo_df = pd.DataFrame(columns=['TotalRepo']) # Return empty DataFrame
        else:
            total_repo_df = total_repo_data.to_frame(name='TotalRepo')
            total_repo_df.index = total_repo_df.index.tz_localize(None)

        return total_repo_df

    except Exception as e:
        logging.error(f"Error getting Total Reverse Repo data: {e}")
        return pd.DataFrame(columns=['TotalRepo']) # Return empty DataFrame in case of error


def get_effr_data():
    """Get Effective Federal Funds Rate data using FRED API"""
    try:
        fred_api_key = os.getenv('YOUR_FRED_API_KEY')
        if not fred_api_key:
            raise ValueError("FRED API key not found in environment variables")
        fred = Fred(api_key=fred_api_key)

        # Fetch Effective Federal Funds Rate data
        effr_data = fred.get_series('DFF')
        if effr_data.empty:
            logging.warning("No Effective Federal Funds Rate data returned")
            effr_df = pd.DataFrame(columns=['EFFR']) # Return empty DataFrame
        else:
            effr_df = effr_data.to_frame(name='EFFR')
            effr_df.index = effr_df.index.tz_localize(None)

        return effr_df

    except Exception as e:
        logging.error(f"Error getting Effective Federal Funds Rate data: {e}")
        return pd.DataFrame(columns=['EFFR']) # Return empty DataFrame in case of error


def get_sofr_data():
    """Get Secured Overnight Financing Rate data using FRED API"""
    try:
        fred_api_key = os.getenv('YOUR_FRED_API_KEY')
        if not fred_api_key:
            raise ValueError("FRED API key not found in environment variables")
        fred = Fred(api_key=fred_api_key)

        # Fetch Secured Overnight Financing Rate data
        sofr_data = fred.get_series('SOFR')
        if sofr_data.empty:
            logging.warning("No Secured Overnight Financing Rate data returned")
            sofr_df = pd.DataFrame(columns=['SOFR']) # Return empty DataFrame
        else:
            sofr_df = sofr_data.to_frame(name='SOFR')
            sofr_df.index = sofr_df.index.tz_localize(None)
        return sofr_df

    except Exception as e:
        logging.error(f"Error getting Secured Overnight Financing Rate data: {e}")
        return pd.DataFrame(columns=['SOFR']) # Return empty DataFrame in case of error


def get_treasury_bill_yield_data():
    """Get 3-Month Treasury Bill Yield data using FRED API"""
    try:
        fred_api_key = os.getenv('YOUR_FRED_API_KEY')
        if not fred_api_key:
            raise ValueError("FRED API key not found in environment variables")
        fred = Fred(api_key=fred_api_key)

        # Fetch 3-Month Treasury Bill Yield data
        tbill_data = fred.get_series('TB3MS')
        if tbill_data.empty:
            logging.warning("No 3-Month Treasury Bill Yield data returned")
            tbill_df = pd.DataFrame(columns=['TBILL3M']) # Return empty DataFrame
        else:
            tbill_df = tbill_data.to_frame(name='TBILL3M')
            tbill_df.index = tbill_df.index.tz_localize(None)

        return tbill_df

    except Exception as e:
        logging.error(f"Error getting 3-Month Treasury Bill Yield data: {e}")
        return pd.DataFrame(columns=['TBILL3M']) # Return empty DataFrame in case of error

def analyze_correlation():
    """Analyze correlation between reverse repo and market with additional data"""
    # Get reverse repo data
    repo_df = get_reverse_repo_data()
    if repo_df is None or repo_df.empty:
        logging.error("Failed to get reverse repo data, correlation analysis stopped.")
        return

    # Get market data for same period
    market_df = get_market_data(
        start_date=repo_df.index.min(),
        end_date=repo_df.index.max()
    )
    if market_df is None or market_df.empty:
        logging.error("Failed to get market data, correlation analysis stopped.")
        return

    # Get additional data
    mmf_repo_df = get_mmf_repo_assets_data()
    total_repo_df = get_total_repo_data()
    effr_df = get_effr_data()
    sofr_df = get_sofr_data()
    tbill_df = get_treasury_bill_yield_data()

    try:
        # Merge dataframes, handling potentially empty dataframes
        dfs_to_merge = [repo_df, market_df, mmf_repo_df, total_repo_df, effr_df, sofr_df, tbill_df]
        df = dfs_to_merge[0] # Start with the first dataframe (repo_df)
        for next_df in dfs_to_merge[1:]:
            if not next_df.empty: # Only merge if the dataframe is not empty
                df = pd.merge(df, next_df, left_index=True, right_index=True, how='inner')

        if df.empty:
            raise ValueError("No overlapping data found after merge")

        # Calculate correlations
        correlation = df.corr()
        print("\nCorrelation Matrix:")
        print(correlation)

        # Plot correlation heatmap
        plt.figure(figsize=(12, 10)) # Adjust figure size to fit more data
        sns.heatmap(
            correlation,
            annot=True,
            cmap='RdYlBu',
            vmin=-1,
            vmax=1,
            center=0
        )
        plt.title('Correlation Matrix: Reverse Repo, S&P 500, and Additional Data') # Updated title
        plt.tight_layout()
        plt.savefig('correlation_matrix.png') # Changed filename to avoid overwrite

        # Calculate rolling correlation (still focusing on ReverseRepo vs SPY for rolling)
        rolling_corr = df['ReverseRepo'].rolling(window=30).corr(df['SPY'])

        # Plot rolling correlation
        plt.figure(figsize=(12, 6))
        rolling_corr.plot()
        plt.title('30-Day Rolling Correlation: Reverse Repo vs S&P 500')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.tight_layout()
        plt.savefig('rolling_correlation.png')

        return {
            'correlation_matrix': correlation, # Return the full correlation matrix
            'rolling_correlation': rolling_corr.mean(),
            'data': df
        }

    except Exception as e:
        logging.error(f"Error analyzing correlation: {e}")
        return None

if __name__ == "__main__":
    results = analyze_correlation()
    if results:
        print(f"\nAverage rolling correlation: {results['rolling_correlation']:.3f}")
