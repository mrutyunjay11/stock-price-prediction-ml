"""
Data Fetcher Module
====================
This module handles fetching stock price data from Yahoo Finance using yfinance API.
It provides functions to download historical stock data with various timeframes.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional


def fetch_stock_data(
    ticker_symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = "5y"
) -> pd.DataFrame:
    """
    Fetch historical stock data from Yahoo Finance.
    
    Args:
        ticker_symbol: Stock symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')
        start_date: Start date in 'YYYY-MM-DD' format (optional if period is specified)
        end_date: End date in 'YYYY-MM-DD' format (optional, defaults to today)
        period: Time period for data ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
    
    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Adj Close, Volume
    
    Example:
        >>> df = fetch_stock_data('AAPL', start_date='2020-01-01', end_date='2024-01-01')
        >>> print(df.head())
    """
    try:
        # Create ticker object
        ticker = yf.Ticker(ticker_symbol)
        
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()
        else:
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        if start_date is None:
            # Calculate start date based on period
            if period == 'max':
                start_date = datetime(2000, 1, 1)
            else:
                # Parse period to calculate start date
                period_map = {
                    '1d': timedelta(days=1),
                    '5d': timedelta(days=5),
                    '1mo': timedelta(days=30),
                    '3mo': timedelta(days=90),
                    '6mo': timedelta(days=180),
                    '1y': timedelta(days=365),
                    '2y': timedelta(days=730),
                    '5y': timedelta(days=1825),
                    '10y': timedelta(days=3650),
                    'ytd': datetime(end_date.year, 1, 1)
                }
                start_date = end_date - period_map.get(period, timedelta(days=1825))
        else:
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        
        # Download historical data
        print(f"Fetching data for {ticker_symbol} from {start_date.date()} to {end_date.date()}...")
        df = ticker.history(start=start_date, end=end_date)
        
        # Check if data is empty
        if df.empty:
            raise ValueError(f"No data found for ticker symbol: {ticker_symbol}")
        
        # Reset index to make Date a column
        df.reset_index(inplace=True)
        
        # Rename columns to standard format
        df.columns = df.columns.str.lower()
        
        # Ensure 'date' column is in datetime format
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"Successfully fetched {len(df)} records for {ticker_symbol}")
        return df
        
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        raise


def get_stock_info(ticker_symbol: str) -> dict:
    """
    Get basic information about a stock.
    
    Args:
        ticker_symbol: Stock symbol
    
    Returns:
        Dictionary containing stock information
    
    Example:
        >>> info = get_stock_info('AAPL')
        >>> print(info['longName'])
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        return info
    except Exception as e:
        print(f"Error fetching stock info: {str(e)}")
        return {}


def get_multiple_stocks_data(
    ticker_symbols: list,
    period: str = "5y"
) -> dict:
    """
    Fetch data for multiple stocks at once.
    
    Args:
        ticker_symbols: List of stock symbols
        period: Time period for data
    
    Returns:
        Dictionary with ticker symbols as keys and DataFrames as values
    
    Example:
        >>> data = get_multiple_stocks_data(['AAPL', 'GOOGL', 'MSFT'])
        >>> print(data['AAPL'].head())
    """
    stock_data = {}
    
    for symbol in ticker_symbols:
        try:
            print(f"Fetching data for {symbol}...")
            df = fetch_stock_data(symbol, period=period)
            stock_data[symbol] = df
        except Exception as e:
            print(f"Failed to fetch data for {symbol}: {str(e)}")
            continue
    
    return stock_data


if __name__ == "__main__":
    # Example usage
    print("Testing data fetcher module...")
    
    # Fetch Apple stock data
    df = fetch_stock_data('AAPL', start_date='2020-01-01', end_date='2024-01-01')
    print(f"\nApple Stock Data Shape: {df.shape}")
    print(df.head())
    print(df.info())
    
    # Get stock information
    info = get_stock_info('AAPL')
    if info:
        print(f"\nCompany Name: {info.get('longName', 'N/A')}")
        print(f"Sector: {info.get('sector', 'N/A')}")
        print(f"Industry: {info.get('industry', 'N/A')}")
