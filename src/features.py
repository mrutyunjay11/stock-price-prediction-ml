"""
Features Module
================
This module calculates technical indicators for stock analysis:
- Moving Average (MA)
- Relative Strength Index (RSI)
"""

import pandas as pd
import numpy as np
from typing import Optional


def calculate_moving_average(
    df: pd.DataFrame,
    column: str = 'close',
    window: int = 20
) -> pd.DataFrame:
    """
    Calculate Simple Moving Average (SMA).
    
    Args:
        df: DataFrame with stock price data
        column: Column to calculate MA for (default: 'close')
        window: Window size for moving average (default: 20 days)
    
    Returns:
        DataFrame with added 'ma_{window}' column
    
    Example:
        >>> df_with_ma = calculate_moving_average(df, window=20)
    """
    print(f"Calculating {window}-day Moving Average for {column}...")
    
    df_copy = df.copy()
    ma_column_name = f'ma_{window}'
    
    # Calculate moving average
    df_copy[ma_column_name] = df_copy[column].rolling(window=window).mean()
    
    print(f"Added column: {ma_column_name}")
    return df_copy


def calculate_rsi(
    df: pd.DataFrame,
    column: str = 'close',
    period: int = 14
) -> pd.DataFrame:
    """
    Calculate Relative Strength Index (RSI).
    
    RSI is a momentum oscillator that measures the speed and magnitude of 
    recent price changes to evaluate overbought or oversold conditions.
    
    RSI values range from 0 to 100:
    - RSI > 70: Overbought (potential sell signal)
    - RSI < 30: Oversold (potential buy signal)
    
    Args:
        df: DataFrame with stock price data
        column: Column to calculate RSI for (default: 'close')
        period: Period for RSI calculation (default: 14 days)
    
    Returns:
        DataFrame with added 'rsi_{period}' column
    
    Example:
        >>> df_with_rsi = calculate_rsi(df, period=14)
    """
    print(f"Calculating {period}-day RSI for {column}...")
    
    df_copy = df.copy()
    rsi_column_name = f'rsi_{period}'
    
    # Calculate price changes
    delta = df_copy[column].diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and average loss
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Calculate RS (Relative Strength)
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    # Handle division by zero
    df_copy[rsi_column_name] = 100 - (100 / (1 + rs))
    
    # Fill NaN values
    df_copy[rsi_column_name] = df_copy[rsi_column_name].fillna(50)
    
    print(f"Added column: {rsi_column_name}")
    return df_copy


def calculate_ema(
    df: pd.DataFrame,
    column: str = 'close',
    span: int = 20
) -> pd.DataFrame:
    """
    Calculate Exponential Moving Average (EMA).
    
    EMA gives more weight to recent prices, making it more responsive 
    to new information compared to simple moving average.
    
    Args:
        df: DataFrame with stock price data
        column: Column to calculate EMA for (default: 'close')
        span: Span/period for EMA (default: 20 days)
    
    Returns:
        DataFrame with added 'ema_{span}' column
    
    Example:
        >>> df_with_ema = calculate_ema(df, span=20)
    """
    print(f"Calculating {span}-day EMA for {column}...")
    
    df_copy = df.copy()
    ema_column_name = f'ema_{span}'
    
    # Calculate EMA
    df_copy[ema_column_name] = df_copy[column].ewm(span=span, adjust=False).mean()
    
    print(f"Added column: {ema_column_name}")
    return df_copy


def calculate_macd(
    df: pd.DataFrame,
    column: str = 'close',
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    MACD is a trend-following momentum indicator that shows the relationship 
    between two moving averages of a security's price.
    
    Components:
    - MACD Line: (12-day EMA - 26-day EMA)
    - Signal Line: 9-day EMA of MACD Line
    - Histogram: MACD Line - Signal Line
    
    Args:
        df: DataFrame with stock price data
        column: Column to calculate MACD for (default: 'close')
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line period (default: 9)
    
    Returns:
        DataFrame with added columns: 'macd', 'macd_signal', 'macd_hist'
    
    Example:
        >>> df_with_macd = calculate_macd(df)
    """
    print(f"Calculating MACD ({fast_period}/{slow_period}/{signal_period})...")
    
    df_copy = df.copy()
    
    # Calculate EMAs
    ema_fast = df_copy[column].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df_copy[column].ewm(span=slow_period, adjust=False).mean()
    
    # Calculate MACD Line
    df_copy['macd'] = ema_fast - ema_slow
    
    # Calculate Signal Line
    df_copy['macd_signal'] = df_copy['macd'].ewm(span=signal_period, adjust=False).mean()
    
    # Calculate Histogram
    df_copy['macd_hist'] = df_copy['macd'] - df_copy['macd_signal']
    
    print("Added columns: macd, macd_signal, macd_hist")
    return df_copy


def add_technical_indicators(
    df: pd.DataFrame,
    ma_window: int = 20,
    rsi_period: int = 14,
    include_ema: bool = True,
    include_macd: bool = False
) -> pd.DataFrame:
    """
    Add all technical indicators to the DataFrame.
    
    This is a convenience function that adds multiple technical indicators at once.
    
    Args:
        df: DataFrame with stock price data
        ma_window: Window for Moving Average (default: 20)
        rsi_period: Period for RSI (default: 14)
        include_ema: Whether to include EMA (default: True)
        include_macd: Whether to include MACD (default: False)
    
    Returns:
        DataFrame with added technical indicator columns
    
    Example:
        >>> df_indicators = add_technical_indicators(df)
        >>> print(df_indicators.columns)
    """
    print("\n=== Adding Technical Indicators ===")
    
    # Start with original DataFrame
    df_result = df.copy()
    
    # Add Moving Average
    df_result = calculate_moving_average(df_result, window=ma_window)
    
    # Add RSI
    df_result = calculate_rsi(df_result, period=rsi_period)
    
    # Add EMA if requested
    if include_ema:
        df_result = calculate_ema(df_result, span=ma_window)
    
    # Add MACD if requested
    if include_macd:
        df_result = calculate_macd(df_result)
    
    print(f"\nTotal indicators added: {len(df_result.columns) - len(df.columns)}")
    print(f"New columns: {[col for col in df_result.columns if col not in df.columns]}")
    
    return df_result


if __name__ == "__main__":
    # Test the features module
    from src.data_fetcher import fetch_stock_data
    
    print("Testing features module...")
    
    # Fetch sample data
    df = fetch_stock_data('AAPL', start_date='2020-01-01', end_date='2023-01-01')
    
    # Add technical indicators
    df_with_indicators = add_technical_indicators(
        df,
        ma_window=20,
        rsi_period=14,
        include_ema=True,
        include_macd=True
    )
    
    print(f"\nDataFrame shape: {df_with_indicators.shape}")
    print(f"Columns: {df_with_indicators.columns.tolist()}")
    print(f"\nLast 5 rows:")
    print(df_with_indicators.tail())
