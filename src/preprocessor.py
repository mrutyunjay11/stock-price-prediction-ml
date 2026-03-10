"""
Preprocessor Module
====================
This module handles data preprocessing including:
- Handling missing values
- Normalization using MinMaxScaler
- Creating time-series sequences for LSTM
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional


def handle_missing_values(df: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame with stock data
        method: Method to handle missing values
                - 'ffill': Forward fill (use previous value)
                - 'bfill': Backward fill (use next value)
                - 'drop': Drop rows with missing values
                - 'interpolate': Linear interpolation
    
    Returns:
        DataFrame with missing values handled
    
    Example:
        >>> df_clean = handle_missing_values(df, method='ffill')
    """
    print(f"Handling missing values using method: {method}")
    
    # Check for missing values
    missing_count = df.isnull().sum()
    total_missing = missing_count.sum()
    print(f"Total missing values before handling: {total_missing}")
    
    if total_missing == 0:
        print("No missing values found!")
        return df
    
    if method == 'ffill':
        df_filled = df.fillna(method='ffill')
    elif method == 'bfill':
        df_filled = df.fillna(method='bfill')
    elif method == 'drop':
        df_filled = df.dropna()
    elif method == 'interpolate':
        # Only interpolate numeric columns
        df_filled = df.copy()
        numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
        df_filled[numeric_cols] = df_filled[numeric_cols].interpolate(method='linear')
        # Fill any remaining NaN at the edges
        df_filled = df_filled.fillna(method='bfill').fillna(method='ffill')
    else:
        raise ValueError(f"Unknown method: {method}. Use 'ffill', 'bfill', 'drop', or 'interpolate'.")
    
    # Verify no missing values remain
    remaining_missing = df_filled.isnull().sum().sum()
    print(f"Missing values after handling: {remaining_missing}")
    
    return df_filled


def normalize_data(
    df: pd.DataFrame,
    feature_columns: list,
    scaler: Optional[MinMaxScaler] = None
) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Normalize specified columns using MinMaxScaler.
    
    Args:
        df: Input DataFrame
        feature_columns: List of column names to normalize
        scaler: Pre-fitted scaler (optional). If None, a new scaler is fitted.
    
    Returns:
        Tuple of (normalized_array, scaler)
    
    Example:
        >>> columns = ['open', 'high', 'low', 'close', 'volume']
        >>> normalized_data, scaler = normalize_data(df, columns)
    """
    print(f"Normalizing features: {feature_columns}")
    
    # Extract features
    features = df[feature_columns].values
    
    # Create and fit scaler if not provided
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = scaler.fit_transform(features)
        print("New scaler fitted.")
    else:
        scaled_features = scaler.transform(features)
        print("Existing scaler used for transformation.")
    
    print(f"Data normalized to range [0, 1]")
    return scaled_features, scaler


def create_sequences(
    data: np.ndarray,
    lookback: int = 60
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM training.
    
    Args:
        data: Normalized data array
        lookback: Number of previous time steps to use for prediction (default: 60 days)
    
    Returns:
        Tuple of (sequences, targets)
        - sequences: Array of shape (num_samples, lookback, num_features)
        - targets: Array of shape (num_samples,)
    
    Example:
        >>> X, y = create_sequences(scaled_data, lookback=60)
        >>> print(f"X shape: {X.shape}, y shape: {y.shape}")
    """
    X, y = [], []
    
    # Create sequences
    for i in range(lookback, len(data)):
        # Input sequence: lookback days
        X.append(data[i-lookback:i])
        # Target: next day's close price (assuming close price is at index 3)
        y.append(data[i, 3])  # Index 3 corresponds to 'close' price
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Created {len(X)} sequences with lookback period of {lookback} days")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    return X, y


def prepare_data(
    df: pd.DataFrame,
    lookback: int = 60,
    feature_columns: Optional[list] = None,
    test_size: float = 0.2,
    shuffle: bool = False
) -> dict:
    """
    Complete data preparation pipeline for LSTM.
    
    Args:
        df: Raw stock data DataFrame
        lookback: Lookback period for sequences
        feature_columns: Columns to use as features (default: OHLCV)
        test_size: Proportion of data for testing (default: 0.2)
        shuffle: Whether to shuffle data (default: False for time series)
    
    Returns:
        Dictionary containing:
        - X_train, X_test: Training and testing sequences
        - y_train, y_test: Training and testing targets
        - scaler: Fitted MinMaxScaler
        - train_indices: Indices for training set
        - test_indices: Indices for testing set
    
    Example:
        >>> data_dict = prepare_data(df, lookback=60, test_size=0.2)
        >>> X_train = data_dict['X_train']
    """
    # Default feature columns
    if feature_columns is None:
        feature_columns = ['open', 'high', 'low', 'close', 'volume']
    
    print("\n=== Data Preparation Pipeline ===")
    print(f"Using features: {feature_columns}")
    print(f"Lookback period: {lookback} days")
    print(f"Test size: {test_size*100}%")
    
    # Step 1: Handle missing values
    df_clean = handle_missing_values(df, method='ffill')
    
    # Step 2: Normalize data
    scaled_data, scaler = normalize_data(df_clean, feature_columns)
    
    # Step 3: Create sequences
    X, y = create_sequences(scaled_data, lookback)
    
    # Step 4: Split data
    split_index = int(len(X) * (1 - test_size))
    
    if shuffle:
        # Shuffle data before splitting (not recommended for time series)
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        split_index = int(len(X) * (1 - test_size))
    
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]
    
    # Store indices for reference
    train_indices = np.arange(split_index) + lookback
    test_indices = np.arange(split_index, len(X)) + lookback
    
    print(f"\nTraining set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'train_indices': train_indices,
        'test_indices': test_indices,
        'df_clean': df_clean
    }


def inverse_transform_predictions(
    predictions: np.ndarray,
    scaler: MinMaxScaler,
    feature_columns: list,
    target_column: str = 'close'
) -> np.ndarray:
    """
    Inverse transform normalized predictions back to original scale.
    
    Args:
        predictions: Normalized predictions
        scaler: Fitted MinMaxScaler
        feature_columns: List of feature columns used during normalization
        target_column: Name of the target column
    
    Returns:
        Denormalized predictions in original scale
    
    Example:
        >>> actual_prices = inverse_transform_predictions(preds, scaler, feature_columns)
    """
    # Get the index of target column
    target_index = feature_columns.index(target_column)
    
    # Create a zero array with shape (len(predictions), num_features)
    dummy = np.zeros((len(predictions), len(feature_columns)))
    
    # Fill in the predictions at the correct index
    dummy[:, target_index] = predictions.flatten()
    
    # Inverse transform
    original_scale = scaler.inverse_transform(dummy)
    
    # Return only the target column
    return original_scale[:, target_index]


if __name__ == "__main__":
    # Test the preprocessor module
    from src.data_fetcher import fetch_stock_data
    
    print("Testing preprocessor module...")
    
    # Fetch sample data
    df = fetch_stock_data('AAPL', start_date='2020-01-01', end_date='2023-01-01')
    
    # Prepare data
    data_dict = prepare_data(df, lookback=60, test_size=0.2)
    
    print(f"\nTraining data shape: {data_dict['X_train'].shape}")
    print(f"Test data shape: {data_dict['X_test'].shape}")
    print(f"Scaler type: {type(data_dict['scaler'])}")
