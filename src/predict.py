"""
Prediction Module
==================
This module handles making predictions using trained LSTM models.
"""

import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional


def predict_prices(
    model: keras.Model,
    X: np.ndarray,
    scaler: MinMaxScaler,
    feature_columns: list,
    target_column: str = 'close'
) -> np.ndarray:
    """
    Make price predictions and inverse transform to original scale.
    
    Args:
        model: Trained LSTM model
        X: Input sequences (normalized)
        scaler: Fitted MinMaxScaler
        feature_columns: List of feature columns used during training
        target_column: Name of the target column (default: 'close')
    
    Returns:
        Array of predicted prices in original scale
    
    Example:
        >>> predictions = predict_prices(model, X_test, scaler, feature_columns)
    """
    print("Making predictions...")
    
    # Make predictions
    predictions_normalized = model.predict(X, verbose=0)
    
    # Inverse transform to get actual prices
    actual_predictions = inverse_transform_predictions(
        predictions_normalized,
        scaler,
        feature_columns,
        target_column
    )
    
    print(f"Generated {len(actual_predictions)} predictions")
    return actual_predictions


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


def predict_next_day(
    model: keras.Model,
    last_sequence: np.ndarray,
    scaler: MinMaxScaler,
    feature_columns: list
) -> float:
    """
    Predict the next day's closing price.
    
    Args:
        model: Trained LSTM model
        last_sequence: Last sequence of data (shape: (1, lookback, num_features))
        scaler: Fitted MinMaxScaler
        feature_columns: List of feature columns
    
    Returns:
        Predicted next day closing price
    
    Example:
        >>> next_price = predict_next_day(model, last_seq, scaler, features)
    """
    print("Predicting next day's price...")
    
    # Make prediction
    prediction_normalized = model.predict(last_sequence, verbose=0)
    
    # Inverse transform
    actual_price = inverse_transform_predictions(
        prediction_normalized,
        scaler,
        feature_columns
    )
    
    predicted_price = float(actual_price[0])
    print(f"Predicted next day price: ${predicted_price:.2f}")
    return predicted_price


def create_prediction_dataframe(
    df: pd.DataFrame,
    predictions: np.ndarray,
    test_indices: np.ndarray,
    lookback: int = 60
) -> pd.DataFrame:
    """
    Create a DataFrame comparing actual vs predicted prices.
    
    Args:
        df: Original DataFrame with stock data
        predictions: Array of predicted prices
        test_indices: Indices of test data points
        lookback: Lookback period used for sequences
    
    Returns:
        DataFrame with columns: Date, Actual_Price, Predicted_Price
    
    Example:
        >>> comparison_df = create_prediction_dataframe(df, preds, test_idx)
    """
    print("Creating prediction comparison DataFrame...")
    
    # Extract dates and actual prices
    dates = df['date'].values[test_indices]
    actual_prices = df['close'].values[test_indices]
    
    # Create DataFrame
    comparison_df = pd.DataFrame({
        'date': dates,
        'actual_price': actual_prices,
        'predicted_price': predictions.flatten()
    })
    
    # Convert date column to datetime
    comparison_df['date'] = pd.to_datetime(comparison_df['date'])
    
    print(f"Created DataFrame with {len(comparison_df)} comparisons")
    return comparison_df


def calculate_prediction_metrics(
    actual: np.ndarray,
    predicted: np.ndarray
) -> dict:
    """
    Calculate various metrics to evaluate prediction quality.
    
    Args:
        actual: Array of actual prices
        predicted: Array of predicted prices
    
    Returns:
        Dictionary containing:
        - MAE: Mean Absolute Error
        - RMSE: Root Mean Squared Error
        - MAPE: Mean Absolute Percentage Error
        - Direction_Accuracy: Percentage of correctly predicted directions
    
    Example:
        >>> metrics = calculate_prediction_metrics(actual, predicted)
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    print("\n=== Calculating Prediction Metrics ===")
    
    # Ensure arrays are 1D
    actual = actual.flatten()
    predicted = predicted.flatten()
    
    # Remove any NaN values
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    actual = actual[mask]
    predicted = predicted[mask]
    
    # Calculate MAE
    mae = mean_absolute_error(actual, predicted)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # Calculate direction accuracy
    if len(actual) > 1:
        actual_direction = np.diff(actual) > 0
        predicted_direction = np.diff(predicted) > 0
        direction_accuracy = np.mean(actual_direction == predicted_direction) * 100
    else:
        direction_accuracy = 0.0
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': f"{mape:.2f}%",
        'Direction_Accuracy': f"{direction_accuracy:.2f}%"
    }
    
    print(f"MAE: ${mae:.2f}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Direction Accuracy: {direction_accuracy:.2f}%")
    
    return metrics


if __name__ == "__main__":
    # Test the prediction module
    from src.data_fetcher import fetch_stock_data
    from src.preprocessor import prepare_data
    from src.model import load_trained_model
    
    print("Testing prediction module...")
    
    # Note: This requires a pre-trained model
    # Uncomment below code if you have a trained model
    """
    # Fetch data
    df = fetch_stock_data('AAPL', start_date='2020-01-01', end_date='2023-01-01')
    
    # Prepare data
    data_dict = prepare_data(df, lookback=60, test_size=0.2)
    
    # Load model
    model = load_trained_model('models/lstm_stock_model.h5')
    
    # Make predictions
    predictions = predict_prices(
        model,
        data_dict['X_test'],
        data_dict['scaler'],
        feature_columns=['open', 'high', 'low', 'close', 'volume']
    )
    
    # Calculate metrics
    metrics = calculate_prediction_metrics(
        data_dict['y_test'],
        predictions
    )
    """
    print("Test complete!")
