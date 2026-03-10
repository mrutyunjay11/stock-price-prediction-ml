"""
Stock Price Prediction using LSTM
==================================
A complete machine learning project for predicting stock price trends.
"""

__version__ = '1.0.0'
__author__ = 'Your Name'

from .data_fetcher import fetch_stock_data, get_stock_info
from .preprocessor import prepare_data, normalize_data, create_sequences
from .features import add_technical_indicators, calculate_moving_average, calculate_rsi
from .model import create_lstm_model, load_trained_model
from .train import train_model
from .predict import predict_prices
from .visualization import plot_actual_vs_predicted, create_visualization_dashboard

__all__ = [
    'fetch_stock_data',
    'get_stock_info',
    'prepare_data',
    'normalize_data',
    'create_sequences',
    'add_technical_indicators',
    'calculate_moving_average',
    'calculate_rsi',
    'create_lstm_model',
    'load_trained_model',
    'train_model',
    'predict_prices',
    'plot_actual_vs_predicted',
    'create_visualization_dashboard'
]
