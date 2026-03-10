"""
Visualization Module
=====================
This module handles all plotting and visualization for the stock prediction project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List
import os


def set_plot_style():
    """Set a professional style for plots."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10


def plot_actual_vs_predicted(
    actual: np.ndarray,
    predicted: np.ndarray,
    dates: Optional[np.ndarray] = None,
    title: str = 'Actual vs Predicted Stock Prices',
    xlabel: str = 'Date',
    ylabel: str = 'Price ($)',
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot actual vs predicted stock prices.
    
    Args:
        actual: Array of actual prices
        predicted: Array of predicted prices
        dates: Array of dates for x-axis labels (optional)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    
    Example:
        >>> plot_actual_vs_predicted(y_test, predictions, dates=test_dates)
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Create date indices if not provided
    if dates is None:
        x_values = np.arange(len(actual))
    else:
        x_values = dates
    
    # Plot actual prices
    ax.plot(x_values, actual, label='Actual Price', linewidth=2, color='blue', alpha=0.7)
    
    # Plot predicted prices
    ax.plot(x_values, predicted, label='Predicted Price', linewidth=2, color='red', alpha=0.7, linestyle='--')
    
    # Customize plot
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels if dates are provided
    if dates is not None:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    # Display plot if requested
    if show:
        plt.show()
    
    plt.close()


def plot_training_history(
    history,
    metrics: List[str] = ['loss', 'mae'],
    title: str = 'Model Training History',
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot training history showing loss and metrics over epochs.
    
    Args:
        history: Keras History object from model.fit()
        metrics: List of metrics to plot
        title: Plot title
        save_path: Path to save the plot
        show: Whether to display the plot
    
    Example:
        >>> plot_training_history(history.history)
    """
    set_plot_style()
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 6))
    
    # Ensure axes is always an array
    if len(metrics) == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Get training and validation values
        train_values = history.history[metric]
        val_values = history.history[f'val_{metric}']
        epochs = range(1, len(train_values) + 1)
        
        # Plot
        ax.plot(epochs, train_values, 'b-', label=f'Training {metric}', linewidth=2)
        ax.plot(epochs, val_values, 'r--', label=f'Validation {metric}', linewidth=2)
        
        ax.set_title(f'{metric.capitalize()} Over Epochs', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.upper())
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, y=1.02, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    # Display if requested
    if show:
        plt.show()
    
    plt.close()


def plot_technical_indicators(
    df: pd.DataFrame,
    indicators: List[str] = ['ma_20', 'rsi_14'],
    title: str = 'Technical Indicators',
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot technical indicators (MA, RSI, etc.).
    
    Args:
        df: DataFrame with stock data and technical indicators
        indicators: List of indicator column names to plot
        title: Plot title
        save_path: Path to save the plot
        show: Whether to display the plot
    
    Example:
        >>> plot_technical_indicators(df, indicators=['ma_20', 'rsi_14'])
    """
    set_plot_style()
    
    # Determine number of subplots
    n_plots = len(indicators) + 1  # +1 for price chart
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots))
    
    # Ensure axes is always an array
    if n_plots == 1:
        axes = [axes]
    
    # Plot price and moving average
    ax_idx = 0
    axes[ax_idx].plot(df['date'], df['close'], label='Close Price', linewidth=2, color='blue')
    
    if 'ma_20' in df.columns:
        axes[ax_idx].plot(df['date'], df['ma_20'], label='20-day MA', linewidth=2, color='orange')
    
    if 'ema_20' in df.columns:
        axes[ax_idx].plot(df['date'], df['ema_20'], label='20-day EMA', linewidth=2, color='green')
    
    axes[ax_idx].set_title('Stock Price with Moving Averages', fontsize=14, fontweight='bold')
    axes[ax_idx].set_ylabel('Price ($)')
    axes[ax_idx].legend(loc='best')
    axes[ax_idx].grid(True, alpha=0.3)
    plt.setp(axes[ax_idx].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot other indicators
    for indicator in indicators:
        ax_idx += 1
        if indicator in df.columns:
            color = 'green' if 'rsi' in indicator else 'purple'
            axes[ax_idx].plot(df['date'], df[indicator], linewidth=2, color=color)
            
            # Add reference lines for RSI
            if 'rsi' in indicator:
                axes[ax_idx].axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought (70)')
                axes[ax_idx].axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold (30)')
            
            axes[ax_idx].set_title(f'{indicator.replace("_", " ").upper()}', fontsize=14, fontweight='bold')
            axes[ax_idx].set_ylabel(indicator.upper())
            axes[ax_idx].legend(loc='best')
            axes[ax_idx].grid(True, alpha=0.3)
            plt.setp(axes[ax_idx].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.suptitle(title, y=1.02, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    # Display if requested
    if show:
        plt.show()
    
    plt.close()


def plot_prediction_comparison(
    df_comparison: pd.DataFrame,
    title: str = 'Prediction Comparison',
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot comparison of actual vs predicted prices from DataFrame.
    
    Args:
        df_comparison: DataFrame with columns: date, actual_price, predicted_price
        title: Plot title
        save_path: Path to save the plot
        show: Whether to display the plot
    
    Example:
        >>> plot_prediction_comparison(comparison_df)
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(df_comparison['date'], df_comparison['actual_price'], 
            label='Actual', linewidth=2, color='blue', alpha=0.7)
    ax.plot(df_comparison['date'], df_comparison['predicted_price'], 
            label='Predicted', linewidth=2, color='red', alpha=0.7, linestyle='--')
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    # Display if requested
    if show:
        plt.show()
    
    plt.close()


def create_visualization_dashboard(
    df: pd.DataFrame,
    predictions: np.ndarray,
    test_indices: np.ndarray,
    history=None,
    save_dir: str = 'outputs/',
    show: bool = True
):
    """
    Create a complete set of visualizations.
    
    Args:
        df: Original DataFrame with stock data
        predictions: Array of predicted prices
        test_indices: Indices of test data points
        history: Training history (optional)
        save_dir: Directory to save plots
        show: Whether to display plots
    
    Example:
        >>> create_visualization_dashboard(df, predictions, test_idx, history)
    """
    print("\n=== Creating Visualizations ===")
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Actual vs Predicted
    plot_actual_vs_predicted(
        actual=df['close'].values[test_indices],
        predicted=predictions,
        dates=df['date'].values[test_indices],
        title='Stock Price Prediction: Actual vs Predicted',
        save_path=os.path.join(save_dir, 'actual_vs_predicted.png'),
        show=show
    )
    
    # 2. Training History (if available)
    if history is not None:
        plot_training_history(
            history=history,
            metrics=['loss', 'mae'],
            title='Model Training Performance',
            save_path=os.path.join(save_dir, 'training_history.png'),
            show=show
        )
    
    # 3. Technical Indicators
    if all(col in df.columns for col in ['ma_20', 'rsi_14']):
        plot_technical_indicators(
            df=df,
            indicators=['ma_20', 'rsi_14'],
            title='Technical Indicators Analysis',
            save_path=os.path.join(save_dir, 'technical_indicators.png'),
            show=show
        )
    
    print(f"All visualizations saved to: {save_dir}")


if __name__ == "__main__":
    # Test the visualization module
    from src.data_fetcher import fetch_stock_data
    from src.features import add_technical_indicators
    
    print("Testing visualization module...")
    
    # Fetch sample data
    df = fetch_stock_data('AAPL', start_date='2020-01-01', end_date='2023-01-01')
    
    # Add technical indicators
    df = add_technical_indicators(df)
    
    # Create sample predictions (for testing)
    test_size = int(len(df) * 0.2)
    test_data = df.tail(test_size)
    predictions = test_data['close'].values * np.random.uniform(0.95, 1.05, test_size)
    
    # Create visualizations
    create_visualization_dashboard(
        df=df,
        predictions=predictions,
        test_indices=np.arange(len(df) - test_size, len(df)),
        history=None,
        save_dir='outputs/test/',
        show=False
    )
    
    print("Test complete!")
