"""
Training Module
================
This module handles the training of LSTM models for stock price prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from src.model import create_lstm_model, build_callbacks
from src.preprocessor import prepare_data


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model: Optional = None,
    input_shape: Optional[tuple] = None,
    epochs: int = 50,
    batch_size: int = 32,
    lstm_units: int = 50,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
    model_save_path: str = 'models/lstm_stock_model.h5',
    patience: int = 10,
    verbose: int = 1
) -> Dict:
    """
    Train an LSTM model on the provided data.
    
    Args:
        X_train: Training input sequences
        y_train: Training target values
        X_val: Validation input sequences
        y_val: Validation target values
        model: Pre-built model (optional). If None, a new model is created.
        input_shape: Input shape for model creation (required if model is None)
        epochs: Number of training epochs (default: 50)
        batch_size: Batch size for training (default: 32)
        lstm_units: Number of LSTM units (used if creating new model)
        dropout_rate: Dropout rate (used if creating new model)
        learning_rate: Learning rate (used if creating new model)
        model_save_path: Path to save the best model
        patience: Early stopping patience
        verbose: Verbosity level (0, 1, or 2)
    
    Returns:
        Dictionary containing:
        - model: Trained model
        - history: Training history object
        - metrics: Final evaluation metrics
    
    Example:
        >>> results = train_model(X_train, y_train, X_val, y_val, 
        ...                       input_shape=(60, 5), epochs=50)
    """
    print("\n" + "="*60)
    print("=== LSTM MODEL TRAINING ===")
    print("="*60)
    
    # Create model if not provided
    if model is None:
        if input_shape is None:
            raise ValueError("input_shape must be provided when model is None")
        
        print(f"\nCreating new LSTM model...")
        model = create_lstm_model(
            input_shape=input_shape,
            lstm_units=lstm_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate
        )
    else:
        print("\nUsing provided pre-built model...")
    
    # Build callbacks
    callbacks = build_callbacks(
        model_save_path=model_save_path,
        patience=patience
    )
    
    # Print training configuration
    print(f"\n=== Training Configuration ===")
    print(f"Training samples: {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Model save path: {model_save_path}")
    print(f"Early stopping patience: {patience} epochs")
    
    # Train the model
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose
    )
    
    # Evaluate final model
    print("\n" + "="*60)
    print("Evaluating trained model...")
    print("="*60)
    
    train_loss = model.evaluate(X_train, y_train, verbose=0)
    val_loss = model.evaluate(X_val, y_val, verbose=0)
    
    print(f"\nFinal Training Loss: {train_loss:.6f}")
    print(f"Final Validation Loss: {val_loss:.6f}")
    
    return {
        'model': model,
        'history': history,
        'train_loss': train_loss,
        'val_loss': val_loss
    }


def train_from_dataframe(
    df: pd.DataFrame,
    lookback: int = 60,
    test_size: float = 0.2,
    feature_columns: Optional[list] = None,
    **train_kwargs
) -> Dict:
    """
    Complete training pipeline from raw DataFrame to trained model.
    
    This function handles the entire pipeline:
    1. Data preparation
    2. Model creation
    3. Training
    4. Evaluation
    
    Args:
        df: Raw stock data DataFrame
        lookback: Lookback period for sequences
        test_size: Proportion for testing
        feature_columns: Columns to use as features
        **train_kwargs: Additional arguments passed to train_model()
    
    Returns:
        Dictionary containing model, history, and preprocessing objects
    
    Example:
        >>> results = train_from_dataframe(df, lookback=60, epochs=50)
        >>> model = results['model']
    """
    print("\n" + "="*60)
    print("=== COMPLETE TRAINING PIPELINE ===")
    print("="*60)
    
    # Prepare data
    data_dict = prepare_data(
        df=df,
        lookback=lookback,
        feature_columns=feature_columns,
        test_size=test_size
    )
    
    # Extract training and validation data
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_test = data_dict['X_test']
    y_test = data_dict['test_indices']
    
    # Use test set as validation for now
    X_val = X_test
    y_val = y_test
    
    # Get input shape
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    print(f"\nInput shape: {input_shape}")
    
    # Train model
    results = train_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        input_shape=input_shape,
        **train_kwargs
    )
    
    # Add preprocessing objects to results
    results['data_dict'] = data_dict
    results['scaler'] = data_dict['scaler']
    results['feature_columns'] = feature_columns or ['open', 'high', 'low', 'close', 'volume']
    
    return results


if __name__ == "__main__":
    # Test the training module
    from src.data_fetcher import fetch_stock_data
    
    print("Testing training module with sample data...")
    
    # Fetch sample data
    df = fetch_stock_data('AAPL', start_date='2020-01-01', end_date='2023-01-01')
    
    # Train model (with minimal epochs for testing)
    results = train_from_dataframe(
        df,
        lookback=60,
        test_size=0.2,
        epochs=5,  # Small number for testing
        batch_size=32,
        model_save_path='models/test_model.h5'
    )
    
    print("\nTraining completed!")
    print(f"Model type: {type(results['model'])}")
    print(f"History keys: {results['history'].history.keys()}")
