"""
Model Module
=============
This module defines the LSTM neural network architecture for stock price prediction.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from typing import Optional, Tuple


def create_lstm_model(
    input_shape: Tuple[int, int],
    lstm_units: int = 50,
    dense_units: int = 25,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
    num_lstm_layers: int = 2
) -> keras.Model:
    """
    Create an LSTM model for stock price prediction.
    
    Args:
        input_shape: Shape of input data (lookback_period, num_features)
        lstm_units: Number of LSTM units per layer (default: 50)
        dense_units: Number of units in dense layer (default: 25)
        dropout_rate: Dropout rate to prevent overfitting (default: 0.2)
        learning_rate: Learning rate for optimizer (default: 0.001)
        num_lstm_layers: Number of LSTM layers (default: 2)
    
    Returns:
        Compiled Keras model
    
    Example:
        >>> model = create_lstm_model(input_shape=(60, 5), lstm_units=50)
        >>> model.summary()
    """
    print(f"\n=== Creating LSTM Model ===")
    print(f"Input shape: {input_shape}")
    print(f"LSTM units: {lstm_units}")
    print(f"Number of LSTM layers: {num_lstm_layers}")
    
    model = Sequential()
    
    # First LSTM layer with return sequences
    model.add(LSTM(
        units=lstm_units,
        return_sequences=True if num_lstm_layers > 1 else False,
        input_shape=input_shape
    ))
    model.add(Dropout(dropout_rate))
    
    # Additional LSTM layers
    for i in range(1, num_lstm_layers):
        model.add(LSTM(
            units=lstm_units,
            return_sequences=(i < num_lstm_layers - 1)
        ))
        model.add(Dropout(dropout_rate))
    
    # Dense layer
    model.add(Dense(units=dense_units, activation='relu'))
    
    # Output layer
    model.add(Dense(units=1, activation='linear'))
    
    # Compile model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    
    print("Model created successfully!")
    return model


def get_model_summary(model: keras.Model) -> str:
    """
    Get a string summary of the model architecture.
    
    Args:
        model: Keras model
    
    Returns:
        String containing model summary
    
    Example:
        >>> summary = get_model_summary(model)
        >>> print(summary)
    """
    # Create a string buffer to capture summary
    import io
    string_buffer = io.StringIO()
    model.summary(print_fn=lambda x: string_buffer.write(x + '\n'))
    return string_buffer.getvalue()


def build_callbacks(
    model_save_path: str,
    patience: int = 10,
    monitor: str = 'val_loss',
    mode: str = 'min'
) -> list:
    """
    Build Keras callbacks for training.
    
    Args:
        model_save_path: Path to save the best model
        patience: Number of epochs with no improvement before early stopping
        monitor: Metric to monitor
        mode: Mode for monitoring ('min' or 'max')
    
    Returns:
        List of Keras callbacks
    
    Example:
        >>> callbacks = build_callbacks('models/best_model.h5', patience=10)
    """
    print(f"\n=== Setting up Training Callbacks ===")
    print(f"Model save path: {model_save_path}")
    print(f"Early stopping patience: {patience} epochs")
    
    callbacks = []
    
    # Model Checkpoint - Save the best model
    checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        monitor=monitor,
        verbose=1,
        save_best_only=True,
        mode=mode,
        save_weights_only=False
    )
    callbacks.append(checkpoint)
    print(f"✓ ModelCheckpoint added - monitoring {monitor}")
    
    # Early Stopping - Stop training when validation loss stops improving
    early_stopping = EarlyStopping(
        monitor=monitor,
        patience=patience,
        verbose=1,
        mode=mode,
        restore_best_weights=True
    )
    callbacks.append(early_stopping)
    print(f"✓ EarlyStopping added - patience={patience}")
    
    return callbacks


def load_trained_model(model_path: str) -> keras.Model:
    """
    Load a pre-trained model from disk.
    
    Args:
        model_path: Path to the saved model file (.h5)
    
    Returns:
        Loaded Keras model
    
    Example:
        >>> model = load_trained_model('models/lstm_stock_model.h5')
    """
    print(f"Loading model from: {model_path}")
    
    try:
        model = keras.models.load_model(model_path)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise


def evaluate_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32
) -> dict:
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained Keras model
        X_test: Test input data
        y_test: Test target data
        batch_size: Batch size for evaluation
    
    Returns:
        Dictionary containing evaluation metrics
    
    Example:
        >>> metrics = evaluate_model(model, X_test, y_test)
        >>> print(f"MSE: {metrics['mse']}, MAE: {metrics['mae']}")
    """
    print("\n=== Evaluating Model ===")
    
    # Evaluate on test data
    results = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
    
    # Get metric names
    metrics_names = model.metrics_names
    metrics = dict(zip(metrics_names, results))
    
    print(f"\nEvaluation Results:")
    for metric_name, value in metrics.items():
        print(f"{metric_name.upper()}: {value:.6f}")
    
    return metrics


if __name__ == "__main__":
    # Test the model module
    print("Testing model module...")
    
    # Create a sample model
    model = create_lstm_model(
        input_shape=(60, 5),
        lstm_units=50,
        num_lstm_layers=2
    )
    
    # Print model summary
    print("\nModel Summary:")
    print(get_model_summary(model))
    
    # Count total parameters
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
