"""
Streamlit Dashboard for Stock Price Prediction
===============================================
This module provides an interactive web interface for stock price prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_fetcher import fetch_stock_data, get_stock_info
from src.features import add_technical_indicators
from src.preprocessor import prepare_data
from src.model import create_lstm_model, load_trained_model
from src.train import train_model
from src.predict import predict_prices, calculate_prediction_metrics
from src.visualization import plot_actual_vs_predicted, plot_training_history


# Page configuration
st.set_page_config(
    page_title="Stock Price Prediction - LSTM",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'data_dict' not in st.session_state:
        st.session_state.data_dict = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None


def main():
    """Main function for Streamlit dashboard."""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<p class="main-header">📈 Stock Price Prediction using LSTM</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Stock symbol input
        stock_symbol = st.text_input(
            "Stock Symbol",
            value="AAPL",
            help="Enter stock ticker symbol (e.g., AAPL, GOOGL, TSLA)"
        ).upper()
        
        # Date range
        col1, col2 = st.columns(2)
        with col1:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                key="end_date"
            )
        with col2:
            start_date = st.date_input(
                "Start Date",
                value=end_date - timedelta(days=365*2),
                key="start_date"
            )
        
        # Model parameters
        st.subheader("Model Parameters")
        lookback = st.slider("Lookback Period (days)", 30, 120, 60, 5)
        epochs = st.slider("Number of Epochs", 10, 200, 50, 10)
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
        lstm_units = st.slider("LSTM Units", 25, 100, 50, 25)
        
        # Action buttons
        st.subheader("Actions")
        fetch_btn = st.button("📊 Fetch Data", use_container_width=True)
        train_btn = st.button("🧠 Train Model", use_container_width=True)
        predict_btn = st.button("🔮 Make Predictions", use_container_width=True)
        
        # Model loading
        st.subheader("Load Pre-trained Model")
        model_path = st.text_input(
            "Model Path",
            value="models/lstm_stock_model.h5",
            placeholder="models/lstm_stock_model.h5"
        )
        if st.button("Load Model", use_container_width=True):
            if os.path.exists(model_path):
                try:
                    st.session_state.model = load_trained_model(model_path)
                    st.success("Model loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
            else:
                st.warning("Model file not found. Please train a new model first.")
    
    # Main content area
    if fetch_btn:
        fetch_and_display_data(stock_symbol, start_date, end_date)
    
    if train_btn:
        train_model_pipeline(lookback, epochs, batch_size, lstm_units)
    
    if predict_btn:
        make_predictions()
    
    # Display current model status
    with st.expander("📊 Model Status"):
        if st.session_state.model:
            st.success("✓ Model is loaded and ready")
            st.info(f"Model type: {type(st.session_state.model)}")
        else:
            st.warning("⚠ No model loaded. Please train or load a model.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>Built with LSTM Neural Networks | For educational purposes only</p>
            <p>⚠️ Not financial advice. Stock market investments carry risks.</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def fetch_and_display_data(symbol, start_date, end_date):
    """Fetch stock data and display it."""
    with st.spinner(f"Fetching data for {symbol}..."):
        try:
            df = fetch_stock_data(
                symbol,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            # Add technical indicators
            df = add_technical_indicators(df, ma_window=20, rsi_period=14)
            
            # Store in session state
            st.session_state.df = df
            
            # Display success message
            st.success(f"✅ Successfully fetched {len(df)} records for {symbol}")
            
            # Show company info if available
            try:
                info = get_stock_info(symbol)
                if info and 'longName' in info:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Company", info.get('longName', 'N/A'))
                    with col2:
                        st.metric("Sector", info.get('sector', 'N/A'))
                    with col3:
                        st.metric("Industry", info.get('industry', 'N/A'))
            except:
                pass
            
            # Display data
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader(f"📈 {symbol} Stock Price History")
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(df['date'], df['close'], label='Close Price', linewidth=2)
                if 'ma_20' in df.columns:
                    ax.plot(df['date'], df['ma_20'], label='20-day MA', linestyle='--', alpha=0.7)
                ax.set_xlabel('Date')
                ax.set_ylabel('Price ($)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                st.pyplot(fig)
            
            with col2:
                st.subheader("Technical Indicators")
                
                # RSI Gauge
                latest_rsi = df['rsi_14'].iloc[-1]
                st.metric("Current RSI (14)", f"{latest_rsi:.2f}")
                
                if latest_rsi > 70:
                    st.warning("Overbought (RSI > 70)")
                elif latest_rsi < 30:
                    st.success("Oversold (RSI < 30)")
                else:
                    st.info("Neutral Zone")
                
                # Current price
                current_price = df['close'].iloc[-1]
                st.metric("Current Price", f"${current_price:.2f}")
            
            # Show raw data
            with st.expander("📋 View Raw Data"):
                st.dataframe(df.tail(100))
            
            # Statistics
            with st.expander("📊 Statistics"):
                st.write(df.describe())
            
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")


def train_model_pipeline(lookback, epochs, batch_size, lstm_units):
    """Train the LSTM model."""
    if st.session_state.df is None:
        st.error("Please fetch data first!")
        return
    
    with st.spinner("Training LSTM model... This may take a few minutes."):
        try:
            df = st.session_state.df
            
            # Prepare data
            feature_columns = ['open', 'high', 'low', 'close', 'volume']
            
            # Check if we have enough data
            if len(df) < lookback + 100:
                st.error(f"Not enough data. Need at least {lookback + 100} records, got {len(df)}.")
                return
            
            # Prepare data for training
            data_dict = prepare_data(
                df=df,
                lookback=lookback,
                feature_columns=feature_columns,
                test_size=0.2
            )
            
            X_train = data_dict['X_train']
            y_train = data_dict['y_train']
            X_test = data_dict['X_test']
            y_test = data_dict['y_test']
            
            # Create validation set from training data
            val_split = int(len(X_train) * 0.8)
            X_val = X_train[val_split:]
            y_val = y_train[val_split:]
            X_train = X_train[:val_split]
            y_train = y_train[:val_split]
            
            # Create model
            input_shape = (X_train.shape[1], X_train.shape[2])
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Creating LSTM model architecture...")
            model = create_lstm_model(
                input_shape=input_shape,
                lstm_units=lstm_units,
                dropout_rate=0.2,
                learning_rate=0.001
            )
            
            # Train model
            status_text.text("Training model...")
            
            # Custom callback to update progress
            class StreamlitProgressCallback(keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(progress)
                    status_text.text(f"Epoch {epoch + 1}/{epochs} - Loss: {logs['loss']:.4f}")
            
            # Import keras here to avoid issues
            from tensorflow import keras
            
            callbacks = [StreamlitProgressCallback()]
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0
            )
            
            progress_bar.progress(1.0)
            status_text.text("✅ Training completed!")
            
            # Save model
            model_save_path = 'models/lstm_stock_model.h5'
            os.makedirs('models', exist_ok=True)
            model.save(model_save_path)
            
            # Store in session state
            st.session_state.model = model
            st.session_state.data_dict = data_dict
            
            st.success(f"Model trained and saved to {model_save_path}")
            
            # Display training history
            with st.expander("📊 Training History", expanded=True):
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                # Loss plot
                axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
                axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
                axes[0].set_title('Loss Over Epochs')
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Loss')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # MAE plot
                axes[1].plot(history.history['mae'], label='Train MAE', linewidth=2)
                axes[1].plot(history.history['val_mae'], label='Val MAE', linewidth=2)
                axes[1].set_title('MAE Over Epochs')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('MAE')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Evaluate model
            st.subheader("Model Evaluation")
            test_loss = model.evaluate(X_test, y_test, verbose=0)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Test Loss (MSE)", f"{test_loss:.4f}")
            with col2:
                st.metric("Test MAE", f"{model.evaluate(X_test, y_test, verbose=0)[1]:.4f}")
            
        except Exception as e:
            st.error(f"Error during training: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


def make_predictions():
    """Make predictions using the trained model."""
    if st.session_state.model is None:
        st.error("Please train or load a model first!")
        return
    
    if st.session_state.data_dict is None:
        st.error("No data available for predictions. Please fetch data and train model.")
        return
    
    with st.spinner("Making predictions..."):
        try:
            model = st.session_state.model
            data_dict = st.session_state.data_dict
            
            # Make predictions
            X_test = data_dict['X_test']
            scaler = data_dict['scaler']
            feature_columns = data_dict.get('feature_columns', ['open', 'high', 'low', 'close', 'volume'])
            
            # Predict
            predictions_normalized = model.predict(X_test, verbose=0)
            
            # Inverse transform
            target_index = feature_columns.index('close')
            dummy = np.zeros((len(predictions_normalized), len(feature_columns)))
            dummy[:, target_index] = predictions_normalized.flatten()
            predictions = scaler.inverse_transform(dummy)[:, target_index]
            
            # Get actual prices
            df = st.session_state.df
            test_indices = data_dict['test_indices']
            actual_prices = df['close'].values[test_indices]
            
            # Calculate metrics
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            mae = mean_absolute_error(actual_prices, predictions)
            rmse = np.sqrt(mean_squared_error(actual_prices, predictions))
            mape = np.mean(np.abs((actual_prices - predictions) / actual_prices)) * 100
            
            # Display metrics
            st.subheader("Prediction Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MAE", f"${mae:.2f}")
            with col2:
                st.metric("RMSE", f"${rmse:.2f}")
            with col3:
                st.metric("MAPE", f"{mape:.2f}%")
            
            # Plot predictions
            st.subheader("📈 Actual vs Predicted Prices")
            fig, ax = plt.subplots(figsize=(14, 7))
            
            dates = df['date'].values[test_indices]
            ax.plot(dates, actual_prices, label='Actual', linewidth=2, color='blue', alpha=0.7)
            ax.plot(dates, predictions, label='Predicted', linewidth=2, color='red', alpha=0.7, linestyle='--')
            
            ax.set_xlabel('Date')
            ax.set_ylabel('Price ($)')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Show comparison table
            with st.expander("📋 Detailed Comparison"):
                comparison_df = pd.DataFrame({
                    'Date': pd.to_datetime(dates),
                    'Actual Price': actual_prices,
                    'Predicted Price': predictions,
                    'Difference': np.abs(actual_prices - predictions)
                })
                st.dataframe(comparison_df)
            
            # Future prediction
            st.subheader("🔮 Next Day Prediction")
            last_sequence = X_test[-1].reshape(1, lookback, -1)
            next_day_pred = model.predict(last_sequence, verbose=0)
            dummy_next = np.zeros((1, len(feature_columns)))
            dummy_next[:, target_index] = next_day_pred.flatten()
            next_day_price = scaler.inverse_transform(dummy_next)[:, target_index][0]
            
            st.success(f"Predicted next day closing price: ${next_day_price:.2f}")
            
        except Exception as e:
            st.error(f"Error making predictions: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


if __name__ == "__main__":
    # Import matplotlib for plotting
    import matplotlib.pyplot as plt
    from tensorflow import keras
    
    main()
