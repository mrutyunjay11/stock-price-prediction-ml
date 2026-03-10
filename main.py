"""
Main Execution Script
======================
This is the main entry point for the Stock Price Prediction project.
It provides a command-line interface to run the complete pipeline.
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def fetch_data(symbol, start_date, end_date):
    """Fetch stock data."""
    from src.data_fetcher import fetch_stock_data
    
    print(f"\n{'='*60}")
    print(f"FETCHING DATA FOR {symbol}")
    print(f"{'='*60}")
    
    df = fetch_stock_data(
        symbol,
        start_date=start_date,
        end_date=end_date
    )
    
    print(f"\nData fetched successfully!")
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df


def preprocess_and_add_indicators(df, lookback=60):
    """Add technical indicators and prepare data."""
    from src.features import add_technical_indicators
    from src.preprocessor import prepare_data
    
    print(f"\n{'='*60}")
    print("PREPROCESSING DATA & ADDING TECHNICAL INDICATORS")
    print(f"{'='*60}")
    
    # Add technical indicators
    df = add_technical_indicators(
        df,
        ma_window=20,
        rsi_period=14,
        include_ema=True,
        include_macd=False
    )
    
    # Prepare data for LSTM
    feature_columns = ['open', 'high', 'low', 'close', 'volume']
    data_dict = prepare_data(
        df=df,
        lookback=lookback,
        feature_columns=feature_columns,
        test_size=0.2
    )
    
    print("\nPreprocessing completed!")
    return df, data_dict


def train_model_pipeline(data_dict, epochs=50, batch_size=32, lstm_units=50):
    """Train the LSTM model."""
    from src.model import create_lstm_model
    from src.train import train_model
    
    print(f"\n{'='*60}")
    print("TRAINING LSTM MODEL")
    print(f"{'='*60}")
    
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    
    # Create validation split
    val_split = int(len(X_train) * 0.8)
    X_val = X_train[val_split:]
    y_val = y_train[val_split:]
    X_train = X_train[:val_split]
    y_train = y_train[:val_split]
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    print(f"\nInput shape: {input_shape}")
    print(f"Training samples: {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    print(f"Test samples: {len(X_test):,}")
    
    # Train model
    results = train_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        input_shape=input_shape,
        epochs=epochs,
        batch_size=batch_size,
        lstm_units=lstm_units,
        dropout_rate=0.2,
        learning_rate=0.001,
        model_save_path='models/lstm_stock_model.h5',
        patience=10,
        verbose=1
    )
    
    print("\nModel training completed!")
    return results


def make_predictions(model, data_dict, df):
    """Make predictions and evaluate."""
    from src.predict import predict_prices, calculate_prediction_metrics
    from src.visualization import create_visualization_dashboard
    
    print(f"\n{'='*60}")
    print("MAKING PREDICTIONS")
    print(f"{'='*60}")
    
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    scaler = data_dict['scaler']
    test_indices = data_dict['test_indices']
    feature_columns = data_dict.get('feature_columns', ['open', 'high', 'low', 'close', 'volume'])
    
    # Make predictions
    predictions = predict_prices(
        model=model,
        X=X_test,
        scaler=scaler,
        feature_columns=feature_columns
    )
    
    # Get actual prices
    actual_prices = df['close'].values[test_indices]
    
    # Calculate metrics
    metrics = calculate_prediction_metrics(actual_prices, predictions)
    
    print("\nPredictions generated!")
    
    # Create visualizations
    create_visualization_dashboard(
        df=df,
        predictions=predictions,
        test_indices=test_indices,
        history=model.history if hasattr(model, 'history') else None,
        save_dir='outputs/',
        show=False
    )
    
    return predictions, metrics


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Stock Price Prediction using LSTM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --symbol AAPL --train
  python main.py --symbol GOOGL --predict
  python main.py --symbol TSLA --train --predict
  python main.py --symbol MSFT --epochs 100 --lookback 90
        """
    )
    
    parser.add_argument(
        '--symbol', '-s',
        type=str,
        default='AAPL',
        help='Stock ticker symbol (default: AAPL)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Start date in YYYY-MM-DD format (default: 5 years ago)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date in YYYY-MM-DD format (default: today)'
    )
    
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train a new model'
    )
    
    parser.add_argument(
        '--predict',
        action='store_true',
        help='Make predictions with trained model'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )
    
    parser.add_argument(
        '--lookback',
        type=int,
        default=60,
        help='Lookback period in days (default: 60)'
    )
    
    parser.add_argument(
        '--lstm-units',
        type=int,
        default=50,
        help='Number of LSTM units (default: 50)'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/lstm_stock_model.h5',
        help='Path to load/save model (default: models/lstm_stock_model.h5)'
    )
    
    args = parser.parse_args()
    
    # Print welcome message
    print("\n" + "="*60)
    print("   STOCK PRICE PREDICTION USING LSTM")
    print("="*60)
    print(f"Symbol: {args.symbol}")
    print(f"Mode: {'Training' if args.train else 'Prediction only'}")
    print("="*60)
    
    # Set default dates
    if args.end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    else:
        end_date = args.end_date
    
    if args.start_date is None:
        start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
    else:
        start_date = args.start_date
    
    # Step 1: Fetch data
    df = fetch_data(args.symbol, start_date, end_date)
    
    # Step 2: Preprocess and add indicators
    df, data_dict = preprocess_and_add_indicators(df, lookback=args.lookback)
    
    # Step 3: Train model if requested
    model = None
    if args.train:
        results = train_model_pipeline(
            data_dict,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lstm_units=args.lstm_units
        )
        model = results['model']
        
        # Save model
        os.makedirs('models', exist_ok=True)
        model.save(args.model_path)
        print(f"\nModel saved to: {args.model_path}")
    
    # Step 4: Make predictions if requested or if we have a model
    if args.predict or (model is not None):
        if model is None:
            # Try to load existing model
            from src.model import load_trained_model
            
            if os.path.exists(args.model_path):
                print(f"\nLoading existing model from: {args.model_path}")
                model = load_trained_model(args.model_path)
            else:
                print("\nNo trained model found. Please train a model first with --train flag.")
                return
        
        # Make predictions
        predictions, metrics = make_predictions(model, data_dict, df)
        
        # Print final summary
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        print(f"Symbol: {args.symbol}")
        print(f"Test Set Size: {len(data_dict['X_test']):,} samples")
        print(f"Mean Absolute Error: ${metrics['MAE']:.2f}")
        print(f"Root Mean Squared Error: ${metrics['RMSE']:.2f}")
        print(f"Mean Absolute Percentage Error: {metrics['MAPE']}")
        print(f"Direction Accuracy: {metrics['Direction_Accuracy']}")
        print("="*60)
        print("\n✅ Process completed successfully!")
        print(f"Visualizations saved to: outputs/")
    
    else:
        print("\nSkipping prediction. Use --predict flag to make predictions.")
    
    print("\nNote: To view interactive dashboard, run:")
    print("      streamlit run app/dashboard.py\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
