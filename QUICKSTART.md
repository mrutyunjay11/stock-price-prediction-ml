# Quick Start Guide - Stock Price Prediction using LSTM

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: Installation may take 5-10 minutes depending on your internet connection.

### Step 2: Train Your First Model

Open terminal/command prompt and run:

```bash
python main.py --symbol AAPL --train --predict
```

This will:
- Fetch Apple stock data (last 5 years)
- Add technical indicators (MA, RSI)
- Train an LSTM model
- Make predictions and show performance metrics
- Save visualizations to `outputs/` folder

### Step 3: Launch Interactive Dashboard

```bash
streamlit run app/dashboard.py
```

Your browser will open automatically at `http://localhost:8501`

In the dashboard you can:
- Enter any stock symbol (AAPL, GOOGL, TSLA, etc.)
- Adjust date ranges
- Train custom models
- View interactive predictions and charts

---

## 📋 Common Commands

### Basic Usage

```bash
# Train model for Tesla
python main.py --symbol TSLA --train

# Make predictions with existing model
python main.py --symbol GOOGL --predict

# Train and predict in one command
python main.py --symbol MSFT --train --predict
```

### Advanced Options

```bash
# Custom training parameters
python main.py --symbol AAPL --train \
  --epochs 100 \
  --batch-size 64 \
  --lookback 90 \
  --lstm-units 100

# Custom date range
python main.py --symbol NVDA --train \
  --start-date 2019-01-01 \
  --end-date 2024-01-01

# Specify model save location
python main.py --symbol AMZN --train \
  --model-path models/my_custom_model.h5
```

---

## 🎯 Using the Python API

You can also use the modules directly in your Python scripts:

```python
from src.data_fetcher import fetch_stock_data
from src.features import add_technical_indicators
from src.preprocessor import prepare_data
from src.model import create_lstm_model
from src.train import train_model
from src.predict import predict_prices

# Fetch data
df = fetch_stock_data('AAPL', start_date='2020-01-01', end_date='2024-01-01')

# Add indicators
df = add_technical_indicators(df)

# Prepare data
data_dict = prepare_data(df, lookback=60, test_size=0.2)

# Create and train model
model = create_lstm_model(input_shape=(60, 5))
results = train_model(
    X_train=data_dict['X_train'],
    y_train=data_dict['y_train'],
    X_val=data_dict['X_test'],
    y_val=data_dict['y_test'],
    epochs=50
)

# Make predictions
predictions = predict_prices(
    model=results['model'],
    X=data_dict['X_test'],
    scaler=data_dict['scaler'],
    feature_columns=['open', 'high', 'low', 'close', 'volume']
)
```

---

## 📊 Understanding the Output

### Training Output
```
=== LSTM MODEL TRAINING ===
Training samples: 1,234
Validation samples: 308
Test samples: 385

Epoch 1/50
50/50 [==============================] - loss: 0.0045 - mae: 0.0521
...
Model saved to: models/lstm_stock_model.h5
```

### Prediction Metrics
```
Prediction Performance:
  MAE: $2.34          # Mean Absolute Error
  RMSE: $3.12         # Root Mean Squared Error
  MAPE: 1.45%         # Mean Absolute Percentage Error
  Direction_Accuracy: 67.89%  # % of correct direction predictions
```

### Generated Files
```
outputs/
├── actual_vs_predicted.png    # Main prediction comparison
├── training_history.png       # Loss/MAE curves
└── technical_indicators.png   # MA and RSI charts
```

---

## 🔧 Troubleshooting

### Issue: "No module named 'tensorflow'"
**Solution**: 
```bash
pip install -r requirements.txt
```

### Issue: "No data found for ticker symbol"
**Solution**: Check if the stock symbol is correct. Use Yahoo Finance to verify symbols.

### Issue: Model training is slow
**Solutions**:
- Reduce number of epochs: `--epochs 30`
- Reduce lookback period: `--lookback 30`
- Use smaller batch size: `--batch-size 16`

### Issue: Out of memory error
**Solutions**:
- Reduce batch size: `--batch-size 16`
- Reduce LSTM units: `--lstm-units 25`
- Use shorter date range

---

## 📈 Project Structure Reference

```
stock-price-trend-prediction-lstm/
├── src/                      # Source code modules
│   ├── data_fetcher.py      # Fetch stock data from Yahoo Finance
│   ├── preprocessor.py      # Data normalization & sequence creation
│   ├── features.py          # Technical indicators (MA, RSI, MACD)
│   ├── model.py             # LSTM model architecture
│   ├── train.py             # Training pipeline
│   ├── predict.py           # Prediction utilities
│   └── visualization.py     # Plotting functions
├── app/
│   └── dashboard.py         # Streamlit web application
├── notebooks/
│   └── demo.ipynb           # Jupyter notebook tutorial
├── models/                   # Saved models (created after training)
├── outputs/                  # Generated plots (created after training)
├── data/                     # Raw data storage
├── main.py                   # Command-line interface
├── requirements.txt          # Dependencies
└── README.md                # Full documentation
```

---

## 💡 Tips for Better Results

1. **More Data**: Use longer historical data (3-5 years minimum)
2. **Tune Parameters**: Experiment with lookback periods (30-90 days)
3. **Multiple Runs**: Train multiple times and average results
4. **Market Conditions**: Model works better in stable markets
5. **Feature Engineering**: Add more technical indicators or sentiment data

---

## ⚠️ Important Disclaimer

This project is for **EDUCATIONAL PURPOSES ONLY**. 

- ❌ NOT financial advice
- ❌ NOT recommended for real trading
- ❌ NO guarantee of future performance
- ✅ OK for learning machine learning concepts
- ✅ OK for academic research
- ✅ OK for portfolio projects

Stock market investments carry inherent risks. Always consult with a qualified financial advisor before making investment decisions.

---

## 🆘 Need Help?

1. Check the full README.md for detailed documentation
2. Review the Jupyter notebook (notebooks/demo.ipynb) for step-by-step guide
3. Examine individual module docstrings for API reference

Happy Learning! 🚀📊
