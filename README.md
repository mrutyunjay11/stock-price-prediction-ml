# Stock Price Trend Prediction using LSTM

A complete machine learning project for predicting stock price trends using Long Short-Term Memory (LSTM) neural networks.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Technical Indicators](#technical-indicators)
- [Examples](#examples)
- [Dependencies](#dependencies)

## 🎯 Project Overview

This project implements a deep learning solution for stock price prediction using LSTM networks. It fetches real-time stock data from Yahoo Finance, preprocesses it, builds an LSTM model, and provides an interactive Streamlit dashboard for visualization.

## ✨ Features

- **Automatic Data Fetching**: Download historical stock data using yfinance API
- **Data Preprocessing**: 
  - Handle missing values
  - Feature scaling using MinMaxScaler
  - Time-series sequence creation
- **Technical Indicators**:
  - Moving Average (MA)
  - Relative Strength Index (RSI)
- **LSTM Neural Network**: Build and train deep learning models for time-series prediction
- **Interactive Dashboard**: Streamlit web application for easy interaction
- **Visualization**: Plot actual vs predicted stock prices
- **Model Saving**: Save and load trained models

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step-by-Step Installation

1. **Clone or navigate to the project directory:**
```bash
cd stock-price-trend-prediction-lstm
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Verify installation:**
```bash
python -c "import tensorflow; import pandas; import streamlit; print('All dependencies installed!')"
```

## 📁 Project Structure

```
stock-price-trend-prediction-lstm/
├── data/                 # Raw and processed data storage
├── models/               # Saved model files
├── notebooks/            # Jupyter notebooks for experimentation
├── src/                  # Source code modules
│   ├── data_fetcher.py   # Stock data fetching from yfinance
│   ├── preprocessor.py   # Data preprocessing utilities
│   ├── features.py       # Technical indicator calculations
│   ├── model.py          # LSTM model architecture
│   ├── train.py          # Model training script
│   ├── predict.py        # Prediction utilities
│   └── visualization.py  # Plotting and visualization
├── app/                  # Streamlit application
│   └── dashboard.py      # Main Streamlit dashboard
├── outputs/              # Generated plots and results
├── main.py               # Main execution script
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🚀 Usage

### Quick Start - Run the Complete Pipeline

1. **Train a new model:**
```bash
python main.py --symbol AAPL --train
```

2. **Run predictions:**
```bash
python main.py --symbol AAPL --predict
```

3. **Launch the Streamlit dashboard:**
```bash
streamlit run app/dashboard.py
```

### Using Individual Modules

#### 1. Fetch Stock Data
```python
from src.data_fetcher import fetch_stock_data

# Download Apple stock data
df = fetch_stock_data('AAPL', start_date='2020-01-01', end_date='2024-01-01')
print(df.head())
```

#### 2. Preprocess Data
```python
from src.preprocessor import prepare_data

# Prepare data for LSTM
X, y, scaler = prepare_data(df, lookback=60)
```

#### 3. Add Technical Indicators
```python
from src.features import add_technical_indicators

# Add MA and RSI
df_with_indicators = add_technical_indicators(df)
```

#### 4. Build and Train Model
```python
from src.model import create_lstm_model
from src.train import train_model

# Create model
model = create_lstm_model(input_shape=(60, 5), lstm_units=50)

# Train model
history = train_model(model, X_train, y_train, X_val, y_val, epochs=50)
```

#### 5. Make Predictions
```python
from src.predict import predict_prices

# Predict future prices
predictions = predict_prices(model, X_test, scaler)
```

### Streamlit Dashboard

The dashboard provides an intuitive interface where you can:

1. **Enter any stock symbol** (e.g., AAPL, GOOGL, TSLA)
2. **View historical price data**
3. **See model predictions**
4. **Analyze interactive charts** showing:
   - Actual vs Predicted prices
   - Technical indicators (MA, RSI)
   - Training history

To access the dashboard, open your browser and navigate to the URL shown in the terminal after running:
```bash
streamlit run app/dashboard.py
```

## 🧠 Model Architecture

The LSTM model consists of:

- **Input Layer**: Accepts sequences of 60 days of stock data
- **LSTM Layers**: 
  - First LSTM layer with 50 units (returns sequences)
  - Second LSTM layer with 50 units
- **Dense Layer**: Fully connected layer for output
- **Output Layer**: Single neuron for price prediction

### Model Parameters

- **Lookback Period**: 60 days
- **LSTM Units**: 50
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)
- **Batch Size**: 32
- **Epochs**: 50 (customizable)

## 📊 Technical Indicators

### Moving Average (MA)
Calculates the average price over a specified period (default: 20 days) to smooth out short-term fluctuations and highlight longer-term trends.

### Relative Strength Index (RSI)
Momentum oscillator that measures the speed and magnitude of recent price changes to evaluate overbought or oversold conditions (range: 0-100).

- **RSI > 70**: Overbought (potential sell signal)
- **RSI < 30**: Oversold (potential buy signal)

## 📈 Example Output

The project generates:

1. **Price Prediction Plots**: Side-by-side comparison of actual and predicted stock prices
2. **Training History**: Loss curves showing model convergence
3. **Technical Indicator Charts**: MA and RSI visualizations

## 📝 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >=1.24.0 | Numerical computations |
| pandas | >=2.0.0 | Data manipulation |
| matplotlib | >=3.7.0 | Plotting |
| seaborn | >=0.12.0 | Enhanced visualizations |
| tensorflow | >=2.13.0 | Deep learning framework |
| scikit-learn | >=1.3.0 | Data preprocessing |
| yfinance | >=0.2.28 | Stock data API |
| streamlit | >=1.28.0 | Web dashboard |
| joblib | >=1.3.0 | Model serialization |

## ⚠️ Disclaimer

This project is for **educational purposes only**. Stock market predictions are inherently uncertain. Do not use this model for actual trading decisions without thorough testing and understanding of the risks involved.

## 📄 License

This project is open-source and available for educational purposes.

## 🤝 Contributing

Feel free to improve the model by:
- Adding more technical indicators
- Implementing different neural network architectures
- Enhancing the Streamlit dashboard
- Adding backtesting capabilities

## 📧 Contact

For questions or suggestions, please open an issue in the repository.

---

**Happy Predicting! 🚀📈**
