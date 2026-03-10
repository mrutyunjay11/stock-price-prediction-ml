#!/bin/bash

# Installation and Testing Script for Stock Price Prediction Project
# This script installs dependencies and performs a basic test

echo "=============================================="
echo "Stock Price Prediction - LSTM"
echo "Installation and Testing Script"
echo "=============================================="
echo ""

# Step 1: Check Python version
echo "Step 1: Checking Python version..."
python --version
if [ $? -ne 0 ]; then
    echo "❌ Error: Python is not installed or not in PATH"
    exit 1
fi
echo "✅ Python found"
echo ""

# Step 2: Create virtual environment (optional but recommended)
read -p "Create a virtual environment? (y/n): " create_venv
if [ "$create_venv" = "y" ]; then
    echo "Step 2: Creating virtual environment..."
    python -m venv venv
    
    echo "Activating virtual environment..."
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    echo "✅ Virtual environment created and activated"
    echo ""
fi

# Step 3: Install dependencies
echo "Step 3: Installing dependencies..."
echo "This may take 5-10 minutes depending on your internet connection..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to install dependencies"
    exit 1
fi
echo "✅ Dependencies installed successfully"
echo ""

# Step 4: Verify installations
echo "Step 4: Verifying installations..."
python -c "import tensorflow; import pandas; import numpy; import streamlit; import yfinance; print('All packages imported successfully!')"
if [ $? -ne 0 ]; then
    echo "❌ Error: Package verification failed"
    exit 1
fi
echo "✅ All packages verified"
echo ""

# Step 5: Run a quick test (optional)
read -p "Run a quick test with Apple stock data? (y/n): " run_test
if [ "$run_test" = "y" ]; then
    echo "Step 5: Running test..."
    echo "Fetching data and training model (this will take a few minutes)..."
    
    # Create directories if they don't exist
    mkdir -p models outputs data
    
    # Run a minimal training
    python main.py --symbol AAPL --train --predict --epochs 10 --lookback 30
    
    if [ $? -eq 0 ]; then
        echo "✅ Test completed successfully!"
        echo ""
        echo "Generated files:"
        ls -lh models/*.h5 2>/dev/null || echo "  No model file found"
        ls -lh outputs/*.png 2>/dev/null || echo "  No output plots found"
    else
        echo "❌ Test failed. Check the error messages above."
        echo "You can still use the project by running main.py manually."
    fi
    echo ""
fi

# Step 6: Instructions for Streamlit dashboard
echo "Step 6: Testing Streamlit Dashboard (Optional)"
echo ""
echo "To launch the interactive web dashboard, run:"
echo "  streamlit run app/dashboard.py"
echo ""
echo "The dashboard will open automatically in your browser at:"
echo "  http://localhost:8501"
echo ""

# Final summary
echo "=============================================="
echo "Installation Complete! ✅"
echo "=============================================="
echo ""
echo "Quick Start Guide:"
echo "------------------"
echo "1. Train a model:"
echo "   python main.py --symbol AAPL --train --predict"
echo ""
echo "2. Launch dashboard:"
echo "   streamlit run app/dashboard.py"
echo ""
echo "3. View documentation:"
echo "   Open README.md and QUICKSTART.md"
echo ""
echo "4. Try the Jupyter notebook:"
echo "   Open notebooks/demo.ipynb"
echo ""
echo "=============================================="
echo "Project Structure:"
echo "==================="
find . -type f \( -name "*.py" -o -name "*.md" -o -name "*.ipynb" \) | head -20
echo ""
echo "=============================================="
echo ""
echo "⚠️  DISCLAIMER: This project is for EDUCATIONAL PURPOSES ONLY."
echo "    NOT financial advice. Do not use for real trading decisions."
echo ""
echo "Happy Learning! 🚀📊"
echo "=============================================="
