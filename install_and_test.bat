@echo off
REM Installation and Testing Script for Stock Price Prediction Project (Windows)
REM This script installs dependencies and performs a basic test

echo ==============================================
echo Stock Price Prediction - LSTM
echo Installation and Testing Script (Windows)
echo ==============================================
echo.

REM Step 1: Check Python version
echo Step 1: Checking Python version...
python --version
if errorlevel 1 (
    echo X Error: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)
echo [OK] Python found
echo.

REM Step 2: Create virtual environment (optional but recommended)
set /p create_venv="Create a virtual environment? (y/n): "
if "%create_venv%"=="y" (
    echo Step 2: Creating virtual environment...
    python -m venv venv
    
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
    echo [OK] Virtual environment created and activated
    echo.
)

REM Step 3: Install dependencies
echo Step 3: Installing dependencies...
echo This may take 5-10 minutes depending on your internet connection...
pip install -r requirements.txt

if errorlevel 1 (
    echo X Error: Failed to install dependencies
    pause
    exit /b 1
)
echo [OK] Dependencies installed successfully
echo.

REM Step 4: Verify installations
echo Step 4: Verifying installations...
python -c "import tensorflow; import pandas; import numpy; import streamlit; import yfinance; print('All packages imported successfully!')"
if errorlevel 1 (
    echo X Error: Package verification failed
    pause
    exit /b 1
)
echo [OK] All packages verified
echo.

REM Step 5: Run a quick test (optional)
set /p run_test="Run a quick test with Apple stock data? (y/n): "
if "%run_test%"=="y" (
    echo Step 5: Running test...
    echo Fetching data and training model (this will take a few minutes)...
    
    REM Create directories if they don't exist
    if not exist models mkdir models
    if not exist outputs mkdir outputs
    if not exist data mkdir data
    
    REM Run a minimal training
    python main.py --symbol AAPL --train --predict --epochs 10 --lookback 30
    
    if errorlevel 0 (
        echo [OK] Test completed successfully!
        echo.
        echo Generated files:
        dir models\*.h5 /b 2>nul || echo   No model file found
        dir outputs\*.png /b 2>nul || echo   No output plots found
    ) else (
        echo X Test failed. Check the error messages above.
        echo You can still use the project by running main.py manually.
    )
    echo.
)

REM Step 6: Instructions for Streamlit dashboard
echo Step 6: Testing Streamlit Dashboard (Optional)
echo.
echo To launch the interactive web dashboard, run:
echo   streamlit run app\dashboard.py
echo.
echo The dashboard will open automatically in your browser at:
echo   http://localhost:8501
echo.

REM Final summary
echo ==============================================
echo Installation Complete! [OK]
echo ==============================================
echo.
echo Quick Start Guide:
echo ------------------
echo 1. Train a model:
echo    python main.py --symbol AAPL --train --predict
echo.
echo 2. Launch dashboard:
echo    streamlit run app\dashboard.py
echo.
echo 3. View documentation:
echo    Open README.md and QUICKSTART.md
echo.
echo 4. Try the Jupyter notebook:
echo    Open notebooks\demo.ipynb
echo.
echo ==============================================
echo Project Structure:
echo ===================
dir /s /b *.py *.md *.ipynb 2>nul | findstr /v ".git"
echo.
echo ==============================================
echo.
echo WARNING: This project is for EDUCATIONAL PURPOSES ONLY.
echo NOT financial advice. Do not use for real trading decisions.
echo.
echo Happy Learning!
echo ==============================================
pause
