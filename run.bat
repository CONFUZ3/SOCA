@echo off
REM Quick start script for Windows

echo Starting Spatial Optimization Conversational Agent...
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate
    echo Installing dependencies...
    pip install -r requirements.txt
    python setup.py
) else (
    call venv\Scripts\activate
)

echo.
echo Starting Streamlit application...
streamlit run app.py

