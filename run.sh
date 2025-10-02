#!/bin/bash
# Quick start script for Unix/Linux/MacOS

echo "Starting Spatial Optimization Conversational Agent..."
echo

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Installing dependencies..."
    pip install -r requirements.txt
    python setup.py
else
    source venv/bin/activate
fi

echo
echo "Starting Streamlit application..."
streamlit run app.py

