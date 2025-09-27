#!/bin/bash

echo "Setting up AgriAssist Chatbot..."

# Create virtual environment
python -m venv agrienv
source agrienv/bin/activate  # On Windows: agrienv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Run the application
echo "Starting AgriAssist Chatbot..."
streamlit run app.py