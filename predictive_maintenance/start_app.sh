#!/bin/bash

echo "🚀 Starting Predictive Maintenance Streamlit App..."
echo "📊 Features:"
echo "  - Synthetic dataset generation"
echo "  - ML model training (RandomForest)"
echo "  - AI scheduling heuristics"
echo "  - GenAI integration (Hugging Face)"
echo "  - Per-machine dashboards"
echo "  - Data export (CSV/JSON/PDF)"
echo ""

# Check if Python 3 is available
if command -v python3 &> /dev/null; then
    echo "✅ Python 3 found: $(python3 --version)"
else
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

# Check if requirements are installed
echo "🔍 Checking dependencies..."
python3 -c "import streamlit, pandas, numpy, sklearn, plotly" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ All dependencies are installed"
else
    echo "⚠️  Installing dependencies..."
    pip3 install -r requirements.txt
fi

echo ""
echo "🌐 Starting Streamlit app..."
echo "📱 Open your browser and go to: http://localhost:8501"
echo "⏹️  Press Ctrl+C to stop the app"
echo ""

# Start the app
streamlit run app.py --server.port 8501 