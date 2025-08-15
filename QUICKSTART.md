# 🚀 Quick Start Guide

## Get Started in 3 Steps

### 1. Install Dependencies
```bash
pip3 install -r requirements.txt
```

### 2. Run the App
```bash
# Option A: Use the startup script
./start_app.sh

# Option B: Run directly
streamlit run app.py
```

### 3. Open Your Browser
Navigate to: **http://localhost:8501**

## 🎯 What You'll See

1. **Generate Data**: Click "Generate New Dataset" in the sidebar
2. **Train ML Model**: Go to "🤖 ML Model" tab and click "Train ML Model"
3. **View Scheduling**: Check "📅 AI Scheduling" tab for maintenance recommendations
4. **Explore Dashboards**: Use "📈 Per-Machine Dashboard" for individual machine views
5. **Export Reports**: Download data from "📋 Reports" tab
6. **Try GenAI**: Load AI models in "🤖 GenAI" tab for advanced features

## 🔧 Demo Mode

Want to test without the full UI? Run:
```bash
python3 demo.py
```

This will:
- Generate 500 synthetic records
- Train ML model
- Show performance comparison
- Save sample data and reports

## 📊 Sample Output

The demo generates:
- **Dataset**: 500 maintenance records
- **ML Accuracy**: ~63% (vs 57% rule-based)
- **Scheduling**: Priority-based maintenance recommendations
- **Files**: `sample_maintenance_data.csv` and `maintenance_report.json`

## 🚨 Troubleshooting

- **Python Version**: Ensure you're using Python 3.8+
- **Dependencies**: Run `pip3 install -r requirements.txt`
- **Port Issues**: Change port in `start_app.sh` if 8501 is busy
- **GenAI Models**: First load may take 10-30 seconds

## 📱 Features Overview

| Feature | Description | Status |
|---------|-------------|---------|
| Synthetic Data | Generate realistic maintenance datasets | ✅ Ready |
| ML Training | Random Forest with performance metrics | ✅ Ready |
| AI Scheduling | Business rules + ML probabilities | ✅ Ready |
| Per-Machine Views | Individual machine monitoring | ✅ Ready |
| Data Export | CSV, JSON, PDF reports | ✅ Ready |
| GenAI Integration | Hugging Face transformers | ✅ Ready |

---

**Ready to start? Run `./start_app.sh` and open your browser!** 🎉 