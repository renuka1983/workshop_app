# 🔧 Predictive Maintenance Streamlit App

A comprehensive predictive maintenance dashboard built with Streamlit, featuring machine learning models, AI scheduling heuristics, and GenAI integration via Hugging Face transformers.

## 🚀 Features

### Core Functionality
- **Synthetic Dataset Generation**: Realistic maintenance data with temperature, vibration, pressure, humidity, and runtime metrics
- **Rule-based Baseline**: Traditional threshold-based maintenance prediction
- **ML Model Training**: Random Forest classifier with performance metrics and feature importance
- **AI Scheduling Heuristic**: Business rules + ML probabilities for maintenance scheduling
- **Per-Machine Dashboard**: Individual machine monitoring with time series visualization
- **Data Export**: CSV, JSON, and PDF report generation

### GenAI Integration (Optional)
- **Sentiment Analysis**: Analyze maintenance log sentiment using RoBERTa
- **Text Classification**: Categorize maintenance issues automatically
- **AI-Powered Insights**: Anomaly detection, trend analysis, and risk assessment
- **Natural Language Summaries**: Human-readable insights generation
- **Batch Log Analysis**: Process multiple maintenance logs simultaneously

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd Workshop_App
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the application**:
```bash
streamlit run app.py
```

## 📊 Usage Guide

### 1. Data Generation
- Use the sidebar to configure the number of samples (500-2000)
- Click "Generate New Dataset" to create synthetic maintenance data
- Data includes 5 machines with realistic failure patterns

### 2. Data Overview
- **Dataset Preview**: View first 10 records
- **Statistical Summary**: Descriptive statistics for all features
- **Maintenance Distribution**: Bar chart showing alerts by machine
- **Feature Distributions**: Histograms for all sensor readings

### 3. Machine Learning Model
- Click "Train ML Model" to train Random Forest classifier
- View model accuracy and cross-validation scores
- Analyze feature importance rankings
- Examine confusion matrix and classification report

### 4. AI Scheduling
- Combines business rules (critical machines, maintenance history) with ML probabilities
- Generates priority scores and maintenance schedules
- Visualizes urgency distribution and priority score histograms

### 5. Per-Machine Dashboard
- Select individual machines for detailed monitoring
- View machine-specific metrics and alert rates
- Analyze time series data for temperature, vibration, and pressure

### 6. Reports & Export
- Download raw data as CSV
- Export comprehensive JSON reports
- Generate PDF reports (placeholder for reportlab/fpdf2 integration)

### 7. GenAI Features
- **Load Models**: Initialize Hugging Face transformers
- **Text Analysis**: Analyze individual maintenance logs
- **Batch Analysis**: Process multiple logs simultaneously
- **AI Insights**: Generate automated insights and recommendations

## 🏗️ Architecture

### Data Structure
```python
{
    'machine_id': str,           # Machine identifier (A-E)
    'timestamp': datetime,        # Hourly timestamps
    'temperature': float,         # Temperature in °C
    'vibration': float,          # Vibration amplitude
    'pressure': float,           # Pressure in PSI
    'humidity': float,           # Humidity percentage
    'runtime_hours': float,      # Cumulative runtime
    'maintenance_history': int,  # Previous maintenance count
    'maintenance_needed': int    # Target variable (0/1)
}
```

### ML Pipeline
1. **Feature Engineering**: 6 sensor features + maintenance history
2. **Data Preprocessing**: StandardScaler normalization
3. **Model Training**: Random Forest with 100 estimators
4. **Evaluation**: Accuracy, CV scores, confusion matrix
5. **Feature Importance**: Ranked feature contributions

### AI Scheduling Algorithm
```python
priority_score = (
    business_rules_weight * business_factors +
    ml_probability_weight * ml_prediction
)

# Business factors:
# - Critical machine: +0.4
# - High maintenance history: +0.3
# - ML probability: +0.3
```

## 🔧 Configuration

### Environment Variables
- No external API keys required
- Models downloaded automatically from Hugging Face
- GPU acceleration if available (CUDA)

### Customization
- Modify `machines` list in `PredictiveMaintenanceApp.__init__()`
- Adjust feature generation parameters in `generate_synthetic_data()`
- Customize business rules in `ai_scheduling_heuristic()`
- Add new ML models in `train_ml_model()`

## 📈 Performance

### Dataset Sizes
- **Small**: 500 samples (~2MB)
- **Medium**: 1000 samples (~4MB)
- **Large**: 2000 samples (~8MB)

### Training Times
- **Random Forest**: 2-5 seconds (100 estimators)
- **GenAI Models**: 10-30 seconds (first load)
- **Inference**: <1 second

## 🚨 Troubleshooting

### Common Issues
1. **GenAI models fail to load**: Check internet connection and Hugging Face access
2. **Memory errors**: Reduce dataset size or use smaller ML models
3. **Slow performance**: Consider using GPU acceleration for GenAI features

### Dependencies
- Python 3.8+
- Streamlit 1.28+
- PyTorch 2.1+
- Transformers 4.35+

## 🔮 Future Enhancements

- **Real-time Data Integration**: Connect to IoT sensors and databases
- **Advanced ML Models**: LSTM, Transformer models for time series
- **Predictive Analytics**: Failure prediction timelines
- **Cost Optimization**: Maintenance cost vs. failure cost analysis
- **Mobile App**: React Native companion app
- **API Endpoints**: RESTful API for external integrations

## 📚 References

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Predictive Maintenance Best Practices](https://www.mckinsey.com/business-functions/operations/our-insights/predictive-maintenance-3-ways-to-get-value-from-your-investment)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement improvements
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built with ❤️ using Streamlit, Scikit-learn, and Hugging Face Transformers** 