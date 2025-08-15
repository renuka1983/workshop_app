# 🚀 Predictive Maintenance Streamlit App - Complete Setup Guide

## 📋 **Prerequisites**
- macOS (tested on macOS 12.6+)
- Python 3.13+ installed
- Terminal/Command Line access

## 🔧 **Step-by-Step Setup**

### **Step 1: Navigate to Project Directory**
```bash
cd /Users/renusapple/Workshop_App
```

### **Step 2: Create Virtual Environment**
```bash
python3 -m venv venv
```

### **Step 3: Activate Virtual Environment**
```bash
source venv/bin/activate
```
**Note**: You should see `(venv)` prefix in your terminal prompt.

### **Step 4: Install Dependencies**
```bash
pip install -r requirements.txt
```
**Expected**: All packages will install successfully with Python 3.13 compatible versions.

### **Step 5: Test the Demo Script**
```bash
python demo.py
```
**Expected Output**: Demo runs successfully showing ML model training and results.

### **Step 6: Start the Streamlit App**
```bash
streamlit run app.py --server.headless true --server.port 8501
```

### **Step 7: Access Your App**
Open your web browser and go to: **http://localhost:8501**

## 🎯 **What You'll See**

### **6 Main Tabs:**
1. **📊 Data Overview** - Generate synthetic datasets, view statistics
2. **🤖 ML Model** - Train Random Forest, view performance metrics
3. **📅 AI Scheduling** - Get maintenance recommendations
4. **📈 Per-Machine Dashboard** - Monitor individual machines
5. **📋 Reports** - Download CSV/JSON reports
6. **🤖 GenAI** - AI-powered insights and text analysis

## 🛠️ **Management Commands**

### **Using the Management Script (Recommended)**
```bash
# Start the app
./manage_venv.sh run-app

# Run demo
./manage_venv.sh run-demo

# Activate venv
source manage_venv.sh activate

# Deactivate venv
deactivate
```

### **Manual Commands**
```bash
# Activate virtual environment
source venv/bin/activate

# Start app
streamlit run app.py --server.headless true --server.port 8501

# Run demo
python demo.py

# Deactivate
deactivate
```

## 🔍 **Troubleshooting**

### **Port Already in Use Error**
```bash
# Find what's using port 8501
lsof -ti:8501

# Kill the process
kill <process_id>

# Or kill all Streamlit processes
pkill -f streamlit
```

### **Virtual Environment Issues**
```bash
# Recreate virtual environment
./manage_venv.sh recreate

# Or manually
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### **Plotly Compatibility Issues**
- ✅ **Fixed**: `px.barh()` → `px.bar(orientation='h')`
- ✅ **Tested**: Works with Plotly 6.2.0+

## 📱 **App Features**

### **Core Functionality**
- ✅ Synthetic dataset generation (500-2000 records)
- ✅ Rule-based baseline maintenance prediction
- ✅ ML model training (Random Forest)
- ✅ AI scheduling with business rules + ML probabilities
- ✅ Per-machine monitoring dashboard
- ✅ Data export (CSV/JSON/PDF)
- ✅ GenAI integration (Hugging Face transformers)

### **Data Science Features**
- ✅ Feature importance analysis
- ✅ Performance metrics (accuracy, cross-validation)
- ✅ Confusion matrix visualization
- ✅ Classification reports
- ✅ Anomaly detection
- ✅ Trend analysis

## 🎉 **Success Indicators**

### **When Everything is Working:**
1. ✅ Virtual environment shows `(venv)` prefix
2. ✅ `python demo.py` runs without errors
3. ✅ Streamlit app starts without port conflicts
4. ✅ Browser shows app at http://localhost:8501
5. ✅ All 6 tabs are accessible
6. ✅ ML model training completes successfully

### **Expected Demo Output:**
```
Dataset Size: 500 records
Rule-based Accuracy: 0.574
ML Model Accuracy: 0.630
Improvement: 9.8%
High Priority Machines: 29
```

## 🚀 **Next Steps**

### **Explore the App:**
1. Generate different dataset sizes
2. Train ML models with various parameters
3. Test the AI scheduling algorithm
4. Explore the GenAI features
5. Download sample reports

### **Customize:**
- Modify `config.py` for different thresholds
- Add new machine types in the data generation
- Extend the ML model with new features
- Customize the business rules in scheduling

## 📞 **Support**

If you encounter issues:
1. Check the troubleshooting section above
2. Verify virtual environment is activated
3. Ensure all dependencies are installed
4. Check for port conflicts
5. Review the error messages for specific issues

---

**🎯 Your Predictive Maintenance App is Ready!**
Open http://localhost:8501 and start exploring the world of AI-powered maintenance prediction! 