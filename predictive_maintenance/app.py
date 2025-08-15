import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class PredictiveMaintenanceApp:
    def __init__(self):
        self.data = None
        self.model = None
        self.scaler = StandardScaler()
        self.machines = ['Machine_A', 'Machine_B', 'Machine_C', 'Machine_D', 'Machine_E']
        
    def generate_synthetic_data(self, num_samples=1000):
        """Generate synthetic predictive maintenance dataset"""
        np.random.seed(42)
        
        # Generate features
        temperature = np.random.normal(65, 15, num_samples)
        vibration = np.random.normal(0.5, 0.3, num_samples)
        pressure = np.random.normal(100, 20, num_samples)
        humidity = np.random.normal(45, 10, num_samples)
        runtime_hours = np.random.uniform(0, 8760, num_samples)  # 1 year in hours
        maintenance_history = np.random.poisson(2, num_samples)
        
        # Generate target variable (maintenance needed)
        # Complex rule-based logic for realistic failure patterns
        maintenance_needed = np.zeros(num_samples, dtype=int)
        
        for i in range(num_samples):
            # High temperature + high vibration + high runtime = high failure probability
            temp_factor = 1 if temperature[i] > 75 else 0
            vib_factor = 1 if vibration[i] > 0.8 else 0
            runtime_factor = 1 if runtime_hours[i] > 6000 else 0
            pressure_factor = 1 if pressure[i] > 120 or pressure[i] < 80 else 0
            
            # Combined risk score
            risk_score = temp_factor + vib_factor + runtime_factor + pressure_factor
            
            # Probability of maintenance needed
            prob = min(0.95, 0.1 + (risk_score * 0.2) + (maintenance_history[i] * 0.1))
            maintenance_needed[i] = np.random.binomial(1, prob)
        
        # Create DataFrame
        data = pd.DataFrame({
            'machine_id': np.random.choice(self.machines, num_samples),
            'timestamp': pd.date_range(start='2023-01-01', periods=num_samples, freq='h'),
            'temperature': temperature,
            'vibration': vibration,
            'pressure': pressure,
            'humidity': humidity,
            'runtime_hours': runtime_hours,
            'maintenance_history': maintenance_history,
            'maintenance_needed': maintenance_needed
        })
        
        # Add some noise and realistic patterns
        data['temperature'] = data['temperature'] + np.random.normal(0, 2, num_samples)
        data['vibration'] = np.abs(data['vibration'])  # Vibration should be positive
        
        return data
    
    def rule_based_baseline(self, data):
        """Traditional rule-based approach for maintenance prediction"""
        predictions = []
        
        for _, row in data.iterrows():
            # Simple rule-based logic
            if (row['temperature'] > 80 or 
                row['vibration'] > 1.0 or 
                row['pressure'] > 130 or 
                row['pressure'] < 70 or
                row['runtime_hours'] > 7000):
                predictions.append(1)
            else:
                predictions.append(0)
        
        return np.array(predictions)
    
    def train_ml_model(self, data):
        """Train Random Forest model"""
        # Prepare features
        feature_cols = ['temperature', 'vibration', 'pressure', 'humidity', 
                       'runtime_hours', 'maintenance_history']
        X = data[feature_cols]
        y = data['maintenance_needed']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        
        return {
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'accuracy': accuracy,
            'cv_scores': cv_scores,
            'feature_importance': dict(zip(feature_cols, self.model.feature_importances_))
        }
    
    def ai_scheduling_heuristic(self, data, ml_probabilities):
        """AI scheduling heuristic combining business rules and ML probabilities"""
        schedule_recommendations = []
        
        for i, (_, row) in enumerate(data.iterrows()):
            # Business rules
            critical_machine = row['machine_id'] in ['Machine_A', 'Machine_B']  # Critical machines
            high_priority = row['maintenance_history'] > 3  # High maintenance history
            
            # ML probability threshold
            ml_prob = ml_probabilities[i] if i < len(ml_probabilities) else 0.5
            
            # Combined priority score
            priority_score = 0
            
            # Business rule weights
            if critical_machine:
                priority_score += 0.4
            if high_priority:
                priority_score += 0.3
            
            # ML probability weight
            priority_score += ml_prob * 0.3
            
            # Determine maintenance schedule
            if priority_score > 0.7:
                schedule = "Immediate (Next 24h)"
                urgency = "High"
            elif priority_score > 0.5:
                schedule = "Within 72h"
                urgency = "Medium"
            elif priority_score > 0.3:
                schedule = "Within 1 week"
                urgency = "Low"
            else:
                schedule = "No maintenance needed"
                urgency = "None"
            
            schedule_recommendations.append({
                'machine_id': row['machine_id'],
                'priority_score': priority_score,
                'schedule': schedule,
                'urgency': urgency,
                'ml_probability': ml_prob,
                'business_factors': {
                    'critical_machine': critical_machine,
                    'high_priority': high_priority
                }
            })
        
        return pd.DataFrame(schedule_recommendations)
    
    def generate_report(self, data, ml_results, schedule_df):
        """Generate comprehensive maintenance report"""
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            return obj
        
        # Convert schedule_df to records with proper type conversion
        schedule_records = []
        for _, row in schedule_df.iterrows():
            record = {}
            for col in schedule_df.columns:
                value = row[col]
                if hasattr(value, 'item'):  # numpy type
                    record[col] = value.item()
                else:
                    record[col] = value
            schedule_records.append(record)
        
        report_data = {
            'summary': {
                'total_machines': len(self.machines),
                'total_records': len(data),
                'maintenance_alerts': int(data['maintenance_needed'].sum()),
                'ml_accuracy': float(ml_results['accuracy']),
                'avg_priority_score': float(schedule_df['priority_score'].mean())
            },
            'machine_breakdown': convert_numpy_types(data.groupby('machine_id')['maintenance_needed'].sum().to_dict()),
            'schedule_recommendations': schedule_records,
            'generated_at': datetime.now().isoformat()
        }
        return report_data

def main():
    st.markdown('<h1 class="main-header">🔧 Predictive Maintenance Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Initialize app
    app = PredictiveMaintenanceApp()
    
    # Sidebar
    st.sidebar.header("Configuration")
    num_samples = st.sidebar.slider("Number of samples", 500, 2000, 1000)
    
    # Generate data
    if st.sidebar.button("Generate New Dataset") or app.data is None:
        with st.spinner("Generating synthetic dataset..."):
            app.data = app.generate_synthetic_data(num_samples)
        st.success(f"Generated {num_samples} records!")
    
    if app.data is not None:
        # Main content
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", len(app.data))
        with col2:
            st.metric("Maintenance Alerts", app.data['maintenance_needed'].sum())
        with col3:
            st.metric("Alert Rate", f"{(app.data['maintenance_needed'].sum() / len(app.data) * 100):.1f}%")
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Data Overview", "🤖 ML Model", "📅 AI Scheduling", "📈 Per-Machine Dashboard", "📋 Reports", "🤖 GenAI"
        ])
        
        with tab1:
            st.header("Data Overview")
            
            # Data preview
            st.subheader("Dataset Preview")
            # Convert timestamp to string for display to avoid PyArrow issues
            display_data = app.data.head(10).copy()
            display_data['timestamp'] = display_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            st.dataframe(display_data)
            
            # Statistical summary
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Statistical Summary")
                st.dataframe(app.data.describe())
            
            with col2:
                st.subheader("Maintenance Distribution by Machine")
                fig = px.bar(
                    app.data.groupby('machine_id')['maintenance_needed'].sum().reset_index(),
                    x='machine_id',
                    y='maintenance_needed',
                    title="Maintenance Alerts by Machine"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature distributions
            st.subheader("Feature Distributions")
            feature_cols = ['temperature', 'vibration', 'pressure', 'humidity', 'runtime_hours']
            
            fig = make_subplots(rows=2, cols=3, subplot_titles=feature_cols)
            
            for i, col in enumerate(feature_cols):
                row = (i // 3) + 1
                col_num = (i % 3) + 1
                
                fig.add_trace(
                    go.Histogram(x=app.data[col], name=col),
                    row=row, col=col_num
                )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.header("Machine Learning Model")
            
            if st.button("Train ML Model"):
                with st.spinner("Training Random Forest model..."):
                    ml_results = app.train_ml_model(app.data)
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Model Accuracy", f"{ml_results['accuracy']:.3f}")
                        st.metric("CV Score (std)", f"{ml_results['cv_scores'].mean():.3f} (±{ml_results['cv_scores'].std():.3f})")
                    
                    with col2:
                        st.subheader("Feature Importance")
                        importance_df = pd.DataFrame(
                            list(ml_results['feature_importance'].items()),
                            columns=['Feature', 'Importance']
                        ).sort_values('Importance', ascending=True)
                        
                        fig = px.bar(
                            importance_df,
                            x='Importance',
                            y='Feature',
                            title="Feature Importance",
                            orientation='h'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Confusion Matrix
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(ml_results['y_test'], ml_results['y_pred'])
                    
                    fig = px.imshow(
                        cm,
                        text_auto=True,
                        aspect="auto",
                        labels=dict(x="Predicted", y="Actual"),
                        title="Confusion Matrix"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Classification Report
                    st.subheader("Classification Report")
                    report = classification_report(ml_results['y_test'], ml_results['y_pred'], output_dict=True)
                    st.json(report)
                    
                    # Store results for scheduling
                    st.session_state.ml_results = ml_results
        
        with tab3:
            st.header("AI Scheduling Heuristic")
            
            if 'ml_results' in st.session_state:
                st.info("ML model results available for scheduling!")
                
                # Generate scheduling recommendations
                schedule_df = app.ai_scheduling_heuristic(
                    app.data, 
                    st.session_state.ml_results['y_pred_proba']
                )
                
                st.subheader("Maintenance Schedule Recommendations")
                # Convert any timestamp columns to strings for display
                display_schedule = schedule_df.copy()
                for col in display_schedule.columns:
                    if display_schedule[col].dtype == 'object' and 'timestamp' in col.lower():
                        display_schedule[col] = display_schedule[col].astype(str)
                st.dataframe(display_schedule)
                
                # Priority distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.pie(
                        schedule_df,
                        names='urgency',
                        title="Maintenance Urgency Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.histogram(
                        schedule_df,
                        x='priority_score',
                        title="Priority Score Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Store for reports
                st.session_state.schedule_df = schedule_df
                
            else:
                st.warning("Please train the ML model first to generate scheduling recommendations.")
        
        with tab4:
            st.header("Per-Machine Dashboard")
            
            # Machine selector
            selected_machine = st.selectbox("Select Machine", app.machines)
            
            if selected_machine:
                machine_data = app.data[app.data['machine_id'] == selected_machine]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Records", len(machine_data))
                    st.metric("Maintenance Alerts", machine_data['maintenance_needed'].sum())
                    st.metric("Alert Rate", f"{(machine_data['maintenance_needed'].sum() / len(machine_data) * 100):.1f}%")
                
                with col2:
                    st.metric("Avg Temperature", f"{machine_data['temperature'].mean():.1f}°C")
                    st.metric("Avg Vibration", f"{machine_data['vibration'].mean():.3f}")
                    st.metric("Avg Pressure", f"{machine_data['pressure'].mean():.1f} PSI")
                
                # Time series plots
                st.subheader(f"Time Series Data - {selected_machine}")
                
                fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=['Temperature', 'Vibration', 'Pressure'],
                    shared_xaxes=True
                )
                
                # Convert timestamp to string for plotly compatibility
                timestamp_str = machine_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                fig.add_trace(
                    go.Scatter(x=timestamp_str, y=machine_data['temperature'], name='Temperature'),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=timestamp_str, y=machine_data['vibration'], name='Vibration'),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Scatter(x=timestamp_str, y=machine_data['pressure'], name='Pressure'),
                    row=3, col=1
                )
                
                fig.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab5:
            st.header("Reports & Export")
            
            if 'schedule_df' in st.session_state:
                # Generate comprehensive report
                report_data = app.generate_report(
                    app.data, 
                    st.session_state.ml_results, 
                    st.session_state.schedule_df
                )
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # CSV Export
                    # Convert timestamp to string for CSV export
                    export_data = app.data.copy()
                    export_data['timestamp'] = export_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    csv_data = export_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name=f"maintenance_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # JSON Export
                    json_data = json.dumps(report_data, indent=2)
                    st.download_button(
                        label="📥 Download JSON Report",
                        data=json_data,
                        file_name=f"maintenance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
                with col3:
                    # PDF Report (simplified)
                    if st.button("📄 Generate PDF Report"):
                        st.info("PDF generation would be implemented here with reportlab/fpdf2")
                
                # Report Summary
                st.subheader("Report Summary")
                st.json(report_data['summary'])
                
            else:
                st.warning("Please complete ML training and scheduling to generate reports.")
        
        with tab6:
            st.header("🤖 GenAI Integration")
            
            # Import and use GenAI module
            try:
                from genai_integration import create_genai_tab
                create_genai_tab()
            except ImportError:
                st.warning("GenAI module not available. Please ensure genai_integration.py is in the same directory.")
                st.info("GenAI features include: sentiment analysis, text classification, and AI-powered insights.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "🔧 Predictive Maintenance Dashboard | Built with Streamlit & ML"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 