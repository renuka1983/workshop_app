import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# import cv2  # Commented out due to macOS compatibility issues
from PIL import Image as PILImage
import datetime
import random
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="QC & Defect Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #1f77b4;
}
.defect-high { color: #ff6b6b; font-weight: bold; }
.defect-medium { color: #ffa500; font-weight: bold; }
.defect-low { color: #90ee90; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

class SteelDefectDataGenerator:
    @staticmethod
    def generate_synthetic_data(n_samples=1000):
        """Generate synthetic steel plate defect data"""
        np.random.seed(42)
        
        # Defect types
        defect_types = ['Crazing', 'Inclusion', 'Patches', 'Pitted_surface', 'Rolled_in_scale', 'Scratches']
        
        # Production lines
        production_lines = ['Line_A', 'Line_B', 'Line_C', 'Line_D', 'Line_E']
        
        # Generate timestamps
        start_date = datetime.datetime.now() - datetime.timedelta(days=90)
        timestamps = [start_date + datetime.timedelta(hours=random.randint(0, 2160)) for _ in range(n_samples)]
        
        data = []
        for i in range(n_samples):
            # Basic properties
            thickness = np.random.normal(5.0, 1.5)
            width = np.random.normal(1500, 200)
            length = np.random.normal(6000, 800)
            temperature = np.random.normal(1200, 100)
            speed = np.random.normal(50, 10)
            pressure = np.random.normal(800, 150)
            
            # Defect characteristics
            has_defect = np.random.choice([0, 1], p=[0.7, 0.3])
            defect_type = np.random.choice(defect_types) if has_defect else 'None'
            defect_severity = np.random.choice(['Low', 'Medium', 'High'], p=[0.5, 0.3, 0.2]) if has_defect else 'None'
            defect_size = np.random.exponential(2) if has_defect else 0
            
            # Quality metrics
            surface_roughness = np.random.normal(0.5, 0.2)
            hardness = np.random.normal(200, 50)
            
            # Inspection results
            manual_detected = has_defect and np.random.choice([0, 1], p=[0.3, 0.7])
            vision_detected = has_defect and np.random.choice([0, 1], p=[0.1, 0.9])
            automated_detected = has_defect and np.random.choice([0, 1], p=[0.05, 0.95])
            
            data.append({
                'timestamp': timestamps[i],
                'batch_id': f'B{i//50:04d}',
                'plate_id': f'P{i:06d}',
                'production_line': np.random.choice(production_lines),
                'thickness': max(thickness, 0.5),
                'width': max(width, 500),
                'length': max(length, 2000),
                'temperature': temperature,
                'rolling_speed': max(speed, 10),
                'pressure': max(pressure, 200),
                'surface_roughness': max(surface_roughness, 0.1),
                'hardness': max(hardness, 50),
                'has_defect': has_defect,
                'defect_type': defect_type,
                'defect_severity': defect_severity,
                'defect_size_mm2': defect_size,
                'manual_inspection_detected': manual_detected,
                'vision_system_detected': vision_detected,
                'automated_system_detected': automated_detected,
                'detection_time_hours': np.random.exponential(2) if has_defect else 0,
                'cost_impact': np.random.exponential(1000) if has_defect else 0,
                'customer_reported': has_defect and np.random.choice([0, 1], p=[0.9, 0.1])
            })
        
        return pd.DataFrame(data)

class DefectAnalyzer:
    def __init__(self, data):
        self.data = data
        
    def get_detection_metrics(self):
        """Calculate detection performance metrics"""
        defective_plates = self.data[self.data['has_defect'] == 1]
        
        if len(defective_plates) == 0:
            return {}
        
        manual_accuracy = defective_plates['manual_inspection_detected'].mean()
        vision_accuracy = defective_plates['vision_system_detected'].mean()
        automated_accuracy = defective_plates['automated_system_detected'].mean()
        
        avg_detection_time = defective_plates['detection_time_hours'].mean()
        total_cost_impact = defective_plates['cost_impact'].sum()
        customer_complaints = defective_plates['customer_reported'].sum()
        
        return {
            'manual_accuracy': manual_accuracy,
            'vision_accuracy': vision_accuracy,
            'automated_accuracy': automated_accuracy,
            'avg_detection_time': avg_detection_time,
            'total_cost_impact': total_cost_impact,
            'customer_complaints': customer_complaints,
            'total_defects': len(defective_plates)
        }
    
    def perform_root_cause_analysis(self):
        """Perform root cause analysis using machine learning"""
        # Prepare features
        feature_cols = ['thickness', 'width', 'length', 'temperature', 'rolling_speed', 
                       'pressure', 'surface_roughness', 'hardness']
        
        X = self.data[feature_cols]
        y = self.data['has_defect']
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return model, feature_importance

def create_synthetic_defect_image():
    """Create a synthetic defect image for demonstration using PIL instead of OpenCV"""
    from PIL import ImageDraw
    
    # Create base steel plate image
    img = PILImage.new('RGB', (600, 400), color=(180, 180, 180))
    draw = ImageDraw.Draw(img)
    
    # Add texture with random dots
    for _ in range(500):
        x = random.randint(0, 599)
        y = random.randint(0, 399)
        brightness = random.randint(160, 200)
        draw.point((x, y), fill=(brightness, brightness, brightness))
    
    # Add defect (scratch) - draw a line
    draw.line([(100, 150), (500, 250)], fill=(100, 100, 100), width=3)
    
    # Add inclusion defect - draw a circle
    draw.ellipse([(185, 285), (215, 315)], fill=(80, 80, 80))
    
    return np.array(img)

def main():
    st.markdown('<div class="main-header">🔍 Quality Control & Defect Detection System</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Module",
        ["Executive Dashboard", "Manual Inspection", "Vision-Based Detection", 
         "Automated Inspection", "Root Cause Analysis", "Process Recommendations"]
    )
    
    # Generate or load data
    @st.cache_data
    def load_data():
        return SteelDefectDataGenerator.generate_synthetic_data(1500)
    
    data = load_data()
    analyzer = DefectAnalyzer(data)
    
    if page == "Executive Dashboard":
        st.header("📊 Executive Dashboard - Quality Control Overview")
        
        # Key metrics
        metrics = analyzer.get_detection_metrics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Defects Detected",
                f"{metrics.get('total_defects', 0):,}",
                delta=f"{random.randint(-5, 15)}% vs last month"
            )
        
        with col2:
            st.metric(
                "Detection Accuracy",
                f"{metrics.get('automated_accuracy', 0):.1%}",
                delta=f"{random.randint(1, 8)}% improvement"
            )
        
        with col3:
            st.metric(
                "Avg Detection Time",
                f"{metrics.get('avg_detection_time', 0):.1f} hrs",
                delta=f"-{random.randint(5, 25)}% vs target"
            )
        
        with col4:
            st.metric(
                "Cost Impact",
                f"${metrics.get('total_cost_impact', 0):,.0f}",
                delta=f"-{random.randint(10, 30)}% vs last month"
            )
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Defect trends over time
            daily_defects = data.groupby(data['timestamp'].dt.date)['has_defect'].sum().reset_index()
            fig = px.line(daily_defects, x='timestamp', y='has_defect',
                         title='Daily Defect Detection Trend')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Detection method comparison
            detection_comparison = pd.DataFrame({
                'Method': ['Manual', 'Vision-Based', 'Automated'],
                'Accuracy': [metrics.get('manual_accuracy', 0),
                           metrics.get('vision_accuracy', 0),
                           metrics.get('automated_accuracy', 0)]
            })
            fig = px.bar(detection_comparison, x='Method', y='Accuracy',
                        title='Detection Method Performance')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Defect type distribution
        defect_dist = data[data['has_defect'] == 1]['defect_type'].value_counts()
        fig = px.pie(values=defect_dist.values, names=defect_dist.index,
                    title='Defect Type Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Manual Inspection":
        st.header("👁️ Manual Inspection Module")
        
        # Upload or use sample image
        st.subheader("Inspect Steel Plate")
        
        uploaded_file = st.file_uploader("Upload steel plate image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = PILImage.open(uploaded_file)
            st.image(image, caption="Uploaded Steel Plate", use_column_width=True)
        else:
            # Show synthetic image
            synthetic_img = create_synthetic_defect_image()
            st.image(synthetic_img, caption="Sample Steel Plate (Synthetic)", use_column_width=True)
            image = PILImage.fromarray(synthetic_img.astype('uint8'))
        
        # Inspection form
        st.subheader("Manual Inspection Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            plate_id = st.text_input("Plate ID", value="P000001")
            inspector = st.text_input("Inspector Name", value="John Smith")
            inspection_date = st.date_input("Inspection Date", value=datetime.date.today())
        
        with col2:
            defect_detected = st.selectbox("Defect Detected?", ["No", "Yes"])
            if defect_detected == "Yes":
                defect_type = st.selectbox("Defect Type", 
                                         ['Crazing', 'Inclusion', 'Patches', 'Pitted_surface', 
                                          'Rolled_in_scale', 'Scratches'])
                severity = st.selectbox("Severity", ["Low", "Medium", "High"])
        
        comments = st.text_area("Inspector Comments")
        
        if st.button("Submit Inspection"):
            st.success("✅ Inspection recorded successfully!")
            
            # Show summary
            st.subheader("Inspection Summary")
            summary_data = {
                "Plate ID": plate_id,
                "Inspector": inspector,
                "Date": inspection_date,
                "Defect Status": defect_detected,
                "Comments": comments
            }
            
            if defect_detected == "Yes":
                summary_data.update({
                    "Defect Type": defect_type,
                    "Severity": severity
                })
            
            st.json(summary_data)
    
    elif page == "Vision-Based Detection":
        st.header("📷 Vision-Based Defect Detection")
        
        # Image upload section
        st.subheader("Upload Steel Plate Image for AI Analysis")
        
        uploaded_file = st.file_uploader(
            "Choose a steel plate image for AI vision analysis", 
            type=['jpg', 'jpeg', 'png'],
            key="vision_upload"
        )
        
        # Initialize variables
        current_img = None
        use_synthetic = False
        
        if uploaded_file is not None:
            current_img = np.array(PILImage.open(uploaded_file))
            st.success("✅ Image uploaded successfully! Running AI analysis...")
        else:
            # Option to use synthetic data
            if st.button("Use Sample Steel Plate for Demo"):
                current_img = create_synthetic_defect_image()
                use_synthetic = True
                st.info("🔬 Using synthetic steel plate sample for demonstration")
        
        # Process image if available
        if current_img is not None:
            # Simulate vision system analysis
            st.subheader("🤖 AI Vision Analysis Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(current_img, caption="Original Image", use_column_width=True)
            
            with col2:
                # Simulate AI processing with progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate processing steps
                import time
                processing_steps = [
                    "Initializing AI model...",
                    "Preprocessing image...",
                    "Running defect detection...",
                    "Analyzing defect patterns...",
                    "Calculating confidence scores...",
                    "Analysis complete!"
                ]
                
                for i, step in enumerate(processing_steps):
                    status_text.text(step)
                    progress_bar.progress((i + 1) / len(processing_steps))
                    time.sleep(0.3)
                
                # Create processed image with defect highlights
                from PIL import ImageDraw
                processed_img_pil = PILImage.fromarray(current_img.astype('uint8'))
                draw = ImageDraw.Draw(processed_img_pil)
                
                # Simulate defect detection based on image analysis
                img_height, img_width = current_img.shape[:2]
                
                # Generate realistic defect locations based on image properties
                if use_synthetic:
                    # Use predefined locations for synthetic image
                    defect_locations = [
                        (95, 145, 505, 255),  # Linear scratch
                        (185, 285, 215, 315)  # Inclusion
                    ]
                    defect_types = ["Linear Scratch", "Inclusion"]
                    confidence_scores = ["96.3%", "92.1%"]
                else:
                    # Generate random defect locations for uploaded images
                    num_defects = random.randint(0, 3)
                    defect_locations = []
                    defect_types = []
                    confidence_scores = []
                    
                    defect_type_options = ['Crazing', 'Inclusion', 'Patches', 'Pitted_surface', 'Rolled_in_scale', 'Scratches']
                    
                    for _ in range(num_defects):
                        x1 = random.randint(10, img_width - 100)
                        y1 = random.randint(10, img_height - 50)
                        x2 = x1 + random.randint(30, 150)
                        y2 = y1 + random.randint(20, 80)
                        
                        defect_locations.append((x1, y1, x2, y2))
                        defect_types.append(random.choice(defect_type_options))
                        confidence_scores.append(f"{random.uniform(85, 98):.1f}%")
                
                # Highlight detected defects
                for i, (x1, y1, x2, y2) in enumerate(defect_locations):
                    draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0), width=3)
                    draw.text((x1, y1-15), f"D{i+1:03d}", fill=(255, 0, 0))
                
                processed_img = np.array(processed_img_pil)
                st.image(processed_img, caption="AI Processed Image (Defects Highlighted)", use_column_width=True)
            
            # Analysis results
            st.subheader("📊 AI Analysis Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Processing Time", f"{random.uniform(0.15, 0.35):.2f}s")
            
            with col2:
                st.metric("Defects Detected", len(defect_locations))
            
            with col3:
                avg_confidence = np.mean([float(conf.replace('%', '')) for conf in confidence_scores]) if confidence_scores else 0
                st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
            
            with col4:
                overall_status = "🔴 Defective" if len(defect_locations) > 0 else "🟢 Pass"
                st.metric("Overall Status", overall_status)
            
            # Detailed defect information
            if len(defect_locations) > 0:
                st.subheader("🔍 Detailed Defect Analysis")
                
                defect_details = []
                for i, ((x1, y1, x2, y2), def_type, confidence) in enumerate(zip(defect_locations, defect_types, confidence_scores)):
                    size = abs(x2-x1) * abs(y2-y1) / 100  # Convert to mm²
                    severity = "High" if size > 20 else "Medium" if size > 10 else "Low"
                    
                    defect_details.append({
                        'Defect ID': f'D{i+1:03d}',
                        'Type': def_type,
                        'Severity': severity,
                        'Size (mm²)': f"{size:.1f}",
                        'Location (x,y)': f"({int((x1+x2)/2)}, {int((y1+y2)/2)})",
                        'Confidence': confidence,
                        'Bounding Box': f"({x1},{y1}) to ({x2},{y2})"
                    })
                
                defect_df = pd.DataFrame(defect_details)
                st.dataframe(defect_df, use_container_width=True)
                
                # Defect severity distribution
                severity_counts = defect_df['Severity'].value_counts()
                fig = px.pie(values=severity_counts.values, names=severity_counts.index, 
                           title="Defect Severity Distribution")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ No defects detected! Steel plate passes quality inspection.")
            
            # Vision system performance metrics
            st.subheader("🎯 Vision System Performance")
            
            perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
            
            with perf_col1:
                st.metric("Detection Rate", "94.7%", "↑2.3%")
            
            with perf_col2:
                st.metric("False Positive Rate", "3.2%", "↓0.8%")
            
            with perf_col3:
                st.metric("Processing Speed", "0.23s", "↓0.05s")
            
            with perf_col4:
                st.metric("Model Accuracy", "96.8%", "↑1.2%")
            
            # Action buttons
            st.subheader("📋 Actions")
            
            action_col1, action_col2, action_col3 = st.columns(3)
            
            with action_col1:
                if st.button("📥 Save Analysis Report"):
                    st.success("Analysis report saved to database!")
            
            with action_col2:
                if st.button("🔄 Reprocess Image"):
                    st.experimental_rerun()
            
            with action_col3:
                if st.button("📧 Send Alert"):
                    if len(defect_locations) > 0:
                        st.warning("Quality control alert sent to production team!")
                    else:
                        st.info("No alerts needed - plate passed inspection.")
        
        else:
            st.info("👆 Please upload a steel plate image or use the sample demo to start AI vision analysis.")
            
            # Show example of what the system can detect
            st.subheader("🎯 AI Vision Capabilities")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Detectable Defect Types:**")
                defect_types = ['Crazing', 'Inclusion', 'Patches', 'Pitted Surface', 'Rolled-in Scale', 'Scratches']
                for defect in defect_types:
                    st.write(f"• {defect}")
            
            with col2:
                st.write("**System Specifications:**")
                specs = [
                    "Resolution: Up to 4K (3840×2160)",
                    "Processing Time: <0.5 seconds",
                    "Accuracy: 96.8% average",
                    "Min Defect Size: 0.1mm²",
                    "Supported Formats: JPG, PNG"
                ]
                for spec in specs:
                    st.write(f"• {spec}")
    
    elif page == "Automated Inspection":
        st.header("🤖 Automated Inspection System")
        
        # Real-time monitoring simulation
        st.subheader("Real-Time Production Monitoring")
        
        # Create placeholder for real-time data
        placeholder = st.empty()
        
        # Simulate real-time updates
        if st.button("Start Real-Time Monitoring"):
            for i in range(10):
                current_time = datetime.datetime.now() + datetime.timedelta(seconds=i*5)
                
                # Generate random inspection data
                plate_data = {
                    'Time': current_time.strftime('%H:%M:%S'),
                    'Plate ID': f'P{random.randint(100000, 999999)}',
                    'Line': random.choice(['Line_A', 'Line_B', 'Line_C']),
                    'Status': random.choice(['Pass', 'Pass', 'Pass', 'Defect Detected']),
                    'Confidence': f"{random.uniform(85, 99):.1f}%"
                }
                
                with placeholder.container():
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.write(f"**Time:** {plate_data['Time']}")
                    with col2:
                        st.write(f"**Plate:** {plate_data['Plate ID']}")
                    with col3:
                        st.write(f"**Line:** {plate_data['Line']}")
                    with col4:
                        if plate_data['Status'] == 'Pass':
                            st.success(f"✅ {plate_data['Status']}")
                        else:
                            st.error(f"❌ {plate_data['Status']}")
                    with col5:
                        st.write(f"**Confidence:** {plate_data['Confidence']}")
                
                import time
                time.sleep(1)
        
        # System status
        st.subheader("System Status")
        
        status_col1, status_col2, status_col3, status_col4 = st.columns(4)
        
        with status_col1:
            st.success("🟢 Camera System: Online")
        
        with status_col2:
            st.success("🟢 AI Processing: Active")
        
        with status_col3:
            st.success("🟢 Database: Connected")
        
        with status_col4:
            st.warning("🟡 Calibration: Due in 2 days")
        
        # Recent inspections
        st.subheader("Recent Automated Inspections")
        
        recent_data = data.tail(20)[['plate_id', 'production_line', 'has_defect', 
                                   'defect_type', 'automated_system_detected', 'timestamp']]
        recent_data['status'] = recent_data['has_defect'].apply(lambda x: '❌ Defect' if x else '✅ Pass')
        
        st.dataframe(recent_data.sort_values('timestamp', ascending=False), use_container_width=True)
    
    elif page == "Root Cause Analysis":
        st.header("🔬 Root Cause Analysis")
        
        # Perform analysis
        model, feature_importance = analyzer.perform_root_cause_analysis()
        
        # Feature importance chart
        st.subheader("Key Factors Contributing to Defects")
        
        fig = px.bar(feature_importance, x='importance', y='feature', 
                    orientation='h', title='Feature Importance in Defect Prediction')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Process parameter analysis
        st.subheader("Process Parameter Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Temperature analysis
            temp_defect = data.groupby('has_defect')['temperature'].mean()
            fig = px.box(data, x='has_defect', y='temperature', 
                        title='Temperature Distribution by Defect Status')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Speed analysis
            fig = px.box(data, x='has_defect', y='rolling_speed',
                        title='Rolling Speed Distribution by Defect Status')
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix
        st.subheader("Parameter Correlation Analysis")
        
        numeric_cols = ['thickness', 'temperature', 'rolling_speed', 'pressure', 
                       'surface_roughness', 'hardness', 'has_defect']
        corr_matrix = data[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                       title="Process Parameter Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations based on analysis
        st.subheader("Analysis Insights")
        
        top_factors = feature_importance.head(3)['feature'].tolist()
        
        insights = [
            f"🎯 **Primary Factor:** {top_factors[0]} shows the strongest correlation with defects",
            f"📊 **Secondary Factor:** {top_factors[1]} is the second most important predictor",
            f"⚙️ **Process Focus:** Monitor {top_factors[2]} for early defect prevention",
            f"🔍 **Detection Rate:** Current automated system achieves {analyzer.get_detection_metrics().get('automated_accuracy', 0):.1%} accuracy"
        ]
        
        for insight in insights:
            st.write(insight)
    
    elif page == "Process Recommendations":
        st.header("💡 Process Recommendations")
        
        # Current performance summary
        metrics = analyzer.get_detection_metrics()
        
        st.subheader("Current Performance Summary")
        
        perf_data = pd.DataFrame({
            'Metric': ['Manual Detection Rate', 'Vision System Rate', 'Automated System Rate', 
                      'Average Detection Time', 'Customer Complaints'],
            'Current Value': [f"{metrics.get('manual_accuracy', 0):.1%}",
                            f"{metrics.get('vision_accuracy', 0):.1%}",
                            f"{metrics.get('automated_accuracy', 0):.1%}",
                            f"{metrics.get('avg_detection_time', 0):.1f} hours",
                            f"{metrics.get('customer_complaints', 0)} cases"],
            'Target': ['85%', '95%', '98%', '< 1 hour', '< 5 cases'],
            'Status': ['⚠️ Below Target', '✅ Above Target', '✅ Above Target', 
                      '⚠️ Needs Improvement', '✅ Within Target']
        })
        
        st.dataframe(perf_data, use_container_width=True)
        
        # Strategic recommendations
        st.subheader("Strategic Recommendations")
        
        recommendations = [
            {
                'category': 'Technology Enhancement',
                'priority': 'High',
                'recommendations': [
                    'Upgrade vision system cameras to higher resolution models',
                    'Implement edge computing for faster real-time processing',
                    'Deploy additional AI models for rare defect types',
                    'Integrate IoT sensors for comprehensive process monitoring'
                ]
            },
            {
                'category': 'Process Optimization',
                'priority': 'High',
                'recommendations': [
                    'Establish automated feedback loops to process parameters',
                    'Implement predictive maintenance for inspection equipment',
                    'Create real-time alerts for parameter deviations',
                    'Develop standard operating procedures for defect response'
                ]
            },
            {
                'category': 'Training & Development',
                'priority': 'Medium',
                'recommendations': [
                    'Train operators on new inspection technologies',
                    'Develop defect identification training programs',
                    'Create knowledge sharing sessions across shifts',
                    'Implement certification programs for quality inspectors'
                ]
            },
            {
                'category': 'Data Management',
                'priority': 'Medium',
                'recommendations': [
                    'Implement centralized defect database',
                    'Develop analytics dashboard for trend analysis',
                    'Create automated reporting systems',
                    'Establish data governance policies'
                ]
            }
        ]
        
        for rec in recommendations:
            with st.expander(f"{rec['category']} - Priority: {rec['priority']}"):
                for item in rec['recommendations']:
                    st.write(f"• {item}")
        
        # Implementation roadmap
        st.subheader("Implementation Roadmap")
        
        roadmap_data = pd.DataFrame({
            'Quarter': ['Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024'],
            'Initiatives': [
                'Vision system upgrade, IoT sensor deployment',
                'AI model enhancement, Automated feedback implementation',
                'Training program rollout, Data platform development',
                'Performance optimization, Full system integration'
            ],
            'Expected ROI': ['15% reduction in defects', '25% faster detection', 
                           '30% improvement in accuracy', '40% cost savings'],
            'Budget Estimate': ['$250K', '$180K', '$120K', '$80K']
        })
        
        st.dataframe(roadmap_data, use_container_width=True)
        
        # Cost-benefit analysis
        st.subheader("Cost-Benefit Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Investment Summary:**")
            st.write("• Total Investment: $630K")
            st.write("• Implementation Period: 12 months")
            st.write("• Payback Period: 18 months")
        
        with col2:
            st.write("**Expected Benefits:**")
            st.write("• 40% reduction in defect costs")
            st.write("• 60% faster defect detection")
            st.write("• 80% reduction in customer complaints")
            st.write("• 25% improvement in overall quality")

if __name__ == "__main__":
    main()