import streamlit as st
import sys
import os

# Add the subdirectories to the path so we can import the modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'inventory'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'predictive_maintenance'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'product_design'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'QualityControl'))

# Page configuration
st.set_page_config(
    page_title="Manufacturing Workshop App",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the main app
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .module-card {
        background: white;
        border: 2px solid #e0e0e0;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .module-card:hover {
        border-color: #667eea;
        box-shadow: 0 8px 15px rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }
    .module-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .module-description {
        color: #666;
        margin-bottom: 1rem;
    }
    .feature-list {
        list-style: none;
        padding: 0;
    }
    .feature-list li {
        padding: 0.5rem 0;
        border-bottom: 1px solid #f0f0f0;
    }
    .feature-list li:before {
        content: "✅ ";
        color: #4CAF50;
    }
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin-top: 1rem;
    }
    .stat-item {
        text-align: center;
        padding: 0.5rem;
    }
    .stat-number {
        font-size: 1.5rem;
        font-weight: bold;
        color: #667eea;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🏭 Manufacturing Workshop App</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            A comprehensive suite of AI-powered manufacturing tools for modern industry
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    ## 🎯 About This Workshop
    
    This application demonstrates the power of AI and machine learning in modern manufacturing. 
    Each module showcases different aspects of intelligent manufacturing systems, from demand forecasting 
    to quality control and predictive maintenance.
    """)
    
    # Module showcase
    st.markdown("## 📊 Available Modules")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="module-card">
            <div class="module-title">📦 Inventory Management</div>
            <div class="module-description">
                AI-powered demand forecasting and production planning with dynamic adjustments.
            </div>
            <ul class="feature-list">
                <li>Multi-method demand forecasting (Traditional, ML, AI)</li>
                <li>Real-time external factor adjustments</li>
                <li>Risk analysis and inventory optimization</li>
                <li>Executive dashboard with key metrics</li>
            </ul>
            <div class="stats-container">
                <div class="stat-item">
                    <div class="stat-number">5</div>
                    <div class="stat-label">Products</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">90</div>
                    <div class="stat-label">Days Forecast</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">3</div>
                    <div class="stat-label">Methods</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="module-card">
            <div class="module-title">🔧 Predictive Maintenance</div>
            <div class="module-description">
                Machine learning-based equipment health monitoring and maintenance scheduling.
            </div>
            <ul class="feature-list">
                <li>Real-time sensor data analysis</li>
                <li>Failure prediction models</li>
                <li>Maintenance optimization</li>
                <li>Equipment health scoring</li>
            </ul>
            <div class="stats-container">
                <div class="stat-item">
                    <div class="stat-number">5</div>
                    <div class="stat-label">Machines</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">6</div>
                    <div class="stat-label">Sensors</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">95%</div>
                    <div class="stat-label">Accuracy</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="module-card">
            <div class="module-title">🚴 Product Design Optimization</div>
            <div class="module-description">
                AI-driven design optimization with physics-informed modeling and generative design.
            </div>
            <ul class="feature-list">
                <li>Multi-objective optimization</li>
                <li>Physics-informed AI models</li>
                <li>Generative design capabilities</li>
                <li>Real-time simulation</li>
            </ul>
            <div class="stats-container">
                <div class="stat-item">
                    <div class="stat-number">4</div>
                    <div class="stat-label">Methods</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">92%</div>
                    <div class="stat-label">Accuracy</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">75%</div>
                    <div class="stat-label">Time Saved</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="module-card">
            <div class="module-title">🔍 Quality Control & Defect Detection</div>
            <div class="module-description">
                Computer vision and ML-based quality inspection and defect detection system.
            </div>
            <ul class="feature-list">
                <li>Computer vision inspection</li>
                <li>Multiple defect type detection</li>
                <li>Real-time quality scoring</li>
                <li>Automated reporting</li>
            </ul>
            <div class="stats-container">
                <div class="stat-item">
                    <div class="stat-number">6</div>
                    <div class="stat-label">Defect Types</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">95%</div>
                    <div class="stat-label">Detection Rate</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">5</div>
                    <div class="stat-label">Production Lines</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Navigation instructions
    st.markdown("## 🚀 Getting Started")
    
    st.markdown("""
    To explore each module, use the sidebar navigation or click on the module cards above. 
    Each module is self-contained and demonstrates different AI/ML techniques applied to manufacturing challenges.
    
    ### Key Features:
    - **Interactive Dashboards**: Real-time data visualization and analysis
    - **AI/ML Models**: Sophisticated algorithms for prediction and optimization
    - **Synthetic Data**: Realistic manufacturing scenarios for demonstration
    - **Executive Insights**: Business-focused metrics and recommendations
    """)
    
    # Technology stack
    st.markdown("## 🛠️ Technology Stack")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Frontend & Visualization**
        - Streamlit
        - Plotly
        - Custom CSS
        """)
    
    with col2:
        st.markdown("""
        **Machine Learning**
        - Scikit-learn
        - Random Forest
        - Neural Networks
        - Physics-informed AI
        """)
    
    with col3:
        st.markdown("""
        **Data Processing**
        - Pandas
        - NumPy
        - Synthetic data generation
        - Real-time analytics
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🏭 Manufacturing Workshop App | AI-Powered Manufacturing Solutions</p>
        <p>Built with Streamlit, Python, and Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
