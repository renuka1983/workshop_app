import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import random

# Page configuration
st.set_page_config(
    page_title="Product Design Optimization",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main > div {
    padding-top: 2rem;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}
.method-card {
    border: 2px solid #e0e0e0;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
    transition: all 0.3s;
}
.method-card:hover {
    border-color: #4CAF50;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'selected_method' not in st.session_state:
    st.session_state.selected_method = None
if 'current_design_params' not in st.session_state:
    st.session_state.current_design_params = None

# Method definitions
METHODS = {
    'traditional': {
        'name': 'Traditional CAD',
        'icon': '🛠️',
        'description': 'Manual CAD design with standard material tables and physical prototyping',
        'expected_weight': 18.0,
        'expected_time': 6.0,
        'cost': 'High',
        'cost_value': 100,
        'accuracy': 70,
        'prototypes': 3,
        'color': '#757575'
    },
    'ml': {
        'name': 'Machine Learning',
        'icon': '📊',
        'description': 'Regression models predict frame properties with optimization',
        'expected_weight': 16.5,
        'expected_time': 4.0,
        'cost': 'Medium',
        'cost_value': 65,
        'accuracy': 82,
        'prototypes': 2,
        'color': '#2196F3'
    },
    'ai': {
        'name': 'Physics-Informed AI',
        'icon': '🧠',
        'description': 'Real-time stress simulation with multi-objective optimization',
        'expected_weight': 15.0,
        'expected_time': 2.0,
        'cost': 'Low',
        'cost_value': 35,
        'accuracy': 92,
        'prototypes': 1,
        'color': '#9C27B0'
    },
    'genai': {
        'name': 'Generative AI',
        'icon': '✨',
        'description': 'Natural language input generates 100+ optimized designs automatically',
        'expected_weight': 13.8,
        'expected_time': 0.75,
        'cost': 'Very Low',
        'cost_value': 20,
        'accuracy': 96,
        'prototypes': 1,
        'color': '#4CAF50'
    }
}

def simulate_optimization(method_key, design_params, progress_bar, status_text):
    """Simulate the optimization process with realistic timing"""
    method = METHODS[method_key]
    
    # Simulation steps based on method complexity
    steps = {
        'traditional': 20,
        'ml': 16,
        'ai': 12,
        'genai': 8
    }
    
    step_names = {
        'traditional': [
            f'Loading CAD software for {design_params["production_scale"].lower()} production',
            f'Creating base geometry (target: {design_params["target_weight"]}kg)',
            f'Applying materials for {design_params["durability_years"]}-year durability',
            f'Running stress analysis for {design_params["aerodynamic_level"].lower()} aerodynamics',
            'Manual adjustments based on requirements', 'Creating prototype 1',
            'Testing prototype 1 against specifications', 'Design modifications for compliance',
            'Creating prototype 2', 'Testing prototype 2 durability',
            'Final weight optimization', 'Creating final prototype',
            'Final testing validation', 'Documentation preparation',
            'Manufacturing prep for production', 'Quality assurance check',
            'Cost analysis review', 'Final design review', 'Approval process', 'Complete'
        ],
        'ml': [
            f'Loading training data for {design_params["target_weight"]}kg targets',
            f'Preprocessing features for {design_params["aerodynamic_level"]} performance',
            f'Training models with {design_params["durability_years"]}-year constraints',
            'Cross-validation with production requirements', 'Parameter tuning optimization',
            'Generating design candidates', 'ML model predictions',
            'Filtering designs by specifications', 'Optimization loop iteration',
            'Validation against requirements', 'Prototype creation',
            'Physical testing verification', 'Results analysis',
            'Model refinement', 'Final optimization', 'Complete'
        ],
        'ai': [
            f'Initializing AI models for {design_params["target_weight"]}kg target',
            f'Physics simulation setup ({design_params["aerodynamic_level"]})',
            f'Multi-objective optimization ({design_params["durability_years"]}-year life)',
            'Stress analysis with AI prediction', 'Vibration modeling integration',
            'Fatigue simulation for durability', 'AI design generation',
            'FEA validation with physics constraints', 'Aerodynamic analysis optimization',
            f'Material optimization for {design_params["production_scale"]}', 'Manufacturing constraints check', 'Complete'
        ],
        'genai': [
            f'Processing requirements: {design_params["target_weight"]}kg, {design_params["durability_years"]}yr, {design_params["aerodynamic_level"]}',
            f'Generating design variants for {design_params["production_scale"]} production',
            'AI evaluation against specifications', 'Structural analysis validation',
            'Manufacturing feasibility filter', 'Cost optimization check',
            'Final selection based on requirements', 'Complete'
        ]
    }
    
    total_steps = steps[method_key]
    step_list = step_names[method_key]
    
    for i in range(total_steps):
        # Update progress
        progress = (i + 1) / total_steps
        progress_bar.progress(progress)
        
        # Update status
        if i < len(step_list):
            status_text.text(f"Step {i+1}/{total_steps}: {step_list[i]}")
        
        # Simulate processing time (faster for more advanced methods)
        delay = {
            'traditional': 0.3,
            'ml': 0.25,
            'ai': 0.2,
            'genai': 0.15
        }
        time.sleep(delay[method_key])
    
    # Generate realistic results based on design parameters
    base_weight = method['expected_weight']
    
    # Adjust weight based on target weight and requirements
    target_weight = design_params['target_weight']
    weight_adjustment = 0
    
    # More aggressive weight reduction if target is very low
    if target_weight < 14:
        weight_adjustment = (14 - target_weight) * 0.3
    
    # Aerodynamic requirements affect weight
    aero_weight_impact = {
        'Standard': 0,
        'High Performance': 0.2,
        'Race Level': 0.5
    }
    weight_adjustment += aero_weight_impact.get(design_params['aerodynamic_level'], 0)
    
    # Durability requirements affect weight
    if design_params['durability_years'] > 7:
        weight_adjustment += 0.3
    elif design_params['durability_years'] < 4:
        weight_adjustment -= 0.2
    
    # Production scale affects optimization capability
    production_multiplier = {
        'Prototype': 0.9,  # Can be more aggressive
        'Small Batch': 1.0,
        'Mass Production': 1.1  # More constraints
    }
    
    variation = (random.random() - 0.5) * 0.4  # ±0.2 kg variation
    adjusted_weight = base_weight - weight_adjustment + variation
    final_weight = max(adjusted_weight * production_multiplier[design_params['production_scale']], 
                      target_weight - 0.5)
    
    # Adjust other metrics based on design parameters
    time_adjustment = 1.0
    
    # Complex requirements increase time
    if design_params['aerodynamic_level'] == 'Race Level':
        time_adjustment *= 1.2
    elif design_params['aerodynamic_level'] == 'High Performance':
        time_adjustment *= 1.1
    
    # Higher durability requirements increase time
    if design_params['durability_years'] > 7:
        time_adjustment *= 1.15
    
    # Production scale affects development time
    scale_time_impact = {
        'Prototype': 0.9,
        'Small Batch': 1.0,  
        'Mass Production': 1.2
    }
    time_adjustment *= scale_time_impact[design_params['production_scale']]
    
    adjusted_time = method['expected_time'] * time_adjustment
    
    designs_generated = {
        'traditional': random.randint(2, 4),
        'ml': random.randint(6, 12),
        'ai': random.randint(12, 20),
        'genai': random.randint(95, 150)
    }
    
    stress_tests = {
        'traditional': random.randint(30, 60),
        'ml': random.randint(80, 150),
        'ai': random.randint(300, 600),
        'genai': random.randint(800, 1500)
    }
    
    # Increase testing based on durability requirements
    durability_multiplier = 1 + (design_params['durability_years'] - 5) * 0.1
    stress_tests[method_key] = int(stress_tests[method_key] * durability_multiplier)
    
    # Aerodynamic requirements increase design iterations
    if design_params['aerodynamic_level'] in ['High Performance', 'Race Level']:
        designs_generated[method_key] = int(designs_generated[method_key] * 1.3)
    
    return {
        'method': method_key,
        'method_name': method['name'],
        'final_weight': round(final_weight, 1),
        'weight_reduction': round((18 - final_weight) / 18 * 100, 1),
        'time_months': round(adjusted_time, 1),
        'cost': method['cost'],
        'cost_reduction': round((100 - method['cost_value']) / 100 * 100, 1),
        'accuracy': method['accuracy'],
        'prototypes': method['prototypes'],
        'designs_generated': designs_generated[method_key],
        'stress_tests': stress_tests[method_key],
        'material_efficiency': min(95, 65 + (4 - list(METHODS.keys()).index(method_key)) * 8),
        'structural_integrity': random.randint(88, 98),
        'aerodynamics': random.randint(82, 95),
        'manufacturability': random.randint(85, 96),
        # Add design parameters to results for reference
        'design_params': design_params
    }

def create_comparison_chart():
    """Create a comparison chart of all methods"""
    methods_data = []
    for key, method in METHODS.items():
        methods_data.append({
            'Method': method['name'],
            'Weight (kg)': method['expected_weight'],
            'Time (months)': method['expected_time'],
            'Cost Value': method['cost_value'],
            'Accuracy (%)': method['accuracy'],
            'Prototypes': method['prototypes']
        })
    
    df = pd.DataFrame(methods_data)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Weight Comparison', 'Time Comparison', 'Cost Comparison', 'Accuracy Comparison'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors = ['#757575', '#2196F3', '#9C27B0', '#4CAF50']
    
    # Weight comparison
    fig.add_trace(
        go.Bar(x=df['Method'], y=df['Weight (kg)'], name='Weight', marker_color=colors),
        row=1, col=1
    )
    
    # Time comparison
    fig.add_trace(
        go.Bar(x=df['Method'], y=df['Time (months)'], name='Time', marker_color=colors),
        row=1, col=2
    )
    
    # Cost comparison
    fig.add_trace(
        go.Bar(x=df['Method'], y=df['Cost Value'], name='Cost', marker_color=colors),
        row=2, col=1
    )
    
    # Accuracy comparison
    fig.add_trace(
        go.Bar(x=df['Method'], y=df['Accuracy (%)'], name='Accuracy', marker_color=colors),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, title_text="Method Comparison Dashboard")
    fig.update_xaxes(tickangle=45)
    
    return fig

def create_performance_radar(results):
    """Create a radar chart for performance metrics"""
    categories = ['Structural Integrity', 'Aerodynamics', 'Manufacturability', 
                 'Material Efficiency', 'Cost Effectiveness', 'Time Efficiency']
    
    values = [
        results['structural_integrity'],
        results['aerodynamics'], 
        results['manufacturability'],
        results['material_efficiency'],
        100 - METHODS[results['method']]['cost_value'],
        100 - (results['time_months'] / 6 * 100)
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=results['method_name'],
        line_color=METHODS[results['method']]['color']
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Performance Analysis"
    )
    
    return fig

# Main App
def main():
    # Header
    st.markdown("""
    # 🚴 Product Design Optimization
    ## Lightweight & Durable Electric Bike Frame Design
    
    This application demonstrates the evolution from traditional CAD methods to generative AI 
    for optimizing bike frame design across weight, durability, cost, and time parameters.
    """)
    
    # Sidebar for design parameters
    st.sidebar.header("🎯 Design Requirements")
    
    design_params = {
        'target_weight': st.sidebar.slider("Target Weight (kg)", 10.0, 20.0, 14.0, 0.5),
        'durability_years': st.sidebar.slider("Durability (years)", 3, 10, 5),
        'aerodynamic_level': st.sidebar.selectbox("Aerodynamic Level", 
                                                 ["Standard", "High Performance", "Race Level"]),
        'production_scale': st.sidebar.selectbox("Production Scale", 
                                                ["Prototype", "Small Batch", "Mass Production"])
    }
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Quick Start Guide:")
    st.sidebar.markdown("""
    1. **Set Requirements** above
    2. **Choose Method** below  
    3. **Click 'Run Optimization'**
    4. **Review Results**
    5. **Try Different Methods**
    """)
    
    # Method selection
    st.header("🛠️ Optimization Methods")
    st.markdown(f"**Current Settings:** Target: {design_params['target_weight']}kg | Durability: {design_params['durability_years']}yrs | Aero: {design_params['aerodynamic_level']} | Scale: {design_params['production_scale']}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Traditional and ML methods
        for method_key in ['traditional', 'ml']:
            method = METHODS[method_key]
            with st.container():
                st.markdown(f"""
                <div class="method-card">
                    <h4>{method['icon']} {method['name']}</h4>
                    <p>{method['description']}</p>
                    <div style="display: flex; justify-content: space-between; font-size: 0.9em;">
                        <span>Weight: {method['expected_weight']} kg</span>
                        <span>Time: {method['expected_time']} months</span>
                        <span>Cost: {method['cost']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"🚀 Run {method['name']}", key=f"btn_{method_key}", 
                           disabled=st.session_state.is_running):
                    st.session_state.selected_method = method_key
                    st.session_state.current_design_params = design_params
                    run_optimization(method_key, design_params)
    
    with col2:
        # AI and GenAI methods
        for method_key in ['ai', 'genai']:
            method = METHODS[method_key]
            with st.container():
                st.markdown(f"""
                <div class="method-card">
                    <h4>{method['icon']} {method['name']}</h4>
                    <p>{method['description']}</p>
                    <div style="display: flex; justify-content: space-between; font-size: 0.9em;">
                        <span>Weight: {method['expected_weight']} kg</span>
                        <span>Time: {method['expected_time']} {'months' if method['expected_time'] >= 1 else 'weeks'}</span>
                        <span>Cost: {method['cost']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"🚀 Run {method['name']}", key=f"btn_{method_key}", 
                           disabled=st.session_state.is_running):
                    st.session_state.selected_method = method_key
                    st.session_state.current_design_params = design_params
                    run_optimization(method_key, design_params)

def run_optimization(method_key, design_params):
    """Run the optimization process"""
    st.session_state.is_running = True
    
    # Create progress section
    st.header(f"⚙️ Running {METHODS[method_key]['name']} Optimization...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Run simulation
    results = simulate_optimization(method_key, design_params, progress_bar, status_text)
    
    # Store results and update state
    st.session_state.optimization_results = results
    st.session_state.is_running = False
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Show results immediately
    show_results()

def show_results():
    """Display optimization results"""
    if st.session_state.optimization_results is None:
        return
    
    results = st.session_state.optimization_results
    
    st.header("📊 Optimization Results")
    
    # Reset button
    if st.button("🔄 Reset Simulation", key=f"reset_button_{hash(str(results))}"):
        st.session_state.optimization_results = None
        st.session_state.selected_method = None
        st.session_state.current_design_params = None
        st.rerun()
    
    # Show design requirements used in optimization
    st.subheader("🎯 Design Requirements Used")
    req_col1, req_col2 = st.columns(2)
    
    with req_col1:
        st.write(f"**Target Weight:** {results['design_params']['target_weight']} kg")
        st.write(f"**Durability:** {results['design_params']['durability_years']} years")
    
    with req_col2:
        st.write(f"**Aerodynamics:** {results['design_params']['aerodynamic_level']}")
        st.write(f"**Production Scale:** {results['design_params']['production_scale']}")
    
    st.markdown("---")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🏋️ Final Weight", 
            value=f"{results['final_weight']} kg",
            delta=f"-{results['weight_reduction']}% vs baseline"
        )
    
    with col2:
        time_unit = 'months' if results['time_months'] >= 1 else 'weeks'
        time_value = results['time_months'] if results['time_months'] >= 1 else round(results['time_months'] * 4.33, 1)
        st.metric(
            label="⏱️ Development Time", 
            value=f"{time_value} {time_unit}"
        )
    
    with col3:
        st.metric(
            label="💰 Cost Level", 
            value=results['cost'],
            delta=f"-{results['cost_reduction']}% reduction"
        )
    
    with col4:
        st.metric(
            label="✅ Accuracy", 
            value=f"{results['accuracy']}%"
        )
    
    # Detailed results in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📈 Design Metrics")
        metrics_df = pd.DataFrame({
            'Metric': ['Designs Generated', 'Stress Tests', 'Material Efficiency', 'Prototypes Needed'],
            'Value': [results['designs_generated'], results['stress_tests'], 
                     f"{results['material_efficiency']}%", results['prototypes']]
        })
        st.dataframe(metrics_df, hide_index=True)
    
    with col2:
        st.subheader("🎯 Performance Scores")
        perf_df = pd.DataFrame({
            'Category': ['Structural Integrity', 'Aerodynamics', 'Manufacturability'],
            'Score (%)': [results['structural_integrity'], results['aerodynamics'], 
                         results['manufacturability']]
        })
        st.dataframe(perf_df, hide_index=True)
    
    with col3:
        st.subheader("💡 Key Innovations")
        innovations = {
            'traditional': ['Standard aluminum alloy frame', 'Proven welding techniques', 'Conservative safety margins'],
            'ml': ['Data-driven material selection', 'Predictive stress modeling', 'Historical performance insights'],
            'ai': ['Physics-informed design topology', 'Multi-material optimization', 'Real-time FEA validation'],
            'genai': ['Hollow carbon fiber tubes', 'AI-optimized joint geometries', 'Integrated cable routing', 'Biomimetic stress distribution']
        }
        
        for innovation in innovations[results['method']]:
            st.write(f"• {innovation}")
    
    # Performance radar chart
    st.subheader("📊 Performance Analysis")
    radar_fig = create_performance_radar(results)
    st.plotly_chart(radar_fig, use_container_width=True)

# Show results if they exist
if st.session_state.optimization_results:
    show_results()

# Comparison section (always visible)
st.header("📋 Method Comparison")

# Create comparison table
comparison_df = pd.DataFrame([
    {
        'Method': f"{method['icon']} {method['name']}",
        'Weight (kg)': method['expected_weight'],
        'Time': f"{method['expected_time']} {'mo' if method['expected_time'] >= 1 else 'wk'}",
        'Cost': method['cost'],
        'Accuracy (%)': method['accuracy'],
        'Prototypes': method['prototypes']
    }
    for method in METHODS.values()
])

st.dataframe(comparison_df, hide_index=True)

# Comparison charts
st.subheader("📊 Visual Comparison")
comparison_fig = create_comparison_chart()
st.plotly_chart(comparison_fig, use_container_width=True)

# Footer with instructions
st.markdown("---")
st.markdown("""
### 🚀 How to Use This Application:

1. **Set Design Parameters**: Use the sidebar to specify your target weight, durability requirements, aerodynamic level, and production scale.

2. **Choose Optimization Method**: Click any "Run" button to start optimization with that method:
   - **Traditional CAD**: Baseline manual approach (18kg, 6 months)
   - **Machine Learning**: Data-driven optimization (16.5kg, 4 months)  
   - **Physics-Informed AI**: Real-time simulation (15kg, 2 months)
   - **Generative AI**: Automated design generation (13.8kg, 3 weeks)

3. **Review Results**: See detailed metrics, performance analysis, and key innovations for your chosen method.

4. **Compare Methods**: Use the comparison table and charts to understand trade-offs between approaches.

5. **Reset and Try Again**: Use the reset button to clear results and try different methods or parameters.

---
**Built with Streamlit** | **Product Design Optimization Demo**
""")

if __name__ == "__main__":
    main()