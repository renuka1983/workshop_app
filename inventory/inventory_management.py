import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Executive Demand Forecasting & Production Planning",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for executive dashboard styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .alert-critical { border-color: #dc3545; background: #f8d7da; }
    .alert-warning { border-color: #ffc107; background: #fff3cd; }
    .alert-success { border-color: #28a745; background: #d4edda; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def generate_synthetic_data():
    """Generate comprehensive synthetic data for multiple products"""
    np.random.seed(42)
    
    # Date range for 2 years of historical data + 6 months future
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 8, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    products = ['Product_A', 'Product_B', 'Product_C', 'Product_D', 'Product_E']
    
    data = []
    
    for product in products:
        base_demand = np.random.randint(500, 2000)
        seasonal_factor = np.random.uniform(0.3, 0.7)
        trend_factor = np.random.uniform(-0.1, 0.2)
        
        for date in dates:
            # Seasonal patterns
            day_of_year = date.timetuple().tm_yday
            seasonal = seasonal_factor * np.sin(2 * np.pi * day_of_year / 365)
            
            # Weekly patterns (lower demand on weekends)
            weekly = -0.2 if date.weekday() >= 5 else 0
            
            # Trend
            days_since_start = (date - start_date).days
            trend = trend_factor * (days_since_start / 365)
            
            # Random noise
            noise = np.random.normal(0, 0.15)
            
            # Calculate demand
            demand = base_demand * (1 + seasonal + weekly + trend + noise)
            demand = max(0, int(demand))
            
            # Production and inventory calculations
            lead_time = np.random.randint(5, 15)
            safety_stock = int(demand * 0.2)
            
            # Current inventory (with some randomness)
            current_inventory = int(demand * np.random.uniform(0.8, 1.5))
            
            # Production cost
            base_cost = np.random.uniform(10, 50)
            production_cost = base_cost * (1 + np.random.normal(0, 0.1))
            
            data.append({
                'Date': date,
                'Product': product,
                'Historical_Demand': demand,
                'Lead_Time': lead_time,
                'Safety_Stock': safety_stock,
                'Current_Inventory': current_inventory,
                'Production_Cost': round(production_cost, 2),
                'Supplier_Reliability': np.random.uniform(0.85, 0.98)
            })
    
    return pd.DataFrame(data)

def seasonal_forecast(df, product, periods=90):
    """Traditional seasonal averaging approach"""
    product_data = df[df['Product'] == product].copy()
    product_data['Month'] = product_data['Date'].dt.month
    product_data['DayOfWeek'] = product_data['Date'].dt.dayofweek
    
    # Calculate seasonal averages
    monthly_avg = product_data.groupby('Month')['Historical_Demand'].mean()
    weekly_avg = product_data.groupby('DayOfWeek')['Historical_Demand'].mean()
    
    # Generate forecast dates
    last_date = product_data['Date'].max()
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods)
    
    forecasts = []
    for date in forecast_dates:
        month_factor = monthly_avg[date.month] / product_data['Historical_Demand'].mean()
        week_factor = weekly_avg[date.dayofweek] / product_data['Historical_Demand'].mean()
        
        base_forecast = product_data['Historical_Demand'].mean()
        seasonal_forecast = base_forecast * (month_factor + week_factor) / 2
        
        forecasts.append({
            'Date': date,
            'Product': product,
            'Forecast': max(0, int(seasonal_forecast)),
            'Method': 'Seasonal_Average'
        })
    
    return pd.DataFrame(forecasts)

def ml_forecast(df, product, periods=90):
    """Machine Learning based forecast using Random Forest"""
    product_data = df[df['Product'] == product].copy()
    
    # Feature engineering
    product_data['Month'] = product_data['Date'].dt.month
    product_data['DayOfWeek'] = product_data['Date'].dt.dayofweek
    product_data['DayOfYear'] = product_data['Date'].dt.dayofyear
    product_data['Week'] = product_data['Date'].dt.isocalendar().week
    
    # Lag features
    for lag in [1, 7, 14, 30]:
        product_data[f'Demand_Lag_{lag}'] = product_data['Historical_Demand'].shift(lag)
    
    # Rolling averages
    for window in [7, 14, 30]:
        product_data[f'MA_{window}'] = product_data['Historical_Demand'].rolling(window=window).mean()
    
    # Drop NaN values
    product_data = product_data.dropna()
    
    # Features and target
    feature_cols = ['Month', 'DayOfWeek', 'DayOfYear', 'Week'] + \
                   [f'Demand_Lag_{lag}' for lag in [1, 7, 14, 30]] + \
                   [f'MA_{window}' for window in [7, 14, 30]]
    
    X = product_data[feature_cols]
    y = product_data['Historical_Demand']
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Generate forecasts
    last_date = product_data['Date'].max()
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods)
    
    forecasts = []
    current_data = product_data.iloc[-1:].copy()
    
    for i, date in enumerate(forecast_dates):
        # Create features for forecast date
        features = {
            'Month': date.month,
            'DayOfWeek': date.dayofweek,
            'DayOfYear': date.timetuple().tm_yday,
            'Week': date.isocalendar().week
        }
        
        # Use last known values for lag and MA features
        for lag in [1, 7, 14, 30]:
            features[f'Demand_Lag_{lag}'] = current_data[f'Demand_Lag_{lag}'].iloc[0]
        
        for window in [7, 14, 30]:
            features[f'MA_{window}'] = current_data[f'MA_{window}'].iloc[0]
        
        # Predict
        X_pred = pd.DataFrame([features])
        pred = model.predict(X_pred)[0]
        
        forecasts.append({
            'Date': date,
            'Product': product,
            'Forecast': max(0, int(pred)),
            'Method': 'ML_RandomForest'
        })
    
    return pd.DataFrame(forecasts)

def ai_dynamic_adjustment(base_forecast, external_factors):
    """AI-driven dynamic adjustment based on external factors"""
    adjusted_forecast = base_forecast.copy()
    
    # Market trend adjustment
    market_multiplier = 1 + (external_factors['market_trend'] - 50) / 100
    
    # Competition adjustment
    competition_multiplier = 1 - (external_factors['competition_intensity'] - 50) / 200
    
    # Economic adjustment
    economic_multiplier = 1 + (external_factors['economic_indicator'] - 50) / 150
    
    # Seasonal event adjustment
    seasonal_multiplier = 1 + external_factors['seasonal_events'] / 100
    
    # Apply adjustments
    total_multiplier = market_multiplier * competition_multiplier * economic_multiplier * seasonal_multiplier
    
    adjusted_forecast['AI_Adjusted_Forecast'] = (adjusted_forecast['Forecast'] * total_multiplier).astype(int)
    
    return adjusted_forecast

def calculate_inventory_metrics(df, forecasts):
    """Calculate inventory and production metrics"""
    metrics = {}
    
    for product in df['Product'].unique():
        product_data = df[df['Product'] == product].iloc[-30:]  # Last 30 days
        product_forecast = forecasts[forecasts['Product'] == product]
        
        avg_demand = product_data['Historical_Demand'].mean()
        current_inventory = product_data['Current_Inventory'].iloc[-1]
        lead_time = product_data['Lead_Time'].iloc[-1]
        safety_stock = product_data['Safety_Stock'].iloc[-1]
        
        # Stock-out risk
        days_of_stock = current_inventory / avg_demand if avg_demand > 0 else 0
        stockout_risk = "High" if days_of_stock < lead_time else "Medium" if days_of_stock < lead_time * 1.5 else "Low"
        
        # Overstock risk
        future_demand = product_forecast['Forecast'].sum() if not product_forecast.empty else 0
        overstock_risk = "High" if current_inventory > future_demand * 1.5 else "Medium" if current_inventory > future_demand * 1.2 else "Low"
        
        metrics[product] = {
            'Current_Inventory': current_inventory,
            'Days_of_Stock': round(days_of_stock, 1),
            'Stockout_Risk': stockout_risk,
            'Overstock_Risk': overstock_risk,
            'Recommended_Production': max(0, int(avg_demand * lead_time + safety_stock - current_inventory))
        }
    
    return metrics

# Main App
def main():
    st.markdown('<h1 class="main-header">Executive Demand Forecasting & Production Planning Dashboard</h1>', unsafe_allow_html=True)
    
    # Generate synthetic data
    with st.spinner("Loading synthetic data..."):
        df = generate_synthetic_data()
    
    # Sidebar for executive controls
    st.sidebar.header("🎯 Executive Controls")
    
    selected_products = st.sidebar.multiselect(
        "Select Products",
        options=df['Product'].unique(),
        default=df['Product'].unique()[:3]
    )
    
    forecast_horizon = st.sidebar.slider("Forecast Horizon (days)", 30, 180, 90)
    
    st.sidebar.header("📊 AI Dynamic Factors")
    market_trend = st.sidebar.slider("Market Trend (%)", 0, 100, 50, help="50 = neutral, >50 = growing, <50 = declining")
    competition_intensity = st.sidebar.slider("Competition Intensity", 0, 100, 50)
    economic_indicator = st.sidebar.slider("Economic Indicator", 0, 100, 60)
    seasonal_events = st.sidebar.slider("Seasonal Events Impact (%)", -20, 50, 0)
    
    external_factors = {
        'market_trend': market_trend,
        'competition_intensity': competition_intensity,
        'economic_indicator': economic_indicator,
        'seasonal_events': seasonal_events
    }
    
    # Executive Summary Dashboard
    st.header("📋 Executive Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_inventory_value = (df.groupby('Product').last()['Current_Inventory'] * 
                           df.groupby('Product').last()['Production_Cost']).sum()
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h3>Total Inventory Value</h3>
            <h2>${total_inventory_value:,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        active_products = len(selected_products)
        st.markdown(f"""
        <div class="metric-container">
            <h3>Active Products</h3>
            <h2>{active_products}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_lead_time = df.groupby('Product').last()['Lead_Time'].mean()
        st.markdown(f"""
        <div class="metric-container">
            <h3>Avg Lead Time</h3>
            <h2>{avg_lead_time:.1f} days</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_reliability = df.groupby('Product').last()['Supplier_Reliability'].mean() * 100
        st.markdown(f"""
        <div class="metric-container">
            <h3>Supplier Reliability</h3>
            <h2>{avg_reliability:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Forecasting Methods Comparison
    st.header("🔮 Demand Forecasting Analysis")
    
    tabs = st.tabs(["📊 Forecast Comparison", "📈 Real-time Adjustments", "⚠️ Risk Analysis", "🤖 AI Planning Assistant"])
    
    with tabs[0]:
        st.subheader("Traditional vs ML vs AI-Enhanced Forecasting")
        
        if selected_products:
            product_to_analyze = st.selectbox("Select Product for Detailed Analysis", selected_products)
            
            # Generate forecasts
            seasonal_pred = seasonal_forecast(df, product_to_analyze, forecast_horizon)
            ml_pred = ml_forecast(df, product_to_analyze, forecast_horizon)
            ai_adjusted = ai_dynamic_adjustment(ml_pred, external_factors)
            
            # Visualization
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Historical vs Forecasted Demand', 'Method Comparison', 
                              'Forecast Accuracy Simulation', 'Production Recommendations'),
                specs=[[{"colspan": 2}, None],
                       [{"type": "bar"}, {"type": "table"}]]
            )
            
            # Historical data for context
            historical = df[df['Product'] == product_to_analyze].tail(90)
            
            # Main forecast plot
            fig.add_trace(
                go.Scatter(x=historical['Date'], y=historical['Historical_Demand'],
                          name='Historical', line=dict(color='blue')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=seasonal_pred['Date'], y=seasonal_pred['Forecast'],
                          name='Seasonal Average', line=dict(color='green', dash='dash')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=ml_pred['Date'], y=ml_pred['Forecast'],
                          name='ML Forecast', line=dict(color='orange', dash='dot')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=ai_adjusted['Date'], y=ai_adjusted['AI_Adjusted_Forecast'],
                          name='AI Enhanced', line=dict(color='red', width=3)),
                row=1, col=1
            )
            
            # Method comparison bar chart
            methods = ['Seasonal', 'ML', 'AI Enhanced']
            avg_forecasts = [
                seasonal_pred['Forecast'].mean(),
                ml_pred['Forecast'].mean(),
                ai_adjusted['AI_Adjusted_Forecast'].mean()
            ]
            
            fig.add_trace(
                go.Bar(x=methods, y=avg_forecasts, name='Avg Forecast',
                      marker_color=['green', 'orange', 'red']),
                row=2, col=1
            )
            
            fig.update_layout(height=800, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        st.subheader("Real-time Dynamic Adjustments")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**External Factor Impact Analysis**")
            
            factors_df = pd.DataFrame({
                'Factor': ['Market Trend', 'Competition', 'Economic', 'Seasonal Events'],
                'Current Value': [market_trend, competition_intensity, economic_indicator, seasonal_events],
                'Impact': ['Positive' if market_trend > 50 else 'Negative',
                          'High' if competition_intensity > 70 else 'Medium' if competition_intensity > 40 else 'Low',
                          'Positive' if economic_indicator > 60 else 'Neutral' if economic_indicator > 40 else 'Negative',
                          'Positive' if seasonal_events > 0 else 'Neutral' if seasonal_events == 0 else 'Negative']
            })
            
            st.dataframe(factors_df, use_container_width=True)
        
        with col2:
            if selected_products:
                # Real-time adjustment visualization
                sample_product = selected_products[0]
                ml_base = ml_forecast(df, sample_product, 30)
                ai_adj = ai_dynamic_adjustment(ml_base, external_factors)
                
                adjustment_pct = ((ai_adj['AI_Adjusted_Forecast'].mean() / ai_adj['Forecast'].mean()) - 1) * 100
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ai_adj['Date'], y=ai_adj['Forecast'], name='Base ML Forecast'))
                fig.add_trace(go.Scatter(x=ai_adj['Date'], y=ai_adj['AI_Adjusted_Forecast'], name='AI Adjusted'))
                fig.update_layout(title=f"AI Adjustment: {adjustment_pct:+.1f}% vs Base Forecast")
                st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        st.subheader("Risk Analysis & Inventory Optimization")
        
        # Calculate metrics for all products
        sample_forecasts = pd.concat([
            ml_forecast(df, product, forecast_horizon) for product in selected_products
        ])
        
        inventory_metrics = calculate_inventory_metrics(df, sample_forecasts)
        
        # Risk summary
        high_stockout_risk = sum(1 for m in inventory_metrics.values() if m['Stockout_Risk'] == 'High')
        high_overstock_risk = sum(1 for m in inventory_metrics.values() if m['Overstock_Risk'] == 'High')
        
        col1, col2 = st.columns(2)
        
        with col1:
            if high_stockout_risk > 0:
                st.markdown(f"""
                <div class="alert-box alert-critical">
                    <strong>⚠️ CRITICAL:</strong> {high_stockout_risk} product(s) at high stockout risk
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="alert-box alert-success">
                    <strong>✅ GOOD:</strong> No high stockout risks detected
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if high_overstock_risk > 0:
                st.markdown(f"""
                <div class="alert-box alert-warning">
                    <strong>⚠️ WARNING:</strong> {high_overstock_risk} product(s) at overstock risk
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="alert-box alert-success">
                    <strong>✅ GOOD:</strong> No overstock risks detected
                </div>
                """, unsafe_allow_html=True)
        
        # Detailed metrics table
        st.write("**Detailed Inventory Analysis**")
        metrics_df = pd.DataFrame(inventory_metrics).T
        st.dataframe(metrics_df, use_container_width=True)
    
    with tabs[3]:
        st.subheader("🤖 AI-Powered Planning Assistant")
        
        st.write("**Conversational Planning & Disruption Simulation**")
        
        # Scenario selector
        scenario = st.selectbox(
            "Select Disruption Scenario",
            [
                "No Disruption (Baseline)",
                "Supply Chain Disruption (30% delay)",
                "Demand Surge (50% increase)",
                "Economic Downturn (20% demand drop)",
                "Competitor Launch (15% market share loss)",
                "Raw Material Shortage (40% cost increase)"
            ]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Scenario Impact Analysis**")
            
            scenario_impacts = {
                "No Disruption (Baseline)": {"demand_change": 0, "cost_change": 0, "lead_time_change": 0},
                "Supply Chain Disruption (30% delay)": {"demand_change": 0, "cost_change": 15, "lead_time_change": 30},
                "Demand Surge (50% increase)": {"demand_change": 50, "cost_change": 10, "lead_time_change": 0},
                "Economic Downturn (20% demand drop)": {"demand_change": -20, "cost_change": 0, "lead_time_change": 0},
                "Competitor Launch (15% market share loss)": {"demand_change": -15, "cost_change": 5, "lead_time_change": 0},
                "Raw Material Shortage (40% cost increase)": {"demand_change": -10, "cost_change": 40, "lead_time_change": 10}
            }
            
            impact = scenario_impacts[scenario]
            
            impact_df = pd.DataFrame({
                'Impact Area': ['Demand Change', 'Cost Change', 'Lead Time Change'],
                'Percentage': [f"{impact['demand_change']:+}%", f"{impact['cost_change']:+}%", f"{impact['lead_time_change']:+}%"]
            })
            
            st.dataframe(impact_df, use_container_width=True)
        
        with col2:
            st.write("**AI Recommendations**")
            
            recommendations = {
                "No Disruption (Baseline)": [
                    "✅ Continue current production plans",
                    "📊 Monitor market trends closely",
                    "🔄 Optimize inventory levels"
                ],
                "Supply Chain Disruption (30% delay)": [
                    "🚨 Increase safety stock by 50%",
                    "🔍 Identify alternative suppliers",
                    "📞 Communicate delays to customers early"
                ],
                "Demand Surge (50% increase)": [
                    "⚡ Ramp up production capacity",
                    "👥 Consider temporary workforce expansion",
                    "🏭 Evaluate overtime production"
                ],
                "Economic Downturn (20% demand drop)": [
                    "📉 Reduce production by 15%",
                    "💰 Focus on cost optimization",
                    "🎯 Target value-conscious segments"
                ],
                "Competitor Launch (15% market share loss)": [
                    "💡 Accelerate product innovation",
                    "📈 Increase marketing investment",
                    "💲 Consider strategic pricing"
                ],
                "Raw Material Shortage (40% cost increase)": [
                    "🔍 Diversify supplier base",
                    "📋 Review product specifications",
                    "💰 Pass through costs where possible"
                ]
            }
            
            for rec in recommendations[scenario]:
                st.write(rec)
        
        # Financial impact simulation
        st.write("**Financial Impact Simulation**")
        
        if selected_products:
            base_revenue = sum(df[df['Product'].isin(selected_products)].groupby('Product').last()['Current_Inventory'] * 
                             df[df['Product'].isin(selected_products)].groupby('Product').last()['Production_Cost'] * 1.3)  # Assuming 30% margin
            
            impact_multiplier = 1 + (impact['demand_change'] / 100)
            cost_multiplier = 1 + (impact['cost_change'] / 100)
            
            projected_revenue = base_revenue * impact_multiplier
            additional_costs = base_revenue * (cost_multiplier - 1)
            net_impact = projected_revenue - base_revenue - additional_costs
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Projected Revenue", f"${projected_revenue:,.0f}", f"{(projected_revenue/base_revenue-1)*100:+.1f}%")
            
            with col2:
                st.metric("Additional Costs", f"${additional_costs:,.0f}")
            
            with col3:
                delta_color = "normal" if net_impact >= 0 else "inverse"
                st.metric("Net Financial Impact", f"${net_impact:,.0f}", delta_color=delta_color)

if __name__ == "__main__":
    main()