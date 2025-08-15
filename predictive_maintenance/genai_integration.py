import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class GenAIIntegration:
    def __init__(self):
        self.sentiment_analyzer = None
        self.text_classifier = None
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        
    def load_models(self):
        """Load pre-trained models from Hugging Face"""
        try:
            with st.spinner("Loading GenAI models..."):
                # Sentiment analysis for maintenance logs
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=0 if torch.cuda.is_available() else -1
                )
                
                # Text classification for maintenance categories
                self.text_classifier = pipeline(
                    "text-classification",
                    model="facebook/bart-large-mnli",
                    device=0 if torch.cuda.is_available() else -1
                )
                
                self.is_loaded = True
                st.success("GenAI models loaded successfully!")
                
        except Exception as e:
            st.error(f"Error loading GenAI models: {str(e)}")
            st.info("Continuing without GenAI features...")
            self.is_loaded = False
    
    def analyze_maintenance_text(self, text: str) -> Dict[str, Any]:
        """Analyze maintenance text using GenAI"""
        if not self.is_loaded:
            return {"error": "GenAI models not loaded"}
        
        try:
            # Sentiment analysis
            sentiment_result = self.sentiment_analyzer(text)
            
            # Text classification for maintenance categories
            candidate_labels = [
                "equipment failure",
                "preventive maintenance",
                "safety concern",
                "performance issue",
                "routine check",
                "emergency repair"
            ]
            
            classification_result = self.text_classifier(
                text,
                candidate_labels=candidate_labels,
                hypothesis_template="This text is about {}."
            )
            
            return {
                "sentiment": sentiment_result[0],
                "maintenance_category": classification_result[0],
                "confidence": classification_result[0]['score'],
                "all_categories": classification_result
            }
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def generate_maintenance_insights(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate AI-powered insights from maintenance data"""
        if not self.is_loaded:
            return {"error": "GenAI models not loaded"}
        
        insights = {
            "anomaly_detection": {},
            "trend_analysis": {},
            "recommendations": [],
            "risk_assessment": {}
        }
        
        try:
            # Anomaly detection using statistical methods
            for col in ['temperature', 'vibration', 'pressure']:
                if col in data.columns:
                    mean_val = data[col].mean()
                    std_val = data[col].std()
                    
                    # Detect anomalies (3-sigma rule)
                    anomalies = data[abs(data[col] - mean_val) > 3 * std_val]
                    
                    insights["anomaly_detection"][col] = {
                        "anomaly_count": len(anomalies),
                        "anomaly_percentage": len(anomalies) / len(data) * 100,
                        "threshold_high": mean_val + 3 * std_val,
                        "threshold_low": mean_val - 3 * std_val
                    }
            
            # Trend analysis
            if 'timestamp' in data.columns:
                data_sorted = data.sort_values('timestamp')
                
                for col in ['temperature', 'vibration', 'pressure']:
                    if col in data.columns:
                        # Calculate trend (simple linear regression)
                        x = np.arange(len(data_sorted))
                        y = data_sorted[col].values
                        
                        if len(y) > 1:
                            slope = np.polyfit(x, y, 1)[0]
                            trend_direction = "increasing" if slope > 0 else "decreasing"
                            trend_strength = abs(slope)
                            
                            insights["trend_analysis"][col] = {
                                "direction": trend_direction,
                                "strength": trend_strength,
                                "slope": slope
                            }
            
            # Generate recommendations based on data patterns
            if 'maintenance_needed' in data.columns:
                maintenance_rate = data['maintenance_needed'].mean()
                
                if maintenance_rate > 0.3:
                    insights["recommendations"].append({
                        "priority": "High",
                        "action": "Increase preventive maintenance frequency",
                        "reason": f"High maintenance rate ({maintenance_rate:.1%}) detected"
                    })
                
                if 'temperature' in data.columns and data['temperature'].max() > 80:
                    insights["recommendations"].append({
                        "priority": "Medium",
                        "action": "Check cooling systems",
                        "reason": "Temperature exceeding 80°C detected"
                    })
                
                if 'vibration' in data.columns and data['vibration'].max() > 1.0:
                    insights["recommendations"].append({
                        "priority": "High",
                        "action": "Inspect mechanical components",
                        "reason": "High vibration levels detected"
                    })
            
            # Risk assessment
            risk_factors = []
            if 'temperature' in data.columns:
                temp_risk = data[data['temperature'] > 75]['temperature'].count() / len(data)
                risk_factors.append(("Temperature", temp_risk))
            
            if 'vibration' in data.columns:
                vib_risk = data[data['vibration'] > 0.8]['vibration'].count() / len(data)
                risk_factors.append(("Vibration", vib_risk))
            
            if 'pressure' in data.columns:
                pressure_risk = data[(data['pressure'] > 120) | (data['pressure'] < 70)]['pressure'].count() / len(data)
                risk_factors.append(("Pressure", pressure_risk))
            
            # Calculate overall risk score
            if risk_factors:
                overall_risk = sum(risk for _, risk in risk_factors) / len(risk_factors)
                risk_level = "High" if overall_risk > 0.3 else "Medium" if overall_risk > 0.1 else "Low"
                
                insights["risk_assessment"] = {
                    "overall_risk": overall_risk,
                    "risk_level": risk_level,
                    "risk_factors": risk_factors
                }
            
        except Exception as e:
            insights["error"] = f"Insight generation failed: {str(e)}"
        
        return insights
    
    def generate_natural_language_summary(self, insights: Dict[str, Any]) -> str:
        """Generate natural language summary of insights"""
        if "error" in insights:
            return f"Unable to generate summary: {insights['error']}"
        
        summary_parts = []
        
        # Risk assessment summary
        if "risk_assessment" in insights and insights["risk_assessment"]:
            risk_info = insights["risk_assessment"]
            summary_parts.append(
                f"The overall system risk level is {risk_info['risk_level']} "
                f"({risk_info['overall_risk']:.1%})."
            )
            
            if risk_info['risk_factors']:
                high_risk_factors = [factor for factor, risk in risk_info['risk_factors'] if risk > 0.2]
                if high_risk_factors:
                    summary_parts.append(
                        f"High risk factors include: {', '.join(high_risk_factors)}."
                    )
        
        # Anomaly summary
        if "anomaly_detection" in insights:
            total_anomalies = sum(info['anomaly_count'] for info in insights["anomaly_detection"].values())
            if total_anomalies > 0:
                summary_parts.append(
                    f"Detected {total_anomalies} anomalies across all sensors."
                )
        
        # Trend summary
        if "trend_analysis" in insights:
            increasing_trends = [col for col, info in insights["trend_analysis"].items() 
                               if info['direction'] == 'increasing']
            if increasing_trends:
                summary_parts.append(
                    f"Concerning increasing trends detected in: {', '.join(increasing_trends)}."
                )
        
        # Recommendations summary
        if "recommendations" in insights and insights["recommendations"]:
            high_priority = [rec for rec in insights["recommendations"] if rec['priority'] == 'High']
            if high_priority:
                summary_parts.append(
                    f"High priority actions required: {len(high_priority)} immediate interventions needed."
                )
        
        if not summary_parts:
            summary_parts.append("No significant issues detected. System operating within normal parameters.")
        
        return " ".join(summary_parts)
    
    def analyze_maintenance_logs(self, logs: List[str]) -> Dict[str, Any]:
        """Analyze multiple maintenance logs using GenAI"""
        if not self.is_loaded:
            return {"error": "GenAI models not loaded"}
        
        results = []
        overall_sentiment = {"positive": 0, "negative": 0, "neutral": 0}
        category_counts = {}
        
        for log in logs:
            analysis = self.analyze_maintenance_text(log)
            if "error" not in analysis:
                results.append(analysis)
                
                # Aggregate sentiment
                sentiment = analysis['sentiment']['label']
                if sentiment == 'POSITIVE':
                    overall_sentiment['positive'] += 1
                elif sentiment == 'NEGATIVE':
                    overall_sentiment['negative'] += 1
                else:
                    overall_sentiment['neutral'] += 1
                
                # Aggregate categories
                category = analysis['maintenance_category']['label']
                category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            "individual_analyses": results,
            "overall_sentiment": overall_sentiment,
            "category_distribution": category_counts,
            "total_logs_analyzed": len(results)
        }

def create_genai_tab():
    """Create the GenAI integration tab for the main app"""
    st.header("🤖 GenAI Integration")
    
    # Initialize GenAI
    if 'genai' not in st.session_state:
        st.session_state.genai = GenAIIntegration()
    
    genai = st.session_state.genai
    
    # Load models
    if not genai.is_loaded:
        if st.button("🚀 Load GenAI Models"):
            genai.load_models()
    
    if genai.is_loaded:
        st.success("✅ GenAI models are loaded and ready!")
        
        # Text analysis section
        st.subheader("Text Analysis")
        maintenance_text = st.text_area(
            "Enter maintenance log text for analysis:",
            placeholder="Example: Machine A showing unusual vibration patterns and temperature spikes..."
        )
        
        if st.button("Analyze Text") and maintenance_text:
            with st.spinner("Analyzing text with GenAI..."):
                analysis = genai.analyze_maintenance_text(maintenance_text)
                
                if "error" not in analysis:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Sentiment", analysis['sentiment']['label'])
                        st.metric("Confidence", f"{analysis['sentiment']['score']:.3f}")
                    
                    with col2:
                        st.metric("Category", analysis['maintenance_category']['label'])
                        st.metric("Category Confidence", f"{analysis['maintenance_category']['score']:.3f}")
                    
                    # Display detailed results
                    st.subheader("Detailed Analysis")
                    st.json(analysis)
                else:
                    st.error(analysis["error"])
        
        # Batch analysis section
        st.subheader("Batch Log Analysis")
        sample_logs = st.text_area(
            "Enter multiple maintenance logs (one per line):",
            placeholder="Log 1: Machine B temperature normal\nLog 2: Machine C vibration alert\nLog 3: Machine A maintenance completed",
            height=100
        )
        
        if st.button("Analyze Batch") and sample_logs:
            logs = [log.strip() for log in sample_logs.split('\n') if log.strip()]
            
            with st.spinner("Analyzing batch with GenAI..."):
                batch_results = genai.analyze_maintenance_logs(logs)
                
                if "error" not in batch_results:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Overall Sentiment")
                        sentiment_df = pd.DataFrame(
                            list(batch_results['overall_sentiment'].items()),
                            columns=['Sentiment', 'Count']
                        )
                        st.dataframe(sentiment_df)
                    
                    with col2:
                        st.subheader("Category Distribution")
                        category_df = pd.DataFrame(
                            list(batch_results['category_distribution'].items()),
                            columns=['Category', 'Count']
                        )
                        st.dataframe(category_df)
                    
                    st.metric("Total Logs Analyzed", batch_results['total_logs_analyzed'])
                else:
                    st.error(batch_results["error"])
        
        # AI Insights section
        st.subheader("AI-Powered Insights")
        if st.button("Generate AI Insights"):
            if 'data' in st.session_state and st.session_state.data is not None:
                with st.spinner("Generating AI insights..."):
                    insights = genai.generate_maintenance_insights(st.session_state.data)
                    
                    if "error" not in insights:
                        # Display insights in organized sections
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Risk Assessment")
                            if "risk_assessment" in insights:
                                risk = insights["risk_assessment"]
                                st.metric("Risk Level", risk['risk_level'])
                                st.metric("Risk Score", f"{risk['overall_risk']:.1%}")
                        
                        with col2:
                            st.subheader("Anomaly Detection")
                            if "anomaly_detection" in insights:
                                total_anomalies = sum(info['anomaly_count'] for info in insights["anomaly_detection"].values())
                                st.metric("Total Anomalies", total_anomalies)
                        
                        # Natural language summary
                        st.subheader("AI Summary")
                        summary = genai.generate_natural_language_summary(insights)
                        st.info(summary)
                        
                        # Detailed insights
                        st.subheader("Detailed Insights")
                        st.json(insights)
                    else:
                        st.error(insights["error"])
            else:
                st.warning("Please generate data first to create AI insights.")
    
    else:
        st.info("Click 'Load GenAI Models' to enable advanced AI features.")
        st.markdown("""
        **GenAI Features Available:**
        - Sentiment analysis of maintenance logs
        - Automatic categorization of maintenance issues
        - AI-powered anomaly detection
        - Natural language insights generation
        - Batch analysis of multiple logs
        """)

if __name__ == "__main__":
    st.title("GenAI Integration Test")
    create_genai_tab() 