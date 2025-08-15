#!/usr/bin/env python3
"""
Demo script for Predictive Maintenance App
Tests core functionality without Streamlit interface
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json
from datetime import datetime

class PredictiveMaintenanceDemo:
    def __init__(self):
        self.machines = ['Machine_A', 'Machine_B', 'Machine_C', 'Machine_D', 'Machine_E']
        
    def generate_synthetic_data(self, num_samples=500):
        """Generate synthetic predictive maintenance dataset"""
        print(f"Generating {num_samples} synthetic maintenance records...")
        
        np.random.seed(42)
        
        # Generate features
        temperature = np.random.normal(65, 15, num_samples)
        vibration = np.random.normal(0.5, 0.3, num_samples)
        pressure = np.random.normal(100, 20, num_samples)
        humidity = np.random.normal(45, 10, num_samples)
        runtime_hours = np.random.uniform(0, 8760, num_samples)
        maintenance_history = np.random.poisson(2, num_samples)
        
        # Generate target variable
        maintenance_needed = np.zeros(num_samples, dtype=int)
        
        for i in range(num_samples):
            temp_factor = 1 if temperature[i] > 75 else 0
            vib_factor = 1 if vibration[i] > 0.8 else 0
            runtime_factor = 1 if runtime_hours[i] > 6000 else 0
            pressure_factor = 1 if pressure[i] > 120 or pressure[i] < 70 else 0
            
            risk_score = temp_factor + vib_factor + runtime_factor + pressure_factor
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
        
        print(f"Generated dataset with {len(data)} records")
        print(f"Maintenance alerts: {data['maintenance_needed'].sum()} ({data['maintenance_needed'].mean():.1%})")
        
        return data
    
    def rule_based_baseline(self, data):
        """Traditional rule-based approach"""
        print("Applying rule-based baseline...")
        
        predictions = []
        for _, row in data.iterrows():
            if (row['temperature'] > 80 or 
                row['vibration'] > 1.0 or 
                row['pressure'] > 130 or 
                row['pressure'] < 70 or
                row['runtime_hours'] > 7000):
                predictions.append(1)
            else:
                predictions.append(0)
        
        accuracy = accuracy_score(data['maintenance_needed'], predictions)
        print(f"Rule-based accuracy: {accuracy:.3f}")
        
        return np.array(predictions), accuracy
    
    def train_ml_model(self, data):
        """Train Random Forest model"""
        print("Training Random Forest model...")
        
        # Prepare features
        feature_cols = ['temperature', 'vibration', 'pressure', 'humidity', 
                       'runtime_hours', 'maintenance_history']
        X = data[feature_cols]
        y = data['maintenance_needed']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"ML model accuracy: {accuracy:.3f}")
        
        # Feature importance
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nFeature Importance:")
        for _, row in importance_df.iterrows():
            print(f"  {row['Feature']}: {row['Importance']:.3f}")
        
        return {
            'model': model,
            'accuracy': accuracy,
            'feature_importance': importance_df,
            'classification_report': classification_report(y_test, y_pred)
        }
    
    def ai_scheduling_heuristic(self, data, ml_probabilities):
        """AI scheduling heuristic"""
        print("Generating AI scheduling recommendations...")
        
        schedule_recommendations = []
        
        for i, (_, row) in enumerate(data.iterrows()):
            # Business rules
            critical_machine = row['machine_id'] in ['Machine_A', 'Machine_B']
            high_priority = row['maintenance_history'] > 3
            
            # ML probability
            ml_prob = ml_probabilities[i] if i < len(ml_probabilities) else 0.5
            
            # Combined priority score
            priority_score = 0
            if critical_machine:
                priority_score += 0.4
            if high_priority:
                priority_score += 0.3
            priority_score += ml_prob * 0.3
            
            # Determine schedule
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
                'urgency': urgency
            })
        
        # Summary
        high_priority = [rec for rec in schedule_recommendations if rec['urgency'] == 'High']
        medium_priority = [rec for rec in schedule_recommendations if rec['urgency'] == 'Medium']
        
        print(f"High priority: {len(high_priority)} machines")
        print(f"Medium priority: {len(medium_priority)} machines")
        
        return pd.DataFrame(schedule_recommendations)
    
    def generate_report(self, data, ml_results, schedule_df):
        """Generate comprehensive report"""
        print("Generating maintenance report...")
        
        report_data = {
            'summary': {
                'total_machines': len(self.machines),
                'total_records': len(data),
                'maintenance_alerts': data['maintenance_needed'].sum(),
                'ml_accuracy': ml_results['accuracy'],
                'avg_priority_score': schedule_df['priority_score'].mean()
            },
            'machine_breakdown': data.groupby('machine_id')['maintenance_needed'].sum().to_dict(),
            'generated_at': datetime.now().isoformat()
        }
        
        print(f"Report generated: {report_data['summary']['total_records']} records, "
              f"{report_data['summary']['maintenance_alerts']} alerts")
        
        return report_data
    
    def run_demo(self):
        """Run complete demo"""
        print("Predictive Maintenance Demo")
        print("=" * 50)
        
        # 1. Generate data
        data = self.generate_synthetic_data(500)
        
        # 2. Rule-based baseline
        rule_predictions, rule_accuracy = self.rule_based_baseline(data)
        
        # 3. Train ML model
        ml_results = self.train_ml_model(data)
        
        # 4. AI scheduling
        schedule_df = self.ai_scheduling_heuristic(data, ml_results['model'].predict_proba(data[['temperature', 'vibration', 'pressure', 'humidity', 'runtime_hours', 'maintenance_history']])[:, 1])
        
        # 5. Generate report
        report = self.generate_report(data, ml_results, schedule_df)
        
        # 6. Summary
        print("\n" + "=" * 50)
        print("DEMO SUMMARY")
        print("=" * 50)
        print(f"Dataset Size: {len(data)} records")
        print(f"Rule-based Accuracy: {rule_accuracy:.3f}")
        print(f"ML Model Accuracy: {ml_results['accuracy']:.3f}")
        print(f"Improvement: {((ml_results['accuracy'] - rule_accuracy) / rule_accuracy * 100):.1f}%")
        print(f"High Priority Machines: {len([rec for rec in schedule_df.to_dict('records') if rec['urgency'] == 'High'])}")
        
        # Save sample data
        data.head(20).to_csv('sample_maintenance_data.csv', index=False)
        print(f"\nSample data saved to 'sample_maintenance_data.csv'")
        
        # Save report
        with open('maintenance_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Report saved to 'maintenance_report.json'")
        
        print("\nDemo completed successfully!")

if __name__ == "__main__":
    demo = PredictiveMaintenanceDemo()
    demo.run_demo() 