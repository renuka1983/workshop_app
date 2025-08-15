"""
Configuration file for Predictive Maintenance App
Customize parameters here for different use cases
"""

# Machine Configuration
MACHINES = {
    'Machine_A': {'critical': True, 'type': 'Production Line', 'location': 'Building 1'},
    'Machine_B': {'critical': True, 'type': 'Assembly Unit', 'location': 'Building 1'},
    'Machine_C': {'critical': False, 'type': 'Packaging Unit', 'location': 'Building 2'},
    'Machine_D': {'critical': False, 'type': 'Quality Control', 'location': 'Building 2'},
    'Machine_E': {'critical': False, 'type': 'Storage System', 'location': 'Building 3'}
}

# Sensor Thresholds
SENSOR_THRESHOLDS = {
    'temperature': {
        'min': 20,      # °C
        'max': 85,      # °C
        'warning_low': 30,
        'warning_high': 75,
        'critical_low': 20,
        'critical_high': 80
    },
    'vibration': {
        'min': 0.0,     # g
        'max': 2.0,     # g
        'warning_low': 0.1,
        'warning_high': 0.8,
        'critical_low': 0.0,
        'critical_high': 1.0
    },
    'pressure': {
        'min': 50,      # PSI
        'max': 150,     # PSI
        'warning_low': 70,
        'warning_high': 120,
        'critical_low': 60,
        'critical_high': 130
    },
    'humidity': {
        'min': 20,      # %
        'max': 80,      # %
        'warning_low': 30,
        'warning_high': 70,
        'critical_low': 20,
        'critical_high': 80
    }
}

# Maintenance Configuration
MAINTENANCE_CONFIG = {
    'preventive_interval_hours': 720,  # 30 days
    'max_runtime_hours': 8760,         # 1 year
    'maintenance_history_weight': 0.3,
    'critical_machine_weight': 0.4,
    'ml_probability_weight': 0.3
}

# ML Model Configuration
ML_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1
}

# GenAI Configuration
GENAI_CONFIG = {
    'sentiment_model': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
    'classification_model': 'facebook/bart-large-mnli',
    'maintenance_categories': [
        'equipment failure',
        'preventive maintenance',
        'safety concern',
        'performance issue',
        'routine check',
        'emergency repair'
    ]
}

# UI Configuration
UI_CONFIG = {
    'page_title': 'Predictive Maintenance Dashboard',
    'page_icon': '🔧',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded',
    'max_samples': 2000,
    'default_samples': 1000
}

# Export Configuration
EXPORT_CONFIG = {
    'csv_include_index': False,
    'json_indent': 2,
    'pdf_page_size': 'A4',
    'pdf_margin': 20
}

# Risk Assessment Configuration
RISK_CONFIG = {
    'anomaly_threshold': 3.0,  # Standard deviations for anomaly detection
    'high_risk_threshold': 0.3,
    'medium_risk_threshold': 0.1,
    'trend_analysis_window': 100  # Number of points for trend calculation
}

def get_machine_info(machine_id):
    """Get machine configuration information"""
    return MACHINES.get(machine_id, {'critical': False, 'type': 'Unknown', 'location': 'Unknown'})

def get_sensor_thresholds(sensor_name):
    """Get sensor threshold configuration"""
    return SENSOR_THRESHOLDS.get(sensor_name, {})

def is_critical_machine(machine_id):
    """Check if machine is critical"""
    return MACHINES.get(machine_id, {}).get('critical', False)

def get_maintenance_priority(machine_id, maintenance_history, ml_probability):
    """Calculate maintenance priority score"""
    score = 0.0
    
    # Critical machine weight
    if is_critical_machine(machine_id):
        score += MAINTENANCE_CONFIG['critical_machine_weight']
    
    # Maintenance history weight
    if maintenance_history > 3:
        score += MAINTENANCE_CONFIG['maintenance_history_weight']
    
    # ML probability weight
    score += ml_probability * MAINTENANCE_CONFIG['ml_probability_weight']
    
    return min(1.0, score)

def get_maintenance_schedule(priority_score):
    """Determine maintenance schedule based on priority score"""
    if priority_score > 0.7:
        return "Immediate (Next 24h)", "High"
    elif priority_score > 0.5:
        return "Within 72h", "Medium"
    elif priority_score > 0.3:
        return "Within 1 week", "Low"
    else:
        return "No maintenance needed", "None" 