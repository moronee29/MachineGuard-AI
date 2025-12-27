"""
==============================================================================
  PREDICTIVE MAINTENANCE SYSTEM - CONFIGURATION FILE
==============================================================================
This file contains ALL configurable settings for the entire project.
Modify these values to customize the system without changing code.
"""

# =============================================================================
# PROJECT METADATA
# =============================================================================
PROJECT = {
    "name": "Predictive Maintenance System",
    "version": "1.0.0",
    "company": "Your Company Name",
    "description": "AI-powered predictive maintenance for industrial machinery"
}

# =============================================================================
# FILE PATHS & DIRECTORIES
# =============================================================================
PATHS = {
    # Data paths
    "data_file": "data/generated/predictive_maintenance_data.csv",
    "generated_data": "data/generated/generated_sensor_data.csv",
    
    # Model paths
    "models_dir": "models",
    "rf_model": "models/random_forest_model.pkl",
    "xgb_model": "models/xgboost_model.pkl",
    "scaler": "models/scaler.pkl",
    "features": "models/selected_features.txt",
    
    # Output paths
    "outputs_dir": "outputs",
    "results_dir": "results",
    "prediction_results": "results/prediction_results.csv",
    "high_risk_machines": "results/high_risk_machines.csv",
    
    # Dashboard paths
    "model_dashboard": "dashboards/model_performance.html",
    "production_dashboard": "dashboards/production_dashboard.html",
    "realtime_dashboard": "dashboards/realtime_dashboard.html",
    "legacy_dashboard": "dashboards/dashboard.html"
}

# =============================================================================
# WEB SERVER CONFIGURATION
# =============================================================================
SERVER = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": False,  # Auto-reload on code changes (development only)
    "log_level": "info"
}

# =============================================================================
# WEBSOCKET CONFIGURATION
# =============================================================================
WEBSOCKET = {
    "endpoint": "/ws/realtime",
    "update_interval": 1.0,  # seconds between updates
    "max_connections": 10,
    "timeout": 300  # seconds
}

# =============================================================================
# MACHINE LEARNING CONFIGURATION
# =============================================================================
ML_CONFIG = {
    # Model selection
    "use_model": "xgboost",  # Options: "random_forest", "xgboost"
    
    # Training parameters
    "test_size": 0.2,
    "random_state": 42,
    "validation_split": 0.15,
    
    # Feature engineering
    "rolling_window": 5,
    "feature_selection_threshold": 0.95  # cumulative importance
}

# =============================================================================
# PREDICTION THRESHOLDS
# =============================================================================
THRESHOLDS = {
    # Risk classification
    "low_risk_max": 0.3,      # Below 30%
    "medium_risk_max": 0.7,   # 30-70%
    "high_risk_min": 0.7,     # Above 70%
    
    # Alert thresholds
    "failure_probability": 0.7,
    "critical_vibration": 8.0,
    "critical_temperature": 90.0,
    "critical_pressure_low": 4.0,
    "critical_pressure_high": 6.0
}

# =============================================================================
# SENSOR CONFIGURATION
# =============================================================================
SENSORS = {
    "enabled_sensors": [
        "vibration",
        "temperature", 
        "pressure",
        "voltage",
        "current",
        "acoustic_emission",
        "rotation_speed",
        "torque",
        "power_consumption"
    ],
    
    # Normal operating ranges
    "normal_ranges": {
        "vibration": {"min": 0.5, "max": 4.0},
        "temperature": {"min": 60, "max": 80},
        "pressure": {"min": 4.5, "max": 6.0},
        "voltage": {"min": 210, "max": 230},
        "current": {"min": 10, "max": 20},
        "acoustic_emission": {"min": 55, "max": 75},
        "rotation_speed": {"min": 1400, "max": 1600},
        "torque": {"min": 130, "max": 165},
        "power_consumption": {"min": 18, "max": 28}
    }
}

# =============================================================================
# REAL-TIME SIMULATION
# =============================================================================
SIMULATION = {
    "duration": 30,  # seconds
    "update_rate": 1.0,  # updates per second
    "degradation_rate": 0.02,  # per time step
    "noise_level": 0.1,  # relative noise
    "failure_threshold": 0.7
}

# =============================================================================
# DASHBOARD CONFIGURATION
# =============================================================================
DASHBOARD = {
    # Theme colors
    "colors": {
        "primary": "#667eea",
        "success": "#11998e",
        "warning": "#f39c12",
        "danger": "#e74c3c",
        "info": "#3498db",
        "dark": "#0f0f1e",
        "light": "#ecf0f1"
    },
    
    # Chart settings
    "max_data_points": 30,
    "animation_duration": 500,  # milliseconds
    "update_interval": 5000,  # milliseconds for status updates
    
    # Display settings
    "show_toolbar": False,
    "static_plots": True,  # Disable zoom/pan/hover
    "dark_theme": True
}

# =============================================================================
# DATA GENERATION (for testing/demo)
# =============================================================================
DATA_GENERATION = {
    "num_machines": 100,
    "samples_per_machine": 100,
    "failure_rate": 0.001,  # 0.1% failure rate
    "seed": 42
}

# =============================================================================
# LOGGING & MONITORING
# =============================================================================
LOGGING = {
    "level": "INFO",  # DEBUG, INFO, WARNING, ERROR
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "save_logs": True,
    "log_file": "logs/system.log"
}

# =============================================================================
# PERFORMANCE & LIMITS
# =============================================================================
PERFORMANCE = {
    "max_batch_size": 1000,
    "enable_caching": True,
    "cache_ttl": 3600,  # seconds
    "max_workers": 4
}

# =============================================================================
# FEATURE FLAGS
# =============================================================================
FEATURES = {
    "enable_real_time": True,
    "enable_batch_prediction": True,
    "enable_model_retraining": True,
    "enable_auto_alerts": True,
    "enable_data_export": True
}

# =============================================================================
# ALERTS & NOTIFICATIONS
# =============================================================================
ALERTS = {
    "enabled": True,
    "email_notifications": False,
    "email_recipients": [],
    "sms_notifications": False,
    "webhook_url": None
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get(key, default=None):
    """Get configuration value by dot notation (e.g., 'SERVER.port')"""
    keys = key.split('.')
    value = globals()
    for k in keys:
        value = value.get(k, default)
        if value is None:
            return default
    return value

def update(key, value):
    """Update configuration value"""
    keys = key.split('.')
    config = globals()
    for k in keys[:-1]:
        config = config[k]
    config[keys[-1]] = value

# =============================================================================
# VALIDATION
# =============================================================================
def validate_config():
    """Validate critical configuration settings"""
    errors = []
    
    # Check required paths exist
    import os
    if not os.path.exists(PATHS["models_dir"]):
        os.makedirs(PATHS["models_dir"], exist_ok=True)
    
    if not os.path.exists(PATHS["outputs_dir"]):
        os.makedirs(PATHS["outputs_dir"], exist_ok=True)
        
    if not os.path.exists(PATHS["results_dir"]):
        os.makedirs(PATHS["results_dir"], exist_ok=True)
    
    # Validate thresholds
    if not (0 < THRESHOLDS["failure_probability"] <= 1):
        errors.append("failure_probability must be between 0 and 1")
    
    # Validate server config
    if not (1024 <= SERVER["port"] <= 65535):
        errors.append("Server port must be between 1024 and 65535")
    
    return errors

# Auto-validate on import
_validation_errors = validate_config()
if _validation_errors:
    print("⚠️  Configuration Validation Warnings:")
    for error in _validation_errors:
        print(f"   - {error}")
else:
    print("✓ Configuration loaded successfully")
