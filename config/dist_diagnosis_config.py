"""
Configuration for the dist_diagnosis integration
"""

# Database node configuration
NODE_CONFIG = {
    'node_list': [
        '10.244.1.14:2881',  # Replace with your actual OceanBase node addresses
        '10.244.3.17:2881',
        '10.244.2.16:2881'
    ],
    'monitor_interval': 5,  # seconds
}

# Monitoring configuration
MONITOR_CONFIG = {
    'metrics': [
        'cpu_usage',
        'memory_usage',
        'disk_io',
        'network_io',
        'qps',
        'tps',
        'latency'
    ],
    'collection_interval': 5,  # seconds
    'window_size': 60  # seconds
}

# Model configuration
MODEL_CONFIG = {
    'model_path': 'models/dist_diagnosis/',
    'training_window': 3600,  # 1 hour
    'prediction_window': 300,  # 5 minutes
    'threshold': 0.8
}

# Integration configuration
INTEGRATION_CONFIG = {
    'enable_realtime_monitoring': True,
    'enable_historical_analysis': True,
    'alert_threshold': 0.9,
    'max_propagation_depth': 3
} 