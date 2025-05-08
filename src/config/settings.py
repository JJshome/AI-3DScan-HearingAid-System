"""
System configuration settings

This module defines default configuration parameters for all components
of the AI-Enhanced 3D Scanning Hearing Aid Manufacturing System.
"""
import os
from typing import Dict, Any

# Scanner hardware configuration
SCANNER_CONFIG: Dict[str, Any] = {
    'model': 'HiRes-3DX-500',
    'manufacturer': 'MedTech Scanning Solutions',
    'firmware_version': '2.4.1',
    'connection_type': 'USB',  # One of: USB, Network, Bluetooth
    'resolution': (0.1, 0.1, 0.1),  # mm (x, y, z)
    'scan_area': (100, 100, 100),  # mm (width, height, depth)
    'calibration_interval_hours': 24,
    'connection_retries': 3,
    'retry_delay_seconds': 2,
    'default_scan_parameters': {
        'point_density': 5000,
        'scan_passes': 1,
        'noise_reduction': True,
        'scan_mode': 'standard',  # One of: standard, high_detail, quick
        'color_capture': False
    }
}

# AI model configuration
AI_MODEL_CONFIG: Dict[str, Any] = {
    'model_path': os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                               'models', 'ear_scan_processor_v3.1.pkl'),
    'preprocessing': {
        'voxel_size': 0.2,  # mm
        'remove_outliers': True,
        'smoothing': True,
        'normalize': True
    },
    'inference': {
        'batch_size': 1,
        'device': 'cuda' if os.environ.get('USE_GPU', 'True').lower() == 'true' else 'cpu',
        'precision': 'float32',
        'confidence_threshold': 0.85
    },
    'postprocessing': {
        'surface_reconstruction': True,
        'hole_filling': True,
        'smoothing_factor': 0.5,
        'minimum_thickness': 1.0,  # mm
        'export_format': 'STL'
    }
}

# 3D Printer configuration
PRINTER_CONFIG: Dict[str, Any] = {
    'model': 'MediPrint Pro-X2',
    'manufacturer': 'Audiology Solutions Inc.',
    'connection_type': 'Network',
    'ip_address': os.environ.get('PRINTER_IP', '192.168.1.100'),
    'port': int(os.environ.get('PRINTER_PORT', '9100')),
    'materials': {
        'shell': 'biocompatible_resin_v2',
        'core': 'flexible_resin_standard'
    },
    'layer_height': 0.05,  # mm
    'print_speed': 'standard',  # One of: standard, high_quality, draft
    'supports': 'minimal',
    'default_print_parameters': {
        'infill_density': 80,  # percentage
        'shell_thickness': 1.2,  # mm
        'temperature': 38.5,  # Celsius
        'platform_adhesion': 'raft'
    }
}

# General system configuration
SYSTEM_CONFIG: Dict[str, Any] = {
    'debug_mode': os.environ.get('DEBUG_MODE', 'False').lower() == 'true',
    'log_level': os.environ.get('LOG_LEVEL', 'INFO'),
    'log_file': os.environ.get('LOG_FILE', 'system.log'),
    'data_storage_path': os.environ.get('DATA_STORAGE_PATH', 'data/'),
    'temp_directory': os.environ.get('TEMP_DIR', 'tmp/'),
    'max_concurrent_processes': int(os.environ.get('MAX_CONCURRENT_PROCESSES', '4')),
    'backup_enabled': os.environ.get('BACKUP_ENABLED', 'True').lower() == 'true',
    'backup_interval_hours': int(os.environ.get('BACKUP_INTERVAL', '24')),
    'database': {
        'type': os.environ.get('DB_TYPE', 'postgresql'),
        'host': os.environ.get('DB_HOST', 'localhost'),
        'port': int(os.environ.get('DB_PORT', '5432')),
        'name': os.environ.get('DB_NAME', 'hearingaid_production'),
        'user': os.environ.get('DB_USER', 'postgres'),
        'password': os.environ.get('DB_PASSWORD', ''),
        'connection_pool_size': int(os.environ.get('DB_POOL_SIZE', '10')),
        'timeout_seconds': int(os.environ.get('DB_TIMEOUT', '30'))
    },
    'api': {
        'host': os.environ.get('API_HOST', '0.0.0.0'),
        'port': int(os.environ.get('API_PORT', '8080')),
        'workers': int(os.environ.get('API_WORKERS', '4')),
        'cors_origins': os.environ.get('CORS_ORIGINS', '*').split(','),
        'rate_limit': int(os.environ.get('RATE_LIMIT', '100')),
        'timeout_seconds': int(os.environ.get('API_TIMEOUT', '60'))
    }
}

# Function to override default settings with environment-specific configuration
def load_environment_config(env_file: str = None) -> None:
    """
    Load environment-specific configuration and override defaults
    
    Args:
        env_file: Optional path to environment file
    """
    # Implementation would load from file or environment variables
    # and update the configuration dictionaries accordingly
    pass
