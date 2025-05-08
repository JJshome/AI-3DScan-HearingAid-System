#!/usr/bin/env python3
"""
System configuration module for the AI-Enhanced 3D Scanning Hearing Aid Manufacturing System.
Handles loading, validation, and access to system-wide configuration.

Author: AI-Enhanced 3D Scanning Hearing Aid System Team
Date: May 8, 2025
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

class SystemConfig:
    """
    System configuration manager for the AI-Enhanced 3D Scanning Hearing Aid Manufacturing System.
    Loads and manages configuration settings for all modules.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the system configuration.
        
        Args:
            config_path: Path to the configuration JSON file
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        self.config_data = {}
        self.standalone_mode = False
        
        self._load_config()
        self._validate_config()
        
    def _load_config(self) -> None:
        """Load configuration from the specified file."""
        try:
            with open(self.config_path, 'r') as config_file:
                self.config_data = json.load(config_file)
                self.logger.info(f"Configuration loaded from {self.config_path}")
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON in configuration file: {self.config_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def _validate_config(self) -> None:
        """Validate the loaded configuration."""
        required_sections = [
            'system', 'modules', 'database', 'network', 'security'
        ]
        
        for section in required_sections:
            if section not in self.config_data:
                self.logger.error(f"Missing required configuration section: {section}")
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate module configurations
        required_modules = [
            'scanning', 'design', 'printing', 'acoustic', 
            'iot', 'integration', 'llm', 'fitting'
        ]
        
        for module in required_modules:
            if module not in self.config_data['modules']:
                self.logger.error(f"Missing configuration for required module: {module}")
                raise ValueError(f"Missing configuration for required module: {module}")
    
    def get_system_config(self) -> Dict[str, Any]:
        """Get general system configuration."""
        return self.config_data['system']
    
    def get_module_config(self, module_name: str) -> Dict[str, Any]:
        """Get configuration for a specific module."""
        if module_name not in self.config_data['modules']:
            self.logger.error(f"No configuration found for module: {module_name}")
            raise ValueError(f"No configuration found for module: {module_name}")
        
        config = self.config_data['modules'][module_name]
        
        # Apply standalone mode modifications if applicable
        if self.standalone_mode:
            if 'standalone' in config:
                # Merge standalone settings with regular settings
                for key, value in config['standalone'].items():
                    config[key] = value
        
        return config
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        return self.config_data['database']
    
    def get_network_config(self) -> Dict[str, Any]:
        """Get network configuration."""
        return self.config_data['network']
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration."""
        return self.config_data['security']
    
    def get_value(self, path: str, default: Optional[Any] = None) -> Any:
        """
        Get a configuration value using a dot-notation path.
        
        Args:
            path: Dot-notation path to the configuration value (e.g., 'modules.scanning.resolution')
            default: Default value to return if path is not found
            
        Returns:
            The configuration value or the default value if not found
        """
        parts = path.split('.')
        value = self.config_data
        
        try:
            for part in parts:
                value = value[part]
            return value
        except (KeyError, TypeError):
            self.logger.debug(f"Configuration path not found: {path}, using default: {default}")
            return default
    
    def set_standalone_mode(self, enabled: bool) -> None:
        """
        Set the standalone mode flag.
        
        Args:
            enabled: True to enable standalone mode, False to disable
        """
        self.standalone_mode = enabled
        self.logger.info(f"Standalone mode {'enabled' if enabled else 'disabled'}")
    
    def is_standalone_mode(self) -> bool:
        """
        Check if the system is running in standalone mode.
        
        Returns:
            True if the system is in standalone mode, False otherwise
        """
        return self.standalone_mode
