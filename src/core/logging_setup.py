#!/usr/bin/env python3
"""
Logging setup module for the AI-Enhanced 3D Scanning Hearing Aid Manufacturing System.
Configures logging for all system components.

Author: AI-Enhanced 3D Scanning Hearing Aid System Team
Date: May 8, 2025
"""

import os
import sys
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path

# Define log format constants
CONSOLE_LOG_FORMAT = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
FILE_LOG_FORMAT = '%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

def setup_logging(log_level_str='INFO', log_dir=None):
    """
    Set up logging for the system.
    
    Args:
        log_level_str: String representation of the log level (DEBUG, INFO, etc.)
        log_dir: Directory to store log files. If None, logs are stored in ./logs
    """
    # Convert log level string to logging constant
    log_level = getattr(logging, log_level_str)
    
    # Create logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(CONSOLE_LOG_FORMAT, DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Set up file logging
    if log_dir is None:
        log_dir = Path('./logs')
    else:
        log_dir = Path(log_dir)
    
    # Create log directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped log file name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'system_{timestamp}.log'
    
    # Create file handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(FILE_LOG_FORMAT, DATE_FORMAT)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Create separate error log file for warnings and above
    error_log_file = log_dir / f'errors_{timestamp}.log'
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_file, maxBytes=10*1024*1024, backupCount=5)
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(file_formatter)
    root_logger.addHandler(error_handler)
    
    # Set up specific loggers for various components
    setup_component_loggers()
    
    # Log initial message
    logging.info(f"Logging initialized at level {log_level_str}")
    logging.info(f"Log files will be written to {log_dir.absolute()}")

def setup_component_loggers():
    """Set up specific loggers for system components with appropriate levels."""
    # Integration Control logger - lower level for detailed integration logs
    integration_logger = logging.getLogger('src.integration_control')
    integration_logger.setLevel(logging.DEBUG)
    
    # Database access logger - may want to limit detailed SQL logging
    db_logger = logging.getLogger('src.database')
    db_logger.setLevel(logging.INFO)
    
    # Security logger - always keep at INFO or higher to capture all security events
    security_logger = logging.getLogger('src.security')
    security_logger.setLevel(logging.INFO)
    
    # Network logger - can be very verbose, so default higher
    network_logger = logging.getLogger('src.network')
    network_logger.setLevel(logging.INFO)

def get_module_logger(module_name):
    """
    Get a logger for a specific module with the appropriate configuration.
    
    Args:
        module_name: Name of the module (e.g., 'scanning', 'design')
        
    Returns:
        Configured logger for the module
    """
    logger_name = f'src.modules.{module_name}'
    logger = logging.getLogger(logger_name)
    
    # All module logs will be written to a module-specific file as well
    log_dir = Path('./logs/modules')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if this module logger already has handlers to avoid duplicates
    if not logger.handlers:
        module_log_file = log_dir / f'{module_name}.log'
        module_handler = logging.handlers.RotatingFileHandler(
            module_log_file, maxBytes=5*1024*1024, backupCount=3)
        module_handler.setLevel(logging.DEBUG)
        module_formatter = logging.Formatter(FILE_LOG_FORMAT, DATE_FORMAT)
        module_handler.setFormatter(module_formatter)
        logger.addHandler(module_handler)
    
    return logger

def log_exception(logger, message, exc_info=True):
    """
    Log an exception with consistent formatting.
    
    Args:
        logger: Logger to use
        message: Message to log
        exc_info: Whether to include exception info
    """
    logger.error(f"EXCEPTION: {message}", exc_info=exc_info)
