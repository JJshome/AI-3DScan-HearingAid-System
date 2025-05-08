#!/usr/bin/env python3
"""
Main entry point for the AI-Enhanced 3D Scanning Hearing Aid Manufacturing System.
This script initializes and starts all system modules, establishing the integration
control system as the central coordination hub.

Author: AI-Enhanced 3D Scanning Hearing Aid System Team
Date: May 8, 2025
"""

import os
import sys
import logging
import argparse
import json
import time
from datetime import datetime

# Add source directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core modules
from src.core.config import SystemConfig
from src.core.logging_setup import setup_logging
from src.integration_control.integration_controller import IntegrationController
from src.integration_control.system_monitor import SystemMonitor
from src.utils.exceptions import SystemInitializationError

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='AI-Enhanced 3D Scanning Hearing Aid Manufacturing System')
    
    parser.add_argument('--config', type=str, default='config/system_config.json',
                        help='Path to system configuration file')
    
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO', help='Set the logging level')
    
    parser.add_argument('--modules', type=str, nargs='+',
                        choices=['scanning', 'design', 'printing', 'acoustic', 'iot', 'llm', 'fitting', 'all'],
                        default=['all'], help='Modules to initialize')
    
    parser.add_argument('--standalone', action='store_true',
                        help='Run in standalone mode (no external connections)')
    
    return parser.parse_args()

def initialize_system(config_path, log_level, modules_to_start, standalone_mode):
    """Initialize the system with the given configuration."""
    try:
        # Set up logging
        setup_logging(log_level)
        logger = logging.getLogger(__name__)
        logger.info(f"Initializing AI-Enhanced 3D Scanning Hearing Aid Manufacturing System")
        
        # Load configuration
        logger.info(f"Loading configuration from {config_path}")
        if not os.path.exists(config_path):
            raise SystemInitializationError(f"Configuration file not found: {config_path}")
        
        config = SystemConfig(config_path)
        if standalone_mode:
            config.set_standalone_mode(True)
            logger.info("System running in standalone mode")
        
        # Initialize integration controller
        logger.info("Initializing Integration Control System")
        controller = IntegrationController(config)
        
        # Initialize system monitor
        logger.info("Initializing System Monitor")
        monitor = SystemMonitor(controller)
        
        # Initialize modules
        initialize_modules(controller, modules_to_start)
        
        # Start system monitor
        monitor.start()
        
        # Start integration controller
        logger.info("Starting Integration Control System")
        controller.start()
        
        return controller, monitor
        
    except Exception as e:
        logger.critical(f"Failed to initialize system: {str(e)}", exc_info=True)
        raise SystemInitializationError(f"System initialization failed: {str(e)}")

def initialize_modules(controller, modules_to_start):
    """Initialize the specified modules."""
    logger = logging.getLogger(__name__)
    
    if 'all' in modules_to_start:
        logger.info("Initializing all modules")
        controller.initialize_all_modules()
    else:
        for module in modules_to_start:
            logger.info(f"Initializing module: {module}")
            controller.initialize_module(module)

def main():
    """Main entry point for the system."""
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        # Initialize the system
        controller, monitor = initialize_system(
            args.config, args.log_level, args.modules, args.standalone)
        
        # Keep the main thread alive
        try:
            while controller.is_running():
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down system...")
        finally:
            # Shutdown procedures
            controller.stop()
            monitor.stop()
            logging.info("System shutdown complete")
            
    except SystemInitializationError as e:
        print(f"ERROR: {str(e)}")
        logging.critical(str(e))
        sys.exit(1)
    except Exception as e:
        print(f"UNHANDLED ERROR: {str(e)}")
        logging.critical(f"Unhandled exception: {str(e)}", exc_info=True)
        sys.exit(2)

if __name__ == "__main__":
    main()
