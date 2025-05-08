#!/usr/bin/env python3
"""
OCT Scanner Driver for the 3D Scanning Module of the AI-Enhanced 3D Scanning Hearing Aid Manufacturing System.
Provides interface to Optical Coherence Tomography (OCT) scanners for high-precision ear canal scanning.

Author: AI-Enhanced 3D Scanning Hearing Aid System Team
Date: May 8, 2025
"""

import os
import sys
import logging
import time
import json
import threading
import serial
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# Import scanner interface
from src.modules.scanning.scanner_interface import ScannerInterface

# Import utility modules
from src.utils.exceptions import ScanningModuleError, CalibrationError, HardwareError

class OCTScannerDriver(ScannerInterface):
    """
    Driver implementation for Optical Coherence Tomography (OCT) scanners.
    Provides high-resolution 3D scanning of ear canals with micron-level precision.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the OCT scanner driver.
        
        Args:
            config: OCT scanner configuration
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.device_id = config.get('device_id', 'OCT-001')
        self.serial_port = config.get('port', '/dev/ttyUSB0')
        self.baud_rate = config.get('baud_rate', 115200)
        self.timeout = config.get('timeout_seconds', 30)
        
        # Scanner specifications
        self.wavelength_nm = config.get('wavelength_nm', 840)
        self.axial_resolution_um = config.get('axial_resolution_um', 5)
        self.lateral_resolution_um = config.get('lateral_resolution_um', 15)
        self.scan_depth_mm = config.get('scan_depth_mm', 7)
        self.max_scan_area_mm = config.get('max_scan_area_mm', 25)
        
        # Internal state
        self.initialized = False
        self.running = False
        self.scanning = False
        self.calibrated = False
        self.serial_connection = None
        self.lock = threading.RLock()
        self.scan_data = None
        self.last_error = None
        self.scan_progress = 0
        
        # Performance metrics
        self.total_scans = 0
        self.successful_scans = 0
        self.scan_times = []
        self.last_calibration_time = None
        
        # Initialize simulation mode if configured
        self.simulation_mode = config.get('simulation_mode', False)
        if self.simulation_mode:
            self.logger.info(f"OCT Scanner Driver initialized in simulation mode")
    
    def initialize(self) -> Dict[str, Any]:
        """
        Initialize the OCT scanner hardware.
        
        Returns:
            Dictionary with initialization results
        """
        with self.lock:
            if self.initialized:
                return {'success': True, 'message': "Scanner already initialized"}
            
            self.logger.info(f"Initializing OCT scanner: {self.device_id}")
            
            try:
                if self.simulation_mode:
                    # Simulate successful initialization
                    time.sleep(2)  # Simulate initialization time
                    self.initialized = True
                    self.logger.info(f"OCT scanner {self.device_id} initialized in simulation mode")
                    return {'success': True, 'message': "Scanner initialized in simulation mode"}
                
                # Open serial connection to scanner
                self.serial_connection = serial.Serial(
                    port=self.serial_port,
                    baudrate=self.baud_rate,
                    timeout=self.timeout
                )
                
                # Send initialization command
                init_command = {
                    'command': 'initialize',
                    'device_id': self.device_id,
                    'timestamp': datetime.now().isoformat()
                }
                
                response = self._send_command(init_command)
                
                if response.get('status') != 'success':
                    error_msg = f"OCT scanner initialization failed: {response.get('message', 'Unknown error')}"
                    self.logger.error(error_msg)
                    self.last_error = error_msg
                    return {'success': False, 'message': error_msg}
                
                # Update state
                self.initialized = True
                self.last_error = None
                
                self.logger.info(f"OCT scanner {self.device_id} initialized successfully")
                
                return {
                    'success': True,
                    'message': "Scanner initialized successfully",
                    'scanner_info': {
                        'device_id': self.device_id,
                        'type': 'OCT',
                        'wavelength_nm': self.wavelength_nm,
                        'axial_resolution_um': self.axial_resolution_um,
                        'lateral_resolution_um': self.lateral_resolution_um
                    }
                }
                
            except serial.SerialException as e:
                error_msg = f"Serial connection error: {str(e)}"
                self.logger.error(error_msg)
                self.last_error = error_msg
                return {'success': False, 'message': error_msg}
                
            except Exception as e:
                error_msg = f"OCT scanner initialization error: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                self.last_error = error_msg
                return {'success': False, 'message': error_msg}
    
    def start(self) -> Dict[str, Any]:
        """
        Start the OCT scanner.
        
        Returns:
            Dictionary with start results
        """
        with self.lock:
            if not self.initialized:
                error_msg = "Cannot start: Scanner not initialized"
                self.logger.error(error_msg)
                return {'success': False, 'message': error_msg}
            
            if self.running:
                return {'success': True, 'message': "Scanner already running"}
            
            self.logger.info(f"Starting OCT scanner: {self.device_id}")
            
            try:
                if self.simulation_mode:
                    # Simulate successful start
                    time.sleep(1)  # Simulate startup time
                    self.running = True
                    self.logger.info(f"OCT scanner {self.device_id} started in simulation mode")
                    return {'success': True, 'message': "Scanner started in simulation mode"}
                
                # Send start command
                start_command = {
                    'command': 'start',
                    'timestamp': datetime.now().isoformat()
                }
                
                response = self._send_command(start_command)
                
                if response.get('status') != 'success':
                    error_msg = f"OCT scanner start failed: {response.get('message', 'Unknown error')}"
                    self.logger.error(error_msg)
                    self.last_error = error_msg
                    return {'success': False, 'message': error_msg}
                
                # Update state
                self.running = True
                self.last_error = None
                
                self.logger.info(f"OCT scanner {self.device_id} started successfully")
                
                return {
                    'success': True,
                    'message': "Scanner started successfully"
                }
                
            except Exception as e:
                error_msg = f"OCT scanner start error: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                self.last_error = error_msg
                return {'success': False, 'message': error_msg}
    
    def stop(self) -> Dict[str, Any]:
        """
        Stop the OCT scanner.
        
        Returns:
            Dictionary with stop results
        """
        with self.lock:
            if not self.initialized:
                error_msg = "Cannot stop: Scanner not initialized"
                self.logger.error(error_msg)
                return {'success': False, 'message': error_msg}
            
            if not self.running:
                return {'success': True, 'message': "Scanner already stopped"}
            
            self.logger.info(f"Stopping OCT scanner: {self.device_id}")
            
            try:
                # If scanning, abort scan first
                if self.scanning:
                    self.abort_scan()
                
                if self.simulation_mode:
                    # Simulate successful stop
                    time.sleep(1)  # Simulate stop time
                    self.running = False
                    self.logger.info(f"OCT scanner {self.device_id} stopped in simulation mode")
                    return {'success': True, 'message': "Scanner stopped in simulation mode"}
                
                # Send stop command
                stop_command = {
                    'command': 'stop',
                    'timestamp': datetime.now().isoformat()
                }
                
                response = self._send_command(stop_command)
                
                if response.get('status') != 'success':
                    error_msg = f"OCT scanner stop failed: {response.get('message', 'Unknown error')}"
                    self.logger.error(error_msg)
                    self.last_error = error_msg
                    return {'success': False, 'message': error_msg}
                
                # Update state
                self.running = False
                self.last_error = None
                
                self.logger.info(f"OCT scanner {self.device_id} stopped successfully")
                
                return {
                    'success': True,
                    'message': "Scanner stopped successfully"
                }
                
            except Exception as e:
                error_msg = f"OCT scanner stop error: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                self.last_error = error_msg
                return {'success': False, 'message': error_msg}
    
    def calibrate(self) -> Dict[str, Any]:
        """
        Calibrate the OCT scanner for optimal performance.
        
        Returns:
            Dictionary with calibration results
        """
        with self.lock:
            if not self.initialized:
                error_msg = "Cannot calibrate: Scanner not initialized"
                self.logger.error(error_msg)
                return {'success': False, 'message': error_msg}
            
            if not self.running:
                error_msg = "Cannot calibrate: Scanner not running"
                self.logger.error(error_msg)
                return {'success': False, 'message': error_msg}
            
            if self.scanning:
                error_msg = "Cannot calibrate: Scanner is currently scanning"
                self.logger.error(error_msg)
                return {'success': False, 'message': error_msg}
            
            self.logger.info(f"Calibrating OCT scanner: {self.device_id}")
            
            try:
                if self.simulation_mode:
                    # Simulate successful calibration
                    time.sleep(3)  # Simulate calibration time
                    self.calibrated = True
                    self.last_calibration_time = datetime.now()
                    self.logger.info(f"OCT scanner {self.device_id} calibrated in simulation mode")
                    
                    # Create simulated calibration data
                    calibration_data = {
                        'reference_intensity': 0.85,
                        'snr_db': 28.5,
                        'wavelength_actual_nm': 842.3,
                        'axial_resolution_actual_um': 4.8,
                        'lateral_resolution_actual_um': 14.7,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    return {
                        'success': True,
                        'message': "Scanner calibrated in simulation mode",
                        'calibration_data': calibration_data
                    }
                
                # Send calibration command
                calibration_command = {
                    'command': 'calibrate',
                    'timestamp': datetime.now().isoformat()
                }
                
                response = self._send_command(calibration_command)
                
                if response.get('status') != 'success':
                    error_msg = f"OCT scanner calibration failed: {response.get('message', 'Unknown error')}"
                    self.logger.error(error_msg)
                    self.last_error = error_msg
                    return {'success': False, 'message': error_msg}
                
                # Extract calibration data
                calibration_data = response.get('calibration_data', {})
                
                # Update state
                self.calibrated = True
                self.last_calibration_time = datetime.now()
                self.last_error = None
                
                self.logger.info(f"OCT scanner {self.device_id} calibrated successfully")
                
                return {
                    'success': True,
                    'message': "Scanner calibrated successfully",
                    'calibration_data': calibration_data
                }
                
            except Exception as e:
                error_msg = f"OCT scanner calibration error: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                self.last_error = error_msg
                return {'success': False, 'message': error_msg}
    
    def check_health(self) -> Dict[str, Any]:
        """
        Check the health and operational status of the OCT scanner.
        
        Returns:
            Dictionary with health check results
        """
        with self.lock:
            if not self.initialized:
                return {
                    'healthy': False,
                    'message': "Scanner not initialized"
                }
            
            try:
                if self.simulation_mode:
                    # Simulate health check
                    return {
                        'healthy': True,
                        'message': "Scanner operational in simulation mode",
                        'details': {
                            'device_id': self.device_id,
                            'status': 'operational',
                            'simulation_mode': True,
                            'last_error': self.last_error
                        }
                    }
                
                # Send health check command
                health_check_command = {
                    'command': 'health_check',
                    'timestamp': datetime.now().isoformat()
                }
                
                response = self._send_command(health_check_command)
                
                if response.get('status') != 'success':
                    return {
                        'healthy': False,
                        'message': response.get('message', 'Unknown error'),
                        'details': response.get('details', {})
                    }
                
                # Extract health data
                health_data = response.get('health_data', {})
                
                # Determine overall health
                all_systems_operational = health_data.get('all_systems_operational', False)
                
                return {
                    'healthy': all_systems_operational,
                    'message': "Scanner operational" if all_systems_operational else "Scanner issues detected",
                    'details': health_data
                }
                
            except Exception as e:
                error_msg = f"OCT scanner health check error: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                self.last_error = error_msg
                return {
                    'healthy': False,
                    'message': error_msg
                }
    
    def recover(self) -> Dict[str, Any]:
        """
        Attempt to recover the OCT scanner from an error state.
        
        Returns:
            Dictionary with recovery results
        """
        with self.lock:
            self.logger.info(f"Attempting to recover OCT scanner: {self.device_id}")
            
            try:
                # Reset internal state
                self.scanning = False
                self.scan_data = None
                self.scan_progress = 0
                
                if self.simulation_mode:
                    # Simulate recovery
                    time.sleep(2)  # Simulate recovery time
                    
                    # Reset state
                    self.running = False
                    self.initialized = False
                    
                    # Reinitialize
                    init_result = self.initialize()
                    if not init_result.get('success', False):
                        return init_result
                    
                    # Restart
                    start_result = self.start()
                    if not start_result.get('success', False):
                        return start_result
                    
                    self.logger.info(f"OCT scanner {self.device_id} recovered in simulation mode")
                    
                    return {
                        'success': True,
                        'message': "Scanner recovered in simulation mode"
                    }
                
                # Close serial connection if open
                if self.serial_connection and self.serial_connection.is_open:
                    self.serial_connection.close()
                
                # Reset state
                self.running = False
                self.initialized = False
                
                # Wait before reconnecting
                time.sleep(2)
                
                # Reinitialize
                init_result = self.initialize()
                if not init_result.get('success', False):
                    return init_result
                
                # Restart
                start_result = self.start()
                if not start_result.get('success', False):
                    return start_result
                
                # Send explicit recovery command if needed
                recovery_command = {
                    'command': 'recover',
                    'timestamp': datetime.now().isoformat()
                }
                
                response = self._send_command(recovery_command)
                
                if response.get('status') != 'success':
                    error_msg = f"OCT scanner recovery command failed: {response.get('message', 'Unknown error')}"
                    self.logger.error(error_msg)
                    self.last_error = error_msg
                    return {'success': False, 'message': error_msg}
                
                self.logger.info(f"OCT scanner {self.device_id} recovered successfully")
                
                return {
                    'success': True,
                    'message': "Scanner recovered successfully"
                }
                
            except Exception as e:
                error_msg = f"OCT scanner recovery error: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                self.last_error = error_msg
                return {'success': False, 'message': error_msg}
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the OCT scanner.
        
        Returns:
            Dictionary with status information
        """
        with self.lock:
            status = {
                'device_id': self.device_id,
                'type': 'OCT',
                'initialized': self.initialized,
                'running': self.running,
                'scanning': self.scanning,
                'calibrated': self.calibrated,
                'scan_progress': self.scan_progress if self.scanning else None,
                'simulation_mode': self.simulation_mode,
                'last_error': self.last_error,
                'last_calibration_time': self.last_calibration_time.isoformat() if self.last_calibration_time else None,
                'specifications': {
                    'wavelength_nm': self.wavelength_nm,
                    'axial_resolution_um': self.axial_resolution_um,
                    'lateral_resolution_um': self.lateral_resolution_um,
                    'scan_depth_mm': self.scan_depth_mm,
                    'max_scan_area_mm': self.max_scan_area_mm
                }
            }
            
            if not self.simulation_mode and self.initialized and self.running:
                try:
                    # Get additional status from device
                    status_command = {
                        'command': 'get_status',
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    response = self._send_command(status_command)
                    
                    if response.get('status') == 'success':
                        status.update(response.get('device_status', {}))
                except Exception as e:
                    self.logger.error(f"Error getting additional device status: {str(e)}")
            
            return status
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics from the OCT scanner.
        
        Returns:
            Dictionary with performance metrics
        """
        with self.lock:
            # Calculate average scan time
            avg_scan_time = None
            if self.scan_times:
                avg_scan_time = sum(self.scan_times) / len(self.scan_times)
            
            # Calculate success rate
            success_rate = None
            if self.total_scans > 0:
                success_rate = self.successful_scans / self.total_scans
            
            metrics = {
                'device_id': self.device_id,
                'type': 'OCT',
                'total_scans': self.total_scans,
                'successful_scans': self.successful_scans,
                'scan_success_rate': success_rate,
                'average_scan_time_seconds': avg_scan_time,
                'last_scan_time_seconds': self.scan_times[-1] if self.scan_times else None,
                'simulation_mode': self.simulation_mode
            }
            
            if not self.simulation_mode and self.initialized and self.running:
                try:
                    # Get additional metrics from device
                    metrics_command = {
                        'command': 'get_metrics',
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    response = self._send_command(metrics_command)
                    
                    if response.get('status') == 'success':
                        metrics.update(response.get('device_metrics', {}))
                except Exception as e:
                    self.logger.error(f"Error getting additional device metrics: {str(e)}")
            
            return metrics
    
    def perform_scan(self, scan_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a 3D OCT scan of the ear.
        
        Args:
            scan_parameters: Dictionary of scan parameters
            
        Returns:
            Dictionary with scan results
        """
        with self.lock:
            if not self.initialized:
                error_msg = "Cannot perform scan: Scanner not initialized"
                self.logger.error(error_msg)
                return {'success': False, 'message': error_msg}
            
            if not self.running:
                error_msg = "Cannot perform scan: Scanner not running"
                self.logger.error(error_msg)
                return {'success': False, 'message': error_msg}
            
            if self.scanning:
                error_msg = "Cannot perform scan: Another scan is already in progress"
                self.logger.error(error_msg)
                return {'success': False, 'message': error_msg}
            
            # Extract scan parameters
            resolution = scan_parameters.get('resolution', 'high')
            scan_mode = scan_parameters.get('scan_mode', 'full')
            job_id = scan_parameters.get('job_id')
            patient_id = scan_parameters.get('patient_id')
            
            self.logger.info(f"Starting OCT scan for job {job_id}, patient {patient_id}, resolution: {resolution}, mode: {scan_mode}")
            
            # Update state
            self.scanning = True
            self.scan_progress = 0
            self.scan_data = None
            
            try:
                scan_start_time = time.time()
                self.total_scans += 1
                
                if self.simulation_mode:
                    # Simulate scanning process
                    total_steps = 100
                    for step in range(total_steps + 1):
                        self.scan_progress = step / total_steps
                        time.sleep(0.05)  # Simulate scanning time
                    
                    # Generate simulated scan data
                    self.scan_data = self._generate_simulated_scan_data(resolution, scan_mode)
                    
                    scan_end_time = time.time()
                    scan_duration = scan_end_time - scan_start_time
                    self.scan_times.append(scan_duration)
                    self.successful_scans += 1
                    
                    # Reset state
                    self.scanning = False
                    self.scan_progress = 1.0
                    
                    self.logger.info(f"OCT scan completed in simulation mode for job {job_id}, duration: {scan_duration:.2f} seconds")
                    
                    return {
                        'success': True,
                        'message': "Scan completed in simulation mode",
                        'scan_data': self.scan_data,
                        'duration_seconds': scan_duration
                    }
                
                # Prepare scan command
                scan_command = {
                    'command': 'perform_scan',
                    'parameters': {
                        'resolution': resolution,
                        'scan_mode': scan_mode,
                        'job_id': job_id,
                        'patient_id': patient_id
                    },
                    'timestamp': datetime.now().isoformat()
                }
                
                # Send scan command
                response = self._send_command(scan_command)
                
                if response.get('status') != 'success':
                    self.scanning = False
                    error_msg = f"OCT scan failed: {response.get('message', 'Unknown error')}"
                    self.logger.error(error_msg)
                    self.last_error = error_msg
                    return {'success': False, 'message': error_msg}
                
                # Monitor scan progress
                scan_id = response.get('scan_id')
                completed = False
                
                while not completed and self.scanning:
                    # Get scan status
                    status_command = {
                        'command': 'get_scan_status',
                        'scan_id': scan_id,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    status_response = self._send_command(status_command)
                    
                    if status_response.get('status') != 'success':
                        self.scanning = False
                        error_msg = f"Error checking scan status: {status_response.get('message', 'Unknown error')}"
                        self.logger.error(error_msg)
                        self.last_error = error_msg
                        return {'success': False, 'message': error_msg}
                    
                    # Update progress
                    self.scan_progress = status_response.get('progress', 0)
                    completed = status_response.get('completed', False)
                    
                    if not completed:
                        time.sleep(0.1)  # Wait before checking again
                
                # If scan was aborted
                if not self.scanning:
                    return {'success': False, 'message': "Scan aborted"}
                
                # Get scan data
                data_command = {
                    'command': 'get_scan_data',
                    'scan_id': scan_id,
                    'timestamp': datetime.now().isoformat()
                }
                
                data_response = self._send_command(data_command)
                
                if data_response.get('status') != 'success':
                    self.scanning = False
                    error_msg = f"Error retrieving scan data: {data_response.get('message', 'Unknown error')}"
                    self.logger.error(error_msg)
                    self.last_error = error_msg
                    return {'success': False, 'message': error_msg}
                
                # Extract scan data
                self.scan_data = data_response.get('scan_data')
                
                # Finalize scan
                scan_end_time = time.time()
                scan_duration = scan_end_time - scan_start_time
                self.scan_times.append(scan_duration)
                self.successful_scans += 1
                
                # Reset state
                self.scanning = False
                self.scan_progress = 1.0
                
                self.logger.info(f"OCT scan completed for job {job_id}, duration: {scan_duration:.2f} seconds")
                
                return {
                    'success': True,
                    'message': "Scan completed successfully",
                    'scan_data': self.scan_data,
                    'duration_seconds': scan_duration
                }
                
            except Exception as e:
                self.scanning = False
                error_msg = f"OCT scanner error during scan: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                self.last_error = error_msg
                return {'success': False, 'message': error_msg}
    
    def get_scan_preview(self) -> Dict[str, Any]:
        """
        Get a preview of the current scan.
        
        Returns:
            Dictionary with preview data
        """
        with self.lock:
            if not self.scanning:
                return {
                    'success': False,
                    'message': "No active scan in progress"
                }
            
            try:
                if self.simulation_mode:
                    # Generate simulated preview
                    preview_data = self._generate_simulated_preview()
                    
                    return {
                        'success': True,
                        'message': "Preview generated in simulation mode",
                        'preview_data': preview_data,
                        'progress': self.scan_progress
                    }
                
                # Send preview command
                preview_command = {
                    'command': 'get_scan_preview',
                    'timestamp': datetime.now().isoformat()
                }
                
                response = self._send_command(preview_command)
                
                if response.get('status') != 'success':
                    error_msg = f"Error getting scan preview: {response.get('message', 'Unknown error')}"
                    self.logger.warning(error_msg)
                    return {'success': False, 'message': error_msg}
                
                # Extract preview data
                preview_data = response.get('preview_data')
                
                return {
                    'success': True,
                    'message': "Preview generated successfully",
                    'preview_data': preview_data,
                    'progress': self.scan_progress
                }
                
            except Exception as e:
                error_msg = f"Error generating scan preview: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                return {'success': False, 'message': error_msg}
    
    def abort_scan(self) -> Dict[str, Any]:
        """
        Abort the current scan operation.
        
        Returns:
            Dictionary with abort results
        """
        with self.lock:
            if not self.scanning:
                return {
                    'success': True,
                    'message': "No active scan to abort"
                }
            
            self.logger.info(f"Aborting OCT scan")
            
            try:
                if self.simulation_mode:
                    # Simulate scan abort
                    time.sleep(0.5)  # Simulate abort time
                    self.scanning = False
                    self.scan_progress = 0
                    self.scan_data = None
                    
                    self.logger.info(f"OCT scan aborted in simulation mode")
                    
                    return {
                        'success': True,
                        'message': "Scan aborted in simulation mode"
                    }
                
                # Send abort command
                abort_command = {
                    'command': 'abort_scan',
                    'timestamp': datetime.now().isoformat()
                }
                
                response = self._send_command(abort_command)
                
                # Even if the command fails, we'll reset the internal state
                self.scanning = False
                self.scan_progress = 0
                self.scan_data = None
                
                if response.get('status') != 'success':
                    error_msg = f"OCT scan abort command failed: {response.get('message', 'Unknown error')}"
                    self.logger.warning(error_msg)
                    return {'success': False, 'message': error_msg}
                
                self.logger.info(f"OCT scan aborted successfully")
                
                return {
                    'success': True,
                    'message': "Scan aborted successfully"
                }
                
            except Exception as e:
                # Ensure scanning state is reset even on error
                self.scanning = False
                error_msg = f"Error aborting scan: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                return {'success': False, 'message': error_msg}
    
    def export_scan(self, format_type: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Export a scan in the specified format.
        
        Args:
            format_type: Type of format to export (STL, OBJ, etc.)
            options: Optional export options
            
        Returns:
            Dictionary with export results
        """
        with self.lock:
            if not self.scan_data:
                return {
                    'success': False,
                    'message': "No scan data available to export"
                }
            
            self.logger.info(f"Exporting OCT scan to {format_type} format")
            
            try:
                if self.simulation_mode:
                    # Simulate export
                    time.sleep(1)  # Simulate export time
                    
                    # Generate dummy export data
                    export_data = self._generate_simulated_export(format_type)
                    
                    self.logger.info(f"OCT scan exported to {format_type} in simulation mode")
                    
                    return {
                        'success': True,
                        'message': f"Scan exported to {format_type} in simulation mode",
                        'exported_data': export_data
                    }
                
                # Prepare export options
                export_options = options or {}
                
                # Send export command
                export_command = {
                    'command': 'export_scan',
                    'format': format_type,
                    'options': export_options,
                    'timestamp': datetime.now().isoformat()
                }
                
                response = self._send_command(export_command)
                
                if response.get('status') != 'success':
                    error_msg = f"OCT scan export failed: {response.get('message', 'Unknown error')}"
                    self.logger.error(error_msg)
                    return {'success': False, 'message': error_msg}
                
                # Extract exported data
                exported_data = response.get('exported_data')
                
                self.logger.info(f"OCT scan exported to {format_type} successfully")
                
                return {
                    'success': True,
                    'message': f"Scan exported to {format_type} successfully",
                    'exported_data': exported_data
                }
                
            except Exception as e:
                error_msg = f"Error exporting scan: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                return {'success': False, 'message': error_msg}
    
    def cleanup(self) -> Dict[str, Any]:
        """
        Clean up resources and prepare for shutdown.
        
        Returns:
            Dictionary with cleanup results
        """
        with self.lock:
            self.logger.info(f"Cleaning up OCT scanner resources")
            
            try:
                # Stop scanning if active
                if self.scanning:
                    self.abort_scan()
                
                # Stop scanner if running
                if self.running:
                    self.stop()
                
                if not self.simulation_mode:
                    # Close serial connection if open
                    if self.serial_connection and self.serial_connection.is_open:
                        self.serial_connection.close()
                
                # Reset state
                self.initialized = False
                self.running = False
                self.scanning = False
                self.calibrated = False
                self.scan_data = None
                self.scan_progress = 0
                
                self.logger.info(f"OCT scanner resources cleaned up successfully")
                
                return {
                    'success': True,
                    'message': "Scanner resources cleaned up successfully"
                }
                
            except Exception as e:
                error_msg = f"Error during scanner cleanup: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                return {'success': False, 'message': error_msg}
    
    def _send_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a command to the OCT scanner.
        
        Args:
            command: Command dictionary
            
        Returns:
            Response dictionary
        """
        if not self.serial_connection or not self.serial_connection.is_open:
            raise ScanningModuleError("Serial connection not established")
        
        try:
            # Convert command to JSON string
            command_str = json.dumps(command) + '\n'
            
            # Send command
            self.serial_connection.write(command_str.encode('utf-8'))
            
            # Read response
            response_str = self.serial_connection.readline().decode('utf-8').strip()
            
            if not response_str:
                raise ScanningModuleError("No response received from scanner")
            
            # Parse response
            try:
                response = json.loads(response_str)
                return response
            except json.JSONDecodeError:
                raise ScanningModuleError(f"Invalid JSON response: {response_str}")
            
        except serial.SerialTimeoutException:
            raise ScanningModuleError("Serial communication timeout")
        except serial.SerialException as e:
            raise ScanningModuleError(f"Serial communication error: {str(e)}")
    
    def _generate_simulated_scan_data(self, resolution: str, scan_mode: str) -> Dict[str, Any]:
        """
        Generate simulated scan data for testing.
        
        Args:
            resolution: Scan resolution
            scan_mode: Scan mode
            
        Returns:
            Simulated scan data
        """
        # Set dimensions based on resolution
        if resolution == 'high':
            dimensions = (512, 512, 256)
        elif resolution == 'medium':
            dimensions = (256, 256, 128)
        else:  # low
            dimensions = (128, 128, 64)
        
        # Create simulated point cloud
        num_points = dimensions[0] * dimensions[1] * 2  # Not all voxels have points
        
        # Generate ear canal shaped point cloud (simplified)
        points = []
        for i in range(num_points):
            # Cylindrical coordinates to simulate ear canal
            theta = np.random.uniform(0, 2 * np.pi)
            r = np.random.normal(5.0, 0.5)  # mm, radius with variation
            z = np.random.uniform(0, 20.0)  # mm, canal length
            
            # Add some constriction towards the inner part
            if z > 10:
                r = r * (1.0 - 0.03 * (z - 10))
            
            # Convert to Cartesian
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            # Add some noise
            x += np.random.normal(0, 0.05)
            y += np.random.normal(0, 0.05)
            z += np.random.normal(0, 0.05)
            
            points.append((x, y, z))
        
        # Create simulated intensity values
        intensities = np.random.uniform(0.2, 0.8, len(points))
        
        # Create metadata
        metadata = {
            'resolution': resolution,
            'scan_mode': scan_mode,
            'dimensions': dimensions,
            'voxel_size_um': 50 if resolution == 'high' else (100 if resolution == 'medium' else 200),
            'point_count': len(points),
            'scanner_type': 'OCT',
            'wavelength_nm': self.wavelength_nm,
            'scan_depth_mm': self.scan_depth_mm,
            'timestamp': datetime.now().isoformat()
        }
        
        # Return structured scan data
        return {
            'metadata': metadata,
            'points': points,
            'intensities': intensities.tolist(),
            'raw_data_path': None  # No raw data in simulation
        }
    
    def _generate_simulated_preview(self) -> Dict[str, Any]:
        """
        Generate a simulated scan preview.
        
        Returns:
            Simulated preview data
        """
        # Create a simple 2D preview image (100x100 grayscale)
        preview_image = np.random.uniform(0.2, 0.8, (100, 100))
        
        # Apply some structure to make it look like an ear canal
        center_x, center_y = 50, 50
        radius = 30
        
        # Create circular pattern for preview
        for x in range(100):
            for y in range(100):
                dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                if 20 < dist < radius:
                    preview_image[y, x] = 0.9 - (dist - 20) / (radius - 20) * 0.5
        
        # Add timestamp
        timestamp = datetime.now().isoformat()
        
        return {
            'preview_image': preview_image.tolist(),
            'dimensions': (100, 100),
            'progress': self.scan_progress,
            'timestamp': timestamp
        }
    
    def _generate_simulated_export(self, format_type: str) -> Dict[str, Any]:
        """
        Generate simulated export data.
        
        Args:
            format_type: Export format type
            
        Returns:
            Simulated export data
        """
        # Create dummy export data based on format
        if format_type.lower() == 'stl':
            # Placeholder for STL binary data
            data = b'\x00' * 1000  # Just a placeholder
            return {
                'format': 'stl',
                'binary': True,
                'data': data.hex(),
                'vertex_count': 500,
                'triangle_count': 1000,
                'timestamp': datetime.now().isoformat()
            }
        elif format_type.lower() == 'obj':
            # Placeholder for OBJ text data
            data = "# OBJ file\n# Simulated export\n"
            for i in range(100):
                data += f"v {np.random.uniform(-5, 5)} {np.random.uniform(-5, 5)} {np.random.uniform(0, 20)}\n"
            for i in range(50):
                data += f"f {np.random.randint(1, 100)} {np.random.randint(1, 100)} {np.random.randint(1, 100)}\n"
            
            return {
                'format': 'obj',
                'binary': False,
                'data': data,
                'vertex_count': 100,
                'face_count': 50,
                'timestamp': datetime.now().isoformat()
            }
        else:
            # Generic format
            return {
                'format': format_type,
                'binary': False,
                'data': f"Simulated export data for {format_type} format",
                'timestamp': datetime.now().isoformat()
            }
