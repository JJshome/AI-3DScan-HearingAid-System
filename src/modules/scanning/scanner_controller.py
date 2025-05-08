#!/usr/bin/env python3
"""
Scanner Controller for the 3D Scanning Module of the AI-Enhanced 3D Scanning Hearing Aid Manufacturing System.
Provides high-level control and coordination of ear scanning operations.

Author: AI-Enhanced 3D Scanning Hearing Aid System Team
Date: May 8, 2025
"""

import os
import sys
import logging
import threading
import time
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np

# Import scanner driver interfaces
from src.modules.scanning.scanner_interface import ScannerInterface
from src.modules.scanning.drivers.oct_scanner_driver import OCTScannerDriver
from src.modules.scanning.drivers.lidar_scanner_driver import LiDARScannerDriver
from src.modules.scanning.drivers.photogrammetry_driver import PhotogrammetryDriver

# Import scan processing components
from src.modules.scanning.processing.point_cloud_processor import PointCloudProcessor
from src.modules.scanning.processing.mesh_generator import MeshGenerator
from src.modules.scanning.processing.anatomical_feature_detector import AnatomicalFeatureDetector
from src.modules.scanning.processing.quality_analyzer import QualityAnalyzer

# Import utility modules
from src.utils.exceptions import ScanningModuleError, CalibrationError, HardwareError

class ScannerController:
    """
    Controller for the 3D Scanning Module.
    Manages scanner hardware, processing, and integration with the overall system.
    """
    
    def __init__(self, config):
        """
        Initialize the scanner controller.
        
        Args:
            config: Scanner module configuration
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.initialized = False
        self.running = False
        self.scanner = None
        self.active_scan = None
        self.lock = threading.RLock()
        self.scan_history = {}
        
        # Initialize processing components
        self.point_cloud_processor = None
        self.mesh_generator = None
        self.feature_detector = None
        self.quality_analyzer = None
        
        self.logger.info("Scanner Controller created")
    
    def initialize(self) -> Dict[str, Any]:
        """
        Initialize the scanner controller and hardware.
        
        Returns:
            Dictionary with initialization results
        """
        if self.initialized:
            self.logger.warning("Scanner Controller is already initialized")
            return {'success': True, 'message': "Already initialized"}
        
        self.logger.info("Initializing Scanner Controller")
        
        try:
            # Initialize scanner hardware based on configuration
            scanner_type = self.config.get('scanner_type', 'oct')
            
            if scanner_type == 'oct':
                self.scanner = OCTScannerDriver(self.config.get('oct_scanner', {}))
            elif scanner_type == 'lidar':
                self.scanner = LiDARScannerDriver(self.config.get('lidar_scanner', {}))
            elif scanner_type == 'photogrammetry':
                self.scanner = PhotogrammetryDriver(self.config.get('photogrammetry', {}))
            else:
                error_msg = f"Unsupported scanner type: {scanner_type}"
                self.logger.error(error_msg)
                return {'success': False, 'message': error_msg}
            
            # Initialize scanner hardware
            scanner_init_result = self.scanner.initialize()
            if not scanner_init_result.get('success', False):
                error_msg = f"Failed to initialize scanner hardware: {scanner_init_result.get('message', 'Unknown error')}"
                self.logger.error(error_msg)
                return {'success': False, 'message': error_msg}
            
            # Initialize processing components
            self.point_cloud_processor = PointCloudProcessor(self.config.get('processing', {}))
            self.mesh_generator = MeshGenerator(self.config.get('mesh_generation', {}))
            self.feature_detector = AnatomicalFeatureDetector(self.config.get('feature_detection', {}))
            self.quality_analyzer = QualityAnalyzer(self.config.get('quality_analysis', {}))
            
            self.initialized = True
            self.logger.info("Scanner Controller initialized successfully")
            
            return {
                'success': True,
                'message': "Scanner Controller initialized successfully",
                'scanner_type': scanner_type
            }
        
        except Exception as e:
            error_msg = f"Error initializing Scanner Controller: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {'success': False, 'message': error_msg}
    
    def start(self) -> Dict[str, Any]:
        """
        Start the scanner system.
        
        Returns:
            Dictionary with start results
        """
        with self.lock:
            if not self.initialized:
                error_msg = "Cannot start: Scanner Controller is not initialized"
                self.logger.error(error_msg)
                return {'success': False, 'message': error_msg}
            
            if self.running:
                self.logger.warning("Scanner Controller is already running")
                return {'success': True, 'message': "Already running"}
            
            self.logger.info("Starting Scanner Controller")
            
            try:
                # Start the scanner hardware
                start_result = self.scanner.start()
                if not start_result.get('success', False):
                    error_msg = f"Failed to start scanner hardware: {start_result.get('message', 'Unknown error')}"
                    self.logger.error(error_msg)
                    return {'success': False, 'message': error_msg}
                
                # Calibrate the scanner if configured to do so on startup
                if self.config.get('calibrate_on_start', True):
                    calibration_result = self.calibrate()
                    if not calibration_result.get('success', False):
                        error_msg = f"Failed to calibrate scanner: {calibration_result.get('message', 'Unknown error')}"
                        self.logger.error(error_msg)
                        # Attempt to continue even if calibration fails
                
                self.running = True
                self.logger.info("Scanner Controller started successfully")
                
                return {
                    'success': True,
                    'message': "Scanner Controller started successfully"
                }
            
            except Exception as e:
                error_msg = f"Error starting Scanner Controller: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                return {'success': False, 'message': error_msg}
    
    def stop(self) -> Dict[str, Any]:
        """
        Stop the scanner system.
        
        Returns:
            Dictionary with stop results
        """
        with self.lock:
            if not self.running:
                self.logger.warning("Scanner Controller is not running")
                return {'success': True, 'message': "Not running"}
            
            self.logger.info("Stopping Scanner Controller")
            
            try:
                # Stop any active scan
                if self.active_scan:
                    self.abort_scan()
                
                # Stop the scanner hardware
                stop_result = self.scanner.stop()
                if not stop_result.get('success', False):
                    error_msg = f"Failed to stop scanner hardware: {stop_result.get('message', 'Unknown error')}"
                    self.logger.error(error_msg)
                    return {'success': False, 'message': error_msg}
                
                self.running = False
                self.logger.info("Scanner Controller stopped successfully")
                
                return {
                    'success': True,
                    'message': "Scanner Controller stopped successfully"
                }
            
            except Exception as e:
                error_msg = f"Error stopping Scanner Controller: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                return {'success': False, 'message': error_msg}
    
    def calibrate(self) -> Dict[str, Any]:
        """
        Calibrate the scanner for optimal performance.
        
        Returns:
            Dictionary with calibration results
        """
        with self.lock:
            if not self.initialized:
                error_msg = "Cannot calibrate: Scanner Controller is not initialized"
                self.logger.error(error_msg)
                return {'success': False, 'message': error_msg}
            
            self.logger.info("Calibrating scanner")
            
            try:
                # Perform scanner calibration
                calibration_result = self.scanner.calibrate()
                if not calibration_result.get('success', False):
                    error_msg = f"Calibration failed: {calibration_result.get('message', 'Unknown error')}"
                    self.logger.error(error_msg)
                    return {'success': False, 'message': error_msg}
                
                self.logger.info("Scanner calibrated successfully")
                
                return {
                    'success': True,
                    'message': "Scanner calibrated successfully",
                    'calibration_data': calibration_result.get('calibration_data', {})
                }
            
            except Exception as e:
                error_msg = f"Error during scanner calibration: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                return {'success': False, 'message': error_msg}
    
    def check_health(self) -> Dict[str, Any]:
        """
        Check the health and operational status of the scanner system.
        
        Returns:
            Dictionary with health check results
        """
        with self.lock:
            if not self.initialized:
                return {
                    'healthy': False,
                    'message': "Scanner Controller is not initialized"
                }
            
            try:
                # Check scanner hardware health
                scanner_health = self.scanner.check_health()
                
                # Check overall system health
                system_healthy = (
                    scanner_health.get('healthy', False) and
                    self.point_cloud_processor is not None and
                    self.mesh_generator is not None and
                    self.feature_detector is not None and
                    self.quality_analyzer is not None
                )
                
                if not system_healthy:
                    message = f"System health check failed: {scanner_health.get('message', 'Unknown issue')}"
                    self.logger.warning(message)
                    return {
                        'healthy': False,
                        'message': message,
                        'scanner_health': scanner_health,
                        'processing_components_healthy': (
                            self.point_cloud_processor is not None and
                            self.mesh_generator is not None and
                            self.feature_detector is not None and
                            self.quality_analyzer is not None
                        )
                    }
                
                return {
                    'healthy': True,
                    'message': "All systems operational",
                    'scanner_health': scanner_health
                }
            
            except Exception as e:
                error_msg = f"Error checking scanner health: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                return {
                    'healthy': False,
                    'message': error_msg
                }
    
    def recover(self) -> Dict[str, Any]:
        """
        Attempt to recover the scanner system from an error state.
        
        Returns:
            Dictionary with recovery results
        """
        with self.lock:
            self.logger.info("Attempting to recover scanner system")
            
            try:
                # Reset active scan if any
                if self.active_scan:
                    self.active_scan = None
                
                # Attempt to stop the scanner
                try:
                    self.scanner.stop()
                except Exception:
                    # Ignore errors during stop in recovery mode
                    pass
                
                # Reinitialize the scanner
                init_result = self.scanner.initialize()
                if not init_result.get('success', False):
                    error_msg = f"Failed to reinitialize scanner during recovery: {init_result.get('message', 'Unknown error')}"
                    self.logger.error(error_msg)
                    return {'success': False, 'message': error_msg}
                
                # Restart the scanner
                start_result = self.scanner.start()
                if not start_result.get('success', False):
                    error_msg = f"Failed to restart scanner during recovery: {start_result.get('message', 'Unknown error')}"
                    self.logger.error(error_msg)
                    return {'success': False, 'message': error_msg}
                
                # Calibrate the scanner
                calibration_result = self.scanner.calibrate()
                if not calibration_result.get('success', False):
                    error_msg = f"Failed to calibrate scanner during recovery: {calibration_result.get('message', 'Unknown error')}"
                    self.logger.error(error_msg)
                    # Continue even if calibration fails
                
                # Update state
                self.running = True
                
                self.logger.info("Scanner system recovered successfully")
                
                return {
                    'success': True,
                    'message': "Scanner system recovered successfully"
                }
            
            except Exception as e:
                error_msg = f"Error during scanner recovery: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                return {'success': False, 'message': error_msg}
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the scanner system.
        
        Returns:
            Dictionary with status information
        """
        with self.lock:
            status = {
                'initialized': self.initialized,
                'running': self.running,
                'active_scan': self.active_scan is not None,
                'scanner_type': self.config.get('scanner_type', 'unknown')
            }
            
            if self.initialized:
                # Get scanner hardware status
                try:
                    scanner_status = self.scanner.get_status()
                    status['scanner'] = scanner_status
                except Exception as e:
                    self.logger.error(f"Error getting scanner status: {str(e)}")
                    status['scanner'] = {'error': str(e)}
            
            return {
                'status': 'running' if self.running else 'stopped',
                'details': status
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics from the scanner system.
        
        Returns:
            Dictionary with performance metrics
        """
        with self.lock:
            try:
                metrics = {
                    'timestamp': datetime.now().isoformat(),
                    'scan_count': len(self.scan_history),
                    'active_scan': self.active_scan is not None
                }
                
                # Get scanner hardware metrics
                if self.initialized:
                    scanner_metrics = self.scanner.get_metrics()
                    metrics['scanner'] = scanner_metrics
                
                # Calculate scan success rate
                if metrics['scan_count'] > 0:
                    successful_scans = sum(1 for scan in self.scan_history.values() 
                                          if scan.get('success', False))
                    metrics['scan_success_rate'] = successful_scans / metrics['scan_count']
                else:
                    metrics['scan_success_rate'] = None
                
                # Calculate average scan time
                if metrics['scan_count'] > 0:
                    scan_times = [scan.get('duration_seconds', 0) for scan in self.scan_history.values() 
                                 if 'duration_seconds' in scan]
                    if scan_times:
                        metrics['average_scan_time'] = sum(scan_times) / len(scan_times)
                    else:
                        metrics['average_scan_time'] = None
                else:
                    metrics['average_scan_time'] = None
                
                return metrics
            
            except Exception as e:
                self.logger.error(f"Error getting scanner metrics: {str(e)}", exc_info=True)
                return {
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e)
                }
    
    def perform_scan(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a 3D scan of the ear for a specific job.
        
        Args:
            job: Job data dictionary containing patient information and scan requirements
            
        Returns:
            Dictionary with scan results
        """
        with self.lock:
            if not self.running:
                error_msg = "Cannot perform scan: Scanner is not running"
                self.logger.error(error_msg)
                return {'success': False, 'message': error_msg}
            
            if self.active_scan:
                error_msg = "Cannot start new scan: Another scan is already in progress"
                self.logger.error(error_msg)
                return {'success': False, 'message': error_msg}
            
            # Extract job information
            job_id = job.get('job_id')
            patient_id = job.get('patient_id')
            
            if not job_id or not patient_id:
                error_msg = "Missing required job information (job_id or patient_id)"
                self.logger.error(error_msg)
                return {'success': False, 'message': error_msg}
            
            self.logger.info(f"Starting ear scan for job {job_id}, patient {patient_id}")
            
            # Create scan parameters
            scan_parameters = {
                'job_id': job_id,
                'patient_id': patient_id,
                'resolution': self.config.get('default_resolution', 'high'),
                'scan_mode': job.get('scan_mode', self.config.get('default_scan_mode', 'full')),
                'timestamp': datetime.now().isoformat()
            }
            
            # Update custom scan parameters if specified in job
            if 'scan_parameters' in job:
                scan_parameters.update(job['scan_parameters'])
            
            # Set active scan
            self.active_scan = {
                'job_id': job_id,
                'patient_id': patient_id,
                'start_time': datetime.now(),
                'parameters': scan_parameters
            }
            
            try:
                # Perform the scan
                scan_start_time = time.time()
                scan_result = self.scanner.perform_scan(scan_parameters)
                scan_end_time = time.time()
                scan_duration = scan_end_time - scan_start_time
                
                if not scan_result.get('success', False):
                    error_msg = f"Scan failed: {scan_result.get('message', 'Unknown error')}"
                    self.logger.error(error_msg)
                    
                    # Add to scan history
                    self.scan_history[job_id] = {
                        'job_id': job_id,
                        'patient_id': patient_id,
                        'timestamp': scan_parameters['timestamp'],
                        'success': False,
                        'error': error_msg,
                        'duration_seconds': scan_duration
                    }
                    
                    self.active_scan = None
                    return {'success': False, 'message': error_msg}
                
                # Get raw scan data
                raw_scan_data = scan_result.get('scan_data')
                
                # Process the point cloud
                point_cloud = self.point_cloud_processor.process(raw_scan_data)
                
                # Generate mesh
                mesh = self.mesh_generator.generate_mesh(point_cloud)
                
                # Detect anatomical features
                features = self.feature_detector.detect_features(mesh)
                
                # Analyze quality
                quality_result = self.quality_analyzer.analyze_quality(mesh, features)
                
                # Check quality thresholds
                if quality_result.get('quality_score', 0) < self.config.get('minimum_quality_score', 70):
                    warning_msg = f"Scan quality below threshold: {quality_result.get('quality_score')}"
                    self.logger.warning(warning_msg)
                    
                    # Return partial success with warning
                    result = {
                        'success': True,
                        'warning': warning_msg,
                        'scan_data': {
                            'job_id': job_id,
                            'patient_id': patient_id,
                            'timestamp': scan_parameters['timestamp'],
                            'point_cloud': point_cloud,
                            'mesh': mesh,
                            'features': features,
                            'quality': quality_result
                        },
                        'scan_parameters': scan_parameters,
                        'quality_score': quality_result.get('quality_score'),
                        'duration_seconds': scan_duration
                    }
                else:
                    # Return success
                    result = {
                        'success': True,
                        'message': "Scan completed successfully",
                        'scan_data': {
                            'job_id': job_id,
                            'patient_id': patient_id,
                            'timestamp': scan_parameters['timestamp'],
                            'point_cloud': point_cloud,
                            'mesh': mesh,
                            'features': features,
                            'quality': quality_result
                        },
                        'scan_parameters': scan_parameters,
                        'quality_score': quality_result.get('quality_score'),
                        'duration_seconds': scan_duration
                    }
                
                # Add to scan history
                self.scan_history[job_id] = {
                    'job_id': job_id,
                    'patient_id': patient_id,
                    'timestamp': scan_parameters['timestamp'],
                    'success': True,
                    'quality_score': quality_result.get('quality_score'),
                    'duration_seconds': scan_duration
                }
                
                # Save scan data to permanent storage
                self._save_scan_data(job_id, result['scan_data'])
                
                # Clear active scan
                self.active_scan = None
                
                self.logger.info(f"Scan completed for job {job_id}, quality score: {quality_result.get('quality_score')}")
                
                # Update job data for next phase
                job_data = {
                    'scan_complete': True,
                    'scan_quality_score': quality_result.get('quality_score'),
                    'scan_timestamp': scan_parameters['timestamp'],
                    'scan_data_path': self._get_scan_data_path(job_id)
                }
                
                # Return combined result with job data for the next module
                return {
                    'success': True,
                    'message': "Scan completed successfully",
                    'job_data': job_data
                }
            
            except Exception as e:
                error_msg = f"Error during scan: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                
                # Add to scan history
                if 'scan_parameters' in locals():
                    self.scan_history[job_id] = {
                        'job_id': job_id,
                        'patient_id': patient_id,
                        'timestamp': scan_parameters['timestamp'],
                        'success': False,
                        'error': str(e),
                        'duration_seconds': time.time() - scan_start_time if 'scan_start_time' in locals() else None
                    }
                
                # Clear active scan
                self.active_scan = None
                
                return {'success': False, 'message': error_msg}
    
    def get_scan_preview(self) -> Dict[str, Any]:
        """
        Get a preview of the current scan.
        
        Returns:
            Dictionary with preview data
        """
        with self.lock:
            if not self.active_scan:
                return {
                    'success': False, 
                    'message': "No active scan in progress"
                }
            
            try:
                preview_result = self.scanner.get_scan_preview()
                return preview_result
            
            except Exception as e:
                error_msg = f"Error getting scan preview: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                return {'success': False, 'message': error_msg}
    
    def abort_scan(self) -> Dict[str, Any]:
        """
        Abort the current scan operation.
        
        Returns:
            Dictionary with abort results
        """
        with self.lock:
            if not self.active_scan:
                return {
                    'success': True, 
                    'message': "No active scan to abort"
                }
            
            job_id = self.active_scan.get('job_id')
            self.logger.info(f"Aborting scan for job {job_id}")
            
            try:
                # Abort the scanner hardware operation
                abort_result = self.scanner.abort_scan()
                
                # Clear active scan regardless of hardware abort result
                self.active_scan = None
                
                return {
                    'success': True,
                    'message': "Scan aborted successfully",
                    'hardware_result': abort_result
                }
            
            except Exception as e:
                error_msg = f"Error aborting scan: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                
                # Still clear active scan on error
                self.active_scan = None
                
                return {'success': False, 'message': error_msg}
    
    def export_scan(self, job_id: str, format_type: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Export a scan in the specified format.
        
        Args:
            job_id: ID of the job/scan to export
            format_type: Type of format to export (STL, OBJ, etc.)
            options: Optional export options
            
        Returns:
            Dictionary with export results
        """
        with self.lock:
            if job_id not in self.scan_history:
                return {
                    'success': False,
                    'message': f"No scan found for job {job_id}"
                }
            
            try:
                # Get scan data path
                scan_data_path = self._get_scan_data_path(job_id)
                
                # Load scan data
                with open(scan_data_path, 'r') as f:
                    scan_data = json.load(f)
                
                # Export using the scanner
                export_options = options or {}
                export_result = self.scanner.export_scan(format_type, {
                    'scan_data': scan_data,
                    **export_options
                })
                
                return export_result
            
            except Exception as e:
                error_msg = f"Error exporting scan: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                return {'success': False, 'message': error_msg}
    
    def _save_scan_data(self, job_id: str, scan_data: Dict[str, Any]) -> bool:
        """
        Save scan data to permanent storage.
        
        Args:
            job_id: ID of the job
            scan_data: Scan data dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create data directory if it doesn't exist
            data_dir = os.path.join(self.config.get('data_directory', './data'), 'scans')
            os.makedirs(data_dir, exist_ok=True)
            
            # Create file path
            file_path = os.path.join(data_dir, f"{job_id}_scan.json")
            
            # Save data
            with open(file_path, 'w') as f:
                json.dump(scan_data, f)
            
            self.logger.info(f"Scan data saved to {file_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error saving scan data: {str(e)}", exc_info=True)
            return False
    
    def _get_scan_data_path(self, job_id: str) -> str:
        """
        Get the path to the saved scan data.
        
        Args:
            job_id: ID of the job
            
        Returns:
            Path to the scan data file
        """
        data_dir = os.path.join(self.config.get('data_directory', './data'), 'scans')
        return os.path.join(data_dir, f"{job_id}_scan.json")
    
    def cleanup(self) -> Dict[str, Any]:
        """
        Clean up resources and prepare for shutdown.
        
        Returns:
            Dictionary with cleanup results
        """
        with self.lock:
            self.logger.info("Cleaning up Scanner Controller resources")
            
            try:
                # Stop if running
                if self.running:
                    stop_result = self.stop()
                    if not stop_result.get('success', False):
                        self.logger.warning(f"Failed to stop scanner during cleanup: {stop_result.get('message', 'Unknown error')}")
                
                # Clean up scanner hardware
                if self.scanner:
                    cleanup_result = self.scanner.cleanup()
                    if not cleanup_result.get('success', False):
                        self.logger.warning(f"Scanner hardware cleanup reported issues: {cleanup_result.get('message', 'Unknown error')}")
                
                # Reset state
                self.initialized = False
                self.running = False
                self.active_scan = None
                
                self.logger.info("Scanner Controller cleanup completed")
                
                return {
                    'success': True,
                    'message': "Scanner Controller cleanup completed"
                }
            
            except Exception as e:
                error_msg = f"Error during Scanner Controller cleanup: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                return {'success': False, 'message': error_msg}
