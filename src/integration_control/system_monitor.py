#!/usr/bin/env python3
"""
System Monitor for the AI-Enhanced 3D Scanning Hearing Aid Manufacturing System.
Monitors system performance, resource usage, and provides real-time status updates.

Author: AI-Enhanced 3D Scanning Hearing Aid System Team
Date: May 8, 2025
"""

import os
import sys
import logging
import threading
import time
import json
import psutil
import platform
from datetime import datetime
from typing import Dict, Any, List, Optional

class SystemMonitor:
    """
    System Monitor for the AI-Enhanced 3D Scanning Hearing Aid Manufacturing System.
    Monitors system resources, module performance, and provides real-time status information.
    """
    
    def __init__(self, integration_controller):
        """
        Initialize the system monitor.
        
        Args:
            integration_controller: Integration controller instance
        """
        self.logger = logging.getLogger(__name__)
        self.integration_controller = integration_controller
        self.running = False
        self.performance_history = {}
        self.system_metrics = {}
        self.lock = threading.RLock()
        
        # Set monitoring intervals
        self.system_metrics_interval = 5  # seconds
        self.module_metrics_interval = 10  # seconds
        self.alert_check_interval = 15  # seconds
        
        # Set resource thresholds
        self.set_default_thresholds()
        
        self.logger.info("System Monitor initialized")
    
    def set_default_thresholds(self):
        """Set default threshold values for system monitoring."""
        self.thresholds = {
            'cpu_percent': 90.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0,
            'job_queue_size': 50,
            'processing_time_multiplier': 2.0  # Multiple of expected processing time
        }
    
    def start(self):
        """Start the system monitor threads."""
        if self.running:
            self.logger.warning("System Monitor is already running")
            return
        
        self.logger.info("Starting System Monitor")
        self.running = True
        
        # Start system metrics monitoring thread
        self.system_metrics_thread = threading.Thread(
            target=self._system_metrics_loop,
            name="SystemMetricsMonitor"
        )
        self.system_metrics_thread.daemon = True
        self.system_metrics_thread.start()
        
        # Start module metrics monitoring thread
        self.module_metrics_thread = threading.Thread(
            target=self._module_metrics_loop,
            name="ModuleMetricsMonitor"
        )
        self.module_metrics_thread.daemon = True
        self.module_metrics_thread.start()
        
        # Start alert monitoring thread
        self.alert_thread = threading.Thread(
            target=self._alert_check_loop,
            name="AlertMonitor"
        )
        self.alert_thread.daemon = True
        self.alert_thread.start()
        
        self.logger.info("System Monitor started")
    
    def stop(self):
        """Stop the system monitor threads."""
        if not self.running:
            self.logger.warning("System Monitor is not running")
            return
        
        self.logger.info("Stopping System Monitor")
        self.running = False
        
        # Wait for threads to finish
        if hasattr(self, 'system_metrics_thread') and self.system_metrics_thread.is_alive():
            self.system_metrics_thread.join(timeout=5)
        
        if hasattr(self, 'module_metrics_thread') and self.module_metrics_thread.is_alive():
            self.module_metrics_thread.join(timeout=5)
        
        if hasattr(self, 'alert_thread') and self.alert_thread.is_alive():
            self.alert_thread.join(timeout=5)
        
        self.logger.info("System Monitor stopped")
    
    def _system_metrics_loop(self):
        """Main loop for monitoring system metrics."""
        self.logger.info("System metrics monitoring thread started")
        
        while self.running:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Sleep until next collection
                time.sleep(self.system_metrics_interval)
            
            except Exception as e:
                self.logger.error(f"Error in system metrics loop: {str(e)}", exc_info=True)
                time.sleep(5)  # Sleep longer on error
        
        self.logger.info("System metrics monitoring thread stopped")
    
    def _module_metrics_loop(self):
        """Main loop for monitoring module metrics."""
        self.logger.info("Module metrics monitoring thread started")
        
        while self.running:
            try:
                # Collect module metrics
                self._collect_module_metrics()
                
                # Sleep until next collection
                time.sleep(self.module_metrics_interval)
            
            except Exception as e:
                self.logger.error(f"Error in module metrics loop: {str(e)}", exc_info=True)
                time.sleep(5)  # Sleep longer on error
        
        self.logger.info("Module metrics monitoring thread stopped")
    
    def _alert_check_loop(self):
        """Main loop for checking and generating alerts."""
        self.logger.info("Alert monitoring thread started")
        
        while self.running:
            try:
                # Check for alert conditions
                self._check_for_alerts()
                
                # Sleep until next check
                time.sleep(self.alert_check_interval)
            
            except Exception as e:
                self.logger.error(f"Error in alert check loop: {str(e)}", exc_info=True)
                time.sleep(5)  # Sleep longer on error
        
        self.logger.info("Alert monitoring thread stopped")
    
    def _collect_system_metrics(self):
        """Collect system-wide performance metrics."""
        try:
            # Collect CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Collect memory metrics
            memory = psutil.virtual_memory()
            
            # Collect disk metrics
            disk = psutil.disk_usage('/')
            
            # Collect network metrics
            network = psutil.net_io_counters()
            
            # Collect process metrics
            process = psutil.Process(os.getpid())
            process_memory = process.memory_info()
            
            # Update system metrics
            with self.lock:
                self.system_metrics = {
                    'timestamp': datetime.now().isoformat(),
                    'system': {
                        'hostname': platform.node(),
                        'platform': platform.platform(),
                        'python_version': platform.python_version(),
                        'uptime': time.time() - psutil.boot_time()
                    },
                    'cpu': {
                        'percent': cpu_percent,
                        'count': cpu_count,
                        'frequency_mhz': cpu_freq.current if cpu_freq else None
                    },
                    'memory': {
                        'total_gb': memory.total / (1024 ** 3),
                        'available_gb': memory.available / (1024 ** 3),
                        'used_gb': memory.used / (1024 ** 3),
                        'percent': memory.percent
                    },
                    'disk': {
                        'total_gb': disk.total / (1024 ** 3),
                        'used_gb': disk.used / (1024 ** 3),
                        'free_gb': disk.free / (1024 ** 3),
                        'percent': disk.percent
                    },
                    'network': {
                        'bytes_sent': network.bytes_sent,
                        'bytes_recv': network.bytes_recv,
                        'packets_sent': network.packets_sent,
                        'packets_recv': network.packets_recv
                    },
                    'process': {
                        'pid': process.pid,
                        'memory_rss_mb': process_memory.rss / (1024 ** 2),
                        'memory_vms_mb': process_memory.vms / (1024 ** 2),
                        'cpu_percent': process.cpu_percent(interval=None),
                        'threads': process.num_threads(),
                        'open_files': len(process.open_files()),
                        'connections': len(process.connections())
                    }
                }
            
            # Log resource usage if it exceeds thresholds
            self._log_resource_warnings()
        
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {str(e)}", exc_info=True)
    
    def _collect_module_metrics(self):
        """Collect performance metrics from all active modules."""
        try:
            # Get module status from integration controller
            module_status = self.integration_controller.get_module_status()
            
            # Collect metrics for each module
            module_metrics = {}
            for module_name, status_info in module_status.items():
                # Skip modules that aren't running
                if status_info['status'] != 'RUNNING':
                    continue
                
                # Get detailed metrics from module if available
                try:
                    module = self.integration_controller.modules.get(module_name)
                    if module and hasattr(module, 'get_metrics'):
                        metrics = module.get_metrics()
                        module_metrics[module_name] = metrics
                except Exception as e:
                    self.logger.error(f"Error getting metrics from module {module_name}: {str(e)}")
            
            # Update performance history
            timestamp = datetime.now().isoformat()
            with self.lock:
                self.performance_history[timestamp] = {
                    'modules': module_metrics,
                    'system': self.system_metrics
                }
                
                # Keep only the latest 100 entries to avoid memory bloat
                if len(self.performance_history) > 100:
                    oldest_key = min(self.performance_history.keys())
                    del self.performance_history[oldest_key]
        
        except Exception as e:
            self.logger.error(f"Error collecting module metrics: {str(e)}", exc_info=True)
    
    def _log_resource_warnings(self):
        """Log warnings for system resources exceeding thresholds."""
        try:
            with self.lock:
                # Check CPU usage
                if self.system_metrics.get('cpu', {}).get('percent', 0) > self.thresholds['cpu_percent']:
                    self.logger.warning(f"High CPU usage: {self.system_metrics['cpu']['percent']}% (threshold: {self.thresholds['cpu_percent']}%)")
                
                # Check memory usage
                if self.system_metrics.get('memory', {}).get('percent', 0) > self.thresholds['memory_percent']:
                    self.logger.warning(f"High memory usage: {self.system_metrics['memory']['percent']}% (threshold: {self.thresholds['memory_percent']}%)")
                
                # Check disk usage
                if self.system_metrics.get('disk', {}).get('percent', 0) > self.thresholds['disk_percent']:
                    self.logger.warning(f"High disk usage: {self.system_metrics['disk']['percent']}% (threshold: {self.thresholds['disk_percent']}%)")
        
        except Exception as e:
            self.logger.error(f"Error logging resource warnings: {str(e)}")
    
    def _check_for_alerts(self):
        """Check for conditions that should trigger alerts."""
        try:
            alerts = []
            
            # Check system metrics for alert conditions
            with self.lock:
                # Critical CPU usage
                if self.system_metrics.get('cpu', {}).get('percent', 0) > 95:
                    alerts.append({
                        'level': 'critical',
                        'type': 'system',
                        'message': f"Critical CPU usage: {self.system_metrics['cpu']['percent']}%"
                    })
                
                # Critical memory usage
                if self.system_metrics.get('memory', {}).get('percent', 0) > 95:
                    alerts.append({
                        'level': 'critical',
                        'type': 'system',
                        'message': f"Critical memory usage: {self.system_metrics['memory']['percent']}%"
                    })
                
                # Critical disk usage
                if self.system_metrics.get('disk', {}).get('percent', 0) > 95:
                    alerts.append({
                        'level': 'critical',
                        'type': 'system',
                        'message': f"Critical disk usage: {self.system_metrics['disk']['percent']}%"
                    })
            
            # Check module status for alert conditions
            module_status = self.integration_controller.get_module_status()
            for module_name, status_info in module_status.items():
                if status_info['status'] == 'ERROR':
                    alerts.append({
                        'level': 'critical',
                        'type': 'module',
                        'module': module_name,
                        'message': f"Module {module_name} in ERROR state"
                    })
                elif status_info['status'] == 'WARNING':
                    alerts.append({
                        'level': 'warning',
                        'type': 'module',
                        'module': module_name,
                        'message': f"Module {module_name} in WARNING state"
                    })
            
            # Log and handle alerts
            for alert in alerts:
                self._handle_alert(alert)
        
        except Exception as e:
            self.logger.error(f"Error checking for alerts: {str(e)}", exc_info=True)
    
    def _handle_alert(self, alert):
        """
        Handle an alert by logging and taking appropriate action.
        
        Args:
            alert: Alert information dictionary
        """
        # Log the alert
        if alert['level'] == 'critical':
            self.logger.critical(f"ALERT: {alert['message']}")
        elif alert['level'] == 'warning':
            self.logger.warning(f"ALERT: {alert['message']}")
        else:
            self.logger.info(f"ALERT: {alert['message']}")
        
        # Take action based on alert type
        if alert['type'] == 'module' and alert['level'] == 'critical':
            # Attempt module recovery for critical module alerts
            if alert.get('module'):
                self._request_module_recovery(alert['module'])
    
    def _request_module_recovery(self, module_name):
        """
        Request recovery of a module that is in an error state.
        
        Args:
            module_name: Name of the module to recover
        """
        self.logger.info(f"Requesting recovery for module {module_name}")
        
        try:
            # Check if this module has attempted recovery recently
            # (to avoid continuous recovery attempts)
            last_recovery_time = getattr(self, f"last_{module_name}_recovery", 0)
            current_time = time.time()
            
            # Allow recovery attempt if at least 5 minutes have passed since last attempt
            if current_time - last_recovery_time > 300:
                # Set last recovery time
                setattr(self, f"last_{module_name}_recovery", current_time)
                
                # Request module recovery from integration controller
                if hasattr(self.integration_controller, '_attempt_module_recovery'):
                    self.integration_controller._attempt_module_recovery(module_name)
                    self.logger.info(f"Recovery request sent for module {module_name}")
                else:
                    self.logger.error(f"Integration controller lacks recovery method for module {module_name}")
            else:
                self.logger.info(f"Skipping recovery for module {module_name} (too soon since last attempt)")
        
        except Exception as e:
            self.logger.error(f"Error requesting module recovery: {str(e)}", exc_info=True)
    
    def get_system_status(self):
        """
        Get the current system status.
        
        Returns:
            Dictionary with system status information
        """
        with self.lock:
            return {
                'timestamp': datetime.now().isoformat(),
                'system_metrics': self.system_metrics,
                'module_status': self.integration_controller.get_module_status()
            }
    
    def get_performance_history(self, duration_minutes=60):
        """
        Get system performance history for the specified duration.
        
        Args:
            duration_minutes: Number of minutes of history to retrieve
            
        Returns:
            Dictionary with performance history
        """
        with self.lock:
            # Calculate cutoff time
            cutoff_time = datetime.now().timestamp() - (duration_minutes * 60)
            
            # Filter history entries newer than cutoff
            filtered_history = {}
            for timestamp_str, metrics in self.performance_history.items():
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    if timestamp.timestamp() >= cutoff_time:
                        filtered_history[timestamp_str] = metrics
                except ValueError:
                    # Skip entries with invalid timestamps
                    continue
            
            return {
                'duration_minutes': duration_minutes,
                'history': filtered_history
            }
    
    def get_resource_usage(self):
        """
        Get current resource usage statistics.
        
        Returns:
            Dictionary with resource usage information
        """
        with self.lock:
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu': self.system_metrics.get('cpu', {}),
                'memory': self.system_metrics.get('memory', {}),
                'disk': self.system_metrics.get('disk', {}),
                'network': self.system_metrics.get('network', {})
            }
    
    def update_thresholds(self, thresholds):
        """
        Update monitoring thresholds.
        
        Args:
            thresholds: Dictionary of threshold values to update
            
        Returns:
            Dictionary with updated thresholds
        """
        with self.lock:
            # Update only the specified thresholds
            for key, value in thresholds.items():
                if key in self.thresholds:
                    self.thresholds[key] = value
            
            self.logger.info(f"Updated monitoring thresholds: {self.thresholds}")
            return {'thresholds': self.thresholds}
