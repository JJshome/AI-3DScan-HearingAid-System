#!/usr/bin/env python3
"""
Integration Controller for the AI-Enhanced 3D Scanning Hearing Aid Manufacturing System.
Central coordination unit for managing all system modules and their interactions.

Author: AI-Enhanced 3D Scanning Hearing Aid System Team
Date: May 8, 2025
"""

import os
import sys
import logging
import threading
import time
import json
from queue import Queue
from typing import Dict, Any, List, Optional
from enum import Enum

# Import module interfaces
from src.modules.scanning.scanner_interface import ScannerInterface
from src.modules.design.designer_interface import DesignerInterface
from src.modules.printing.printer_interface import PrinterInterface
from src.modules.acoustic.acoustic_interface import AcousticInterface
from src.modules.iot.iot_interface import IoTInterface
from src.modules.llm.llm_interface import LLMInterface
from src.modules.fitting.fitting_interface import FittingInterface

# Import utility modules
from src.core.config import SystemConfig
from src.utils.exceptions import (
    SystemInitializationError, IntegrationModuleError, ModuleError
)
from src.database.job_repository import JobRepository
from src.database.patient_repository import PatientRepository
from src.security.access_controller import AccessController

class ModuleStatus(Enum):
    """Enumeration of possible module statuses."""
    UNINITIALIZED = 0
    INITIALIZING = 1
    READY = 2
    RUNNING = 3
    WARNING = 4
    ERROR = 5
    STOPPING = 6
    STOPPED = 7

class IntegrationController:
    """
    Central integration controller for the AI-Enhanced 3D Scanning Hearing Aid System.
    Manages all modules, coordinates data flow, and orchestrates the manufacturing process.
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize the integration controller.
        
        Args:
            config: System configuration
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.running = False
        self.job_queue = Queue()
        self.active_jobs = {}
        self.modules = {}
        self.module_status = {}
        self.locks = {}
        
        # Initialize repositories
        self.job_repository = JobRepository(config.get_database_config())
        self.patient_repository = PatientRepository(config.get_database_config())
        
        # Initialize security controller
        self.access_controller = AccessController(config.get_security_config())
        
        # Create module status tracking
        self._initialize_module_status()
        
        # Create module locks for thread safety
        self._initialize_locks()
        
        self.logger.info("Integration Controller initialized")
    
    def _initialize_module_status(self):
        """Initialize the status of all modules to UNINITIALIZED."""
        modules = [
            'scanning', 'design', 'printing', 'acoustic', 
            'iot', 'llm', 'fitting'
        ]
        
        for module in modules:
            self.module_status[module] = ModuleStatus.UNINITIALIZED
    
    def _initialize_locks(self):
        """Initialize locks for thread-safe module operations."""
        self.locks = {
            'scanning': threading.RLock(),
            'design': threading.RLock(),
            'printing': threading.RLock(),
            'acoustic': threading.RLock(),
            'iot': threading.RLock(),
            'llm': threading.RLock(),
            'fitting': threading.RLock(),
            'job_queue': threading.RLock(),
            'active_jobs': threading.RLock()
        }
        self.global_lock = threading.RLock()
    
    def start(self):
        """Start the integration controller and all initialized modules."""
        if self.running:
            self.logger.warning("Integration Controller is already running")
            return
        
        self.logger.info("Starting Integration Controller")
        
        # Start job processing thread
        self.running = True
        self.job_processor_thread = threading.Thread(
            target=self._job_processor_loop, 
            name="JobProcessor"
        )
        self.job_processor_thread.daemon = True
        self.job_processor_thread.start()
        
        # Start module health check thread
        self.health_check_thread = threading.Thread(
            target=self._health_check_loop, 
            name="HealthCheck"
        )
        self.health_check_thread.daemon = True
        self.health_check_thread.start()
        
        # Start each initialized module
        self._start_all_modules()
        
        self.logger.info("Integration Controller started")
    
    def stop(self):
        """Stop the integration controller and all modules."""
        if not self.running:
            self.logger.warning("Integration Controller is not running")
            return
        
        self.logger.info("Stopping Integration Controller")
        
        # Stop running flag to terminate threads
        self.running = False
        
        # Stop all modules
        self._stop_all_modules()
        
        # Wait for threads to finish
        if hasattr(self, 'job_processor_thread') and self.job_processor_thread.is_alive():
            self.job_processor_thread.join(timeout=10)
        
        if hasattr(self, 'health_check_thread') and self.health_check_thread.is_alive():
            self.health_check_thread.join(timeout=10)
        
        self.logger.info("Integration Controller stopped")
    
    def initialize_all_modules(self):
        """Initialize all system modules."""
        self.logger.info("Initializing all modules")
        
        try:
            self.initialize_module('scanning')
            self.initialize_module('design')
            self.initialize_module('printing')
            self.initialize_module('acoustic')
            self.initialize_module('iot')
            self.initialize_module('llm')
            self.initialize_module('fitting')
            
            self.logger.info("All modules initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize all modules: {str(e)}", exc_info=True)
            raise SystemInitializationError(f"Failed to initialize all modules: {str(e)}")
    
    def initialize_module(self, module_name: str):
        """
        Initialize a specific module.
        
        Args:
            module_name: Name of the module to initialize
        """
        if module_name not in self.locks:
            raise ValueError(f"Unknown module: {module_name}")
        
        with self.locks[module_name]:
            if self.module_status.get(module_name) in [ModuleStatus.INITIALIZING, ModuleStatus.READY, ModuleStatus.RUNNING]:
                self.logger.warning(f"Module {module_name} is already initialized or being initialized")
                return
            
            self.logger.info(f"Initializing module: {module_name}")
            self.module_status[module_name] = ModuleStatus.INITIALIZING
            
            try:
                module_config = self.config.get_module_config(module_name)
                
                # Initialize the appropriate module based on the name
                if module_name == 'scanning':
                    from src.modules.scanning.scanner_controller import ScannerController
                    self.modules[module_name] = ScannerController(module_config)
                
                elif module_name == 'design':
                    from src.modules.design.designer_controller import DesignerController
                    self.modules[module_name] = DesignerController(module_config)
                
                elif module_name == 'printing':
                    from src.modules.printing.printer_controller import PrinterController
                    self.modules[module_name] = PrinterController(module_config)
                
                elif module_name == 'acoustic':
                    from src.modules.acoustic.acoustic_controller import AcousticController
                    self.modules[module_name] = AcousticController(module_config)
                
                elif module_name == 'iot':
                    from src.modules.iot.iot_controller import IoTController
                    self.modules[module_name] = IoTController(module_config)
                
                elif module_name == 'llm':
                    from src.modules.llm.llm_controller import LLMController
                    self.modules[module_name] = LLMController(module_config)
                
                elif module_name == 'fitting':
                    from src.modules.fitting.fitting_controller import FittingController
                    self.modules[module_name] = FittingController(module_config)
                
                # Initialize the module
                init_result = self.modules[module_name].initialize()
                if not init_result.get('success', False):
                    error_msg = init_result.get('message', f"Failed to initialize {module_name} module")
                    self.logger.error(error_msg)
                    self.module_status[module_name] = ModuleStatus.ERROR
                    raise IntegrationModuleError(error_msg)
                
                self.module_status[module_name] = ModuleStatus.READY
                self.logger.info(f"Module {module_name} initialized successfully")
            
            except Exception as e:
                self.module_status[module_name] = ModuleStatus.ERROR
                self.logger.error(f"Error initializing {module_name} module: {str(e)}", exc_info=True)
                raise IntegrationModuleError(f"Error initializing {module_name} module: {str(e)}")
    
    def _start_all_modules(self):
        """Start all initialized modules."""
        for module_name, status in self.module_status.items():
            if status == ModuleStatus.READY:
                self._start_module(module_name)
    
    def _stop_all_modules(self):
        """Stop all running modules."""
        for module_name, status in self.module_status.items():
            if status == ModuleStatus.RUNNING:
                self._stop_module(module_name)
    
    def _start_module(self, module_name: str):
        """
        Start a specific module.
        
        Args:
            module_name: Name of the module to start
        """
        with self.locks[module_name]:
            if self.module_status.get(module_name) != ModuleStatus.READY:
                self.logger.warning(f"Cannot start module {module_name}: not in READY state")
                return
            
            self.logger.info(f"Starting module: {module_name}")
            
            try:
                start_result = self.modules[module_name].start()
                if not start_result.get('success', False):
                    error_msg = start_result.get('message', f"Failed to start {module_name} module")
                    self.logger.error(error_msg)
                    self.module_status[module_name] = ModuleStatus.ERROR
                    raise IntegrationModuleError(error_msg)
                
                self.module_status[module_name] = ModuleStatus.RUNNING
                self.logger.info(f"Module {module_name} started successfully")
            
            except Exception as e:
                self.module_status[module_name] = ModuleStatus.ERROR
                self.logger.error(f"Error starting {module_name} module: {str(e)}", exc_info=True)
                raise IntegrationModuleError(f"Error starting {module_name} module: {str(e)}")
    
    def _stop_module(self, module_name: str):
        """
        Stop a specific module.
        
        Args:
            module_name: Name of the module to stop
        """
        with self.locks[module_name]:
            if self.module_status.get(module_name) != ModuleStatus.RUNNING:
                self.logger.warning(f"Cannot stop module {module_name}: not in RUNNING state")
                return
            
            self.logger.info(f"Stopping module: {module_name}")
            self.module_status[module_name] = ModuleStatus.STOPPING
            
            try:
                stop_result = self.modules[module_name].stop()
                if not stop_result.get('success', False):
                    error_msg = stop_result.get('message', f"Failed to stop {module_name} module")
                    self.logger.error(error_msg)
                    self.module_status[module_name] = ModuleStatus.ERROR
                    raise IntegrationModuleError(error_msg)
                
                self.module_status[module_name] = ModuleStatus.STOPPED
                self.logger.info(f"Module {module_name} stopped successfully")
            
            except Exception as e:
                self.module_status[module_name] = ModuleStatus.ERROR
                self.logger.error(f"Error stopping {module_name} module: {str(e)}", exc_info=True)
                raise IntegrationModuleError(f"Error stopping {module_name} module: {str(e)}")
    
    def get_module_status(self, module_name: str = None) -> Dict[str, str]:
        """
        Get the status of all modules or a specific module.
        
        Args:
            module_name: Name of the module to get status for, or None for all modules
            
        Returns:
            Dictionary of module status information
        """
        if module_name:
            if module_name not in self.module_status:
                raise ValueError(f"Unknown module: {module_name}")
            
            with self.locks[module_name]:
                status = self.module_status[module_name]
                return {
                    'name': module_name,
                    'status': status.name,
                    'details': self._get_module_details(module_name)
                }
        else:
            # Get status of all modules
            result = {}
            for module, status in self.module_status.items():
                with self.locks[module]:
                    result[module] = {
                        'status': status.name,
                        'details': self._get_module_details(module)
                    }
            return result
    
    def _get_module_details(self, module_name: str) -> Dict[str, Any]:
        """
        Get detailed status information from a module.
        
        Args:
            module_name: Name of the module
            
        Returns:
            Dictionary of detailed status information
        """
        if module_name not in self.modules:
            return {'error': 'Module not initialized'}
        
        try:
            status_result = self.modules[module_name].get_status()
            return status_result.get('details', {})
        except Exception as e:
            self.logger.error(f"Error getting status details for {module_name}: {str(e)}")
            return {'error': str(e)}
    
    def _job_processor_loop(self):
        """Main job processing loop that runs in a separate thread."""
        self.logger.info("Job processor thread started")
        
        while self.running:
            try:
                # Process any jobs in the queue
                self._process_jobs()
                
                # Check for completed jobs
                self._check_completed_jobs()
                
                # Sleep briefly before next iteration
                time.sleep(0.1)
            
            except Exception as e:
                self.logger.error(f"Error in job processor loop: {str(e)}", exc_info=True)
                time.sleep(1)  # Sleep longer on error
        
        self.logger.info("Job processor thread stopped")
    
    def _health_check_loop(self):
        """Health check loop that runs in a separate thread."""
        self.logger.info("Health check thread started")
        
        check_interval = self.config.get_value(
            'system.health_check_interval_seconds', 30)
        
        while self.running:
            try:
                # Check health of all modules
                self._check_module_health()
                
                # Sleep until next check
                time.sleep(check_interval)
            
            except Exception as e:
                self.logger.error(f"Error in health check loop: {str(e)}", exc_info=True)
                time.sleep(5)  # Sleep longer on error
        
        self.logger.info("Health check thread stopped")
    
    def _process_jobs(self):
        """Process jobs from the job queue."""
        with self.locks['job_queue']:
            if self.job_queue.empty():
                return
            
            # Get the next job from the queue
            job = self.job_queue.get()
            job_id = job['job_id']
            
            self.logger.info(f"Processing job: {job_id}")
            
            # Add to active jobs
            with self.locks['active_jobs']:
                self.active_jobs[job_id] = job
            
            # Start job processing in a new thread
            job_thread = threading.Thread(
                target=self._process_job,
                args=(job,),
                name=f"Job-{job_id}"
            )
            job_thread.daemon = True
            job_thread.start()
    
    def _process_job(self, job):
        """
        Process a single manufacturing job through all stages.
        
        Args:
            job: Job data dictionary
        """
        job_id = job['job_id']
        current_stage = job.get('current_stage', 'scanning')
        
        self.logger.info(f"Starting job {job_id} at stage {current_stage}")
        
        try:
            # Update job status in database
            self.job_repository.update_job_status(job_id, 'processing', current_stage)
            
            # Process job through each stage in sequence
            while current_stage != 'completed':
                # Check if the required module is available
                if not self._is_module_ready(current_stage):
                    self.logger.error(f"Module {current_stage} not ready for job {job_id}")
                    self.job_repository.update_job_status(job_id, 'error', current_stage, 
                                                         "Required module not available")
                    return
                
                # Process the current stage
                result = self._process_job_stage(job, current_stage)
                
                if not result.get('success', False):
                    error_message = result.get('message', f"Error processing stage {current_stage}")
                    self.logger.error(f"Job {job_id} failed at stage {current_stage}: {error_message}")
                    self.job_repository.update_job_status(job_id, 'error', current_stage, error_message)
                    return
                
                # Update job data with results from this stage
                job.update(result.get('job_data', {}))
                
                # Move to next stage
                previous_stage = current_stage
                current_stage = self._get_next_stage(current_stage)
                self.logger.info(f"Job {job_id} moving from {previous_stage} to {current_stage}")
                
                # Update job status in database
                self.job_repository.update_job_status(job_id, 'processing', current_stage)
            
            # Job completed successfully
            self.logger.info(f"Job {job_id} completed successfully")
            self.job_repository.update_job_status(job_id, 'completed', 'completed')
        
        except Exception as e:
            self.logger.error(f"Error processing job {job_id}: {str(e)}", exc_info=True)
            self.job_repository.update_job_status(job_id, 'error', current_stage, str(e))
        
        finally:
            # Remove from active jobs
            with self.locks['active_jobs']:
                if job_id in self.active_jobs:
                    del self.active_jobs[job_id]
    
    def _process_job_stage(self, job, stage):
        """
        Process a specific stage of a job.
        
        Args:
            job: Job data dictionary
            stage: Current processing stage
            
        Returns:
            Result dictionary with success flag and updated job data
        """
        job_id = job['job_id']
        
        self.logger.info(f"Processing job {job_id} stage {stage}")
        
        try:
            # Get the appropriate module for this stage
            module = self.modules.get(stage)
            if not module:
                return {
                    'success': False,
                    'message': f"Module {stage} not initialized"
                }
            
            # Process the stage using the appropriate module
            with self.locks[stage]:
                if stage == 'scanning':
                    result = module.perform_scan(job)
                elif stage == 'design':
                    result = module.design_hearing_aid(job)
                elif stage == 'printing':
                    result = module.print_hearing_aid(job)
                elif stage == 'acoustic':
                    result = module.optimize_acoustics(job)
                elif stage == 'iot':
                    result = module.configure_monitoring(job)
                elif stage == 'llm':
                    result = module.process_natural_language(job)
                elif stage == 'fitting':
                    result = module.perform_fitting(job)
                else:
                    return {
                        'success': False,
                        'message': f"Unknown stage: {stage}"
                    }
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error processing job {job_id} stage {stage}: {str(e)}", exc_info=True)
            return {
                'success': False,
                'message': str(e)
            }
    
    def _check_completed_jobs(self):
        """Check for completed jobs and perform cleanup."""
        with self.locks['active_jobs']:
            jobs_to_remove = []
            
            for job_id, job in self.active_jobs.items():
                if job.get('status') in ['completed', 'error']:
                    jobs_to_remove.append(job_id)
            
            for job_id in jobs_to_remove:
                del self.active_jobs[job_id]
    
    def _check_module_health(self):
        """Check the health of all running modules."""
        for module_name, status in self.module_status.items():
            if status == ModuleStatus.RUNNING:
                with self.locks[module_name]:
                    try:
                        # Check module health
                        health_result = self.modules[module_name].check_health()
                        
                        if not health_result.get('healthy', False):
                            self.logger.warning(f"Module {module_name} health check failed: {health_result.get('message', 'Unknown error')}")
                            self.module_status[module_name] = ModuleStatus.WARNING
                            
                            # Attempt recovery if module is in warning state
                            if self._should_attempt_recovery(module_name):
                                self._attempt_module_recovery(module_name)
                        else:
                            # Module is healthy
                            if self.module_status[module_name] == ModuleStatus.WARNING:
                                self.logger.info(f"Module {module_name} is now healthy")
                                self.module_status[module_name] = ModuleStatus.RUNNING
                    
                    except Exception as e:
                        self.logger.error(f"Error checking health of module {module_name}: {str(e)}", exc_info=True)
                        self.module_status[module_name] = ModuleStatus.ERROR
    
    def _should_attempt_recovery(self, module_name):
        """
        Determine if recovery should be attempted for a module.
        
        Args:
            module_name: Name of the module
            
        Returns:
            True if recovery should be attempted, False otherwise
        """
        # Only attempt recovery for modules in WARNING state
        if self.module_status[module_name] != ModuleStatus.WARNING:
            return False
        
        # Check module-specific recovery settings
        module_config = self.config.get_module_config(module_name)
        return module_config.get('auto_recovery_enabled', True)
    
    def _attempt_module_recovery(self, module_name):
        """
        Attempt to recover a module that is in a warning state.
        
        Args:
            module_name: Name of the module to recover
        """
        self.logger.info(f"Attempting recovery of module {module_name}")
        
        try:
            # Call module's recovery method
            recovery_result = self.modules[module_name].recover()
            
            if recovery_result.get('success', False):
                self.logger.info(f"Successfully recovered module {module_name}")
                self.module_status[module_name] = ModuleStatus.RUNNING
            else:
                self.logger.error(f"Failed to recover module {module_name}: {recovery_result.get('message', 'Unknown error')}")
                self.module_status[module_name] = ModuleStatus.ERROR
        
        except Exception as e:
            self.logger.error(f"Error during recovery of module {module_name}: {str(e)}", exc_info=True)
            self.module_status[module_name] = ModuleStatus.ERROR
    
    def _is_module_ready(self, module_name):
        """
        Check if a module is ready for processing.
        
        Args:
            module_name: Name of the module
            
        Returns:
            True if the module is ready, False otherwise
        """
        return self.module_status.get(module_name) == ModuleStatus.RUNNING
    
    def _get_next_stage(self, current_stage):
        """
        Determine the next stage in the manufacturing process.
        
        Args:
            current_stage: Current processing stage
            
        Returns:
            Name of the next stage
        """
        stage_sequence = {
            'scanning': 'design',
            'design': 'printing',
            'printing': 'acoustic',
            'acoustic': 'iot',
            'iot': 'llm',
            'llm': 'fitting',
            'fitting': 'completed'
        }
        
        return stage_sequence.get(current_stage, 'completed')
    
    def submit_job(self, job_data):
        """
        Submit a new job for processing.
        
        Args:
            job_data: Job data dictionary
            
        Returns:
            Dictionary with job submission results
        """
        try:
            # Validate job data
            if 'patient_id' not in job_data:
                return {
                    'success': False, 
                    'message': "Missing required field: patient_id"
                }
            
            # Create a new job record
            job_id = self.job_repository.create_job(job_data)
            
            # Add job to processing queue
            job_data['job_id'] = job_id
            job_data['status'] = 'queued'
            job_data['current_stage'] = 'scanning'
            
            with self.locks['job_queue']:
                self.job_queue.put(job_data)
            
            self.logger.info(f"Job {job_id} submitted successfully")
            
            return {
                'success': True,
                'job_id': job_id,
                'message': "Job submitted successfully"
            }
        
        except Exception as e:
            self.logger.error(f"Error submitting job: {str(e)}", exc_info=True)
            return {
                'success': False,
                'message': f"Error submitting job: {str(e)}"
            }
    
    def get_job_status(self, job_id):
        """
        Get the status of a specific job.
        
        Args:
            job_id: ID of the job
            
        Returns:
            Dictionary with job status information
        """
        try:
            # Check if job is in active jobs
            with self.locks['active_jobs']:
                if job_id in self.active_jobs:
                    return {
                        'success': True,
                        'job_id': job_id,
                        'status': self.active_jobs[job_id].get('status', 'unknown'),
                        'current_stage': self.active_jobs[job_id].get('current_stage', 'unknown'),
                        'details': self.active_jobs[job_id].get('details', {})
                    }
            
            # Check job status in database
            job_data = self.job_repository.get_job(job_id)
            if not job_data:
                return {
                    'success': False,
                    'message': f"Job {job_id} not found"
                }
            
            return {
                'success': True,
                'job_id': job_id,
                'status': job_data.get('status', 'unknown'),
                'current_stage': job_data.get('current_stage', 'unknown'),
                'details': job_data.get('details', {})
            }
        
        except Exception as e:
            self.logger.error(f"Error getting job status for {job_id}: {str(e)}", exc_info=True)
            return {
                'success': False,
                'message': f"Error getting job status: {str(e)}"
            }
    
    def is_running(self):
        """
        Check if the integration controller is running.
        
        Returns:
            True if running, False otherwise
        """
        return self.running
