#!/usr/bin/env python3
"""
Scanner Interface for the 3D Scanning Module of the AI-Enhanced 3D Scanning Hearing Aid Manufacturing System.
Defines the interface for interacting with 3D ear scanning hardware.

Author: AI-Enhanced 3D Scanning Hearing Aid System Team
Date: May 8, 2025
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class ScannerInterface(ABC):
    """
    Abstract base class defining the interface for all 3D ear scanner implementations.
    This ensures consistent functionality across different scanner hardware.
    """
    
    @abstractmethod
    def initialize(self) -> Dict[str, Any]:
        """
        Initialize the scanner hardware and software.
        
        Returns:
            Dictionary with initialization results including success flag
        """
        pass
    
    @abstractmethod
    def start(self) -> Dict[str, Any]:
        """
        Start the scanner system.
        
        Returns:
            Dictionary with start results including success flag
        """
        pass
    
    @abstractmethod
    def stop(self) -> Dict[str, Any]:
        """
        Stop the scanner system.
        
        Returns:
            Dictionary with stop results including success flag
        """
        pass
    
    @abstractmethod
    def calibrate(self) -> Dict[str, Any]:
        """
        Calibrate the scanner for optimal performance.
        
        Returns:
            Dictionary with calibration results including success flag
        """
        pass
    
    @abstractmethod
    def check_health(self) -> Dict[str, Any]:
        """
        Check the health and operational status of the scanner.
        
        Returns:
            Dictionary with health check results including healthy flag
        """
        pass
    
    @abstractmethod
    def recover(self) -> Dict[str, Any]:
        """
        Attempt to recover the scanner from an error state.
        
        Returns:
            Dictionary with recovery results including success flag
        """
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the scanner.
        
        Returns:
            Dictionary with status information
        """
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics from the scanner.
        
        Returns:
            Dictionary with performance metrics
        """
        pass
    
    @abstractmethod
    def perform_scan(self, scan_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a 3D scan of the ear.
        
        Args:
            scan_parameters: Dictionary of scan parameters 
                             (patient ID, scan resolution, etc.)
            
        Returns:
            Dictionary with scan results including success flag and scan data
        """
        pass
    
    @abstractmethod
    def get_scan_preview(self) -> Dict[str, Any]:
        """
        Get a preview of the current scan.
        
        Returns:
            Dictionary with preview data including preview image
        """
        pass
    
    @abstractmethod
    def abort_scan(self) -> Dict[str, Any]:
        """
        Abort the current scan operation.
        
        Returns:
            Dictionary with abort results including success flag
        """
        pass
    
    @abstractmethod
    def export_scan(self, format_type: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Export a scan in the specified format.
        
        Args:
            format_type: Type of format to export (STL, OBJ, etc.)
            options: Optional export options
            
        Returns:
            Dictionary with export results including success flag and exported data
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> Dict[str, Any]:
        """
        Clean up resources and prepare for shutdown.
        
        Returns:
            Dictionary with cleanup results including success flag
        """
        pass
