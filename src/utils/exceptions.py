"""
Custom exceptions for the AI-Enhanced 3D Scanner system
"""

class HardwareConnectionError(Exception):
    """
    Raised when there's an issue connecting to hardware devices
    such as scanners, printers, or other peripherals
    """
    pass


class ScanningError(Exception):
    """
    Raised when there's an issue during the scanning process,
    such as calibration failures, scan interruptions, or data quality issues
    """
    pass


class ProcessingError(Exception):
    """
    Raised when there's an issue processing the scanned data,
    such as mesh generation, point cloud processing, or data transformation errors
    """
    pass


class PrintingError(Exception):
    """
    Raised when there's an issue during the 3D printing process,
    such as printer connection failures, material issues, or print job errors
    """
    pass


class ModelError(Exception):
    """
    Raised when there's an issue with AI model operations,
    such as loading failures, inference errors, or unsupported operations
    """
    pass


class ValidationError(Exception):
    """
    Raised when data validation fails,
    such as invalid parameters, data formats, or constraint violations
    """
    pass


class ConfigurationError(Exception):
    """
    Raised when there are issues with system configuration,
    such as missing required settings, invalid values, or conflicting parameters
    """
    pass
