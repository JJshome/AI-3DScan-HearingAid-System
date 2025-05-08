#!/usr/bin/env python3
"""
Exceptions module for the AI-Enhanced 3D Scanning Hearing Aid Manufacturing System.
Defines custom exceptions for various system components.

Author: AI-Enhanced 3D Scanning Hearing Aid System Team
Date: May 8, 2025
"""

class SystemException(Exception):
    """Base exception for all system exceptions."""
    def __init__(self, message="System error occurred", error_code=None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)
    
    def __str__(self):
        if self.error_code:
            return f"{self.message} (Error code: {self.error_code})"
        return self.message

# Core System Exceptions
class SystemInitializationError(SystemException):
    """Exception raised when the system fails to initialize."""
    def __init__(self, message="Failed to initialize system", error_code=None):
        super().__init__(message, error_code)

class ConfigurationError(SystemException):
    """Exception raised for configuration-related errors."""
    def __init__(self, message="Configuration error", error_code=None):
        super().__init__(message, error_code)

class ModuleError(SystemException):
    """Base exception for module-specific errors."""
    def __init__(self, module, message="Module error", error_code=None):
        self.module = module
        full_message = f"[{module}] {message}"
        super().__init__(full_message, error_code)

# Module-Specific Exceptions
class ScanningModuleError(ModuleError):
    """Exception raised by the 3D Scanning Module."""
    def __init__(self, message="3D Scanning error", error_code=None):
        super().__init__("scanning", message, error_code)

class DesignModuleError(ModuleError):
    """Exception raised by the AI Design Module."""
    def __init__(self, message="AI Design error", error_code=None):
        super().__init__("design", message, error_code)

class PrintingModuleError(ModuleError):
    """Exception raised by the 3D Printing Module."""
    def __init__(self, message="3D Printing error", error_code=None):
        super().__init__("printing", message, error_code)

class AcousticModuleError(ModuleError):
    """Exception raised by the Acoustic Optimization Module."""
    def __init__(self, message="Acoustic Optimization error", error_code=None):
        super().__init__("acoustic", message, error_code)

class IoTModuleError(ModuleError):
    """Exception raised by the IoT Monitoring Module."""
    def __init__(self, message="IoT Monitoring error", error_code=None):
        super().__init__("iot", message, error_code)

class IntegrationModuleError(ModuleError):
    """Exception raised by the Integration Control System."""
    def __init__(self, message="Integration Control error", error_code=None):
        super().__init__("integration", message, error_code)

class LLMModuleError(ModuleError):
    """Exception raised by the LLM Integration Module."""
    def __init__(self, message="LLM Integration error", error_code=None):
        super().__init__("llm", message, error_code)

class FittingModuleError(ModuleError):
    """Exception raised by the Rapid Fitting Module."""
    def __init__(self, message="Rapid Fitting error", error_code=None):
        super().__init__("fitting", message, error_code)

# Functional Exceptions
class DatabaseError(SystemException):
    """Exception raised for database-related errors."""
    def __init__(self, message="Database error", error_code=None):
        super().__init__(message, error_code)

class NetworkError(SystemException):
    """Exception raised for network-related errors."""
    def __init__(self, message="Network error", error_code=None):
        super().__init__(message, error_code)

class SecurityError(SystemException):
    """Exception raised for security-related errors."""
    def __init__(self, message="Security error", error_code=None):
        super().__init__(message, error_code)

class HardwareError(SystemException):
    """Exception raised for hardware-related errors."""
    def __init__(self, device, message="Hardware error", error_code=None):
        self.device = device
        full_message = f"[{device}] {message}"
        super().__init__(full_message, error_code)

# API Exceptions
class APIError(SystemException):
    """Base exception for API-related errors."""
    def __init__(self, message="API error", status_code=500, error_code=None):
        self.status_code = status_code
        super().__init__(message, error_code)

class ValidationError(APIError):
    """Exception raised for input validation errors."""
    def __init__(self, message="Validation error", field=None, error_code=None):
        self.field = field
        full_message = message
        if field:
            full_message = f"{message} (field: {field})"
        super().__init__(full_message, 400, error_code)

class AuthenticationError(APIError):
    """Exception raised for authentication errors."""
    def __init__(self, message="Authentication error", error_code=None):
        super().__init__(message, 401, error_code)

class AuthorizationError(APIError):
    """Exception raised for authorization errors."""
    def __init__(self, message="Authorization error", error_code=None):
        super().__init__(message, 403, error_code)

class ResourceNotFoundError(APIError):
    """Exception raised when a requested resource is not found."""
    def __init__(self, resource_type=None, resource_id=None, error_code=None):
        message = "Resource not found"
        if resource_type and resource_id:
            message = f"{resource_type} with ID {resource_id} not found"
        elif resource_type:
            message = f"{resource_type} not found"
        super().__init__(message, 404, error_code)

# Data Processing Exceptions
class DataProcessingError(SystemException):
    """Exception raised for data processing errors."""
    def __init__(self, message="Data processing error", error_code=None):
        super().__init__(message, error_code)

class ModelInferenceError(SystemException):
    """Exception raised for AI model inference errors."""
    def __init__(self, model_name, message="Model inference error", error_code=None):
        self.model_name = model_name
        full_message = f"[{model_name}] {message}"
        super().__init__(full_message, error_code)

class OptimizationError(SystemException):
    """Exception raised for optimization-related errors."""
    def __init__(self, message="Optimization error", error_code=None):
        super().__init__(message, error_code)

# Manufacturing Exceptions
class ManufacturingError(SystemException):
    """Base exception for manufacturing-related errors."""
    def __init__(self, message="Manufacturing error", error_code=None):
        super().__init__(message, error_code)

class MaterialError(ManufacturingError):
    """Exception raised for material-related errors."""
    def __init__(self, material_type=None, message="Material error", error_code=None):
        full_message = message
        if material_type:
            full_message = f"[{material_type}] {message}"
        super().__init__(full_message, error_code)

class QualityControlError(ManufacturingError):
    """Exception raised for quality control failures."""
    def __init__(self, component=None, message="Quality control failure", error_code=None):
        full_message = message
        if component:
            full_message = f"[{component}] {message}"
        super().__init__(full_message, error_code)

class CalibrationError(ManufacturingError):
    """Exception raised for calibration-related errors."""
    def __init__(self, device=None, message="Calibration error", error_code=None):
        full_message = message
        if device:
            full_message = f"[{device}] {message}"
        super().__init__(full_message, error_code)
