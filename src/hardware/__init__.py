"""
Hardware components module for the AI-Enhanced 3D Scanner system
Includes scanner drivers, printer interfaces, and other hardware abstractions
"""

from .scanner_driver import Scanner3DDriver, HighResolutionScanner

__all__ = ['Scanner3DDriver', 'HighResolutionScanner']
