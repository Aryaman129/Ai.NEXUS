"""
Screen Analysis Module for NEXUS

This module provides computer vision capabilities for screen capture,
UI element detection, and automated interaction.

Key components:
- Screen capture with DirectX optimization for Windows
- UI element detection using optimized vision models
- Visual memory for learning interface patterns
- Safe automation tools for mouse/keyboard control
"""

# Import core components
from .visual_memory import VisualMemorySystem

# Import detector components if available
try:
    from .detectors import (
        DetectorRegistry, ConfidenceCalibrator,
        AutoGluonDetector, HuggingFaceDetector, OpenCVDetector
    )
    from .visualization import DetectionVisualizer
    DETECTORS_AVAILABLE = True
except ImportError:
    DETECTORS_AVAILABLE = False

__all__ = [
    'VisualMemorySystem'
]

if DETECTORS_AVAILABLE:
    __all__.extend([
        'DetectorRegistry', 
        'ConfidenceCalibrator',
        'AutoGluonDetector', 
        'HuggingFaceDetector', 
        'OpenCVDetector',
        'DetectionVisualizer'
    ])
