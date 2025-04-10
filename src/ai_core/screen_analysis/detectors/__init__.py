"""
UI Element Detectors for NEXUS.

This package provides a flexible, pluggable system for detecting UI elements in 
screenshots. It implements NEXUS's adaptive philosophy by allowing different detection 
methods to be used interchangeably, with runtime selection of the best available detector.

The detector system supports:
- Multiple detection methods (ML-based and traditional CV)
- Performance tracking and adaptive selection
- Specialized detectors for specific element types
- Confidence calibration based on real-world accuracy
- Parallel detection with consensus voting for critical actions
"""

from .base import UIDetectorInterface, SpecializedDetectorInterface
from .registry import DetectorRegistry
from .calibration import ConfidenceCalibrator

# Import detectors (these imports are conditional - they'll only succeed if dependencies are available)
try:
    from .autogluon_detector import AutoGluonDetector
except ImportError:
    AutoGluonDetector = None

try:
    from .huggingface_detector import HuggingFaceDetector
except ImportError:
    HuggingFaceDetector = None

try:
    from .opencv_detector import OpenCVDetector
except ImportError:
    OpenCVDetector = None

# Convenience functions
def get_detector_registry():
    """Get a new detector registry instance"""
    return DetectorRegistry()

def get_best_available_detector(registry=None):
    """
    Get the best available detector
    
    Args:
        registry: Optional registry to use (creates new one if not provided)
        
    Returns:
        Best available detector or None if no detectors available
    """
    if registry is None:
        registry = DetectorRegistry()
        _register_available_detectors(registry)
        
    return registry.get_detector()

def _register_available_detectors(registry):
    """Register all available detectors with the registry"""
    # Register AutoGluon (highest priority)
    if AutoGluonDetector is not None:
        registry.register_detector("autogluon", AutoGluonDetector, priority=100)
        
    # Register HuggingFace (medium priority)
    if HuggingFaceDetector is not None:
        registry.register_detector("huggingface", HuggingFaceDetector, priority=80)
        
    # Register OpenCV (lowest priority, but always available)
    if OpenCVDetector is not None:
        registry.register_detector("opencv", OpenCVDetector, priority=10)
        
# Export primary classes and functions
__all__ = [
    'UIDetectorInterface',
    'SpecializedDetectorInterface',
    'DetectorRegistry',
    'ConfidenceCalibrator',
    'AutoGluonDetector',
    'HuggingFaceDetector',
    'OpenCVDetector',
    'get_detector_registry',
    'get_best_available_detector'
]
