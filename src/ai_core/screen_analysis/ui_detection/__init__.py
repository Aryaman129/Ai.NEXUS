"""
NEXUS Unified UI Detection Framework

This module provides a unified framework for UI element detection with multiple backends.
It consolidates previously scattered UI detection implementations into a single, modular system
that embodies the NEXUS philosophy of adaptation over rigid rules.
"""

from .detector_registry import UIDetectorRegistry
from .detector_interface import UIDetectorInterface
from .confidence_calibrator import ConfidenceCalibrator

# Import available detectors
try:
    from .autogluon_detector import AutoGluonDetector
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False

try:
    from .huggingface_detector import HuggingFaceDetector
    HUGGINGFACE_DETECTOR_AVAILABLE = True
except ImportError:
    HUGGINGFACE_DETECTOR_AVAILABLE = False

try:
    from .yolo_detector import YOLODetector
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Always available fallback
from .opencv_detector import OpenCVDetector
OPENCV_AVAILABLE = True

# Version tracking
__version__ = "0.2.0"
