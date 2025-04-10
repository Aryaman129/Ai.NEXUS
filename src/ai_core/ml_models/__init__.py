"""
Machine Learning Models for NEXUS
Advanced ML models for enhanced AI capabilities
"""

try:
    from .yolo_detector import YOLODetector
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

__all__ = ['YOLODetector'] if YOLO_AVAILABLE else []
