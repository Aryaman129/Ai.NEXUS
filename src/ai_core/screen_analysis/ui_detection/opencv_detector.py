"""
OpenCV-based UI Detector

This module provides a basic UI element detector using OpenCV.
It serves as a reliable fallback when ML-based detectors are not available.
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Any, Optional
import time

from .detector_interface import UIDetectorInterface

logger = logging.getLogger(__name__)

class OpenCVDetector(UIDetectorInterface):
    """UI element detector using OpenCV for basic detection"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.initialized = False
        
    def initialize(self, config: Optional[Dict] = None) -> bool:
        """Initialize the detector with optional configuration"""
        if config:
            self.config.update(config)
            
        # OpenCV is always available as it's a core dependency
        self.initialized = True
        return True
        
    def detect_elements(self, screenshot: Any, context: Optional[Dict] = None) -> List[Dict]:
        """
        Detect UI elements in the screenshot using OpenCV
        
        Args:
            screenshot: Image as numpy array or PIL Image
            context: Optional context information
            
        Returns:
            List of detected elements
        """
        if not self.initialized:
            logger.warning("OpenCV detector not initialized")
            return []
            
        # Preprocess image
        img = self.preprocess_image(screenshot)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Detect elements using multiple methods
        elements = []
        
        # Method 1: Contour detection for buttons and UI elements
        elements.extend(self._detect_contours(gray, img))
        
        # Method 2: Template matching for common UI elements
        elements.extend(self._detect_templates(gray, img, context))
        
        # Method 3: Text detection for labels and text fields
        elements.extend(self._detect_text_areas(gray, img))
        
        # Normalize all elements
        normalized_elements = [self.normalize_element(e) for e in elements]
        
        return normalized_elements
        
    def _detect_contours(self, gray: np.ndarray, original: np.ndarray) -> List[Dict]:
        """Detect UI elements using contour detection"""
        elements = []
        
        try:
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Find contours
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Filter and classify contours
            for contour in contours:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter out very small elements
                if w < 20 or h < 20:
                    continue
                    
                # Filter out very large elements (likely background)
                if w > original.shape[1] * 0.9 or h > original.shape[0] * 0.9:
                    continue
                    
                # Calculate aspect ratio
                aspect_ratio = w / h
                
                # Classify based on shape and size
                element_type = "unknown"
                confidence = 0.5  # Base confidence
                
                if 0.9 < aspect_ratio < 1.1 and w < 50 and h < 50:
                    # Square elements are likely checkboxes or small buttons
                    element_type = "checkbox"
                    confidence = 0.6
                elif aspect_ratio > 3:
                    # Wide elements are likely text fields or buttons
                    element_type = "text_field"
                    confidence = 0.7
                elif 2 < aspect_ratio < 3:
                    # Medium-wide elements are likely buttons
                    element_type = "button"
                    confidence = 0.7
                    
                # Create element
                element = {
                    "type": element_type,
                    "bbox": (x, y, x + w, y + h),
                    "confidence": confidence,
                    "detector": "opencv"
                }
                
                elements.append(element)
                
        except Exception as e:
            logger.error(f"Error in contour detection: {e}")
            
        return elements
        
    def _detect_templates(self, gray: np.ndarray, original: np.ndarray, 
                         context: Optional[Dict] = None) -> List[Dict]:
        """Detect UI elements using template matching"""
        elements = []
        
        # This would use template matching with common UI elements
        # For a real implementation, we would have a library of templates
        # For now, we'll return an empty list
        
        return elements
        
    def _detect_text_areas(self, gray: np.ndarray, original: np.ndarray) -> List[Dict]:
        """Detect text areas that might be labels or text fields"""
        elements = []
        
        try:
            # Apply MSER (Maximally Stable Extremal Regions) for text detection
            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(gray)
            
            # Convert regions to bounding boxes
            for region in regions:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(region)
                
                # Filter out very small elements
                if w < 30 or h < 10:
                    continue
                    
                # Filter out very large elements
                if w > original.shape[1] * 0.8 or h > original.shape[0] * 0.5:
                    continue
                    
                # Calculate aspect ratio
                aspect_ratio = w / h
                
                # Text areas typically have a wide aspect ratio
                if aspect_ratio > 2:
                    element = {
                        "type": "text_field",
                        "bbox": (x, y, x + w, y + h),
                        "confidence": 0.6,
                        "detector": "opencv"
                    }
                    
                    elements.append(element)
                    
        except Exception as e:
            logger.error(f"Error in text area detection: {e}")
            
        return elements
        
    def supports_incremental_learning(self) -> bool:
        """Whether this detector supports incremental learning"""
        return False
        
    def get_capabilities(self) -> Dict[str, Any]:
        """Get detector capabilities and metadata"""
        capabilities = super().get_capabilities()
        
        # Add OpenCV-specific capabilities
        capabilities.update({
            "name": "opencv",
            "requires_gpu": False,
            "version": "0.1.0",
            "supports_batch_processing": False,
            "max_elements_per_image": 50
        })
        
        return capabilities
