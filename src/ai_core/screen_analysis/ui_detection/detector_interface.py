"""
UI Detector Interface

This module defines the common interface that all UI element detectors must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class UIDetectorInterface(ABC):
    """Interface for all UI element detectors."""
    
    @abstractmethod
    def initialize(self, config: Optional[Dict] = None) -> bool:
        """
        Initialize the detector with optional configuration.
        
        Args:
            config: Optional configuration dictionary.
            
        Returns:
            True if initialization was successful, False otherwise.
        """
        pass
        
    @abstractmethod
    def detect_elements(self, screenshot: Any, context: Optional[Dict] = None) -> List[Dict]:
        """
        Detect UI elements in the screenshot.
        
        Args:
            screenshot: Image as numpy array (RGB format) or PIL Image.
            context: Optional context information (app name, window title, etc.).
            
        Returns:
            List of detected elements, each with:
            - type: Element type (button, text_field, etc.).
            - bbox: Bounding box (x1, y1, x2, y2).
            - center: Center point (x, y).
            - confidence: Detection confidence (0-1).
            - text: Extracted text (if any).
            - additional metadata.
        """
        pass
        
    @abstractmethod
    def supports_incremental_learning(self) -> bool:
        """
        Whether this detector supports incremental learning.
        
        Returns:
            True if the detector supports incremental learning, False otherwise.
        """
        pass
        
    def update_model(self, examples: List[Dict]) -> bool:
        """
        Update the detector with new examples (only for incremental learning).
        
        Args:
            examples: List of labeled examples.
                
        Returns:
            Success status.
        """
        return False
        
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get detector capabilities and metadata.
        
        Returns:
            Dictionary of capabilities.
        """
        return {
            "supports_incremental_learning": self.supports_incremental_learning(),
            "element_types": self.get_supported_element_types(),
            "requires_gpu": False,
            "version": "0.1.0"
        }
        
    def get_supported_element_types(self) -> List[str]:
        """
        Get the element types this detector can identify.
        
        Returns:
            List of supported element types.
        """
        return ["button", "text_field", "checkbox", "dropdown", "icon"]
        
    def preprocess_image(self, screenshot: Any) -> np.ndarray:
        """
        Preprocess the image to a standard format (numpy array).
        
        Args:
            screenshot: Image as numpy array or PIL Image.
            
        Returns:
            Preprocessed image as numpy array.
        """
        if isinstance(screenshot, Image.Image):
            return np.array(screenshot)
        elif isinstance(screenshot, np.ndarray):
            return screenshot
        else:
            raise ValueError(f"Unsupported image type: {type(screenshot)}")
            
    def normalize_element(self, element: Dict) -> Dict:
        """
        Normalize a detected element to the standard format.
        
        Args:
            element: Raw detected element.
            
        Returns:
            Normalized element.
        """
        # Ensure all required fields are present
        if "bbox" in element and "type" in element:
            x1, y1, x2, y2 = element["bbox"]
            
            # Add center point if not present
            if "center" not in element:
                element["center"] = ((x1 + x2) // 2, (y1 + y2) // 2)
                
            # Add confidence if not present
            if "confidence" not in element:
                element["confidence"] = 1.0
                
            # Add text field if not present
            if "text" not in element:
                element["text"] = ""
                
            # Add size information
            element["width"] = x2 - x1
            element["height"] = y2 - y1
                
            return element
        else:
            logger.warning(f"Invalid element format: {element}")
            return element
