"""
Base interfaces for UI element detectors in NEXUS.

This module defines the core interfaces that all UI element detectors must implement.
The interfaces are designed to support the adaptive philosophy of NEXUS,
allowing different detection methods to be used interchangeably.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)

class UIDetectorInterface(ABC):
    """Interface for all UI element detectors"""
    
    @abstractmethod
    def initialize(self, config: Optional[Dict] = None) -> bool:
        """
        Initialize the detector with optional configuration
        
        Args:
            config: Optional configuration dictionary
            
        Returns:
            Success status
        """
        pass
        
    @abstractmethod
    def detect_elements(self, screenshot: np.ndarray, context: Optional[Dict] = None) -> List[Dict]:
        """
        Detect UI elements in the screenshot
        
        Args:
            screenshot: Image as numpy array (RGB format)
            context: Optional context information (app name, window title, etc.)
            
        Returns:
            List of detected elements, each with:
            - type: Element type (button, text_field, etc.)
            - bbox: Bounding box (x1, y1, x2, y2)
            - center: Center point (x, y)
            - confidence: Detection confidence (0-1)
            - text: Extracted text (if any)
            - additional metadata
        """
        pass
        
    def batch_detect_elements(self, screenshots: List[np.ndarray], 
                             contexts: Optional[List[Dict]] = None) -> List[List[Dict]]:
        """
        Detect UI elements in multiple screenshots (batch processing)
        
        Default implementation processes each screenshot individually.
        Detectors that support efficient batch processing should override this.
        
        Args:
            screenshots: List of images as numpy arrays
            contexts: Optional list of context information (1:1 with screenshots)
            
        Returns:
            List of detection results, one per screenshot
        """
        results = []
        for i, screenshot in enumerate(screenshots):
            context = contexts[i] if contexts and i < len(contexts) else None
            elements = self.detect_elements(screenshot, context)
            results.append(elements)
        return results
        
    @abstractmethod
    def supports_incremental_learning(self) -> bool:
        """Whether this detector supports incremental learning"""
        pass
        
    def update_model(self, examples: List[Dict]) -> bool:
        """
        Update the detector with new examples (only for incremental learning)
        
        Args:
            examples: List of labeled examples with:
                - screenshot: The image
                - elements: List of labeled UI elements
                
        Returns:
            Success status
        """
        return False
        
    def get_capabilities(self) -> Dict[str, Any]:
        """Get detector capabilities and metadata"""
        return {
            "name": self._get_name(),
            "supports_incremental_learning": self.supports_incremental_learning(),
            "element_types": self.get_supported_element_types(),
            "requires_gpu": False,
            "supports_batch_processing": False,
            "version": "0.1.0"
        }
        
    def get_supported_element_types(self) -> List[str]:
        """Get the element types this detector can identify"""
        return ["button", "text_field", "checkbox", "dropdown", "icon", "menu_item"]
    
    def _get_name(self) -> str:
        """Get the name of this detector"""
        return self.__class__.__name__.lower().replace('detector', '')
        
    def _measure_execution_time(self, func, *args, **kwargs) -> Tuple[Any, float]:
        """Utility to measure execution time of a function"""
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        return result, execution_time
        
class SpecializedDetectorInterface(UIDetectorInterface):
    """Interface for detectors that specialize in specific UI element types"""
    
    def __init__(self, element_type: str):
        """
        Initialize specialized detector
        
        Args:
            element_type: The UI element type this detector specializes in
        """
        super().__init__()
        self.element_type = element_type
        
    def get_supported_element_types(self) -> List[str]:
        """Specialized detectors typically support only one element type"""
        return [self.element_type]
        
    def get_capabilities(self) -> Dict[str, Any]:
        """Add specialization info to capabilities"""
        capabilities = super().get_capabilities()
        capabilities["specialized_for"] = self.element_type
        return capabilities
