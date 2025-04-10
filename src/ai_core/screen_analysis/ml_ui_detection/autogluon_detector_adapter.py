"""
AutoGluon Detector Adapter

This module provides an adapter that integrates the existing
AutoGluon detector with the enhanced detector registry.
"""

import logging
import os
import sys
import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# Import the detector interface
from ..ui_detection.detector_interface import UIDetectorInterface

# Dynamic import to avoid circular dependencies
import importlib

logger = logging.getLogger(__name__)


class AutoGluonDetectorAdapter(UIDetectorInterface):
    """
    Adapter for AutoGluon-based UI element detection that integrates
    with the enhanced detector registry system.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the AutoGluon detector adapter
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.detector = None
        self.initialized = False
        self.specialized_elements = self.config.get("specialized_elements", ["button", "checkbox", "dropdown"])
        
    def initialize(self, config: Optional[Dict] = None) -> bool:
        """
        Initialize the detector with optional configuration
        
        Args:
            config: Optional configuration dictionary
            
        Returns:
            Success status
        """
        if config:
            self.config.update(config)
            
        try:
            # Import the original AutoGluon detector dynamically
            autogluon_module = importlib.import_module("..screen_analysis.detectors.autogluon_detector", package="src.ai_core")
            AutoGluonDetector = getattr(autogluon_module, "AutoGluonDetector")
            
            # Create an instance of the AutoGluon detector
            self.detector = AutoGluonDetector(self.config)
            
            # Initialize the detector
            success = self.detector.initialize(self.config)
            if success:
                self.initialized = True
                logger.info("AutoGluon detector adapter initialized successfully")
                return True
            else:
                logger.warning("Failed to initialize AutoGluon detector")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing AutoGluon detector adapter: {e}")
            return False
            
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get detector capabilities
        
        Returns:
            Dictionary of capabilities
        """
        if not self.initialized or not self.detector:
            return {
                "name": "autogluon_adapter",
                "version": "1.0.0",
                "initialized": False,
                "specialized_for": self.specialized_elements
            }
            
        # Get capabilities from the underlying detector
        detector_capabilities = self.detector.get_capabilities() if hasattr(self.detector, "get_capabilities") else {}
        
        # Add adapter-specific capabilities
        capabilities = {
            "name": "autogluon_adapter",
            "version": "1.0.0",
            "initialized": self.initialized,
            "specialized_for": self.specialized_elements,
            **detector_capabilities
        }
        
        return capabilities
        
    def detect_elements(self, screenshot: np.ndarray, context: Optional[Dict] = None) -> List[Dict]:
        """
        Detect UI elements in the screenshot
        
        Args:
            screenshot: Screenshot as numpy array
            context: Optional context information
            
        Returns:
            List of detected elements
        """
        if not self.initialized or not self.detector:
            logger.warning("AutoGluon detector adapter not initialized")
            return []
            
        try:
            # Measure detection time
            start_time = time.time()
            
            # Call the underlying detector
            elements = self.detector.detect_elements(screenshot, context)
            
            # Calculate detection time
            detection_time = time.time() - start_time
            
            # Enhance detections with additional metadata
            for element in elements:
                # Add detection method
                element["detection_method"] = "autogluon"
                
                # Add timing information
                element["detection_time"] = detection_time / len(elements) if elements else 0
                
                # Add context-based confidence boost for specialized elements
                if element.get("type") in self.specialized_elements:
                    # Boost confidence for elements this detector specializes in
                    element["confidence"] = min(0.99, element.get("confidence", 0.5) * 1.2)
                    element["specialized"] = True
            
            return elements
            
        except Exception as e:
            logger.error(f"Error detecting elements with AutoGluon: {e}")
            return []
            
    def enhance_with_metadata(self, elements: List[Dict], context: Optional[Dict] = None) -> List[Dict]:
        """
        Enhance detected elements with additional metadata
        
        Args:
            elements: List of detected elements
            context: Optional context information
            
        Returns:
            Enhanced list of elements
        """
        # Skip if no elements or not initialized
        if not elements or not self.initialized:
            return elements
            
        # Get application context if available
        app_context = context.get("app_name", "") if context else ""
        
        for element in elements:
            # Add confidence boost for specialized elements in known applications
            if app_context and element.get("type") in self.specialized_elements:
                element["context_relevance"] = 0.8
                
            # Add metadata about detection quality
            confidence = element.get("confidence", 0.5)
            element["detection_quality"] = {
                "confidence": confidence,
                "reliability": min(1.0, confidence * 1.2) if element.get("specialized") else confidence,
                "detector": "autogluon"
            }
            
        return elements
        
    def supports_incremental_learning(self) -> bool:
        """
        Whether this detector supports incremental learning
        
        Returns:
            True if incremental learning is supported, False otherwise
        """
        # If the underlying detector supports learning, we do too
        if self.detector and hasattr(self.detector, "supports_incremental_learning"):
            return self.detector.supports_incremental_learning()
        # Otherwise we have our own basic learning capabilities
        return True
        
    def learn_from_feedback(self, elements: List[Dict], feedback: Dict):
        """
        Learn from user feedback to improve future detections
        
        Args:
            elements: List of detected elements
            feedback: Feedback information
        """
        # Pass feedback to underlying detector if it supports learning
        if hasattr(self.detector, "learn_from_feedback"):
            try:
                self.detector.learn_from_feedback(elements, feedback)
                logger.info("Passed feedback to AutoGluon detector for learning")
            except Exception as e:
                logger.error(f"Error passing feedback to AutoGluon detector: {e}")
                
        # Update specialized elements based on feedback
        if feedback and "element_type_accuracy" in feedback:
            type_accuracy = feedback["element_type_accuracy"]
            # Add element types with high accuracy to specialized list
            for element_type, accuracy in type_accuracy.items():
                if accuracy > 0.8 and element_type not in self.specialized_elements:
                    self.specialized_elements.append(element_type)
                    logger.info(f"Added {element_type} to specialized elements based on feedback")
                    
            # Update configuration
            self.config["specialized_elements"] = self.specialized_elements
