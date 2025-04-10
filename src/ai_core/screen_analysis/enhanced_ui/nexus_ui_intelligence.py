"""
NEXUS UI Intelligence System

This module integrates all enhanced UI understanding components into a unified system that
provides advanced UI understanding and interaction capabilities through adaptive learning.

Key capabilities:
1. Multimodal UI detection using advanced computer vision
2. Semantic understanding of UI elements using LLMs
3. Adaptive learning from user interactions
4. Hierarchical understanding of interface structure
5. Interaction assistance and automation
"""

import os
import cv2
import numpy as np
import logging
import json
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import threading

# Import component modules
from .vision_layer import EnhancedVisionDetector, UIElement
from .semantic_layer import SemanticAnalyzer
from .confidence_calibrator import AdaptiveConfidenceCalibrator
from .reasoning_engine import UIReasoningEngine, UIAnalysisResult

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NexusUIIntelligence:
    """
    Main entry point for NEXUS UI Intelligence capabilities.
    
    This class provides a unified interface to all enhanced UI understanding
    capabilities, including detection, analysis, and interaction.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the NEXUS UI Intelligence system.
        
        Args:
            config_path: Optional path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize the reasoning engine (which initializes all sub-components)
        self.reasoning_engine = UIReasoningEngine(self.config.get("reasoning_engine"))
        
        # Optional direct access to components for advanced usage
        self.vision_detector = self.reasoning_engine.vision_detector
        self.semantic_analyzer = self.reasoning_engine.semantic_analyzer
        self.confidence_calibrator = self.reasoning_engine.confidence_calibrator
        
        # State tracking
        self.current_analysis = None
        self.analysis_history = []
        self.max_history = self.config.get("max_history", 10)
        
        # Performance metrics
        self.total_analyses = 0
        self.total_processing_time = 0
        
        # Initialize result visualization
        self.visualization_enabled = self.config.get("enable_visualization", True)
        
        # Create output directories
        os.makedirs(self.config.get("output_dir", "ui_intelligence_results"), exist_ok=True)
        
        logger.info("NEXUS UI Intelligence initialized")
        
        # Log capabilities based on component availability
        self._log_capabilities()
    
    def analyze_screenshot(self, 
                        screenshot: np.ndarray,
                        context: Optional[Dict] = None) -> UIAnalysisResult:
        """
        Analyze a screenshot to detect and understand UI elements.
        
        Args:
            screenshot: Screenshot image to analyze
            context: Optional context information to enhance analysis
            
        Returns:
            Complete UI analysis result
        """
        if screenshot is None or screenshot.size == 0:
            logger.error("Invalid screenshot provided")
            return None
            
        # Track performance
        start_time = time.time()
        
        # Perform comprehensive analysis
        analysis = self.reasoning_engine.analyze_screen(screenshot, context)
        
        # Update state and history
        self.current_analysis = analysis
        self._update_history(analysis)
        
        # Update performance metrics
        self.total_analyses += 1
        processing_time = time.time() - start_time
        self.total_processing_time += processing_time
        
        # Generate visualization if enabled
        if self.visualization_enabled:
            self._visualize_analysis(screenshot, analysis)
            
        logger.info(f"Screenshot analysis completed in {processing_time:.3f}s")
        return analysis
    
    def provide_feedback(self, 
                       element_id: str, 
                       was_correct: bool,
                       correct_type: Optional[str] = None) -> None:
        """
        Provide feedback on element detection for adaptive learning.
        
        Args:
            element_id: ID of the element to provide feedback for
            was_correct: Whether the detection was correct
            correct_type: If incorrect, the correct element type
        """
        self.reasoning_engine.provide_feedback(element_id, was_correct, correct_type)
    
    def get_element_by_id(self, element_id: str) -> Optional[UIElement]:
        """
        Get a UI element by its ID from the current analysis.
        
        Args:
            element_id: ID of the element to retrieve
            
        Returns:
            The UI element if found, None otherwise
        """
        if not self.current_analysis:
            return None
            
        for element in self.current_analysis.elements:
            if element.element_id == element_id:
                return element
                
        return None
    
    def suggest_interactions(self, 
                          element_id: Optional[str] = None) -> List[Dict]:
        """
        Suggest possible interactions with the UI.
        
        Args:
            element_id: Optional ID of a specific element to get suggestions for
            
        Returns:
            List of interaction suggestions
        """
        if element_id:
            # Get suggestions for a specific element
            return self.reasoning_engine.get_interaction_suggestions(element_id)
        else:
            # No specific element - suggest general interactions
            if not self.current_analysis:
                return []
                
            # Prioritize interactive elements
            interactive_types = [
                "button", "link", "menu_item", "text_field", 
                "checkbox", "radio_button", "dropdown"
            ]
            
            # Find elements by priority
            suggestions = []
            
            for type_name in interactive_types:
                for element in self.current_analysis.elements:
                    if element.element_type == type_name:
                        element_suggestions = self.reasoning_engine.get_interaction_suggestions(element.element_id)
                        if element_suggestions:
                            # Add element information to each suggestion
                            for suggestion in element_suggestions:
                                suggestion["element_id"] = element.element_id
                                suggestion["element_type"] = element.element_type
                                suggestion["element_text"] = element.text if hasattr(element, "text") else ""
                                
                            suggestions.extend(element_suggestions[:2])  # Limit to top 2 per element
                            
                            # Stop after finding enough suggestions
                            if len(suggestions) >= 10:
                                break
                                
                if len(suggestions) >= 10:
                    break
                    
            return suggestions
    
    def plan_interaction(self, goal_description: str) -> List[Dict]:
        """
        Plan a sequence of interactions to achieve a specific goal.
        
        Args:
            goal_description: Description of the goal to achieve
            
        Returns:
            Sequence of interaction steps
        """
        return self.reasoning_engine.analyze_interaction_path(goal_description)
    
    def get_performance_metrics(self) -> Dict:
        """
        Get performance metrics for the UI intelligence system.
        
        Returns:
            Dictionary of performance metrics
        """
        metrics = {
            "total_analyses": self.total_analyses,
            "average_processing_time": self.total_processing_time / max(1, self.total_analyses),
            "component_metrics": self.reasoning_engine.get_metrics()
        }
        
        return metrics
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            "output_dir": "ui_intelligence_results",
            "enable_visualization": True,
            "max_history": 10,
            "reasoning_engine": {
                "min_detection_confidence": 0.5,
                "min_elements_required": 3,
                "async_processing": False,
                "max_history_size": 20,
                "vision_detector_config": {
                    "enabled_detectors": ["contour", "color", "template", "feature"],
                    "min_confidence": 0.5,
                    "template_dir": os.path.join("data", "ui_templates")
                },
                "semantic_analyzer_config": {
                    "ocr_config": {
                        "tesseract_path": None,  # Set this to your Tesseract path if needed
                        "lang": "eng"
                    },
                    "connectors": {
                        "together_ai": {
                            "api_key": None,  # Will use environment variable
                            "model": "meta-llama/Llama-3-70b-instruct"
                        }
                    }
                },
                "confidence_calibrator_config": {
                    "storage_path": os.path.join("data", "calibration"),
                    "learning_rate": 0.1,
                    "min_samples": 5
                }
            }
        }
        
        # If config path provided, load and merge with defaults
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    
                # Deep merge configs (this is a simple version)
                for key, value in loaded_config.items():
                    if isinstance(value, dict) and key in default_config and isinstance(default_config[key], dict):
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
                        
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading config from {config_path}: {e}")
        
        return default_config
    
    def _update_history(self, analysis: UIAnalysisResult) -> None:
        """Update analysis history"""
        self.analysis_history.append(analysis)
        
        # Trim history if needed
        if len(self.analysis_history) > self.max_history:
            self.analysis_history = self.analysis_history[-self.max_history:]
    
    def _visualize_analysis(self, 
                         screenshot: np.ndarray, 
                         analysis: UIAnalysisResult) -> None:
        """Visualize analysis results"""
        if not self.visualization_enabled:
            return
            
        try:
            # Create a copy of the screenshot
            visualization = screenshot.copy()
            
            # Draw bounding boxes for each element
            for element in analysis.elements:
                x1, y1, x2, y2 = element.bbox
                
                # Determine color based on element type
                color_map = {
                    "button": (0, 165, 255),    # Orange
                    "text_field": (0, 255, 0),  # Green
                    "icon": (255, 0, 0),        # Blue
                    "control": (255, 0, 255),   # Magenta
                    "list_item": (255, 255, 0), # Cyan
                    "checkbox": (0, 255, 255),  # Yellow
                    "dropdown": (128, 0, 128),  # Purple
                    "menu_item": (0, 128, 255), # Light blue
                    "link": (255, 128, 0),      # Light orange
                    "text": (200, 200, 200)     # Light gray
                }
                
                color = color_map.get(element.element_type, (128, 128, 128))  # Default: gray
                
                # Adjust color brightness based on confidence
                brightness = max(0.4, min(1.0, element.confidence))
                color = tuple(int(c * brightness) for c in color)
                
                # Draw rectangle
                cv2.rectangle(visualization, (x1, y1), (x2, y2), color, 2)
                
                # Draw element type and ID
                label = f"{element.element_type} ({element.confidence:.2f})"
                cv2.putText(visualization, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Draw text content if available
                if hasattr(element, "text") and element.text:
                    text_label = element.text[:20] + "..." if len(element.text) > 20 else element.text
                    cv2.putText(visualization, text_label, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Save visualization
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                self.config.get("output_dir", "ui_intelligence_results"),
                f"ui_analysis_{timestamp}.png"
            )
            cv2.imwrite(output_path, visualization)
            
            logger.info(f"Saved visualization to {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
    
    def _log_capabilities(self) -> None:
        """Log system capabilities based on available components"""
        capabilities = []
        
        # Vision capabilities
        if hasattr(self.vision_detector, "feature_detector") and self.vision_detector.feature_detector:
            capabilities.append("Advanced feature detection")
            
        if hasattr(self.vision_detector, "templates") and self.vision_detector.templates:
            capabilities.append(f"Template matching ({len(self.vision_detector.templates)} templates)")
        
        # Semantic capabilities
        if hasattr(self.semantic_analyzer, "ocr") and self.semantic_analyzer.ocr.ocr_available:
            capabilities.append("OCR text extraction")
            
        if hasattr(self.semantic_analyzer, "connectors") and self.semantic_analyzer.connectors:
            for connector in self.semantic_analyzer.connectors:
                if connector._available:
                    capabilities.append(f"LLM integration ({connector.name})")
        
        # Learning capabilities
        if hasattr(self.confidence_calibrator, "calibration_data"):
            detector_count = len(self.confidence_calibrator.calibration_data)
            if detector_count > 0:
                capabilities.append(f"Adaptive learning ({detector_count} detectors calibrated)")
        
        if capabilities:
            logger.info("NEXUS UI Intelligence capabilities:")
            for capability in capabilities:
                logger.info(f"  - {capability}")
        else:
            logger.info("NEXUS UI Intelligence initialized with basic capabilities")


# Convenience function for quick initialization
def create_ui_intelligence(config_path: Optional[str] = None) -> NexusUIIntelligence:
    """
    Create and initialize a NEXUS UI Intelligence instance.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Initialized NexusUIIntelligence instance
    """
    return NexusUIIntelligence(config_path)
