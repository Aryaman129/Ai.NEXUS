"""
UI Reasoning Engine

This module provides high-level reasoning capabilities about UI elements
by combining visual detection, semantic understanding, and adaptive learning.
"""

import os
import logging
import json
import time
import threading
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from dataclasses import dataclass, field
import queue

# Import components
from .vision_layer import UIElement, EnhancedVisionDetector
from .semantic_layer import SemanticAnalyzer
from .confidence_calibrator import AdaptiveConfidenceCalibrator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UIAnalysisResult:
    """Complete result of UI analysis"""
    elements: List[UIElement]
    structure: Dict[str, Any] = field(default_factory=dict) 
    application_type: str = ""
    confidence: float = 0.0
    processing_time: float = 0.0
    analyzer_version: str = "0.1.0"

class UIReasoningEngine:
    """
    High-level reasoning engine for UI understanding.
    
    This component integrates visual detection, semantic understanding,
    and adaptive learning to provide comprehensive UI intelligence.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the reasoning engine"""
        self.config = config or {}
        
        # Initialize components
        self.vision_detector = EnhancedVisionDetector(
            self.config.get("vision_detector_config")
        )
        
        self.semantic_analyzer = SemanticAnalyzer(
            self.config.get("semantic_analyzer_config")
        )
        
        self.confidence_calibrator = AdaptiveConfidenceCalibrator(
            self.config.get("confidence_calibrator_config")
        )
        
        # Analysis history for adaptive learning
        self.analysis_history = []
        self.max_history_size = self.config.get("max_history_size", 100)
        
        # Quality thresholds
        self.min_detection_confidence = self.config.get("min_detection_confidence", 0.5)
        self.min_elements_required = self.config.get("min_elements_required", 3)
        
        # Asynchronous processing
        self.async_processing = self.config.get("async_processing", False)
        
        # State tracking
        self.current_application = None
        self.current_screen_id = None
        
        logger.info("UIReasoningEngine initialized")
    
    def analyze_screen(self, 
                     screenshot: np.ndarray,
                     context: Optional[Dict] = None) -> UIAnalysisResult:
        """
        Perform comprehensive analysis of a UI screenshot.
        
        Args:
            screenshot: Screenshot image to analyze
            context: Optional context information to enhance analysis
            
        Returns:
            Complete UI analysis result
        """
        if screenshot is None or screenshot.size == 0:
            logger.error("Invalid screenshot provided")
            return UIAnalysisResult(elements=[])
        
        start_time = time.time()
        
        # 1. Visual detection of UI elements
        elements = self.vision_detector.detect_elements(screenshot)
        
        # Early exit if no elements detected
        if len(elements) < self.min_elements_required:
            logger.warning(f"Too few elements detected: {len(elements)}")
            return UIAnalysisResult(
                elements=elements,
                confidence=0.3,
                processing_time=time.time() - start_time
            )
        
        # 2. Semantic understanding
        elements = self.semantic_analyzer.extract_text_from_elements(screenshot, elements)
        
        if not self.async_processing:
            # Synchronous processing
            elements = self.semantic_analyzer.analyze_elements(screenshot, elements)
            structure = self.semantic_analyzer.understand_ui_structure(screenshot, elements)
        else:
            # Asynchronous processing
            self.semantic_analyzer.analyze_elements(screenshot, elements, async_mode=True)
            structure = {}  # Will be filled in asynchronously
        
        # 3. Calibrate confidence scores
        for element in elements:
            calibrated_confidence = self.confidence_calibrator.calibrate_confidence(
                detector_name=element.detection_method,
                element_type=element.element_type,
                raw_confidence=element.confidence
            )
            element.confidence = calibrated_confidence
        
        # 4. Determine application type
        application_type = self._determine_application_type(elements, structure, context)
        
        # 5. Filter low-confidence elements
        filtered_elements = [e for e in elements if e.confidence >= self.min_detection_confidence]
        
        # 6. Create analysis result
        analysis = UIAnalysisResult(
            elements=filtered_elements,
            structure=structure,
            application_type=application_type,
            confidence=self._calculate_overall_confidence(filtered_elements),
            processing_time=time.time() - start_time
        )
        
        # 7. Update history
        self._update_analysis_history(analysis)
        
        logger.info(f"Screen analysis completed with {len(filtered_elements)} elements in {analysis.processing_time:.3f}s")
        return analysis
    
    def provide_feedback(self, 
                       element_id: str, 
                       was_correct: bool,
                       correct_type: Optional[str] = None) -> None:
        """
        Provide feedback on detection accuracy for adaptive learning.
        
        Args:
            element_id: ID of the element to provide feedback for
            was_correct: Whether the detection was correct
            correct_type: If detection was incorrect, provide the correct type
        """
        # Find element in recent analyses
        element = None
        for analysis in reversed(self.analysis_history):
            for e in analysis.elements:
                if e.element_id == element_id:
                    element = e
                    break
            if element:
                break
        
        if not element:
            logger.warning(f"Element {element_id} not found in history for feedback")
            return
        
        # Update calibrator with feedback
        self.confidence_calibrator.update_calibration(
            detector_name=element.detection_method,
            element_type=element.element_type,
            raw_confidence=element.confidence,
            was_correct=was_correct
        )
        
        # If a correct type was provided, use it for additional learning
        if not was_correct and correct_type:
            logger.info(f"Learning correction: {element.element_type} -> {correct_type}")
            # Here you could implement additional learning logic
        
        logger.info(f"Feedback recorded for element {element_id}: correct={was_correct}")
    
    def get_interaction_suggestions(self, 
                                 element_id: str) -> List[Dict]:
        """
        Get suggestions for how to interact with a UI element.
        
        Args:
            element_id: ID of the element to get suggestions for
            
        Returns:
            List of interaction suggestions
        """
        # Find element in most recent analysis
        element = None
        if self.analysis_history:
            latest_analysis = self.analysis_history[-1]
            for e in latest_analysis.elements:
                if e.element_id == element_id:
                    element = e
                    break
        
        if not element:
            logger.warning(f"Element {element_id} not found for interaction suggestions")
            return []
        
        # Generate suggestions based on element type
        suggestions = []
        
        # Basic interactions by element type
        if element.element_type in ["button", "link", "menu_item"]:
            suggestions.append({
                "action": "click",
                "description": f"Click the {element.element_type}",
                "confidence": 0.9
            })
            
        elif element.element_type in ["text_field", "search_box", "input"]:
            suggestions.append({
                "action": "type",
                "description": f"Type text into the {element.element_type}",
                "confidence": 0.9
            })
            suggestions.append({
                "action": "click",
                "description": f"Click to focus the {element.element_type}",
                "confidence": 0.8
            })
            
        elif element.element_type in ["checkbox", "radio_button", "toggle"]:
            suggestions.append({
                "action": "toggle",
                "description": f"Toggle the state of the {element.element_type}",
                "confidence": 0.9
            })
            
        elif element.element_type in ["dropdown", "select", "combo_box"]:
            suggestions.append({
                "action": "expand",
                "description": f"Expand the {element.element_type} to show options",
                "confidence": 0.9
            })
            suggestions.append({
                "action": "select",
                "description": f"Select an option from the {element.element_type}",
                "confidence": 0.8
            })
            
        elif element.element_type in ["scrollbar", "slider"]:
            suggestions.append({
                "action": "drag",
                "description": f"Drag the {element.element_type} to scroll/adjust",
                "confidence": 0.9
            })
            
        # Add context-sensitive interactions based on element attributes
        if "purpose" in element.attributes:
            purpose = element.attributes["purpose"]
            suggestions.append({
                "action": "interact",
                "description": f"Interact with element to {purpose}",
                "confidence": 0.7
            })
        
        # Add text-based interactions if element has text
        if hasattr(element, "text") and element.text:
            suggestions.append({
                "action": "read",
                "description": f"Read the text: '{element.text}'",
                "confidence": 0.8
            })
        
        return suggestions
    
    def analyze_interaction_path(self, 
                              goal_description: str) -> List[Dict]:
        """
        Analyze possible interaction paths to achieve a specified goal.
        
        Args:
            goal_description: Description of the goal to achieve
            
        Returns:
            List of interaction steps to achieve the goal
        """
        if not self.analysis_history:
            logger.warning("No analysis history available for interaction path analysis")
            return []
        
        # Get most recent analysis
        latest_analysis = self.analysis_history[-1]
        
        # Basic interaction path analysis based on goal keywords
        lower_goal = goal_description.lower()
        
        # Extract key elements from the UI
        buttons = [e for e in latest_analysis.elements if e.element_type == "button"]
        inputs = [e for e in latest_analysis.elements if e.element_type in ["text_field", "input", "search_box"]]
        clickables = [e for e in latest_analysis.elements if e.element_type in ["link", "menu_item", "button"]]
        
        # Build a path based on goal and available elements
        path = []
        
        # Check for common operations in the goal
        if "search" in lower_goal:
            # Search operation
            search_inputs = [e for e in inputs if "search" in e.element_type.lower() or 
                           (hasattr(e, "text") and "search" in e.text.lower())]
            
            if search_inputs:
                search_input = search_inputs[0]
                path.append({
                    "step": 1,
                    "element_id": search_input.element_id,
                    "action": "click",
                    "description": "Click on the search field"
                })
                
                path.append({
                    "step": 2,
                    "element_id": search_input.element_id,
                    "action": "type",
                    "description": f"Type search terms related to: {goal_description}"
                })
                
                # Look for a search button
                search_buttons = [b for b in buttons if hasattr(b, "text") and 
                               ("search" in b.text.lower() or "find" in b.text.lower())]
                
                if search_buttons:
                    path.append({
                        "step": 3,
                        "element_id": search_buttons[0].element_id,
                        "action": "click",
                        "description": "Click the search button"
                    })
        
        elif "login" in lower_goal or "sign in" in lower_goal:
            # Login operation
            username_inputs = [e for e in inputs if any(term in e.element_type.lower() or 
                                                    (hasattr(e, "text") and term in e.text.lower())
                                                    for term in ["user", "email", "login", "username"])]
            
            password_inputs = [e for e in inputs if "password" in e.element_type.lower() or 
                             (hasattr(e, "text") and "password" in e.text.lower())]
            
            login_buttons = [b for b in buttons if hasattr(b, "text") and 
                           any(term in b.text.lower() for term in ["login", "sign in", "log in"])]
            
            if username_inputs:
                path.append({
                    "step": 1,
                    "element_id": username_inputs[0].element_id,
                    "action": "type",
                    "description": "Enter username or email"
                })
            
            if password_inputs:
                path.append({
                    "step": 2,
                    "element_id": password_inputs[0].element_id,
                    "action": "type",
                    "description": "Enter password"
                })
            
            if login_buttons:
                path.append({
                    "step": 3,
                    "element_id": login_buttons[0].element_id,
                    "action": "click",
                    "description": "Click the login button"
                })
        
        else:
            # Generic goal - look for relevant clickable elements
            relevant_elements = []
            
            for element in clickables:
                if hasattr(element, "text") and element.text:
                    # Calculate relevance based on keyword matching
                    words = goal_description.lower().split()
                    element_words = element.text.lower().split()
                    
                    # Count matching words
                    matches = sum(1 for w in words if any(w in ew for ew in element_words))
                    
                    if matches > 0:
                        relevant_elements.append((element, matches))
            
            # Sort by relevance
            relevant_elements.sort(key=lambda x: x[1], reverse=True)
            
            # Add top relevant elements to the path
            for i, (element, _) in enumerate(relevant_elements[:3]):
                path.append({
                    "step": i + 1,
                    "element_id": element.element_id,
                    "action": "click",
                    "description": f"Click on '{element.text}'"
                })
        
        return path
    
    def _determine_application_type(self, 
                                 elements: List[UIElement], 
                                 structure: Dict,
                                 context: Optional[Dict]) -> str:
        """Determine the application type from UI elements and structure"""
        # Use structure information if available
        if structure and "application_type" in structure:
            return structure["application_type"]
        
        # Use context information if available
        if context and "application_type" in context:
            return context["application_type"]
        
        # Count element types to infer application type
        type_counts = {}
        for element in elements:
            if element.element_type not in type_counts:
                type_counts[element.element_type] = 0
            type_counts[element.element_type] += 1
        
        # Look for text content in elements
        text_content = " ".join([
            e.text for e in elements 
            if hasattr(e, "text") and e.text
        ])
        
        # Detect application type based on element distribution and text
        if "menu_item" in type_counts and type_counts["menu_item"] > 5:
            return "desktop_application"
            
        if "table_cell" in type_counts and type_counts["table_cell"] > 10:
            return "data_application"
            
        if "link" in type_counts and type_counts["link"] > 5:
            return "web_application"
            
        text_lower = text_content.lower()
        if any(word in text_lower for word in ["login", "sign in", "create account"]):
            return "authentication_screen"
            
        if any(word in text_lower for word in ["search", "find", "filter"]):
            return "search_interface"
            
        # Default to generic UI
        return "generic_interface"
    
    def _calculate_overall_confidence(self, elements: List[UIElement]) -> float:
        """Calculate overall confidence in the analysis"""
        if not elements:
            return 0.0
            
        # Weight elements by area (larger elements have more impact)
        weighted_confidences = []
        total_area = 0
        
        for element in elements:
            area = element.width * element.height
            weighted_confidences.append((element.confidence, area))
            total_area += area
        
        # Calculate weighted average confidence
        if total_area == 0:
            return sum(c for c, _ in weighted_confidences) / len(weighted_confidences)
            
        weighted_sum = sum(confidence * area for confidence, area in weighted_confidences)
        return weighted_sum / total_area
    
    def _update_analysis_history(self, analysis: UIAnalysisResult) -> None:
        """Update the analysis history"""
        self.analysis_history.append(analysis)
        
        # Trim history if needed
        if len(self.analysis_history) > self.max_history_size:
            self.analysis_history = self.analysis_history[-self.max_history_size:]
    
    def get_metrics(self) -> Dict:
        """Get performance metrics for the reasoning engine"""
        metrics = {
            "total_analyses": len(self.analysis_history),
            "average_elements_detected": 0,
            "average_confidence": 0,
            "average_processing_time": 0,
            "detector_metrics": self.confidence_calibrator.get_performance_metrics()
        }
        
        if self.analysis_history:
            metrics["average_elements_detected"] = sum(len(a.elements) for a in self.analysis_history) / len(self.analysis_history)
            metrics["average_confidence"] = sum(a.confidence for a in self.analysis_history) / len(self.analysis_history)
            metrics["average_processing_time"] = sum(a.processing_time for a in self.analysis_history) / len(self.analysis_history)
        
        return metrics
