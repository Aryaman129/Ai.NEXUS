"""
Visualization tools for UI element detection in NEXUS.

This module provides visualization capabilities for detected UI elements,
useful for debugging and demonstrating the detection results. It supports
different coloring schemes for different element types and detectors.
"""

import logging
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class DetectionVisualizer:
    """Visualizer for UI element detection results"""
    
    def __init__(self):
        """Initialize the visualizer with default color schemes"""
        # Color scheme for element types (BGR format for OpenCV)
        self.type_colors = {
            "button": (0, 165, 255),      # Orange
            "text_field": (0, 255, 0),    # Green
            "checkbox": (255, 0, 0),      # Blue
            "dropdown": (255, 0, 255),    # Magenta
            "radio_button": (255, 255, 0), # Cyan
            "slider": (0, 0, 255),        # Red
            "toggle": (128, 0, 128),      # Purple
            "icon": (255, 255, 255),      # White
            "menu_item": (0, 255, 255),   # Yellow
            "unknown": (128, 128, 128)    # Gray
        }
        
        # Color scheme for detectors
        self.detector_colors = {
            "autogluon": (0, 165, 255),   # Orange
            "huggingface": (0, 255, 0),   # Green
            "opencv": (255, 0, 0),        # Blue
            "gemini": (255, 0, 255),      # Magenta
            "legacy": (128, 128, 128),    # Gray
            "unknown": (255, 255, 255)    # White
        }
        
        # Line thickness
        self.thickness = 2
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 1
        
    def visualize_detections(self, image: np.ndarray, elements: List[Dict],
                            color_by: str = "type",
                            show_confidence: bool = True,
                            show_text: bool = False,
                            show_detector: bool = True) -> np.ndarray:
        """
        Visualize detected UI elements on an image
        
        Args:
            image: Original image as numpy array
            elements: List of detected elements
            color_by: What to use for coloring ('type' or 'detector')
            show_confidence: Whether to show confidence scores
            show_text: Whether to show extracted text
            show_detector: Whether to show detector name
            
        Returns:
            Image with visualized detections
        """
        # Make a copy of the image to avoid modifying the original
        vis_image = image.copy()
        
        # Draw each element
        for i, element in enumerate(elements):
            # Get bounding box
            x1, y1, x2, y2 = element["bbox"]
            
            # Determine color
            if color_by == "detector":
                detector = element.get("detector", "unknown")
                color = self.detector_colors.get(detector, self.detector_colors["unknown"])
            else:
                element_type = element.get("type", "unknown")
                color = self.type_colors.get(element_type, self.type_colors["unknown"])
                
            # Draw rectangle
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, self.thickness)
            
            # Prepare label
            label_parts = []
            
            # Add element type
            element_type = element.get("type", "unknown")
            label_parts.append(element_type)
            
            # Add confidence if requested
            if show_confidence and "confidence" in element:
                confidence = element["confidence"]
                label_parts.append(f"{confidence:.2f}")
                
            # Add detector if requested
            if show_detector and "detector" in element:
                detector = element["detector"]
                label_parts.append(detector)
                
            # Combine parts
            label = ":".join(label_parts)
            
            # Draw label background
            label_size, _ = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)
            cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(vis_image, label, (x1, y1 - 5), self.font, self.font_scale, (255, 255, 255), self.font_thickness)
            
            # Draw text if requested and available
            if show_text and "text" in element and element["text"]:
                text = element["text"]
                # Truncate long text
                if len(text) > 20:
                    text = text[:17] + "..."
                text_y = y2 + 15
                cv2.putText(vis_image, text, (x1, text_y), self.font, self.font_scale, color, self.font_thickness)
                
        return vis_image
        
    def create_comparison_visualization(self, image: np.ndarray, detections_by_detector: Dict[str, List[Dict]],
                                      show_confidence: bool = True) -> np.ndarray:
        """
        Create a side-by-side comparison of different detectors
        
        Args:
            image: Original image as numpy array
            detections_by_detector: Dictionary mapping detector names to their results
            show_confidence: Whether to show confidence scores
            
        Returns:
            Composite image with visualized comparisons
        """
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Determine grid layout
        num_detectors = len(detections_by_detector) + 1  # +1 for original image
        cols = min(3, num_detectors)
        rows = (num_detectors + cols - 1) // cols
        
        # Create canvas
        canvas_width = width * cols
        canvas_height = height * rows
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        # Add original image in first position
        canvas[0:height, 0:width] = image.copy()
        
        # Add label for original
        label = "Original"
        cv2.putText(canvas, label, (10, 30), self.font, 1.0, (255, 255, 255), 2)
        
        # Add visualizations for each detector
        i = 1
        for detector_name, elements in detections_by_detector.items():
            # Calculate position
            row = i // cols
            col = i % cols
            y_start = row * height
            x_start = col * width
            
            # Create visualization
            vis = self.visualize_detections(image, elements, color_by="type", 
                                           show_confidence=show_confidence,
                                           show_detector=False)
            
            # Add to canvas
            canvas[y_start:y_start+height, x_start:x_start+width] = vis
            
            # Add label
            element_count = len(elements)
            label = f"{detector_name} ({element_count})"
            cv2.putText(canvas, label, (x_start + 10, y_start + 30), self.font, 1.0, (255, 255, 255), 2)
            
            i += 1
            
        return canvas
        
    def create_consensus_visualization(self, image: np.ndarray, consensus_elements: List[Dict],
                                     individual_results: List[List[Dict]],
                                     detector_names: List[str]) -> np.ndarray:
        """
        Create visualization showing consensus detection results
        
        Args:
            image: Original image as numpy array
            consensus_elements: Elements detected with consensus voting
            individual_results: List of element lists from individual detectors
            detector_names: Names of the detectors, in same order as individual_results
            
        Returns:
            Visualization image
        """
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Determine layout: 2x2 grid with original, consensus, and top 2 detectors
        canvas_width = width * 2
        canvas_height = height * 2
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        # Original image (top-left)
        canvas[0:height, 0:width] = image.copy()
        cv2.putText(canvas, "Original", (10, 30), self.font, 1.0, (255, 255, 255), 2)
        
        # Consensus results (top-right)
        consensus_vis = self.visualize_detections(image, consensus_elements, color_by="type", 
                                               show_confidence=True, show_detector=False)
        canvas[0:height, width:width*2] = consensus_vis
        cv2.putText(canvas, f"Consensus ({len(consensus_elements)})", (width + 10, 30), 
                   self.font, 1.0, (255, 255, 255), 2)
        
        # Top individual detectors (bottom row)
        for i in range(min(2, len(individual_results))):
            col = i
            detector_name = detector_names[i] if i < len(detector_names) else f"Detector {i}"
            elements = individual_results[i]
            
            vis = self.visualize_detections(image, elements, color_by="type", 
                                          show_confidence=True, show_detector=False)
            
            y_start = height
            x_start = col * width
            canvas[y_start:y_start+height, x_start:x_start+width] = vis
            
            cv2.putText(canvas, f"{detector_name} ({len(elements)})", (x_start + 10, y_start + 30), 
                       self.font, 1.0, (255, 255, 255), 2)
            
        return canvas
        
    def save_visualization(self, image: np.ndarray, file_path: str) -> bool:
        """
        Save visualization to a file
        
        Args:
            image: Visualization image
            file_path: Path to save the image
            
        Returns:
            Success status
        """
        try:
            cv2.imwrite(file_path, image)
            logger.info(f"Saved visualization to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving visualization: {e}")
            return False
