"""
HuggingFace UI Element Detector

This module provides a UI detector implementation that leverages
HuggingFace's vision models for advanced UI element detection.
"""

import time
import logging
import os
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# Import the detector interface
from ..ui_detection.detector_interface import UIDetectorInterface

logger = logging.getLogger(__name__)

class HuggingFaceDetector(UIDetectorInterface):
    """
    UI element detector using HuggingFace vision models
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the HuggingFace detector
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.initialized = False
        self.model = None
        self.processor = None
        self.is_available = self._check_huggingface_available()
        # Use a proper object detection model instead of LayoutLM
        self.model_name = self.config.get("model_name", "facebook/detr-resnet-50")
        self.token = self.config.get("token")
        
        # Cache to avoid repeated downloads
        self.model_cache = {}
        
        # Element type mapping
        self.category_mapping = {
            "paragraph": "text_field",
            "title": "text_field",
            "list": "text_field",
            "table": "text_field",
            "header": "text_field",
            "footer": "text_field",
            "page-header": "text_field",
            "button": "button",
            "checkbox": "checkbox",
            "radio": "radio_button",
            "field": "text_field",
            "dropdown": "dropdown",
            "menu": "menu",
            "navigation": "menu",
            "image": "icon",
            "figure": "icon"
        }
        
    def _check_huggingface_available(self) -> bool:
        """
        Check if HuggingFace Transformers and APIs are available
        
        Returns:
            True if available, False otherwise
        """
        try:
            import transformers
            return True
        except ImportError:
            logger.warning("HuggingFace Transformers not available. Install with: pip install transformers")
            return False
            
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
            
        if not self.is_available:
            logger.warning("HuggingFace not available, can't initialize detector")
            return False
            
        try:
            # Import here to avoid errors if HuggingFace isn't available
            from transformers import AutoProcessor, AutoModelForObjectDetection
            import torch
            
            # Set HuggingFace token if provided
            token = self.config.get("token", self.token)
            if token:
                os.environ["HF_TOKEN"] = token
            
            # Use model from config
            model_name = self.config.get("model_name", self.model_name)
            
            # Check if model is already loaded
            if model_name in self.model_cache:
                self.model = self.model_cache[model_name]["model"]
                self.processor = self.model_cache[model_name]["processor"]
                self.initialized = True
                logger.info(f"Loaded cached HuggingFace model: {model_name}")
                return True
            
            # Load processor and model
            logger.info(f"Loading HuggingFace model: {model_name}")
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForObjectDetection.from_pretrained(model_name)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.model.to("cuda")
                logger.info("Using GPU for HuggingFace model")
            
            # Cache the model
            self.model_cache[model_name] = {
                "model": self.model,
                "processor": self.processor
            }
            
            self.initialized = True
            logger.info(f"Initialized HuggingFace detector with model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing HuggingFace detector: {e}")
            return False
            
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get detector capabilities
        
        Returns:
            Dictionary of capabilities
        """
        return {
            "name": "huggingface_detector",
            "version": "1.0.0",
            "model": self.model_name,
            "element_types": list(set(self.category_mapping.values())),
            "supports_ocr": True,
            "supports_layout": True
        }
        
    def detect_elements(self, screenshot: np.ndarray, context: Optional[Dict] = None) -> List[Dict]:
        """
        Detect UI elements in the screenshot
        
        Args:
            screenshot: Screenshot as numpy array
            context: Optional context information
            
        Returns:
            List of detected elements
        """
        if not self.is_available or not self.initialized:
            logger.warning("HuggingFace detector not initialized or available")
            return []
            
        try:
            import torch
            
            # Start timing
            start_time = time.time()
            
            # Convert if needed
            if isinstance(screenshot, np.ndarray):
                # Convert OpenCV BGR to RGB if needed
                if screenshot.shape[2] == 3 and screenshot[0, 0, 0] > screenshot[0, 0, 2]:
                    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
            
            # Process image through model
            inputs = self.processor(images=screenshot, return_tensors="pt")
            
            # Move inputs to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Process results
            target_sizes = torch.tensor([screenshot.shape[:2]])
            results = self.processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=0.5  # Confidence threshold
            )[0]
            
            # Convert to standard format
            ui_elements = []
            
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                # Skip low confidence detections
                confidence = score.item()
                if confidence < 0.5:
                    continue
                
                # Get class name and map to element type
                class_name = self.model.config.id2label[label.item()]
                element_type = self.category_mapping.get(class_name, "unknown")
                
                # Get box coordinates
                x1, y1, x2, y2 = box.tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Calculate center
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                
                ui_elements.append({
                    "type": element_type,
                    "bbox": (x1, y1, x2, y2),
                    "center": (center_x, center_y),
                    "confidence": confidence,
                    "detection_method": "huggingface",
                    "original_label": class_name
                })
            
            return ui_elements
            
        except Exception as e:
            logger.error(f"Error detecting elements with HuggingFace: {e}")
            return []
            
    def filter_by_element_type(self, elements: List[Dict], element_type: str) -> List[Dict]:
        """
        Filter elements by type
        
        Args:
            elements: List of detected elements
            element_type: Element type to filter for
            
        Returns:
            Filtered list of elements
        """
        return [e for e in elements if e["type"] == element_type]
        
    def get_most_confident(self, elements: List[Dict], count: int = 1) -> List[Dict]:
        """
        Get the most confident elements
        
        Args:
            elements: List of detected elements
            count: Number of elements to return
            
        Returns:
            List of most confident elements
        """
        sorted_elements = sorted(elements, key=lambda e: e["confidence"], reverse=True)
        return sorted_elements[:count]
        
    def supports_incremental_learning(self) -> bool:
        """
        Whether this detector supports incremental learning
        
        Returns:
            True if incremental learning is supported, False otherwise
        """
        return False  # HuggingFace models require full retraining by default
