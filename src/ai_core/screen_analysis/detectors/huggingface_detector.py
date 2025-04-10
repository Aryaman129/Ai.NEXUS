"""
UI element detector using Hugging Face models.

This detector leverages pre-trained object detection models from the Hugging Face
model hub to detect UI elements in screenshots. It provides a good balance of
accuracy and accessibility, especially when AutoGluon is not available.
"""

import logging
import time
import os
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import cv2

from .base import UIDetectorInterface

logger = logging.getLogger(__name__)

class HuggingFaceDetector(UIDetectorInterface):
    """UI element detector using Hugging Face Vision models"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Hugging Face detector
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.detector = None
        self.processor = None
        self.is_available = self._check_transformers_available()
        self.model_name = self.config.get("model_name", "microsoft/table-transformer-detection")
        self.initialized = False
        
        # Default label mapping (can be overridden by config)
        self.label_map = self.config.get("label_map", {
            0: "button",
            1: "text_field",
            2: "checkbox",
            3: "dropdown",
            4: "radio_button",
            5: "icon",
            6: "menu_item"
        })
        
    def _check_transformers_available(self) -> bool:
        """
        Check if transformers library is available
        
        Returns:
            True if available, False otherwise
        """
        try:
            import transformers
            return True
        except ImportError:
            logger.warning("Hugging Face Transformers not available. Install with: pip install transformers torch")
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
            self.model_name = self.config.get("model_name", self.model_name)
            
        if not self.is_available:
            logger.warning("Hugging Face Transformers not available, can't initialize detector")
            return False
            
        try:
            # Import here to avoid errors if transformers isn't available
            from transformers import AutoFeatureExtractor, AutoModelForObjectDetection
            
            logger.info(f"Loading Hugging Face model: {self.model_name}")
            self.processor = AutoFeatureExtractor.from_pretrained(self.model_name)
            self.detector = AutoModelForObjectDetection.from_pretrained(self.model_name)
            
            # Update label map if provided in config
            if "label_map" in self.config:
                self.label_map = self.config["label_map"]
                
            logger.info(f"Initialized Hugging Face model: {self.model_name}")
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Hugging Face detector: {e}")
            return False
            
    def detect_elements(self, screenshot: np.ndarray, context: Optional[Dict] = None) -> List[Dict]:
        """
        Detect UI elements in the screenshot
        
        Args:
            screenshot: Image as numpy array (RGB format)
            context: Optional context information
            
        Returns:
            List of detected elements
        """
        if not self.is_available or not self.detector or not self.processor or not self.initialized:
            logger.warning("Hugging Face detector not initialized or available")
            return []
            
        try:
            # Start timing
            start_time = time.time()
            
            # Convert to RGB if needed
            if len(screenshot.shape) == 3 and screenshot.shape[2] == 3:
                # Check if it's BGR (OpenCV format) and convert to RGB
                if screenshot[0, 0, 0] > screenshot[0, 0, 2]:  # Simple BGR check
                    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
            
            # Prepare image for model
            inputs = self.processor(images=screenshot, return_tensors="pt")
            
            # Run detection
            import torch
            with torch.no_grad():
                outputs = self.detector(**inputs)
            
            # Convert outputs to standard format
            target_sizes = torch.tensor([screenshot.shape[:2]])
            results = self.processor.post_process_object_detection(
                outputs, 
                threshold=self.config.get("confidence_threshold", 0.5), 
                target_sizes=target_sizes
            )[0]
            
            # Process results into standard format
            ui_elements = []
            
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                x1, y1, x2, y2 = box.tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Map model label to element type
                label_id = int(label.item())
                element_type = self.label_map.get(label_id, "unknown")
                
                ui_elements.append({
                    "type": element_type,
                    "bbox": (x1, y1, x2, y2),
                    "center": (center_x, center_y),
                    "confidence": float(score.item()),
                    "text": "",  # To be filled by OCR in a second pass
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "detector": "huggingface",
                    "label_id": label_id
                })
                
            # Calculate timing
            elapsed_time = time.time() - start_time
            
            # Add context if provided
            if context:
                for element in ui_elements:
                    element["context"] = context
                    
            logger.debug(f"Hugging Face detected {len(ui_elements)} elements in {elapsed_time:.3f}s")
            
            return ui_elements
            
        except Exception as e:
            logger.error(f"Error in Hugging Face detection: {e}")
            return []
            
    def batch_detect_elements(self, screenshots: List[np.ndarray], 
                             contexts: Optional[List[Dict]] = None) -> List[List[Dict]]:
        """
        Detect UI elements in multiple screenshots
        
        Args:
            screenshots: List of images as numpy arrays
            contexts: Optional list of context information
            
        Returns:
            List of detection results, one per screenshot
        """
        if not self.is_available or not self.detector or not self.processor or not self.initialized:
            logger.warning("Hugging Face detector not initialized or available")
            return [[] for _ in range(len(screenshots))]
            
        try:
            # Start timing
            start_time = time.time()
            
            # Process each screenshot individually
            # Could be optimized for batch processing in the future
            all_elements = []
            
            for i, screenshot in enumerate(screenshots):
                elements = self.detect_elements(screenshot, contexts[i] if contexts and i < len(contexts) else None)
                all_elements.append(elements)
                
            # Calculate timing
            elapsed_time = time.time() - start_time
            logger.debug(f"Hugging Face detected elements in {len(screenshots)} screenshots in {elapsed_time:.3f}s")
            
            return all_elements
            
        except Exception as e:
            logger.error(f"Error in Hugging Face batch detection: {e}")
            return [[] for _ in range(len(screenshots))]
            
    def supports_incremental_learning(self) -> bool:
        """Whether this detector supports incremental learning"""
        return False  # Standard implementation doesn't support incremental learning
        
    def get_capabilities(self) -> Dict[str, Any]:
        """Get detector capabilities and metadata"""
        capabilities = super().get_capabilities()
        
        # Add Hugging Face-specific capabilities
        capabilities.update({
            "name": "huggingface",
            "requires_gpu": self.config.get("requires_gpu", False),  # Some models can run on CPU
            "version": "0.1.0",
            "model_name": self.model_name,
            "supports_batch_processing": False  # Current implementation doesn't optimize batch processing
        })
        
        return capabilities
