"""
UI element detector using AutoGluon.

This detector leverages AutoGluon's automated machine learning capabilities
to detect UI elements in screenshots. It supports both pre-trained models
and training new models from labeled data.
"""

import logging
import time
import os
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import cv2

from .base import UIDetectorInterface

logger = logging.getLogger(__name__)

class AutoGluonDetector(UIDetectorInterface):
    """UI element detector using AutoGluon for automated ML"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the AutoGluon detector
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.detector = None
        self.is_available = self._check_autogluon_available()
        self.model_path = self.config.get("model_path")
        self.initialized = False
        
        # Default label mapping (can be overridden by model)
        self.label_map = {
            0: "button",
            1: "text_field",
            2: "checkbox",
            3: "dropdown",
            4: "radio_button",
            5: "slider",
            6: "toggle",
            7: "icon",
            8: "menu_item"
        }
        
    def _check_autogluon_available(self) -> bool:
        """
        Check if AutoGluon is available
        
        Returns:
            True if AutoGluon is available, False otherwise
        """
        try:
            import autogluon.vision
            return True
        except ImportError:
            logger.warning("AutoGluon not available. Install with: pip install autogluon.vision")
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
            logger.warning("AutoGluon not available, can't initialize detector")
            return False
            
        try:
            # Import here to avoid errors if AutoGluon isn't available
            from autogluon.vision import ObjectDetector
            
            # Use model path from config if provided
            model_path = self.config.get("model_path", self.model_path)
            
            if model_path and os.path.exists(model_path):
                # Load existing model
                logger.info(f"Loading AutoGluon model from {model_path}")
                try:
                    self.detector = ObjectDetector.load(model_path)
                    
                    # Try to load label map from model path
                    label_map_path = os.path.join(model_path, "label_map.json")
                    if os.path.exists(label_map_path):
                        import json
                        with open(label_map_path, 'r') as f:
                            self.label_map = json.load(f)
                            
                    logger.info(f"Loaded AutoGluon model with {len(self.label_map)} classes")
                    self.initialized = True
                    return True
                except Exception as e:
                    logger.error(f"Error loading AutoGluon model: {e}")
                    return False
                
            # Check if we should train a new model
            training_data_path = self.config.get("training_data_path")
            if training_data_path and os.path.exists(training_data_path):
                logger.info(f"Training new AutoGluon model with data from {training_data_path}")
                try:
                    self.detector = ObjectDetector()
                    
                    # Training parameters
                    epochs = self.config.get("training_epochs", 10)
                    batch_size = self.config.get("batch_size", 16)
                    learning_rate = self.config.get("learning_rate", 1e-4)
                    
                    # Train model
                    self.detector.fit(
                        train_data=training_data_path,
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=learning_rate
                    )
                    
                    # Save model if path provided
                    save_path = self.config.get("save_path")
                    if save_path:
                        # Create directory if it doesn't exist
                        os.makedirs(save_path, exist_ok=True)
                        
                        # Save model
                        self.detector.save(save_path)
                        self.model_path = save_path
                        
                        # Save label map
                        label_map_path = os.path.join(save_path, "label_map.json")
                        import json
                        with open(label_map_path, 'w') as f:
                            json.dump(self.label_map, f, indent=2)
                        
                    logger.info(f"Trained new AutoGluon model")
                    self.initialized = True
                    return True
                except Exception as e:
                    logger.error(f"Error training AutoGluon model: {e}")
                    return False
                
            # No model path or training data
            logger.warning("No model path or training data provided for AutoGluon detector")
            return False
            
        except Exception as e:
            logger.error(f"Error initializing AutoGluon detector: {e}")
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
        if not self.is_available or not self.detector or not self.initialized:
            logger.warning("AutoGluon detector not initialized or available")
            return []
            
        try:
            # Start timing
            start_time = time.time()
            
            # Convert screenshot to format expected by AutoGluon
            if isinstance(screenshot, np.ndarray):
                # Convert OpenCV BGR to RGB if needed
                if screenshot.shape[2] == 3 and screenshot[0, 0, 0] > screenshot[0, 0, 2]:
                    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
                    
            # Run detection
            predictions = self.detector.predict(screenshot)
            
            # Process predictions into standard format
            ui_elements = []
            
            for i, row in predictions.iterrows():
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Get class ID and map to element type
                class_id = int(row['class_id'])
                element_type = self.label_map.get(class_id, "unknown")
                
                ui_elements.append({
                    "type": element_type,
                    "bbox": (x1, y1, x2, y2),
                    "center": (center_x, center_y),
                    "confidence": float(row['score']),
                    "text": "",  # To be filled by OCR in a second pass
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "detector": "autogluon",
                    "class_id": class_id
                })
                
            # Calculate timing
            elapsed_time = time.time() - start_time
            
            # Add context if provided
            if context:
                for element in ui_elements:
                    element["context"] = context
                    
            logger.debug(f"AutoGluon detected {len(ui_elements)} elements in {elapsed_time:.3f}s")
            
            return ui_elements
            
        except Exception as e:
            logger.error(f"Error in AutoGluon detection: {e}")
            return []
            
    def batch_detect_elements(self, screenshots: List[np.ndarray], 
                             contexts: Optional[List[Dict]] = None) -> List[List[Dict]]:
        """
        Detect UI elements in multiple screenshots efficiently
        
        AutoGluon supports batch processing for more efficient detection
        across multiple screenshots.
        
        Args:
            screenshots: List of images as numpy arrays
            contexts: Optional list of context information
            
        Returns:
            List of detection results, one per screenshot
        """
        if not self.is_available or not self.detector or not self.initialized:
            logger.warning("AutoGluon detector not initialized or available")
            return [[] for _ in range(len(screenshots))]
            
        try:
            # Start timing
            start_time = time.time()
            
            # Convert all screenshots to RGB if needed
            processed_screenshots = []
            for screenshot in screenshots:
                if isinstance(screenshot, np.ndarray):
                    # Convert OpenCV BGR to RGB if needed
                    if screenshot.shape[2] == 3 and screenshot[0, 0, 0] > screenshot[0, 0, 2]:
                        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
                processed_screenshots.append(screenshot)
                
            # Run batch detection
            batch_predictions = self.detector.predict_batch(processed_screenshots)
            
            # Process predictions for each screenshot
            all_elements = []
            
            for i, predictions in enumerate(batch_predictions):
                ui_elements = []
                context = contexts[i] if contexts and i < len(contexts) else None
                
                for j, row in predictions.iterrows():
                    x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    # Get class ID and map to element type
                    class_id = int(row['class_id'])
                    element_type = self.label_map.get(class_id, "unknown")
                    
                    element = {
                        "type": element_type,
                        "bbox": (x1, y1, x2, y2),
                        "center": (center_x, center_y),
                        "confidence": float(row['score']),
                        "text": "",  # To be filled by OCR in a second pass
                        "width": x2 - x1,
                        "height": y2 - y1,
                        "detector": "autogluon",
                        "class_id": class_id
                    }
                    
                    # Add context if provided
                    if context:
                        element["context"] = context
                        
                    ui_elements.append(element)
                    
                all_elements.append(ui_elements)
                
            # Calculate timing
            elapsed_time = time.time() - start_time
            logger.debug(f"AutoGluon batch detected elements in {len(screenshots)} screenshots in {elapsed_time:.3f}s")
            
            return all_elements
            
        except Exception as e:
            logger.error(f"Error in AutoGluon batch detection: {e}")
            return [[] for _ in range(len(screenshots))]
            
    def supports_incremental_learning(self) -> bool:
        """Whether this detector supports incremental learning"""
        return True
        
    def update_model(self, examples: List[Dict]) -> bool:
        """
        Update the detector with new examples
        
        Args:
            examples: List of labeled examples
                
        Returns:
            Success status
        """
        if not self.is_available or not self.detector or not self.initialized:
            return False
            
        try:
            # Import here to avoid errors if AutoGluon isn't available
            from autogluon.vision import ObjectDetector
            import pandas as pd
            
            # Convert examples to format expected by AutoGluon
            # This is a simplified implementation; actual implementation would need
            # to convert the data format appropriately
            
            # Train with examples
            self.detector.fit(
                train_data=examples,
                epochs=5,  # Shorter epochs for incremental learning
                batch_size=8,
                learning_rate=5e-5  # Lower learning rate for fine-tuning
            )
            
            # Save updated model if path available
            if self.model_path:
                self.detector.save(self.model_path)
                
                # Save updated label map
                label_map_path = os.path.join(self.model_path, "label_map.json")
                import json
                with open(label_map_path, 'w') as f:
                    json.dump(self.label_map, f, indent=2)
                
            return True
            
        except Exception as e:
            logger.error(f"Error updating AutoGluon model: {e}")
            return False
            
    def get_capabilities(self) -> Dict[str, Any]:
        """Get detector capabilities and metadata"""
        capabilities = super().get_capabilities()
        
        # Add AutoGluon-specific capabilities
        capabilities.update({
            "name": "autogluon",
            "requires_gpu": True,
            "version": "0.1.0",
            "model_path": self.model_path,
            "supports_batch_processing": True,
            "max_elements_per_image": 100,
            "pretrained_backbone": "faster_rcnn"
        })
        
        return capabilities
