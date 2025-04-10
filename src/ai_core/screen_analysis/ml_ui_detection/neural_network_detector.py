#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Neural Network Detector for UI element detection.
This module implements a neural network based detector that can be fine-tuned
with examples from successful detections to continuously improve over time.
"""

import os
import json
import logging
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from collections import deque

# Configure logging
logger = logging.getLogger("adaptive_ui_detection")

class NeuralNetworkDetector:
    """
    A UI element detector using neural networks that can be fine-tuned on successful detections.
    This creates specialized neural detectors that go beyond template matching.
    """
    
    def __init__(self, model_path: str = None, memory_size: int = 1000):
        """Initialize the neural network detector.
        
        Args:
            model_path: Path to load pre-trained model (optional)
            memory_size: Size of memory buffer for fine-tuning examples
        """
        self.name = "NeuralNetworkDetector"
        self.description = "Neural network detector with fine-tuning capabilities"
        
        # Set up paths
        self.models_dir = Path('ui_neural_models')
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Create element type specific directories
        for element_type in ['button', 'text_input', 'checkbox', 'icon', 'dropdown', 'toggle']:
            os.makedirs(self.models_dir / element_type, exist_ok=True)
        
        # Set up memory buffer for each element type
        self.memory_buffers = {
            'button': deque(maxlen=memory_size),
            'text_input': deque(maxlen=memory_size),
            'checkbox': deque(maxlen=memory_size),
            'icon': deque(maxlen=memory_size),
            'dropdown': deque(maxlen=memory_size),
            'toggle': deque(maxlen=memory_size)
        }
        
        # Load or create base models
        self.models = {}
        self.initialize_models(model_path)
        
        # Training stats
        self.fine_tuning_sessions = 0
        self.training_examples_seen = 0
        
        logger.info(f"Initialized neural network detector with {len(self.models)} element type models")
    
    def initialize_models(self, model_path: str = None) -> None:
        """Initialize neural network models for each element type.
        
        Args:
            model_path: Path to load pre-trained model (optional)
        """
        # Element types we'll create specialized models for
        element_types = ['button', 'text_input', 'checkbox', 'icon', 'dropdown', 'toggle']
        
        # Try to load existing models first
        if model_path:
            model_dir = Path(model_path)
            for element_type in element_types:
                model_file = model_dir / f"{element_type}_model"
                if os.path.exists(model_file):
                    try:
                        self.models[element_type] = tf.keras.models.load_model(model_file)
                        logger.info(f"Loaded model for {element_type} from {model_file}")
                        continue
                    except Exception as e:
                        logger.warning(f"Error loading model for {element_type}: {e}")
        
        # Create new models for any missing element types
        for element_type in element_types:
            if element_type not in self.models:
                self.models[element_type] = self._create_base_model()
                logger.info(f"Created new model for {element_type}")
    
    def _create_base_model(self) -> tf.keras.Model:
        """Create a base neural network model using transfer learning from MobileNetV2.
        
        Returns:
            A Keras model ready for fine-tuning
        """
        # Use MobileNetV2 as base model (lightweight and efficient)
        base_model = MobileNetV2(
            input_shape=(128, 128, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze the base model
        base_model.trainable = False
        
        # Create the UI element detection model
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # Binary: is this a UI element or not?
        ])
        
        # Compile the model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess an image for neural network input.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image ready for neural network
        """
        # Resize to expected input size
        image_resized = cv2.resize(image, (128, 128))
        
        # Convert to RGB if needed
        if len(image_resized.shape) == 3 and image_resized.shape[2] == 3:
            if image_resized.dtype != np.float32:
                image_resized = image_resized.astype(np.float32) / 255.0
        else:
            # Convert grayscale to RGB
            image_resized = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
            
        # Add batch dimension
        return np.expand_dims(image_resized, axis=0)
    
    def detect_elements(self, image: np.ndarray, context: Dict = None) -> List[Dict]:
        """Detect UI elements in the image using neural networks.
        
        Args:
            image: Input image as numpy array
            context: Optional context information
            
        Returns:
            List of detected UI elements
        """
        detected_elements = []
        
        # Determine which element types to detect based on context
        element_types = list(self.models.keys())
        if context and 'target_element_type' in context and context['target_element_type'] in element_types:
            element_types = [context['target_element_type']]
        
        # Perform sliding window detection
        window_sizes = [(80, 40), (60, 60), (100, 40), (200, 50), (40, 40)]
        confidence_threshold = 0.7
        
        for element_type in element_types:
            if element_type not in self.models:
                continue
                
            model = self.models[element_type]
            
            for window_size in window_sizes:
                w, h = window_size
                
                # Skip if window is larger than image
                if w > image.shape[1] or h > image.shape[0]:
                    continue
                    
                # Slide the window across the image
                for y in range(0, image.shape[0] - h, h // 2):
                    for x in range(0, image.shape[1] - w, w // 2):
                        # Extract window
                        window = image[y:y+h, x:x+w]
                        
                        # Preprocess window
                        processed_window = self.preprocess_image(window)
                        
                        # Predict
                        confidence = model.predict(processed_window, verbose=0)[0][0]
                        
                        if confidence >= confidence_threshold:
                            # Create element data
                            element = {
                                'id': f"neural_{element_type}_{len(detected_elements)}",
                                'type': element_type,
                                'rect': {
                                    'x': x,
                                    'y': y,
                                    'width': w,
                                    'height': h
                                },
                                'confidence': float(confidence),
                                'text': '',  # Neural detector doesn't extract text
                                'source_detector': self.name
                            }
                            
                            # Check for overlapping detections
                            overlap = False
                            for existing_elem in detected_elements:
                                if self._calculate_iou(element['rect'], existing_elem['rect']) > 0.5:
                                    # If overlapping and current has higher confidence, replace
                                    if element['confidence'] > existing_elem['confidence']:
                                        detected_elements.remove(existing_elem)
                                    else:
                                        overlap = True
                                    break
                            
                            if not overlap:
                                detected_elements.append(element)
        
        return detected_elements
    
    def _calculate_iou(self, rect1: Dict, rect2: Dict) -> float:
        """Calculate Intersection over Union for two rectangles.
        
        Args:
            rect1: First rectangle dict with x, y, width, height
            rect2: Second rectangle dict with x, y, width, height
            
        Returns:
            IoU score between 0 and 1
        """
        # Convert to x1, y1, x2, y2 format
        x1_1, y1_1 = rect1['x'], rect1['y']
        x2_1, y2_1 = x1_1 + rect1['width'], y1_1 + rect1['height']
        
        x1_2, y1_2 = rect2['x'], rect2['y']
        x2_2, y2_2 = x1_2 + rect2['width'], y1_2 + rect2['height']
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        rect1_area = rect1['width'] * rect1['height']
        rect2_area = rect2['width'] * rect2['height']
        union_area = rect1_area + rect2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def add_training_example(self, image: np.ndarray, element_type: str, is_positive: bool = True) -> None:
        """Add a training example to the memory buffer for fine-tuning.
        
        Args:
            image: UI element image
            element_type: Type of UI element
            is_positive: Whether this is a positive example (True) or negative (False)
        """
        if element_type not in self.memory_buffers:
            logger.warning(f"Unknown element type: {element_type}, skipping")
            return
            
        # Preprocess the image
        try:
            processed_image = cv2.resize(image, (128, 128))
            if len(processed_image.shape) == 3 and processed_image.shape[2] == 3:
                if processed_image.dtype != np.float32:
                    processed_image = processed_image.astype(np.float32) / 255.0
            else:
                # Convert grayscale to RGB
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
                
            # Add to memory buffer
            label = 1.0 if is_positive else 0.0
            self.memory_buffers[element_type].append((processed_image, label))
            self.training_examples_seen += 1
            
            logger.info(f"Added {'positive' if is_positive else 'negative'} training example for {element_type}")
            
            # Check if we should fine-tune the model
            if len(self.memory_buffers[element_type]) >= 10:  # Fine-tune after collecting 10 examples
                self._fine_tune_model(element_type)
        except Exception as e:
            logger.error(f"Error adding training example: {e}")
    
    def _fine_tune_model(self, element_type: str) -> None:
        """Fine-tune the model for a specific element type.
        
        Args:
            element_type: Type of UI element to fine-tune for
        """
        if len(self.memory_buffers[element_type]) < 10:
            logger.info(f"Not enough examples for {element_type} to fine-tune (need at least 10)")
            return
            
        if element_type not in self.models:
            logger.warning(f"No model found for {element_type}")
            return
            
        try:
            # Create training data
            examples = list(self.memory_buffers[element_type])
            X = np.array([example[0] for example in examples])
            y = np.array([example[1] for example in examples])
            
            # Create validation split
            split_idx = int(len(examples) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Fine-tune the model
            model = self.models[element_type]
            
            # Unfreeze a few top layers for fine-tuning if it's a pre-trained model
            if isinstance(model.layers[0], tf.keras.Model):  # Check if first layer is a model (like MobileNetV2)
                base_model = model.layers[0]
                # Unfreeze the top 5 layers
                for layer in base_model.layers[-5:]:
                    layer.trainable = True
            
            # Fine-tune
            model.fit(
                X_train, y_train,
                epochs=5,
                batch_size=8,
                validation_data=(X_val, y_val),
                verbose=0
            )
            
            # Save the fine-tuned model
            model_save_path = self.models_dir / element_type / f"{element_type}_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model.save(model_save_path)
            
            # Update stats
            self.fine_tuning_sessions += 1
            
            logger.info(f"Fine-tuned model for {element_type} with {len(examples)} examples")
            
            # Reset memory buffer to avoid retraining on the same examples
            self.memory_buffers[element_type].clear()
            
        except Exception as e:
            logger.error(f"Error fine-tuning model for {element_type}: {e}")
    
    def add_correct_detection(self, image: np.ndarray, element_data: Dict) -> None:
        """Add a correct detection example for fine-tuning.
        
        Args:
            image: Full screenshot image
            element_data: Data for the correctly detected element
        """
        element_type = element_data.get('type', '')
        if not element_type or element_type not in self.memory_buffers:
            return
            
        try:
            # Extract the element image
            rect = element_data['rect']
            x, y = rect['x'], rect['y']
            w, h = rect['width'], rect['height']
            
            # Ensure coordinates are within image bounds
            if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
                return
                
            element_image = image[y:y+h, x:x+w]
            
            # Add as positive example
            self.add_training_example(element_image, element_type, True)
            
            # Also add some negative examples from surrounding areas
            self._add_negative_examples(image, rect, element_type)
            
        except Exception as e:
            logger.error(f"Error adding correct detection: {e}")
    
    def _add_negative_examples(self, image: np.ndarray, rect: Dict, element_type: str) -> None:
        """Add negative examples from surrounding areas of a positive detection.
        
        Args:
            image: Full screenshot image
            rect: Rectangle dict with x, y, width, height of positive detection
            element_type: Type of UI element
        """
        x, y = rect['x'], rect['y']
        w, h = rect['width'], rect['height']
        
        # Add up to 3 negative examples from nearby regions
        offsets = [(w, 0), (0, h), (w, h)]
        
        for dx, dy in offsets:
            nx, ny = x + dx, y + dy
            
            # Skip if outside image bounds
            if nx + w > image.shape[1] or ny + h > image.shape[0]:
                continue
                
            # Extract negative example
            negative_example = image[ny:ny+h, nx:nx+w]
            
            # Add as negative example
            self.add_training_example(negative_example, element_type, False)
    
    def supports_incremental_learning(self) -> bool:
        """Whether this detector supports incremental learning."""
        return True
    
    def provide_feedback(self, detection_results: List[Dict], correct_results: List[Dict]) -> None:
        """Process feedback to improve detection in future iterations.
        
        Args:
            detection_results: Detection results from this detector
            correct_results: Correct detection results (ground truth)
        """
        # This would be implemented using the add_correct_detection method
        logger.info(f"Received feedback with {len(correct_results)} correct results")
    
    def get_training_stats(self) -> Dict:
        """Get statistics about the neural network detector training.
        
        Returns:
            Dictionary with training statistics
        """
        return {
            'fine_tuning_sessions': self.fine_tuning_sessions,
            'training_examples_seen': self.training_examples_seen,
            'models': list(self.models.keys()),
            'memory_buffer_sizes': {k: len(v) for k, v in self.memory_buffers.items()},
            'models_directory': str(self.models_dir)
        }
