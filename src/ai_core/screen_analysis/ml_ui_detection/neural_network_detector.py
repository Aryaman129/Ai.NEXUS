#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Neural Network Detector for UI element detection.
This module implements a neural network based detector that can be fine-tuned
with examples from successful detections to continuously improve over time.
"""

import os
import uuid
import json
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union, Any

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pickle

from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UIElementDataset(Dataset):
    """Dataset for UI elements with positive and negative examples."""
    
    def __init__(self, transform=None):
        self.samples = []
        self.transform = transform
        self.class_weights = None
        self.update_class_weights()
        
    def add_sample(self, image: np.ndarray, element_type: str, is_positive: bool):
        """Add a sample to the dataset."""
        if image is None:
            return
            
        # Convert to PIL Image for torchvision transforms
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
        self.samples.append({
            'image': image,
            'element_type': element_type,
            'label': 1 if is_positive else 0
        })
        
        # Update class weights after adding a sample
        self.update_class_weights()
        
    def update_class_weights(self):
        """Update class weights for balanced training."""
        if not self.samples:
            self.class_weights = None
            return
            
        # Count positive and negative examples
        positive_count = sum(1 for sample in self.samples if sample['label'] == 1)
        negative_count = len(self.samples) - positive_count
        
        if positive_count == 0 or negative_count == 0:
            self.class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32)
        else:
            # Calculate weights inversely proportional to class frequencies
            total = positive_count + negative_count
            self.class_weights = torch.tensor(
                [total / (2.0 * negative_count), total / (2.0 * positive_count)], 
                dtype=torch.float32
            )
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = sample['image']
        label = sample['label']
        element_type = sample['element_type']
        
        if self.transform:
            image = self.transform(image)
            
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.float32),
            'element_type': element_type
        }


class UIElementDetectorNetwork(nn.Module):
    """Neural network for UI element detection using transfer learning."""
    
    def __init__(self, num_classes=1):
        super(UIElementDetectorNetwork, self).__init__()
        
        # Use a pre-trained model like ResNet18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Replace the final fully connected layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
            nn.Sigmoid()  # For binary classification (is it this element type or not)
        )
        
    def forward(self, x):
        return self.backbone(x)


class NeuralNetworkDetector:
    """Neural network-based UI element detector with incremental learning capabilities."""
    
    def __init__(self, model_path: Optional[str] = None, element_types: Optional[List[str]] = None):
        self.name = "NeuralNetworkDetector"
        self.model_path = model_path
        self.element_types = element_types or ["button", "checkbox", "text_input", "dropdown", "icon", "link"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define image transform for preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create a dataset for continual learning
        self.dataset = UIElementDataset(transform=self.transform)
        
        # Create models for each element type
        self.models = {}
        self.optimizers = {}
        
        for element_type in self.element_types:
            self.models[element_type] = UIElementDetectorNetwork().to(self.device)
            self.optimizers[element_type] = optim.Adam(self.models[element_type].parameters(), lr=0.001)
            
        # Load existing models if path is provided
        if model_path and os.path.exists(model_path):
            self.load_models()
            
        # Training parameters
        self.batch_size = 4
        self.min_training_examples = 10  # Minimum examples before training
        self.loss_fn = nn.BCELoss()
        
        # Statistics for tracking performance
        self.training_stats = {
            "training_runs": 0,
            "examples_by_type": {t: {"positive": 0, "negative": 0} for t in self.element_types},
            "model_versions": {t: 1 for t in self.element_types},
            "accuracy_by_type": {t: [] for t in self.element_types}
        }
            
        logger.info(f"Initialized NeuralNetworkDetector with {len(self.element_types)} element types on {self.device}")
        
    def load_models(self):
        """Load saved model weights and stats."""
        try:
            # Load models
            for element_type in self.element_types:
                model_file = os.path.join(self.model_path, f"{element_type}_model.pth")
                if os.path.exists(model_file):
                    self.models[element_type].load_state_dict(
                        torch.load(model_file, map_location=self.device)
                    )
                    logger.info(f"Loaded model for {element_type}")
            
            # Load stats if they exist
            stats_file = os.path.join(self.model_path, "training_stats.json")
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    self.training_stats = json.load(f)
                    
            # Load dataset if it exists
            dataset_file = os.path.join(self.model_path, "training_dataset.pkl")
            if os.path.exists(dataset_file):
                with open(dataset_file, 'rb') as f:
                    self.dataset = pickle.load(f)
                
            logger.info(f"Successfully loaded models and training data")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
        
    def save_models(self):
        """Save model weights and training stats."""
        if not self.model_path:
            return
            
        # Create directory if it doesn't exist
        os.makedirs(self.model_path, exist_ok=True)
        
        try:
            # Save models
            for element_type in self.element_types:
                torch.save(
                    self.models[element_type].state_dict(),
                    os.path.join(self.model_path, f"{element_type}_model.pth")
                )
            
            # Save stats
            with open(os.path.join(self.model_path, "training_stats.json"), 'w') as f:
                json.dump(self.training_stats, f)
                
            # Save dataset
            with open(os.path.join(self.model_path, "training_dataset.pkl"), 'wb') as f:
                pickle.dump(self.dataset, f)
                
            logger.info(f"Saved models and training data to {self.model_path}")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def supports_incremental_learning(self) -> bool:
        """Return whether this detector supports incremental learning."""
        return True
        
    def add_training_example(self, image: np.ndarray, element_type: str, is_positive: bool) -> bool:
        """Add a training example for continual learning."""
        if element_type not in self.element_types:
            self.element_types.append(element_type)
            self.models[element_type] = UIElementDetectorNetwork().to(self.device)
            self.optimizers[element_type] = optim.Adam(self.models[element_type].parameters(), lr=0.001)
            self.training_stats["examples_by_type"][element_type] = {"positive": 0, "negative": 0}
            self.training_stats["model_versions"][element_type] = 1
            self.training_stats["accuracy_by_type"][element_type] = []
            
        # Add to dataset
        self.dataset.add_sample(image, element_type, is_positive)
        
        # Update statistics
        example_type = "positive" if is_positive else "negative"
        self.training_stats["examples_by_type"][element_type][example_type] += 1
        
        # Check if we have enough examples to train
        element_examples = sum(
            self.training_stats["examples_by_type"][element_type].values()
        )
        
        if element_examples >= self.min_training_examples:
            self._train_model(element_type)
            return True
            
        return False
    
    def add_correct_detection(self, image: np.ndarray, detection: Dict[str, Any]) -> bool:
        """Add a correct detection as a positive training example."""
        element_type = detection.get('type')
        if not element_type or element_type not in self.element_types:
            return False
            
        # Extract ROI from the image based on detection coordinates
        x, y, w, h = detection.get('x', 0), detection.get('y', 0), detection.get('width', 0), detection.get('height', 0)
        
        # Handle case where detection doesn't have valid coordinates
        if w <= 0 or h <= 0 or x < 0 or y < 0:
            return False
            
        # Extract ROI
        try:
            roi = image[y:y+h, x:x+w]
            # Add some padding around the element (20%)
            pad_x, pad_y = int(w * 0.2), int(h * 0.2)
            x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
            x2, y2 = min(image.shape[1], x + w + pad_x), min(image.shape[0], y + h + pad_y)
            context_roi = image[y1:y2, x1:x2]
            
            # Add as positive example
            self.add_training_example(roi, element_type, True)
            
            # Also add the context as it helps with detection
            if context_roi.size > 0:
                self.add_training_example(context_roi, element_type, True)
                
            return True
        except Exception as e:
            logger.error(f"Error adding correct detection: {e}")
            return False
    
    def _train_model(self, element_type: str):
        """Train the model for the specified element type."""
        if len(self.dataset) < self.batch_size:
            return
            
        logger.info(f"Training model for {element_type} with {len(self.dataset)} examples")
        
        # Create DataLoader for batch training
        dataloader = DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        # Get the model and optimizer for this element type
        model = self.models[element_type]
        optimizer = self.optimizers[element_type]
        
        # Set model to training mode
        model.train()
        
        # Get class weights for balanced training
        class_weights = self.dataset.class_weights
        
        if class_weights is not None:
            weighted_loss_fn = nn.BCELoss(weight=class_weights[1].expand(self.batch_size))
        else:
            weighted_loss_fn = self.loss_fn
        
        # Train for a few epochs
        epochs = 5
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            for batch in dataloader:
                # Skip batches that don't have the current element type
                if all(et != element_type for et in batch['element_type']):
                    continue
                
                # Extract elements of the current type
                mask = [et == element_type for et in batch['element_type']]
                if not any(mask):
                    continue
                    
                images = batch['image'][mask].to(self.device)
                labels = batch['label'][mask].to(self.device)
                
                # Forward pass
                outputs = model(images)
                loss = weighted_loss_fn(outputs, labels.unsqueeze(1))
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Track accuracy
                predictions = (outputs > 0.5).float()
                correct_predictions += (predictions == labels.unsqueeze(1)).sum().item()
                total_predictions += labels.size(0)
            
            total_loss += epoch_loss
            
        # Update statistics
        self.training_stats["training_runs"] += 1
        self.training_stats["model_versions"][element_type] += 1
        
        # Calculate accuracy
        accuracy = correct_predictions / max(1, total_predictions)
        self.training_stats["accuracy_by_type"][element_type].append(accuracy)
        
        # Save updated models
        self.save_models()
        
        logger.info(f"Completed training for {element_type}. Accuracy: {accuracy:.4f}")
    
    def detect(self, image: np.ndarray, min_confidence: float = 0.7) -> List[Dict]:
        """Detect UI elements in the image using the neural network models."""
        if image is None or image.size == 0:
            return []
            
        # Convert image to RGB for torchvision
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Sliding window parameters
        window_sizes = [(224, 224), (112, 112), (168, 168)]
        stride_ratio = 0.5
        
        all_detections = []
        
        # Set models to evaluation mode
        for element_type in self.element_types:
            self.models[element_type].eval()
        
        # Analyze the image with sliding windows of different sizes
        for window_size in window_sizes:
            window_w, window_h = window_size
            stride_w, stride_h = int(window_w * stride_ratio), int(window_h * stride_ratio)
            
            # Scan the image
            for y in range(0, image.shape[0] - window_h + 1, stride_h):
                for x in range(0, image.shape[1] - window_w + 1, stride_w):
                    # Extract window
                    window = image[y:y+window_h, x:x+window_w]
                    
                    # Convert to PIL for transformation
                    window_pil = Image.fromarray(cv2.cvtColor(window, cv2.COLOR_BGR2RGB))
                    
                    # Apply transforms
                    window_tensor = self.transform(window_pil).unsqueeze(0).to(self.device)
                    
                    # Check each element type
                    for element_type in self.element_types:
                        with torch.no_grad():
                            confidence = self.models[element_type](window_tensor).item()
                            
                        if confidence >= min_confidence:
                            detection = {
                                'type': element_type,
                                'x': x,
                                'y': y,
                                'width': window_w,
                                'height': window_h,
                                'confidence': confidence,
                                'detector': self.name
                            }
                            all_detections.append(detection)
        
        # Apply non-maximum suppression to remove overlapping detections
        final_detections = self._non_maximum_suppression(all_detections)
        
        return final_detections
    
    def _non_maximum_suppression(self, detections: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
        """Apply non-maximum suppression to remove overlapping detections."""
        if not detections:
            return []
            
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        final_detections = []
        
        while detections:
            best = detections.pop(0)
            final_detections.append(best)
            
            # Remove overlapping detections
            detections = [
                d for d in detections
                if self._calculate_iou(best, d) < iou_threshold
                or best['type'] != d['type']  # Allow overlapping of different types
            ]
            
        return final_detections
    
    def _calculate_iou(self, detection1: Dict, detection2: Dict) -> float:
        """Calculate IoU (Intersection over Union) between two detections."""
        # Get the coordinates of the intersecting rectangle
        x1 = max(detection1['x'], detection2['x'])
        y1 = max(detection1['y'], detection2['y'])
        x2 = min(detection1['x'] + detection1['width'], detection2['x'] + detection2['width'])
        y2 = min(detection1['y'] + detection1['height'], detection2['y'] + detection2['height'])
        
        # Area of intersection
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        intersection = width * height
        
        # Area of union
        area1 = detection1['width'] * detection1['height']
        area2 = detection2['width'] * detection2['height']
        union = area1 + area2 - intersection
        
        # Calculate IoU
        iou = intersection / union if union > 0 else 0
        
        return iou
    
    def get_stats(self) -> Dict:
        """Get statistics about the detector's training and performance."""
        return {
            "name": self.name,
            "device": str(self.device),
            "element_types": self.element_types,
            "training_stats": self.training_stats,
            "dataset_size": len(self.dataset),
            "timestamp": datetime.now().isoformat()
        }
