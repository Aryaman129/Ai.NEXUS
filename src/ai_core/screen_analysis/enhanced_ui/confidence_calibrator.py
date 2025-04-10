"""
Adaptive Confidence Calibration for UI Element Detection

This module provides a self-learning confidence calibration system
that improves detection accuracy over time based on feedback.
"""

import os
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import pickle
from collections import defaultdict

logger = logging.getLogger(__name__)

class AdaptiveConfidenceCalibrator:
    """
    Calibrates confidence scores for UI element detections based on historical performance.
    
    This adaptive system:
    1. Tracks detection success/failure rates per element type and detector
    2. Learns optimal confidence thresholds over time
    3. Adjusts raw confidence scores based on historical accuracy
    4. Supports cross-session learning through persistence
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the adaptive confidence calibrator"""
        self.config = config or {}
        self.storage_path = self.config.get("storage_path", os.path.join("data", "calibration"))
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Calibration data structure
        # {detector_name: {element_type: {bucket: {"correct": X, "total": Y}}}}
        self.calibration_data = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: {"correct": 0, "total": 0}
                )
            )
        )
        
        # Number of confidence buckets (0.0-0.1, 0.1-0.2, ..., 0.9-1.0)
        self.num_buckets = 10
        
        # Learning rate for updating calibration
        self.learning_rate = self.config.get("learning_rate", 0.1)
        
        # Minimum feedback samples needed for calibration
        self.min_samples = self.config.get("min_samples", 5)
        
        # Load existing calibration data if available
        self._load_calibration_data()
        
        logger.info("AdaptiveConfidenceCalibrator initialized")
    
    def calibrate_confidence(self, 
                          detector_name: str, 
                          element_type: str, 
                          raw_confidence: float) -> float:
        """
        Calibrate a raw confidence score based on historical accuracy.
        
        Args:
            detector_name: Name of the detector that produced the confidence
            element_type: Type of UI element being detected
            raw_confidence: Raw confidence score from detector (0.0 to 1.0)
            
        Returns:
            Calibrated confidence score
        """
        # If we don't have enough data for this detector/type, return raw confidence
        if not self._has_sufficient_data(detector_name, element_type):
            return raw_confidence
        
        # Get the bucket for this confidence score
        bucket = min(self.num_buckets - 1, int(raw_confidence * self.num_buckets))
        
        # Get historical accuracy for this bucket
        bucket_data = self.calibration_data[detector_name][element_type][bucket]
        historical_accuracy = bucket_data["correct"] / bucket_data["total"] if bucket_data["total"] > 0 else 0.5
        
        # Blend raw confidence with historical accuracy
        # The more data we have, the more we rely on historical accuracy
        weight = min(0.8, bucket_data["total"] / 100)  # Cap influence at 80%
        calibrated_confidence = (1 - weight) * raw_confidence + weight * historical_accuracy
        
        logger.debug(f"Calibrated {detector_name} {element_type} confidence: {raw_confidence:.2f} → {calibrated_confidence:.2f}")
        return calibrated_confidence
    
    def update_calibration(self, 
                        detector_name: str, 
                        element_type: str, 
                        raw_confidence: float, 
                        was_correct: bool) -> None:
        """
        Update calibration data based on feedback.
        
        Args:
            detector_name: Name of the detector
            element_type: Type of UI element
            raw_confidence: Raw confidence score that was produced
            was_correct: Whether the detection was correct
        """
        # Determine bucket for this confidence
        bucket = min(self.num_buckets - 1, int(raw_confidence * self.num_buckets))
        
        # Update bucket data
        self.calibration_data[detector_name][element_type][bucket]["total"] += 1
        if was_correct:
            self.calibration_data[detector_name][element_type][bucket]["correct"] += 1
        
        # Save updated calibration data
        self._save_calibration_data()
        
        logger.debug(f"Updated calibration for {detector_name} {element_type} bucket {bucket}, correct: {was_correct}")
    
    def get_recommended_threshold(self, 
                               detector_name: str, 
                               element_type: str) -> float:
        """
        Get the recommended confidence threshold for accepting detections.
        
        Args:
            detector_name: Name of the detector
            element_type: Type of UI element
            
        Returns:
            Recommended threshold value
        """
        if not self._has_sufficient_data(detector_name, element_type):
            # Default thresholds for different element types
            default_thresholds = {
                "button": 0.6,
                "text_field": 0.65,
                "icon": 0.7,
                "checkbox": 0.65,
                "dropdown": 0.7,
                "radio_button": 0.65,
                "scrollbar": 0.7,
                "menu_item": 0.6,
                "panel": 0.55
            }
            return default_thresholds.get(element_type, 0.6)
        
        # Find the bucket where accuracy exceeds 0.75
        for bucket in range(self.num_buckets):
            bucket_data = self.calibration_data[detector_name][element_type][bucket]
            if bucket_data["total"] >= self.min_samples:
                accuracy = bucket_data["correct"] / bucket_data["total"]
                if accuracy >= 0.75:
                    # Return the lower bound of this bucket as threshold
                    return bucket / self.num_buckets
        
        # If no bucket has sufficient accuracy, return a higher threshold
        return 0.7
    
    def get_performance_metrics(self, 
                             detector_name: Optional[str] = None, 
                             element_type: Optional[str] = None) -> Dict:
        """
        Get performance metrics for detectors and element types.
        
        Args:
            detector_name: Optional name of detector to filter by
            element_type: Optional element type to filter by
            
        Returns:
            Dictionary of performance metrics
        """
        metrics = {}
        
        # Filter by detector if specified
        detectors = [detector_name] if detector_name else self.calibration_data.keys()
        
        for detector in detectors:
            detector_metrics = {}
            
            # Filter by element type if specified
            element_types = [element_type] if element_type else self.calibration_data[detector].keys()
            
            for e_type in element_types:
                # Calculate overall performance for this type
                correct = 0
                total = 0
                
                for bucket in range(self.num_buckets):
                    bucket_data = self.calibration_data[detector][e_type][bucket]
                    correct += bucket_data["correct"]
                    total += bucket_data["total"]
                
                if total > 0:
                    detector_metrics[e_type] = {
                        "accuracy": correct / total,
                        "total_samples": total,
                        "recommended_threshold": self.get_recommended_threshold(detector, e_type)
                    }
            
            if detector_metrics:
                metrics[detector] = detector_metrics
        
        return metrics
    
    def _has_sufficient_data(self, detector_name: str, element_type: str) -> bool:
        """Check if we have sufficient calibration data for a detector/element type"""
        if detector_name not in self.calibration_data:
            return False
            
        if element_type not in self.calibration_data[detector_name]:
            return False
            
        # Check total samples across all buckets
        total_samples = sum(
            self.calibration_data[detector_name][element_type][bucket]["total"]
            for bucket in range(self.num_buckets)
        )
        
        return total_samples >= self.min_samples
    
    def _save_calibration_data(self) -> None:
        """Save calibration data to disk for persistence"""
        try:
            file_path = os.path.join(self.storage_path, "calibration_data.pkl")
            with open(file_path, 'wb') as f:
                pickle.dump(dict(self.calibration_data), f)
                
            # Also save a human-readable summary
            summary_path = os.path.join(self.storage_path, "calibration_summary.json")
            metrics = self.get_performance_metrics()
            with open(summary_path, 'w') as f:
                json.dump({
                    "last_updated": datetime.now().isoformat(),
                    "metrics": metrics
                }, f, indent=2)
                
            logger.debug(f"Saved calibration data to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save calibration data: {e}")
    
    def _load_calibration_data(self) -> None:
        """Load calibration data from disk"""
        file_path = os.path.join(self.storage_path, "calibration_data.pkl")
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    loaded_data = pickle.load(f)
                    
                # Convert loaded dict to defaultdict structure
                for detector, type_data in loaded_data.items():
                    for element_type, bucket_data in type_data.items():
                        for bucket, data in bucket_data.items():
                            self.calibration_data[detector][element_type][bucket] = data
                            
                logger.info(f"Loaded calibration data for {len(loaded_data)} detectors")
            except Exception as e:
                logger.error(f"Failed to load calibration data: {e}")
    
    def reset_calibration(self, 
                       detector_name: Optional[str] = None, 
                       element_type: Optional[str] = None) -> None:
        """
        Reset calibration data.
        
        Args:
            detector_name: Optional name of detector to reset (None for all)
            element_type: Optional element type to reset (None for all)
        """
        if detector_name is None:
            # Reset all calibration data
            self.calibration_data = defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: {"correct": 0, "total": 0}
                    )
                )
            )
            logger.info("Reset all calibration data")
        elif element_type is None:
            # Reset calibration for specific detector
            self.calibration_data[detector_name] = defaultdict(
                lambda: defaultdict(
                    lambda: {"correct": 0, "total": 0}
                )
            )
            logger.info(f"Reset calibration data for detector: {detector_name}")
        else:
            # Reset calibration for specific detector and element type
            self.calibration_data[detector_name][element_type] = defaultdict(
                lambda: {"correct": 0, "total": 0}
            )
            logger.info(f"Reset calibration data for {detector_name}, {element_type}")
        
        # Save updated (reset) calibration
        self._save_calibration_data()
