"""
Confidence calibration system for UI element detectors.

This module provides calibration of raw detector confidence scores based on
historical accuracy. ML models often output confidence scores that don't 
directly correspond to actual accuracy, and this calibration system corrects
this discrepancy by tracking the relationship between predicted confidence
and actual correctness.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import json
import os

logger = logging.getLogger(__name__)

class ConfidenceCalibrator:
    """Calibrates detector confidence based on historical accuracy"""
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the confidence calibrator
        
        Args:
            storage_path: Optional path to store calibration data
        """
        self.detector_calibration = {}  # detector_name -> calibration data
        self.storage_path = storage_path
        self.min_samples_for_calibration = 10
        
        # Load existing calibration data if available
        if storage_path and os.path.exists(storage_path):
            self.load_calibration_data()
        
    def register_detector(self, detector_name: str) -> None:
        """
        Register a new detector for calibration
        
        Args:
            detector_name: Name of the detector to register
        """
        if detector_name not in self.detector_calibration:
            self.detector_calibration[detector_name] = {
                "confidence_buckets": [[] for _ in range(10)],  # 10 buckets of 0.1 range
                "calibration_map": [0.0] * 10,  # Calibrated confidence for each bucket
                "last_updated": time.time()
            }
            logger.info(f"Registered detector '{detector_name}' for confidence calibration")
            
    def update_calibration(self, detector_name: str, raw_confidence: float, was_correct: bool) -> None:
        """
        Update calibration with new data point
        
        Args:
            detector_name: Name of the detector
            raw_confidence: Raw confidence score from detector (0-1)
            was_correct: Whether the detection was correct
        """
        if detector_name not in self.detector_calibration:
            self.register_detector(detector_name)
            
        # Determine bucket (0.0-0.1 -> 0, 0.1-0.2 -> 1, etc.)
        bucket = min(9, int(raw_confidence * 10))
        
        # Add result to bucket
        self.detector_calibration[detector_name]["confidence_buckets"][bucket].append(was_correct)
        
        # Recalculate calibration for this bucket
        bucket_results = self.detector_calibration[detector_name]["confidence_buckets"][bucket]
        if bucket_results:
            calibrated = sum(bucket_results) / len(bucket_results)
            self.detector_calibration[detector_name]["calibration_map"][bucket] = calibrated
            
        # Update timestamp
        self.detector_calibration[detector_name]["last_updated"] = time.time()
        
        # Save calibration data if storage path is set
        if self.storage_path:
            self.save_calibration_data()
            
    def calibrate_confidence(self, detector_name: str, raw_confidence: float) -> float:
        """
        Get calibrated confidence value
        
        Args:
            detector_name: Name of the detector
            raw_confidence: Raw confidence score from detector (0-1)
            
        Returns:
            Calibrated confidence score (0-1)
        """
        if detector_name not in self.detector_calibration:
            return raw_confidence
            
        # Determine bucket
        bucket = min(9, int(raw_confidence * 10))
        
        # Get calibrated confidence for this bucket
        calibrated = self.detector_calibration[detector_name]["calibration_map"][bucket]
        
        # If we don't have enough data for this bucket, blend with raw confidence
        bucket_size = len(self.detector_calibration[detector_name]["confidence_buckets"][bucket])
        if bucket_size < self.min_samples_for_calibration:
            # Blend based on sample count
            weight = bucket_size / self.min_samples_for_calibration
            return (calibrated * weight) + (raw_confidence * (1 - weight))
            
        return calibrated

    def calibrate_elements(self, detector_name: str, elements: List[Dict]) -> List[Dict]:
        """
        Calibrate confidence scores for a list of detected elements
        
        Args:
            detector_name: Name of the detector
            elements: List of detected elements
            
        Returns:
            Elements with calibrated confidence scores
        """
        if detector_name not in self.detector_calibration:
            return elements
            
        for element in elements:
            if "confidence" in element:
                raw_confidence = element["confidence"]
                element["raw_confidence"] = raw_confidence  # Preserve original
                element["confidence"] = self.calibrate_confidence(detector_name, raw_confidence)
                
        return elements

    def get_calibration_stats(self, detector_name: str) -> Optional[List[Dict]]:
        """
        Get calibration statistics for a detector
        
        Args:
            detector_name: Name of the detector
            
        Returns:
            List of statistics for each confidence bucket, or None if not registered
        """
        if detector_name not in self.detector_calibration:
            return None
            
        stats = []
        for i, bucket in enumerate(self.detector_calibration[detector_name]["confidence_buckets"]):
            if bucket:
                raw_confidence = (i * 0.1) + 0.05  # Middle of bucket
                calibrated = self.detector_calibration[detector_name]["calibration_map"][i]
                count = len(bucket)
                stats.append({
                    "raw_confidence_range": f"{i * 0.1:.1f}-{(i + 1) * 0.1:.1f}",
                    "calibrated_confidence": calibrated,
                    "sample_count": count,
                    "actual_accuracy": calibrated
                })
                
        return stats
        
    def save_calibration_data(self) -> bool:
        """
        Save calibration data to disk
        
        Returns:
            Success status
        """
        if not self.storage_path:
            return False
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            # Convert data to serializable format
            serializable_data = {}
            for detector_name, data in self.detector_calibration.items():
                serializable_data[detector_name] = {
                    "calibration_map": data["calibration_map"],
                    "confidence_buckets": [[bool(x) for x in bucket] for bucket in data["confidence_buckets"]],
                    "last_updated": data["last_updated"]
                }
                
            # Save to file
            with open(self.storage_path, 'w') as f:
                json.dump(serializable_data, f, indent=2)
                
            logger.info(f"Saved confidence calibration data to {self.storage_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving calibration data: {e}")
            return False
            
    def load_calibration_data(self) -> bool:
        """
        Load calibration data from disk
        
        Returns:
            Success status
        """
        if not self.storage_path or not os.path.exists(self.storage_path):
            return False
            
        try:
            with open(self.storage_path, 'r') as f:
                serialized_data = json.load(f)
                
            # Convert to internal format
            for detector_name, data in serialized_data.items():
                self.detector_calibration[detector_name] = {
                    "calibration_map": data["calibration_map"],
                    "confidence_buckets": [[bool(x) for x in bucket] for bucket in data["confidence_buckets"]],
                    "last_updated": data.get("last_updated", time.time())
                }
                
            logger.info(f"Loaded confidence calibration data from {self.storage_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading calibration data: {e}")
            return False
