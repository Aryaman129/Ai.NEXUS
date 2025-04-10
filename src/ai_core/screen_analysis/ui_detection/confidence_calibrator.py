"""
Confidence Calibrator

This module provides a confidence calibration system that adjusts raw confidence
scores based on historical accuracy.
"""

import logging
import json
import os
import time
from typing import Dict, List, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

class ConfidenceCalibrator:
    """
    Calibrates confidence scores based on historical accuracy
    
    This class adjusts raw confidence scores from detectors based on their
    historical accuracy, ensuring more reliable confidence values.
    """
    
    def __init__(self, memory_path: str = "memory/ui_detection"):
        self.memory_path = memory_path
        self.calibration_data = {}  # detector -> element_type -> calibration
        
        # Create memory directory if it doesn't exist
        os.makedirs(memory_path, exist_ok=True)
        
        # Load calibration data if available
        self._load_calibration_data()
        
    def calibrate_confidence(self, 
                           detector_name: str, 
                           element_type: str, 
                           raw_confidence: float) -> float:
        """
        Calibrate a raw confidence score
        
        Args:
            detector_name: Name of the detector
            element_type: Type of UI element
            raw_confidence: Raw confidence score (0-1)
            
        Returns:
            Calibrated confidence score (0-1)
        """
        # Get calibration data for this detector and element type
        if detector_name not in self.calibration_data:
            return raw_confidence
            
        if element_type not in self.calibration_data[detector_name]:
            return raw_confidence
            
        calibration = self.calibration_data[detector_name][element_type]
        
        # Apply calibration formula
        # This is a simple linear adjustment; more sophisticated methods could be used
        adjusted = (raw_confidence * calibration["scale"]) + calibration["offset"]
        
        # Ensure confidence stays in [0,1] range
        return max(0.0, min(1.0, adjusted))
        
    def update_calibration(self, 
                         detector_name: str, 
                         element_type: str, 
                         raw_confidence: float,
                         actual_correct: bool) -> None:
        """
        Update calibration based on a new observation
        
        Args:
            detector_name: Name of the detector
            element_type: Type of UI element
            raw_confidence: Raw confidence score (0-1)
            actual_correct: Whether the detection was actually correct
        """
        # Initialize calibration data if needed
        if detector_name not in self.calibration_data:
            self.calibration_data[detector_name] = {}
            
        if element_type not in self.calibration_data[detector_name]:
            self.calibration_data[detector_name][element_type] = {
                "scale": 1.0,
                "offset": 0.0,
                "observations": [],
                "last_updated": time.time()
            }
            
        calibration = self.calibration_data[detector_name][element_type]
        
        # Add observation
        calibration["observations"].append({
            "raw_confidence": raw_confidence,
            "actual_correct": actual_correct,
            "timestamp": time.time()
        })
        
        # Limit observations to last 100
        if len(calibration["observations"]) > 100:
            calibration["observations"] = calibration["observations"][-100:]
            
        # Recalculate calibration parameters
        self._recalculate_calibration(detector_name, element_type)
        
        # Save calibration data periodically
        if len(calibration["observations"]) % 10 == 0:
            self._save_calibration_data()
            
    def _recalculate_calibration(self, detector_name: str, element_type: str) -> None:
        """
        Recalculate calibration parameters based on observations
        
        Args:
            detector_name: Name of the detector
            element_type: Type of UI element
        """
        calibration = self.calibration_data[detector_name][element_type]
        observations = calibration["observations"]
        
        if len(observations) < 5:
            # Not enough data for reliable calibration
            return
            
        # Extract raw confidences and actual outcomes
        confidences = [obs["raw_confidence"] for obs in observations]
        outcomes = [1.0 if obs["actual_correct"] else 0.0 for obs in observations]
        
        # Calculate correlation between confidence and correctness
        try:
            correlation = np.corrcoef(confidences, outcomes)[0, 1]
        except:
            correlation = 0.0
            
        # If correlation is positive, confidence is somewhat predictive
        if correlation > 0.1:
            # Calculate average confidence and average correctness
            avg_confidence = np.mean(confidences)
            avg_correctness = np.mean(outcomes)
            
            # Calculate scale and offset to align confidence with correctness
            if avg_confidence > 0:
                scale = avg_correctness / avg_confidence
            else:
                scale = 1.0
                
            offset = avg_correctness - (scale * avg_confidence)
            
            # Update calibration parameters
            calibration["scale"] = scale
            calibration["offset"] = offset
        else:
            # If correlation is low or negative, use conservative values
            calibration["scale"] = 0.5
            calibration["offset"] = 0.2
            
        calibration["last_updated"] = time.time()
        
    def _save_calibration_data(self) -> None:
        """Save calibration data to disk"""
        try:
            # Create a simplified version for storage
            storage_data = {}
            
            for detector_name, detector_data in self.calibration_data.items():
                storage_data[detector_name] = {}
                
                for element_type, calibration in detector_data.items():
                    storage_data[detector_name][element_type] = {
                        "scale": calibration["scale"],
                        "offset": calibration["offset"],
                        "observation_count": len(calibration["observations"]),
                        "last_updated": calibration["last_updated"]
                    }
                    
            # Save to file
            calibration_path = os.path.join(self.memory_path, "confidence_calibration.json")
            with open(calibration_path, 'w') as f:
                json.dump(storage_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving calibration data: {e}")
            
    def _load_calibration_data(self) -> None:
        """Load calibration data from disk"""
        try:
            calibration_path = os.path.join(self.memory_path, "confidence_calibration.json")
            
            if os.path.exists(calibration_path):
                with open(calibration_path, 'r') as f:
                    storage_data = json.load(f)
                    
                # Convert to full calibration data
                for detector_name, detector_data in storage_data.items():
                    self.calibration_data[detector_name] = {}
                    
                    for element_type, calibration in detector_data.items():
                        self.calibration_data[detector_name][element_type] = {
                            "scale": calibration["scale"],
                            "offset": calibration["offset"],
                            "observations": [],
                            "last_updated": calibration.get("last_updated", time.time())
                        }
                        
        except Exception as e:
            logger.error(f"Error loading calibration data: {e}")
            self.calibration_data = {}
