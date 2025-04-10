"""
UI Detector Registry

This module provides a registry for UI element detectors with adaptive selection
based on performance metrics and availability.
"""

import time
import logging
import json
import os
from typing import Dict, List, Any, Optional, Type, Tuple
import numpy as np

from .detector_interface import UIDetectorInterface

logger = logging.getLogger(__name__)

class UIDetectorRegistry:
    """Registry for UI element detectors with adaptive selection"""
    
    def __init__(self, memory_path: str = "memory/ui_detection"):
        self.detectors = {}  # name -> detector instance
        self.specialized_detectors = {}  # element_type -> {name -> detector}
        self.performance_metrics = {}  # name -> metrics
        self.detector_versions = {}  # name -> version history
        self.detector_classes = {}  # name -> class
        self.memory_path = memory_path
        
        # Create memory directory if it doesn't exist
        os.makedirs(memory_path, exist_ok=True)
        
        # Load performance metrics if available
        self._load_performance_metrics()
        
    def register_detector(self, name: str, detector_class: Type[UIDetectorInterface], 
                         specialized_for: Optional[str] = None,
                         priority: int = 0,
                         config: Optional[Dict] = None) -> bool:
        """
        Register a detector with optional specialization
        
        Args:
            name: Unique detector name
            detector_class: Detector class (not instance)
            specialized_for: Optional element type specialization
            priority: Priority (higher = preferred)
            config: Optional configuration
            
        Returns:
            Success status
        """
        # Store the class for later instantiation
        self.detector_classes[name] = {
            "class": detector_class,
            "priority": priority,
            "config": config or {},
            "specialized_for": specialized_for
        }
        
        # Initialize metrics if needed
        if name not in self.performance_metrics:
            self.performance_metrics[name] = {
                "calls": 0,
                "successes": 0,
                "failures": 0,
                "total_elements_detected": 0,
                "average_confidence": 0.0,
                "average_latency": 0.0,
                "version_history": []
            }
            
        # Initialize version history
        if name not in self.detector_versions:
            self.detector_versions[name] = []
            
        logger.info(f"Registered UI detector: {name}")
        return True
        
    def initialize_detector(self, name: str) -> Optional[UIDetectorInterface]:
        """
        Initialize a specific detector
        
        Args:
            name: Detector name
            
        Returns:
            Initialized detector or None if failed
        """
        if name not in self.detector_classes:
            logger.warning(f"Detector {name} not registered")
            return None
            
        # Create instance
        detector_info = self.detector_classes[name]
        detector_class = detector_info["class"]
        config = detector_info["config"]
        
        try:
            # Create detector instance
            detector = detector_class(config)
            
            # Try to initialize
            success = detector.initialize(config)
            if success:
                # Store instance
                if detector_info["specialized_for"]:
                    element_type = detector_info["specialized_for"]
                    if element_type not in self.specialized_detectors:
                        self.specialized_detectors[element_type] = {}
                    self.specialized_detectors[element_type][name] = detector
                else:
                    self.detectors[name] = detector
                    
                # Record version
                capabilities = detector.get_capabilities()
                version = capabilities.get("version", "unknown")
                self.detector_versions[name].append({
                    "version": version,
                    "timestamp": time.time(),
                    "capabilities": capabilities
                })
                
                logger.info(f"Initialized UI detector {name} v{version}")
                return detector
            else:
                logger.warning(f"Failed to initialize UI detector {name}")
                return None
        except Exception as e:
            logger.error(f"Error initializing UI detector {name}: {e}")
            return None
        
    def get_detector(self, name: Optional[str] = None, 
                    element_type: Optional[str] = None,
                    required_capabilities: Optional[List[str]] = None) -> Optional[UIDetectorInterface]:
        """
        Get a detector by name, type, or capabilities
        
        Args:
            name: Optional specific detector name
            element_type: Optional element type for specialized detector
            required_capabilities: Optional list of required capabilities
            
        Returns:
            Best matching detector or None if none available
        """
        # Case 1: Specific detector requested
        if name:
            # Try specialized first if element_type provided
            if element_type and element_type in self.specialized_detectors and name in self.specialized_detectors[element_type]:
                return self.specialized_detectors[element_type][name]
                
            # Try general detectors
            if name in self.detectors:
                return self.detectors[name]
                
            # Not found, try to initialize
            return self.initialize_detector(name)
            
        # Case 2: Specialized detector for element type
        if element_type and element_type in self.specialized_detectors:
            # Get highest performing specialized detector
            best_detector = None
            best_score = -1
            
            for detector_name, detector in self.specialized_detectors[element_type].items():
                score = self._calculate_detector_score(detector_name, element_type)
                if score > best_score:
                    best_score = score
                    best_detector = detector
                    
            if best_detector:
                return best_detector
                
        # Case 3: General detector with required capabilities
        best_detector = None
        best_score = -1
        
        for detector_name, detector in self.detectors.items():
            # Check capabilities if required
            if required_capabilities:
                capabilities = detector.get_capabilities()
                if not all(cap in capabilities for cap in required_capabilities):
                    continue
                    
            score = self._calculate_detector_score(detector_name)
            if score > best_score:
                best_score = score
                best_detector = detector
                
        return best_detector
        
    def _calculate_detector_score(self, detector_name: str, element_type: Optional[str] = None) -> float:
        """
        Calculate a score for detector selection based on performance metrics
        
        Higher score = preferred detector
        """
        if detector_name not in self.performance_metrics:
            return 0.0
            
        metrics = self.performance_metrics[detector_name]
        
        # Base score from priority
        priority = self.detector_classes[detector_name]["priority"]
        score = priority * 10  # Priority is a major factor
        
        # Add performance metrics if we have sufficient data
        if metrics["calls"] > 0:
            # Success rate (0-100)
            success_rate = metrics["successes"] / metrics["calls"] * 100
            score += success_rate
            
            # Average confidence (0-100)
            score += metrics["average_confidence"] * 50
            
            # Penalize for high latency
            latency_penalty = min(20, metrics["average_latency"])
            score -= latency_penalty
            
        return score
        
    def update_metrics(self, detector_name: str, metrics: Dict[str, Any]) -> None:
        """Update performance metrics for a detector"""
        if detector_name not in self.performance_metrics:
            logger.warning(f"Detector {detector_name} not in registry, can't update metrics")
            return
            
        current = self.performance_metrics[detector_name]
        
        # Update calls and success counts
        current["calls"] += 1
        if metrics.get("success", False):
            current["successes"] += 1
        else:
            current["failures"] += 1
            
        # Update detection metrics
        elements_detected = metrics.get("elements_detected", 0)
        current["total_elements_detected"] += elements_detected
        
        # Update average confidence
        if "confidence" in metrics and elements_detected > 0:
            # Weighted average with previous values
            old_weight = current["total_elements_detected"] - elements_detected
            old_value = current["average_confidence"] * old_weight if old_weight > 0 else 0
            new_value = metrics["confidence"] * elements_detected
            total_weight = current["total_elements_detected"]
            
            if total_weight > 0:
                current["average_confidence"] = (old_value + new_value) / total_weight
                
        # Update latency metrics
        if "latency" in metrics:
            # Exponential moving average for latency
            alpha = 0.1  # Weight for new values
            current["average_latency"] = (alpha * metrics["latency"] + 
                                        (1 - alpha) * current["average_latency"])
                                        
        # Save metrics periodically
        if current["calls"] % 10 == 0:
            self._save_performance_metrics()
            
    def detect_ui_elements(self, screenshot: Any, context: Optional[Dict] = None) -> List[Dict]:
        """
        Detect UI elements using the best available detector
        
        Args:
            screenshot: Screenshot image
            context: Optional context information
            
        Returns:
            List of detected UI elements
        """
        # Get the best detector
        detector = self.get_detector()
        
        if not detector:
            logger.warning("No UI detector available")
            return []
            
        # Start timing
        start_time = time.time()
        
        try:
            # Detect elements
            elements = detector.detect_elements(screenshot, context)
            
            # Calculate metrics
            elapsed_time = time.time() - start_time
            success = len(elements) > 0
            
            # Update metrics
            detector_name = detector.get_capabilities().get("name", "unknown")
            self.update_metrics(detector_name, {
                "success": success,
                "elements_detected": len(elements),
                "confidence": sum(e.get("confidence", 0) for e in elements) / len(elements) if elements else 0,
                "latency": elapsed_time
            })
            
            return elements
            
        except Exception as e:
            # Calculate metrics
            elapsed_time = time.time() - start_time
            
            # Update metrics for failure
            detector_name = detector.get_capabilities().get("name", "unknown")
            self.update_metrics(detector_name, {
                "success": False,
                "elements_detected": 0,
                "confidence": 0,
                "latency": elapsed_time
            })
            
            logger.error(f"Error detecting UI elements: {e}")
            return []
            
    def get_all_detectors(self) -> Dict[str, Any]:
        """Get all registered detectors with metrics"""
        result = {
            "general_detectors": {},
            "specialized_detectors": {}
        }
        
        # Add general detectors
        for name, detector in self.detectors.items():
            result["general_detectors"][name] = {
                "metrics": self.performance_metrics.get(name, {}),
                "capabilities": detector.get_capabilities(),
                "versions": self.detector_versions.get(name, [])
            }
            
        # Add specialized detectors
        for element_type, detectors in self.specialized_detectors.items():
            result["specialized_detectors"][element_type] = {}
            for name, detector in detectors.items():
                result["specialized_detectors"][element_type][name] = {
                    "metrics": self.performance_metrics.get(name, {}),
                    "capabilities": detector.get_capabilities(),
                    "versions": self.detector_versions.get(name, [])
                }
                
        return result
        
    def _save_performance_metrics(self) -> None:
        """Save performance metrics to disk"""
        try:
            metrics_path = os.path.join(self.memory_path, "performance_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(self.performance_metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving performance metrics: {e}")
            
    def _load_performance_metrics(self) -> None:
        """Load performance metrics from disk"""
        try:
            metrics_path = os.path.join(self.memory_path, "performance_metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    self.performance_metrics = json.load(f)
        except Exception as e:
            logger.error(f"Error loading performance metrics: {e}")
            self.performance_metrics = {}
