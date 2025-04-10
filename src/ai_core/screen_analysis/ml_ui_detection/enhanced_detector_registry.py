"""
Enhanced UI Detector Registry

This module provides an advanced registry for UI element detectors with:
- Adaptive detector selection based on performance metrics
- Multi-model intelligence fusion
- Continuous learning capabilities
- Performance tracking and analysis
"""

import os
import time
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Type, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import threading
from collections import defaultdict

# Import the detector interface
from ..ui_detection.detector_interface import UIDetectorInterface

# Import detectors
from .adaptive_detector import AdaptiveDetector
from .huggingface_detector import HuggingFaceDetector
from .autogluon_detector_adapter import AutoGluonDetectorAdapter
from .askui_detector_adapter import AskUIDetectorAdapter

logger = logging.getLogger(__name__)

@dataclass
class DetectorPerformanceMetrics:
    """Performance metrics for a UI detector"""
    name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_elements_detected: int = 0
    average_latency: float = 0.0
    average_confidence: float = 0.0
    last_update_time: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        """Calculate the success rate"""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls

    @property
    def elements_per_call(self) -> float:
        """Calculate average elements detected per call"""
        if self.successful_calls == 0:
            return 0.0
        return self.total_elements_detected / self.successful_calls

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'total_calls': self.total_calls,
            'successful_calls': self.successful_calls,
            'failed_calls': self.failed_calls,
            'success_rate': self.success_rate,
            'total_elements_detected': self.total_elements_detected,
            'elements_per_call': self.elements_per_call,
            'average_latency': self.average_latency,
            'average_confidence': self.average_confidence,
            'last_update_time': self.last_update_time
        }


class EnhancedDetectorRegistry:
    """
    Advanced registry for UI element detectors with adaptive selection,
    performance tracking, and multi-model intelligence fusion.
    """
    
    def __init__(self, memory_path: str = "memory/ml_ui_detection"):
        self.detectors = {}  # name -> detector instance
        self.specialized_detectors = {}  # element_type -> {name -> detector}
        self.detector_classes = {}  # name -> class info
        self.performance_metrics = {}  # name -> DetectorPerformanceMetrics
        self.fusion_strategies = {}  # strategy_name -> fusion strategy
        self.memory_path = memory_path
        self.lock = threading.RLock()  # Thread safety for metrics updates
        
        # Default exploration rate for adaptive selection
        self.exploration_rate = 0.1
        
        # Create memory directory if it doesn't exist
        os.makedirs(memory_path, exist_ok=True)
        
        # Load performance metrics if available
        self._load_performance_metrics()
        
        # Register fusion strategies
        self._register_default_fusion_strategies()
        
        # Initialize built-in detectors
        detector_classes = [
            AdaptiveDetector,
            HuggingFaceDetector,
            AutoGluonDetectorAdapter,
            AskUIDetectorAdapter
        ]
        
        for detector_class in detector_classes:
            self.register_detector(detector_class.__name__, detector_class)
    
    def register_detector(self, name: str, detector_class: Type[UIDetectorInterface],
                          specialized_for: Optional[List[str]] = None,
                          priority: int = 0,
                          config: Optional[Dict] = None) -> bool:
        """
        Register a detector with the registry
        
        Args:
            name: Unique detector name
            detector_class: Detector class (not instance)
            specialized_for: Optional list of element types this detector specializes in
            priority: Priority for selection (higher = preferred)
            config: Optional configuration for detector
            
        Returns:
            Success status
        """
        with self.lock:
            # Validate detector doesn't already exist
            if name in self.detector_classes:
                logger.warning(f"Detector {name} already registered")
                return False
            
            # Store the detector class and configuration
            self.detector_classes[name] = {
                "class": detector_class,
                "priority": priority,
                "config": config or {},
                "specialized_for": specialized_for or [],
                "registration_time": time.time()
            }
            
            # Initialize performance metrics
            self.performance_metrics[name] = DetectorPerformanceMetrics(name=name)
            
            logger.info(f"Registered UI detector: {name}")
            
            # Save updated metrics
            self._save_performance_metrics()
            
            return True
    
    def initialize_detector(self, name: str) -> Optional[UIDetectorInterface]:
        """
        Initialize a detector by name
        
        Args:
            name: Detector name
            
        Returns:
            Initialized detector or None if initialization fails
        """
        if name not in self.detector_classes:
            logger.warning(f"Detector {name} not registered")
            return None
        
        # Check if already initialized
        if name in self.detectors:
            logger.info(f"Detector {name} already initialized")
            return self.detectors[name]
        
        # Get class information
        detector_info = self.detector_classes[name]
        detector_class = detector_info["class"]
        config = detector_info["config"]
        
        try:
            # Create detector instance
            detector = detector_class(config)
            
            # Initialize the detector
            success = detector.initialize(config)
            if not success:
                logger.warning(f"Failed to initialize detector {name}")
                return None
            
            # Store the initialized detector
            self.detectors[name] = detector
            
            # Register for specialized element types if specified
            specialized_for = detector_info.get("specialized_for", [])
            for element_type in specialized_for:
                if element_type not in self.specialized_detectors:
                    self.specialized_detectors[element_type] = {}
                self.specialized_detectors[element_type][name] = detector
            
            logger.info(f"Initialized detector {name}")
            return detector
            
        except Exception as e:
            logger.error(f"Error initializing detector {name}: {e}")
            return None
    
    def get_best_detector(self, element_type: Optional[str] = None, 
                         context: Optional[Dict] = None) -> Tuple[Optional[UIDetectorInterface], str]:
        """
        Get the best detector based on performance metrics and context
        
        Args:
            element_type: Optional element type for specialized detector
            context: Optional context information
            
        Returns:
            Tuple of (best detector, detector name) or (None, "") if none available
        """
        # Exploration vs. exploitation (epsilon-greedy strategy)
        if np.random.random() < self.exploration_rate:
            # Exploration: try a random detector
            return self._get_random_detector(element_type)
        
        # Exploitation: use best performing detector
        return self._get_highest_performing_detector(element_type, context)
    
    def _get_random_detector(self, element_type: Optional[str] = None) -> Tuple[Optional[UIDetectorInterface], str]:
        """
        Get a random detector for exploration
        
        Args:
            element_type: Optional element type for specialized detector
            
        Returns:
            Tuple of (random detector, detector name) or (None, "") if none available
        """
        available_detectors = {}
        
        # If element type specified, check specialized detectors first
        if element_type and element_type in self.specialized_detectors:
            available_detectors.update(self.specialized_detectors[element_type])
        
        # Add general detectors
        available_detectors.update(self.detectors)
        
        if not available_detectors:
            logger.warning("No detectors available for selection")
            return None, ""
        
        # Select random detector
        detector_name = np.random.choice(list(available_detectors.keys()))
        return available_detectors[detector_name], detector_name
    
    def _get_highest_performing_detector(self, element_type: Optional[str] = None,
                                       context: Optional[Dict] = None) -> Tuple[Optional[UIDetectorInterface], str]:
        """
        Get the highest performing detector based on metrics
        
        Args:
            element_type: Optional element type for specialized detector
            context: Optional context information for contextual selection
            
        Returns:
            Tuple of (best detector, detector name) or (None, "") if none available
        """
        candidates = {}
        scores = {}
        
        # If element type specified, check specialized detectors first
        if element_type and element_type in self.specialized_detectors:
            for name, detector in self.specialized_detectors[element_type].items():
                candidates[name] = detector
                scores[name] = self._calculate_detector_score(name, element_type, context)
        
        # Add general detectors if no specialized ones available or scores too low
        if not candidates or max(scores.values()) < 0.5:
            for name, detector in self.detectors.items():
                # Skip if already added as specialized
                if name in candidates:
                    continue
                    
                candidates[name] = detector
                scores[name] = self._calculate_detector_score(name, None, context)
        
        if not candidates:
            logger.warning("No detectors available for selection")
            return None, ""
        
        # Get the highest scoring detector
        best_name = max(scores, key=scores.get)
        return candidates[best_name], best_name
    
    def _calculate_detector_score(self, detector_name: str, 
                               element_type: Optional[str] = None,
                               context: Optional[Dict] = None) -> float:
        """
        Calculate a score for detector selection based on performance metrics
        
        Args:
            detector_name: Detector name
            element_type: Optional element type
            context: Optional context information
            
        Returns:
            Score (higher = better)
        """
        if detector_name not in self.performance_metrics:
            return 0.0
        
        metrics = self.performance_metrics[detector_name]
        
        # Base score based on success rate and elements per call
        base_score = metrics.success_rate * 0.6 + min(1.0, metrics.elements_per_call / 10) * 0.3
        
        # Adjust for average confidence
        confidence_factor = metrics.average_confidence * 0.2
        
        # Adjust for latency (lower is better)
        latency_factor = 0.0
        if metrics.average_latency > 0:
            # Normalize latency (assuming 2000ms as high threshold)
            norm_latency = min(1.0, metrics.average_latency / 2000.0)
            latency_factor = (1.0 - norm_latency) * 0.1
        
        # Adjust for priority from registration
        priority = self.detector_classes[detector_name].get("priority", 0)
        priority_factor = min(0.2, priority * 0.05)
        
        # Context-aware scoring if context provided
        context_factor = 0.0
        if context:
            # App-specific scoring based on past performance in this app
            app_name = context.get("app_name", "")
            # This would require tracking per-app metrics, simplified here
            context_factor = 0.1 if app_name else 0.0
        
        # Calculate final score
        score = base_score + confidence_factor + latency_factor + priority_factor + context_factor
        
        return min(1.0, max(0.0, score))
    
    def update_metrics(self, detector_name: str, detection_time: float, 
                      detected_elements: List[Dict], success: bool = True):
        """
        Update performance metrics for a detector after a detection run
        
        Args:
            detector_name: Detector name
            detection_time: Time taken for detection (ms)
            detected_elements: List of detected elements
            success: Whether the detection was successful
        """
        with self.lock:
            if detector_name not in self.performance_metrics:
                logger.warning(f"No metrics found for detector {detector_name}")
                return
            
            metrics = self.performance_metrics[detector_name]
            metrics.total_calls += 1
            metrics.last_update_time = time.time()
            
            if success:
                metrics.successful_calls += 1
                elements_count = len(detected_elements)
                metrics.total_elements_detected += elements_count
                
                # Update average latency with exponential moving average
                if metrics.average_latency == 0:
                    metrics.average_latency = detection_time
                else:
                    metrics.average_latency = 0.9 * metrics.average_latency + 0.1 * detection_time
                
                # Update average confidence with exponential moving average
                if elements_count > 0:
                    avg_confidence = sum(e.get("confidence", 0) for e in detected_elements) / elements_count
                    if metrics.average_confidence == 0:
                        metrics.average_confidence = avg_confidence
                    else:
                        metrics.average_confidence = 0.9 * metrics.average_confidence + 0.1 * avg_confidence
            else:
                metrics.failed_calls += 1
            
            # Save updated metrics
            self._save_performance_metrics()
    
    def detect_with_fusion(self, screenshot: np.ndarray, 
                         strategy: str = "weighted_confidence",
                         context: Optional[Dict] = None) -> List[Dict]:
        """
        Detect UI elements using multiple detectors and fusion
        
        Args:
            screenshot: Screenshot as numpy array
            strategy: Fusion strategy name
            context: Optional context information
            
        Returns:
            List of detected elements after fusion
        """
        if not self.detectors:
            logger.warning("No detectors initialized")
            return []
        
        # Get fusion strategy
        fusion_func = self.fusion_strategies.get(strategy)
        if not fusion_func:
            logger.warning(f"Unknown fusion strategy: {strategy}, using weighted_confidence")
            fusion_func = self.fusion_strategies.get("weighted_confidence")
            if not fusion_func:
                logger.error("No fusion strategies available")
                return []
        
        # Run all detectors
        detector_results = {}
        for name, detector in self.detectors.items():
            try:
                start_time = time.time()
                elements = detector.detect_elements(screenshot, context)
                detection_time = (time.time() - start_time) * 1000  # convert to ms
                
                # Update metrics
                self.update_metrics(name, detection_time, elements, success=True)
                
                detector_results[name] = {
                    "elements": elements,
                    "metrics": self.performance_metrics[name].to_dict()
                }
            except Exception as e:
                logger.error(f"Error running detector {name}: {e}")
                self.update_metrics(name, 0, [], success=False)
        
        # Apply fusion strategy
        fused_elements = fusion_func(detector_results, context)
        
        return fused_elements
    
    def get_all_detectors(self) -> Dict[str, UIDetectorInterface]:
        """
        Get all registered and initialized detectors
        
        Returns:
            Dictionary mapping detector names to detector instances
        """
        # Return only initialized detectors
        initialized_detectors = {}
        
        with self.lock:
            for name, detector in self.detectors.items():
                if hasattr(detector, 'initialized') and detector.initialized:
                    initialized_detectors[name] = detector
                    
        return initialized_detectors
        
    def select_best_detector(self) -> str:
        """
        Select the best detector based on performance metrics
        
        Returns:
            Name of the best detector or empty string if none available
        """
        detector, name = self.get_best_detector()
        return name
        
    def select_detector_for_context(self, context: Dict) -> str:
        """
        Select the best detector for the given context
        
        Args:
            context: Context information for detector selection
            
        Returns:
            Name of the selected detector or empty string if none available
        """
        element_type = context.get('element_type')
        detector, name = self.get_best_detector(element_type=element_type, context=context)
        return name
        
    def update_detector_performance(self, detector_name: str, performance_data: Dict):
        """
        Update performance metrics for a detector manually
        
        Args:
            detector_name: Name of the detector
            performance_data: Performance data to update
        """
        if detector_name not in self.performance_metrics:
            self.performance_metrics[detector_name] = DetectorPerformanceMetrics(name=detector_name)
            
        metrics = self.performance_metrics[detector_name]
        
        # Update metrics with provided data
        success_rate = performance_data.get('success_rate')
        if success_rate is not None:
            # Calculate implied total calls based on success rate
            successful_calls = performance_data.get('successful_calls')
            if successful_calls is not None:
                total_implied = int(successful_calls / success_rate) if success_rate > 0 else 0
                metrics.successful_calls = successful_calls
                metrics.failed_calls = total_implied - metrics.successful_calls
                metrics.total_calls = total_implied
            
        avg_confidence = performance_data.get('avg_confidence')
        if avg_confidence is not None:
            metrics.average_confidence = avg_confidence
            
        avg_detection_time = performance_data.get('avg_detection_time')
        if avg_detection_time is not None:
            metrics.average_latency = avg_detection_time
            
        # Update timestamp
        metrics.last_update_time = time.time()
        
        # Save metrics
        self._save_performance_metrics()
        
    def update_specialization(self, detector_name: str, element_type: str, success_rate: float) -> None:
        """
        Update detector specialization for a specific element type
        
        Args:
            detector_name: Name of the detector
            element_type: Type of UI element
            success_rate: Success rate for this element type
        """
        detector = self.registry.get(detector_name)
        if not detector:
            logger.warning(f"Cannot update specialization for unknown detector {detector_name}")
            return
            
        # Initialize specializations dict in registry_meta if needed
        if "registry_meta" not in self.__dict__:
            self.registry_meta = {}
            
        if detector_name not in self.registry_meta:
            self.registry_meta[detector_name] = {}
            
        if "specializations" not in self.registry_meta[detector_name]:
            self.registry_meta[detector_name]["specializations"] = {}
            
        # Update specialization with exponential moving average
        current = self.registry_meta[detector_name]["specializations"].get(element_type, success_rate)
        alpha = 0.2  # Weight for new value (0.2 means 20% weight to new value)
        updated = (alpha * success_rate) + ((1 - alpha) * current)
        self.registry_meta[detector_name]["specializations"][element_type] = updated
        
        # Also update specialized_detectors registry for faster lookup
        if element_type not in self.specialized_detectors:
            self.specialized_detectors[element_type] = {}
        self.specialized_detectors[element_type][detector_name] = detector
        
        logger.info(f"Updated specialization for detector {detector_name} on element type {element_type}: {updated:.2f}")
        
    def update_context_association(self, detector_name: str, context_type: str, success_rate: float) -> None:
        """
        Update detector association with a specific context type
        
        Args:
            detector_name: Name of the detector
            context_type: Type of context (e.g., task type)
            success_rate: Success rate for this context
        """
        detector = self.registry.get(detector_name)
        if not detector:
            logger.warning(f"Cannot update context association for unknown detector {detector_name}")
            return
            
        # Initialize registry_meta if needed
        if "registry_meta" not in self.__dict__:
            self.registry_meta = {}
            
        if detector_name not in self.registry_meta:
            self.registry_meta[detector_name] = {}
            
        # Initialize context associations dict if needed
        if "context_associations" not in self.registry_meta[detector_name]:
            self.registry_meta[detector_name]["context_associations"] = {}
            
        # Update context association with exponential moving average
        current = self.registry_meta[detector_name]["context_associations"].get(context_type, success_rate)
        alpha = 0.25  # Weight for new value (0.25 means 25% weight to new value)
        updated = (alpha * success_rate) + ((1 - alpha) * current)
        self.registry_meta[detector_name]["context_associations"][context_type] = updated
        
        logger.info(f"Updated context association for detector {detector_name} with context {context_type}: {updated:.2f}")
        
    def get_detector_metrics(self, detector_name: str) -> Dict:
        """
        Get performance metrics for a detector
        
        Args:
            detector_name: Name of the detector
            
        Returns:
            Dictionary of metrics or empty dict if detector not found
        """
        if detector_name not in self.performance_metrics:
            return {}
            
        return self.performance_metrics[detector_name].to_dict()
    
    def _register_default_fusion_strategies(self):
        """
        Register default fusion strategies
        """
        self.fusion_strategies["weighted_confidence"] = self._weighted_confidence_fusion
        self.fusion_strategies["ensemble_voting"] = self._ensemble_voting_fusion
        self.fusion_strategies["highest_confidence"] = self._highest_confidence_fusion
    
    def _weighted_confidence_fusion(self, detector_results: Dict[str, Dict], 
                                 context: Optional[Dict] = None) -> List[Dict]:
        """
        Fuse results by weighting element confidence by detector performance
        
        Args:
            detector_results: Dictionary mapping detector names to results
            context: Optional context information
            
        Returns:
            Fused list of elements
        """
        all_elements = []
        detector_weights = {}
        
        # Calculate weights based on performance metrics
        total_weight = 0
        for name, result in detector_results.items():
            metrics = result["metrics"]
            weight = metrics["success_rate"] * 0.5 + metrics["average_confidence"] * 0.5
            detector_weights[name] = max(0.1, weight)  # Minimum weight of 0.1
            total_weight += detector_weights[name]
        
        # Normalize weights
        if total_weight > 0:
            for name in detector_weights:
                detector_weights[name] /= total_weight
        
        # Apply weights to element confidence
        element_map = {}  # map from element location to merged element
        
        for name, result in detector_results.items():
            weight = detector_weights[name]
            elements = result["elements"]
            
            for element in elements:
                # Create a unique key based on element bbox
                x1, y1, x2, y2 = element["bbox"]
                # Create fuzzy key to handle slight variations in bbox
                fuzzy_key = f"{x1//10}_{y1//10}_{x2//10}_{y2//10}_{element['type']}"
                
                # Calculate weighted confidence
                weighted_confidence = element["confidence"] * weight
                
                if fuzzy_key in element_map:
                    # Merge with existing element
                    existing = element_map[fuzzy_key]
                    # Update confidence as weighted average
                    total_conf = existing["confidence"] + weighted_confidence
                    # Merge other properties
                    for k, v in element.items():
                        if k == "confidence":
                            continue
                        existing[k] = v
                    # Update confidence
                    existing["confidence"] = total_conf
                else:
                    # Create new element with weighted confidence
                    new_element = element.copy()
                    new_element["confidence"] = weighted_confidence
                    element_map[fuzzy_key] = new_element
        
        # Convert map to list and sort by confidence
        fused_elements = list(element_map.values())
        fused_elements.sort(key=lambda e: e["confidence"], reverse=True)
        
        return fused_elements
    
    def _ensemble_voting_fusion(self, detector_results: Dict[str, Dict],
                               context: Optional[Dict] = None) -> List[Dict]:
        """
        Fuse results by ensemble voting (element must be detected by multiple detectors)
        
        Args:
            detector_results: Dictionary mapping detector names to results
            context: Optional context information
            
        Returns:
            Fused list of elements
        """
        # Count votes for each element
        element_votes = defaultdict(int)
        element_instances = defaultdict(list)
        
        for name, result in detector_results.items():
            elements = result["elements"]
            
            for element in elements:
                # Create a unique key based on element bbox
                x1, y1, x2, y2 = element["bbox"]
                # Create fuzzy key to handle slight variations in bbox
                fuzzy_key = f"{x1//10}_{y1//10}_{x2//10}_{y2//10}_{element['type']}"
                
                element_votes[fuzzy_key] += 1
                element_instances[fuzzy_key].append(element)
        
        # Determine minimum votes needed (at least 2 or 30% of detectors)
        min_votes = max(2, round(0.3 * len(detector_results)))
        
        # Select elements with sufficient votes
        fused_elements = []
        for key, votes in element_votes.items():
            if votes >= min_votes:
                # Merge all instances of this element
                instances = element_instances[key]
                merged = instances[0].copy()
                
                # Average confidence and merge other properties
                avg_confidence = sum(e["confidence"] for e in instances) / len(instances)
                merged["confidence"] = avg_confidence * (votes / len(detector_results))  # Scale by vote ratio
                
                fused_elements.append(merged)
        
        # Sort by confidence
        fused_elements.sort(key=lambda e: e["confidence"], reverse=True)
        
        return fused_elements
    
    def _highest_confidence_fusion(self, detector_results: Dict[str, Dict],
                                 context: Optional[Dict] = None) -> List[Dict]:
        """
        Fuse results by taking elements with highest confidence for each region
        
        Args:
            detector_results: Dictionary mapping detector names to results
            context: Optional context information
            
        Returns:
            Fused list of elements
        """
        element_map = {}  # map from element location to highest confidence element
        
        for name, result in detector_results.items():
            elements = result["elements"]
            
            for element in elements:
                # Create a unique key based on element bbox
                x1, y1, x2, y2 = element["bbox"]
                # Create fuzzy key to handle slight variations in bbox
                fuzzy_key = f"{x1//10}_{y1//10}_{x2//10}_{y2//10}_{element['type']}"
                
                # Check if we already have an element for this region
                if fuzzy_key in element_map:
                    # Keep element with higher confidence
                    if element["confidence"] > element_map[fuzzy_key]["confidence"]:
                        element_map[fuzzy_key] = element
                else:
                    # New element
                    element_map[fuzzy_key] = element
        
        # Convert map to list and sort by confidence
        fused_elements = list(element_map.values())
        fused_elements.sort(key=lambda e: e["confidence"], reverse=True)
        
        return fused_elements
        
    def _load_performance_metrics(self):
        """
        Load performance metrics from memory
        """
        metrics_path = os.path.join(self.memory_path, "detector_metrics.json")
        if not os.path.exists(metrics_path):
            logger.info("No saved metrics found")
            return
        
        try:
            with open(metrics_path, 'r') as f:
                metrics_data = json.load(f)
                
            # Convert to DetectorPerformanceMetrics instances
            for name, data in metrics_data.items():
                metrics = DetectorPerformanceMetrics(name=name)
                metrics.total_calls = data.get("total_calls", 0)
                metrics.successful_calls = data.get("successful_calls", 0)
                metrics.failed_calls = data.get("failed_calls", 0)
                metrics.total_elements_detected = data.get("total_elements_detected", 0)
                metrics.average_latency = data.get("average_latency", 0.0)
                metrics.average_confidence = data.get("average_confidence", 0.0)
                metrics.last_update_time = data.get("last_update_time", time.time())
                self.performance_metrics[name] = metrics
                
            logger.info(f"Loaded performance metrics for {len(metrics_data)} detectors")
        except Exception as e:
            logger.error(f"Error loading performance metrics: {e}")
    
    def _save_performance_metrics(self):
        """
        Save performance metrics to memory
        """
        metrics_path = os.path.join(self.memory_path, "detector_metrics.json")
        
        try:
            # Convert metrics to dictionary
            metrics_data = {name: metrics.to_dict() for name, metrics in self.performance_metrics.items()}
            
            # Save to file
            with open(metrics_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            logger.debug(f"Saved performance metrics for {len(metrics_data)} detectors")
        except Exception as e:
            logger.error(f"Error saving performance metrics: {e}")
