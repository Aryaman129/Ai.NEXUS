"""
Registry for UI element detectors in NEXUS.

This module provides a central registry for UI element detectors,
enabling adaptive selection of the best detector based on performance metrics.
It implements the core adaptive philosophy of NEXUS by allowing the system
to automatically choose the most effective detector for each task.
"""

from typing import Dict, List, Any, Optional, Type, Tuple
import numpy as np
import time
import logging
from .base import UIDetectorInterface

logger = logging.getLogger(__name__)

class DetectorRegistry:
    """Registry for UI element detectors with adaptive selection"""
    
    def __init__(self):
        self.detectors = {}  # name -> detector instance
        self.specialized_detectors = {}  # element_type -> {name -> detector}
        self.performance_metrics = {}  # name -> metrics
        self.detector_versions = {}  # name -> version history
        self.detector_classes = {}  # name -> class info
        
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
                "average_latency": 0.0
            }
            
        # Initialize version history
        if name not in self.detector_versions:
            self.detector_versions[name] = []
            
        logger.info(f"Registered detector {name} with priority {priority}")
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
        try:
            detector = detector_info["class"]()
        except Exception as e:
            logger.error(f"Error instantiating detector {name}: {e}")
            return None
        
        # Try to initialize
        try:
            success = detector.initialize(detector_info["config"])
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
                
                logger.info(f"Initialized detector {name} v{version}")
                return detector
            else:
                logger.warning(f"Failed to initialize detector {name}")
                return None
        except Exception as e:
            logger.error(f"Error initializing detector {name}: {e}")
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
                
        # If no detector is initialized yet, initialize the highest priority one
        if not best_detector and self.detector_classes:
            # Find highest priority uninitialized detector
            highest_priority = -1
            highest_priority_name = None
            
            for name, info in self.detector_classes.items():
                # Skip specialized detectors if we're looking for general
                if info["specialized_for"] and not element_type:
                    continue
                    
                # Skip if specialized but not for the requested type
                if info["specialized_for"] and element_type and info["specialized_for"] != element_type:
                    continue
                    
                if info["priority"] > highest_priority:
                    highest_priority = info["priority"]
                    highest_priority_name = name
                    
            if highest_priority_name:
                best_detector = self.initialize_detector(highest_priority_name)
                
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
        if detector_name in self.detector_classes:
            priority = self.detector_classes[detector_name]["priority"]
        else:
            priority = 0
            
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
                                        
        # Update version-specific metrics if version provided
        if "version" in metrics:
            self.record_version_performance(detector_name, metrics["version"], metrics)
            
    def record_version_performance(self, detector_name, version, metrics):
        """Record performance metrics for a specific detector version"""
        if detector_name not in self.detector_versions:
            return
            
        # Find the version entry
        for version_entry in self.detector_versions[detector_name]:
            if version_entry["version"] == version:
                # Update performance metrics
                if "performance" not in version_entry:
                    version_entry["performance"] = {
                        "calls": 0,
                        "successes": 0,
                        "average_confidence": 0.0
                    }
                    
                perf = version_entry["performance"]
                perf["calls"] += 1
                
                if metrics.get("success", False):
                    perf["successes"] += 1
                    
                # Update average confidence
                if "confidence" in metrics:
                    old_avg = perf.get("average_confidence", 0.0)
                    old_weight = perf["calls"] - 1
                    new_value = metrics["confidence"]
                    
                    if perf["calls"] > 0:
                        perf["average_confidence"] = (old_avg * old_weight + new_value) / perf["calls"]
                
                break
                                        
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
        
    def detect_elements_with_consensus(self, screenshot, context=None, min_confidence=0.6):
        """
        Run multiple detectors in parallel and combine results via consensus voting
        
        This approach provides higher accuracy for critical UI actions by using
        multiple detectors and combining their results based on agreement.
        
        Args:
            screenshot: Image as numpy array
            context: Optional context information
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of UI elements with consensus confidence
        """
        all_results = []
        detector_weights = {}
        
        # Get top 3 general detectors by performance score
        detectors = []
        for name, detector in self.detectors.items():
            score = self._calculate_detector_score(name)
            detectors.append((name, detector, score))
        
        # Sort by score and take top 3
        detectors.sort(key=lambda x: x[2], reverse=True)
        top_detectors = detectors[:3]
        
        # Run detectors in parallel and collect results
        for name, detector, score in top_detectors:
            try:
                start_time = time.time()
                elements = detector.detect_elements(screenshot, context)
                elapsed_time = time.time() - start_time
                
                # Update metrics
                confidence = sum(e.get("confidence", 0) for e in elements) / len(elements) if elements else 0
                self.update_metrics(name, {
                    "success": True,
                    "elements_detected": len(elements),
                    "confidence": confidence,
                    "latency": elapsed_time
                })
                
                all_results.append(elements)
                detector_weights[name] = score
                
            except Exception as e:
                logger.error(f"Error running detector {name}: {e}")
                self.update_metrics(name, {
                    "success": False,
                    "elements_detected": 0,
                    "confidence": 0,
                    "latency": 0
                })
        
        # Combine results using weighted voting
        return self._combine_detection_results(all_results, detector_weights, min_confidence)

    def _combine_detection_results(self, all_results, detector_weights, min_confidence):
        """
        Combine results from multiple detectors using weighted voting
        
        Args:
            all_results: List of element lists from different detectors
            detector_weights: Dictionary of detector name -> weight
            min_confidence: Minimum confidence threshold
            
        Returns:
            Combined list of elements with consensus confidence
        """
        if not all_results:
            return []
            
        # Normalize weights
        total_weight = sum(detector_weights.values())
        norm_weights = {k: v/total_weight for k, v in detector_weights.items()}
        
        # Merge similar elements (those with significant overlap)
        merged_elements = []
        for i, elements in enumerate(all_results):
            weight = list(norm_weights.values())[i]
            
            for element in elements:
                # Skip low confidence elements
                if element["confidence"] < min_confidence:
                    continue
                    
                # Check if this element overlaps with any existing merged element
                matched = False
                for merged in merged_elements:
                    if self._calculate_iou(element["bbox"], merged["bbox"]) > 0.5:
                        # Found overlap, update the merged element
                        merged["confidence"] += element["confidence"] * weight
                        merged["vote_count"] += 1
                        
                        # Average the center points
                        cx1, cy1 = merged["center"]
                        cx2, cy2 = element["center"]
                        merged["center"] = ((cx1 + cx2) / 2, (cy1 + cy2) / 2)
                        
                        # If this detector has higher confidence, use its element type
                        if element["confidence"] > merged["confidence"] / merged["vote_count"]:
                            merged["type"] = element["type"]
                            
                        matched = True
                        break
                    
                if not matched:
                    # New element, add it to merged list
                    element["vote_count"] = 1
                    element["confidence"] *= weight
                    element["consensus"] = True
                    merged_elements.append(element)
        
        # Normalize confidence based on vote count
        for element in merged_elements:
            # Adjust confidence based on vote count and consensus
            element["confidence"] /= element["vote_count"]
            element["confidence"] *= element["vote_count"] / len(all_results)  # Weight by consensus
            
        # Filter final results by confidence
        final_results = [e for e in merged_elements if e["confidence"] >= min_confidence]
        
        return final_results
        
    def _calculate_iou(self, bbox1, bbox2):
        """
        Calculate intersection over union for two bounding boxes
        
        Args:
            bbox1: First bounding box (x1, y1, x2, y2)
            bbox2: Second bounding box (x1, y1, x2, y2)
            
        Returns:
            IoU score (0-1)
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - intersection_area
        
        if union_area == 0:
            return 0.0
            
        return intersection_area / union_area
