"""
Visual Memory System for NEXUS

This module provides a persistent memory system for storing and retrieving
visual patterns related to UI elements. It enables NEXUS to learn from
interaction experiences and improve recognition over time.

Key features:
- Storage of UI element visual patterns
- Association with interaction success/failure
- Similarity-based retrieval of known patterns
- Continuous learning from new observations
"""
import os
import time
import json
import logging
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import pickle
from datetime import datetime

# Check if detector modules are available
try:
    from .detectors import (
        DetectorRegistry, ConfidenceCalibrator,
        AutoGluonDetector, HuggingFaceDetector, OpenCVDetector
    )
    from .visualization import DetectionVisualizer
    DETECTORS_AVAILABLE = True
except ImportError:
    DETECTORS_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)

class VisualMemorySystem:
    """
    Visual memory system for UI patterns
    
    This class provides methods for storing, retrieving, and learning from
    visual patterns related to UI elements. It enables NEXUS to improve
    recognition and interaction over time.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the visual memory system
        
        Args:
            config: Optional configuration dictionary with the following keys:
                - memory_path: Path to store visual memory data
                - max_patterns: Maximum number of patterns to store
                - similarity_threshold: Threshold for pattern matching (0-1)
                - enable_learning: Whether to enable continuous learning
                - use_ml_detection: Whether to use ML-based UI detection (default: True)
                - use_blended_detection: Whether to blend ML and legacy detection (default: True)
                - detector_config: Configuration for specific detectors
        """
        self.config = config or {}
        
        # Set default configuration
        self.memory_path = self.config.get("memory_path", "memory/visual")
        self.max_patterns = self.config.get("max_patterns", 10000)
        self.similarity_threshold = self.config.get("similarity_threshold", 0.7)
        self.enable_learning = self.config.get("enable_learning", True)
        self.use_ml_detection = self.config.get("use_ml_detection", True) 
        self.use_blended_detection = self.config.get("use_blended_detection", True)
        
        # Initialize memory storage
        self.patterns = []
        self.interaction_history = []
        
        # Create memory directory if it doesn't exist
        os.makedirs(self.memory_path, exist_ok=True)
        
        # Initialize ML detection components if available
        self.detector_registry = None
        self.current_detector = None
        self.confidence_calibrator = None
        self.visualizer = None
        
        if DETECTORS_AVAILABLE and self.use_ml_detection:
            self._initialize_detectors()
        
        # Load existing memory if available
        self._load_memory()
        
        logger.info(f"VisualMemorySystem initialized with {len(self.patterns)} patterns")
    
    def _load_memory(self):
        """Load existing visual memory from disk"""
        patterns_path = os.path.join(self.memory_path, "patterns.pkl")
        history_path = os.path.join(self.memory_path, "interaction_history.json")
        
        # Load patterns
        if os.path.exists(patterns_path):
            try:
                with open(patterns_path, "rb") as f:
                    self.patterns = pickle.load(f)
                logger.info(f"Loaded {len(self.patterns)} patterns from memory")
            except Exception as e:
                logger.error(f"Error loading visual patterns: {e}")
                self.patterns = []
        
        # Load interaction history
        if os.path.exists(history_path):
            try:
                with open(history_path, "r") as f:
                    self.interaction_history = json.load(f)
                logger.info(f"Loaded {len(self.interaction_history)} interaction records")
            except Exception as e:
                logger.error(f"Error loading interaction history: {e}")
                self.interaction_history = []
    
    def _save_memory(self):
        """Save visual memory to disk"""
        patterns_path = os.path.join(self.memory_path, "patterns.pkl")
        history_path = os.path.join(self.memory_path, "interaction_history.json")
        
        # Create backup of existing files
        for path in [patterns_path, history_path]:
            if os.path.exists(path):
                backup_path = f"{path}.bak"
                try:
                    import shutil
                    shutil.copy2(path, backup_path)
                except Exception as e:
                    logger.warning(f"Failed to create backup of {path}: {e}")
        
        # Save patterns
        try:
            with open(patterns_path, "wb") as f:
                pickle.dump(self.patterns, f)
        except Exception as e:
            logger.error(f"Error saving visual patterns: {e}")
        
        # Save interaction history
        try:
            with open(history_path, "w") as f:
                json.dump(self.interaction_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving interaction history: {e}")
    
    def store_pattern(self, pattern: Dict) -> str:
        """
        Store a new visual pattern
        
        Args:
            pattern: Dictionary containing pattern information:
                - visual_signature: NumPy array of visual features
                - type: UI element type (button, checkbox, etc.)
                - text: Optional text content of the element
                - metadata: Additional information (app name, window title, etc.)
                
        Returns:
            ID of the stored pattern
        """
        # Generate a unique ID for the pattern
        pattern_id = f"pattern_{int(time.time())}_{len(self.patterns)}"
        
        # Add ID and timestamp to pattern
        pattern_with_meta = pattern.copy()
        pattern_with_meta["id"] = pattern_id
        pattern_with_meta["timestamp"] = time.time()
        pattern_with_meta["creation_date"] = datetime.now().isoformat()
        pattern_with_meta["successful_interactions"] = 0
        pattern_with_meta["failed_interactions"] = 0
        
        # Add to patterns list
        self.patterns.append(pattern_with_meta)
        
        # Trim patterns list if it exceeds maximum size
        if len(self.patterns) > self.max_patterns:
            # Sort by success rate and remove least successful patterns
            self.patterns.sort(key=lambda p: p.get("successful_interactions", 0) / 
                              (p.get("successful_interactions", 0) + p.get("failed_interactions", 1)),
                              reverse=True)
            self.patterns = self.patterns[:self.max_patterns]
        
        # Save memory after adding new pattern
        if len(self.patterns) % 10 == 0:  # Save every 10 patterns to avoid excessive disk I/O
            self._save_memory()
            
        return pattern_id
    
    def find_similar_patterns(self, visual_signature: np.ndarray, 
                             element_type: Optional[str] = None,
                             text_query: Optional[str] = None,
                             metadata_filter: Optional[Dict] = None,
                             max_results: int = 5) -> List[Dict]:
        """
        Find patterns similar to the given visual signature
        
        Args:
            visual_signature: NumPy array of visual features
            element_type: Optional filter by UI element type
            text_query: Optional filter by text content (substring match)
            metadata_filter: Optional filter by metadata fields
            max_results: Maximum number of results to return
            
        Returns:
            List of matching patterns with similarity scores
        """
        matches = []
        
        for pattern in self.patterns:
            # Skip if element type doesn't match
            if element_type and pattern.get("type") != element_type:
                continue
                
            # Skip if text doesn't match
            if text_query and text_query.lower() not in pattern.get("text", "").lower():
                continue
                
            # Skip if metadata doesn't match
            if metadata_filter:
                pattern_metadata = pattern.get("metadata", {})
                if not all(pattern_metadata.get(k) == v for k, v in metadata_filter.items()):
                    continue
            
            # Calculate visual similarity
            pattern_signature = pattern.get("visual_signature")
            if pattern_signature is not None:
                similarity = self._calculate_similarity(visual_signature, pattern_signature)
                
                # If similarity is above threshold, add to matches
                if similarity >= self.similarity_threshold:
                    match = {
                        "id": pattern.get("id"),
                        "type": pattern.get("type"),
                        "text": pattern.get("text"),
                        "similarity": similarity,
                        "successful_interactions": pattern.get("successful_interactions", 0),
                        "failed_interactions": pattern.get("failed_interactions", 0),
                        "creation_date": pattern.get("creation_date"),
                        "metadata": pattern.get("metadata", {})
                    }
                    matches.append(match)
        
        # Sort by similarity and return top matches
        matches.sort(key=lambda m: m["similarity"], reverse=True)
        return matches[:max_results]
    
    def get_pattern_by_id(self, pattern_id: str) -> Optional[Dict]:
        """
        Get a pattern by its ID
        
        Args:
            pattern_id: ID of the pattern to retrieve
            
        Returns:
            Pattern dictionary or None if not found
        """
        for pattern in self.patterns:
            if pattern.get("id") == pattern_id:
                return pattern.copy()
        return None
    
    def record_interaction(self, pattern_id: str, action: str, 
                          success: bool, context: Optional[Dict] = None) -> bool:
        """
        Record an interaction with a UI element
        
        Args:
            pattern_id: ID of the pattern that was interacted with
            action: Type of interaction (click, type, etc.)
            success: Whether the interaction was successful
            context: Additional context about the interaction
            
        Returns:
            True if the interaction was recorded, False otherwise
        """
        # Find the pattern
        pattern = None
        pattern_index = -1
        
        for i, p in enumerate(self.patterns):
            if p.get("id") == pattern_id:
                pattern = p
                pattern_index = i
                break
                
        if pattern is None:
            logger.warning(f"Pattern not found for ID: {pattern_id}")
            return False
            
        # Update pattern success/failure counters
        if success:
            pattern["successful_interactions"] = pattern.get("successful_interactions", 0) + 1
        else:
            pattern["failed_interactions"] = pattern.get("failed_interactions", 0) + 1
            
        # Update the pattern in the list
        self.patterns[pattern_index] = pattern
        
        # Record interaction in history
        interaction = {
            "pattern_id": pattern_id,
            "action": action,
            "success": success,
            "timestamp": time.time(),
            "date": datetime.now().isoformat(),
            "context": context or {}
        }
        
        self.interaction_history.append(interaction)
        
        # Trim history if it gets too large
        max_history = 1000
        if len(self.interaction_history) > max_history:
            self.interaction_history = self.interaction_history[-max_history:]
            
        # Save memory periodically
        if len(self.interaction_history) % 10 == 0:
            self._save_memory()
            
        return True
    
    def enhance_detection(self, layout_results: Dict, 
                          ui_elements: List[Dict], 
                          screen_image: np.ndarray) -> List[Dict]:
        """
        Enhance UI element detection with visual memory
        
        Args:
            layout_results: Results from layout analysis
            ui_elements: Detected UI elements from object detection
            screen_image: Screen image as NumPy array
            
        Returns:
            Enhanced list of UI elements with memory-based improvements
        """
        enhanced_elements = ui_elements.copy()
        
        # Extract context information from layout
        context = self._extract_context(layout_results)
        
        # Process each detected element
        for i, element in enumerate(enhanced_elements):
            # Extract element image patch
            bbox = element.get("bbox")
            if bbox:
                x1, y1, x2, y2 = bbox
                element_image = screen_image[y1:y2, x1:x2]
                
                # Create visual signature for the element
                visual_signature = self._create_visual_signature(element_image)
                
                # Find similar patterns in memory
                similar_patterns = self.find_similar_patterns(
                    visual_signature=visual_signature,
                    element_type=element.get("type")
                )
                
                # If we found similar patterns, enhance the detection
                if similar_patterns:
                    best_match = similar_patterns[0]
                    
                    # Add memory-based information
                    enhanced_elements[i]["memory_match"] = {
                        "pattern_id": best_match["id"],
                        "similarity": best_match["similarity"],
                        "successful_interactions": best_match["successful_interactions"],
                        "failed_interactions": best_match["failed_interactions"]
                    }
                    
                    # If the element has no text but the memory does, add it
                    if not element.get("text") and best_match.get("text"):
                        enhanced_elements[i]["text"] = best_match["text"]
                        enhanced_elements[i]["text_source"] = "memory"
                        
                    # Calculate confidence adjustment based on interaction history
                    success_rate = self._calculate_success_rate(best_match)
                    memory_confidence = success_rate * best_match["similarity"]
                    
                    # Blend detection and memory confidence
                    orig_confidence = element.get("confidence", 0.5)
                    enhanced_elements[i]["confidence"] = (orig_confidence + memory_confidence) / 2
                
                # If learning is enabled, store new patterns
                if self.enable_learning and visual_signature is not None:
                    if not similar_patterns or similar_patterns[0]["similarity"] < 0.9:
                        # This is a new or significantly different pattern
                        new_pattern = {
                            "visual_signature": visual_signature,
                            "type": element.get("type", "unknown"),
                            "text": element.get("text", ""),
                            "bbox_size": (x2 - x1, y2 - y1),
                            "metadata": {
                                "context": context,
                                "detection_confidence": element.get("confidence", 0)
                            }
                        }
                        self.store_pattern(new_pattern)
        
        return enhanced_elements
    
    def _calculate_similarity(self, signature1: np.ndarray, signature2: np.ndarray) -> float:
        """
        Calculate similarity between two visual signatures
        
        Args:
            signature1: First visual signature
            signature2: Second visual signature
            
        Returns:
            Similarity score (0-1)
        """
        # Ensure signatures are valid
        if signature1 is None or signature2 is None:
            return 0.0
            
        # Ensure signatures have the same shape
        if signature1.shape != signature2.shape:
            # Resize to match if needed
            try:
                signature2 = np.resize(signature2, signature1.shape)
            except:
                return 0.0
        
        try:
            # Calculate cosine similarity
            dot_product = np.dot(signature1.flatten(), signature2.flatten())
            norm1 = np.linalg.norm(signature1.flatten())
            norm2 = np.linalg.norm(signature2.flatten())
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            return dot_product / (norm1 * norm2)
        except:
            # Fallback to Mean Squared Error if cosine similarity fails
            mse = np.mean((signature1 - signature2) ** 2)
            return 1.0 / (1.0 + mse)  # Convert MSE to similarity (0-1)
    
    def _create_visual_signature(self, image: np.ndarray) -> np.ndarray:
        """
        Create a visual signature from an image
        
        Args:
            image: Image as NumPy array
            
        Returns:
            Visual signature as NumPy array
        """
        try:
            # Resize to standard size
            from skimage.transform import resize
            resized = resize(image, (64, 64), anti_aliasing=True)
            
            # Convert to grayscale if color
            if len(resized.shape) == 3 and resized.shape[2] == 3:
                grayscale = np.mean(resized, axis=2)
            else:
                grayscale = resized
                
            # Normalize
            normalized = (grayscale - np.mean(grayscale)) / (np.std(grayscale) + 1e-10)
            
            return normalized
        except Exception as e:
            logger.error(f"Error creating visual signature: {e}")
            return None
    
    def _extract_context(self, layout_results: Dict) -> Dict:
        """
        Extract context information from layout analysis
        
        Args:
            layout_results: Results from layout analysis
            
        Returns:
            Context dictionary
        """
        context = {}
        
        # Extract window title if available
        if "window_title" in layout_results:
            context["window_title"] = layout_results["window_title"]
            
        # Extract application name if available
        if "application" in layout_results:
            context["application"] = layout_results["application"]
            
        return context
    
    def _calculate_success_rate(self, pattern: Dict) -> float:
        """
        Calculate success rate for a pattern
        
        Args:
            pattern: Pattern dictionary
            
        Returns:
            Success rate (0-1)
        """
        successes = pattern.get("successful_interactions", 0)
        failures = pattern.get("failed_interactions", 0)
        
        # If no interactions, assume neutral success rate
        if successes + failures == 0:
            return 0.5
            
        return successes / (successes + failures)
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the visual memory system
        
        Returns:
            Dictionary with statistics
        """
        element_types = {}
        total_successes = 0
        total_failures = 0
        
        # Calculate statistics
        for pattern in self.patterns:
            # Count by element type
            element_type = pattern.get("type", "unknown")
            if element_type not in element_types:
                element_types[element_type] = 0
            element_types[element_type] += 1
            
            # Count interactions
            total_successes += pattern.get("successful_interactions", 0)
            total_failures += pattern.get("failed_interactions", 0)
            
        # Calculate overall success rate
        overall_success_rate = 0.5
        if total_successes + total_failures > 0:
            overall_success_rate = total_successes / (total_successes + total_failures)
            
        return {
            "total_patterns": len(self.patterns),
            "total_interactions": len(self.interaction_history),
            "element_types": element_types,
            "successful_interactions": total_successes,
            "failed_interactions": total_failures,
            "overall_success_rate": overall_success_rate
        }
        
    def clean_memory(self, older_than_days: Optional[int] = None, 
                    max_patterns: Optional[int] = None) -> int:
        """
        Clean up old or unused patterns
        
        Args:
            older_than_days: Remove patterns older than this many days
            max_patterns: Limit memory to this many patterns (removes least successful)
            
        Returns:
            Number of patterns removed
        """
        original_count = len(self.patterns)
        
        # Remove old patterns
        if older_than_days is not None:
            cutoff_time = time.time() - (older_than_days * 86400)
            self.patterns = [p for p in self.patterns if p.get("timestamp", 0) >= cutoff_time]
            
        # Limit to max patterns
        if max_patterns is not None and len(self.patterns) > max_patterns:
            # Sort by success rate
            self.patterns.sort(key=lambda p: p.get("successful_interactions", 0) / 
                              (p.get("successful_interactions", 0) + p.get("failed_interactions", 1)),
                              reverse=True)
            self.patterns = self.patterns[:max_patterns]
            
        # Save memory after cleaning
        self._save_memory()
        
        return original_count - len(self.patterns)
        
    def detect_ui_elements(self, screenshot: np.ndarray, context: Optional[Dict] = None) -> List[Dict]:
        """
        Detect UI elements in the screenshot using ML-based detection with fallback
        
        Args:
            screenshot: Screenshot as numpy array
            context: Optional context information
            
        Returns:
            List of detected UI elements
        """
        # If ML detection is disabled or not available, use legacy detection
        if not DETECTORS_AVAILABLE or not self.use_ml_detection or not self.current_detector:
            return self._legacy_detect_ui_elements(screenshot, context)
            
        # Try ML-based detection
        try:
            # Start timing
            start_time = time.time()
            
            # Run detection with current detector
            elements = self.current_detector.detect_elements(screenshot, context)
            
            # Calculate metrics
            elapsed_time = time.time() - start_time
            avg_confidence = sum(e.get("confidence", 0) for e in elements) / len(elements) if elements else 0
            
            # Update detector metrics
            if self.detector_registry:
                detector_name = self.current_detector.get_capabilities().get("name", "unknown")
                self.detector_registry.update_metrics(detector_name, {
                    "success": True,
                    "elements_detected": len(elements),
                    "confidence": avg_confidence,
                    "latency": elapsed_time
                })
            
            # If using blended approach, combine with legacy detection
            if self.use_blended_detection:
                legacy_elements = self._legacy_detect_ui_elements(screenshot, context)
                elements = self._blend_detection_results(elements, legacy_elements)
                
            # Apply confidence calibration if available
            if self.confidence_calibrator and elements:
                detector_name = self.current_detector.get_capabilities().get("name", "unknown")
                elements = self.confidence_calibrator.calibrate_elements(detector_name, elements)
                
            logger.debug(f"ML detection found {len(elements)} elements in {elapsed_time:.3f}s")
            return elements
            
        except Exception as e:
            logger.error(f"Error in ML UI element detection: {e}")
            
            # Update metrics for failure if registry available
            if self.detector_registry and self.current_detector:
                detector_name = self.current_detector.get_capabilities().get("name", "unknown")
                self.detector_registry.update_metrics(detector_name, {
                    "success": False,
                    "elements_detected": 0,
                    "confidence": 0,
                    "latency": 0
                })
                
            # Fall back to legacy detection
            return self._legacy_detect_ui_elements(screenshot, context)
            
    def _legacy_detect_ui_elements(self, screenshot: np.ndarray, context: Optional[Dict] = None) -> List[Dict]:
        """
        Original UI element detection method using OpenCV
        
        Args:
            screenshot: Screenshot as numpy array
            context: Optional context information
            
        Returns:
            List of detected UI elements
        """
        # Convert to grayscale if needed
        if len(screenshot.shape) == 3:
            grayscale = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        else:
            grayscale = screenshot
            
        # Apply edge detection
        edges = cv2.Canny(grayscale, 50, 150)
        
        # Dilate edges to connect broken lines
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Extract potential UI elements
        ui_elements = []
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter small noise
            if w < 30 or h < 20:
                continue
                
            # Calculate aspect ratio to help classify the element
            aspect_ratio = w / h if h > 0 else 0
            
            # Determine element type based on aspect ratio and size
            if aspect_ratio > 3.0:
                # Very wide elements are likely text fields
                element_type = "text_field"
                confidence = 0.7
            elif 0.9 < aspect_ratio < 1.1 and w < 40 and h < 40:
                # Square elements are likely checkboxes or radio buttons
                element_type = "checkbox"
                confidence = 0.7
            elif aspect_ratio < 0.3 and h > 100:
                # Tall, narrow elements might be scrollbars
                element_type = "scrollbar"
                confidence = 0.6
            else:
                # Default to button for other cases
                element_type = "button"
                confidence = 0.6
                
            # Create element dict
            element = {
                "type": element_type,
                "bbox": (x, y, x + w, y + h),
                "center": (x + w // 2, y + h // 2),
                "confidence": confidence,
                "width": w,
                "height": h,
                "detector": "legacy",
                "text": ""
            }
            
            # Add context if provided
            if context:
                element["context"] = context
                
            ui_elements.append(element)
            
        logger.debug(f"Legacy detection found {len(ui_elements)} elements")
        return ui_elements
        
    def _blend_detection_results(self, ml_elements: List[Dict], legacy_elements: List[Dict]) -> List[Dict]:
        """
        Blend ML and legacy detection results for more robust detection
        
        Args:
            ml_elements: Elements detected by ML methods
            legacy_elements: Elements detected by legacy methods
            
        Returns:
            Combined list of elements
        """
        if not ml_elements:
            return legacy_elements
            
        if not legacy_elements:
            return ml_elements
            
        # Start with all ML elements with high confidence
        high_conf_threshold = 0.8
        high_conf_elements = [e for e in ml_elements if e.get("confidence", 0) >= high_conf_threshold]
        
        # Add legacy elements that don't overlap with high confidence ML elements
        for legacy_element in legacy_elements:
            # Check if this legacy element overlaps with any high confidence ML element
            overlaps = False
            for ml_element in high_conf_elements:
                iou = self._calculate_iou(legacy_element["bbox"], ml_element["bbox"])
                if iou > 0.3:  # 30% overlap threshold
                    overlaps = True
                    break
                    
            if not overlaps:
                # Add non-overlapping legacy element, but mark it as legacy
                legacy_element["detector"] = "legacy"
                legacy_element["confidence"] = legacy_element.get("confidence", 0.5)  # Default confidence
                high_conf_elements.append(legacy_element)
        
        # Add remaining ML elements with lower confidence if they add new information
        low_conf_elements = [e for e in ml_elements if e.get("confidence", 0) < high_conf_threshold]
        for ml_element in low_conf_elements:
            # Check if this ML element adds new information
            overlaps = False
            for element in high_conf_elements:
                iou = self._calculate_iou(ml_element["bbox"], element["bbox"])
                if iou > 0.3:  # 30% overlap threshold
                    overlaps = True
                    break
                    
            if not overlaps:
                high_conf_elements.append(ml_element)
        
        return high_conf_elements
        
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
        
    def _initialize_detectors(self):
        """
        Initialize ML-based UI element detectors
        """
        try:
            # Create detector registry
            self.detector_registry = DetectorRegistry()
            
            # Register available detectors
            detector_config = self.config.get("detector_config", {})
            
            # AutoGluon (highest priority)
            if AutoGluonDetector is not None:
                self.detector_registry.register_detector(
                    "autogluon", 
                    AutoGluonDetector, 
                    priority=100,
                    config=detector_config.get("autogluon")
                )
                
            # HuggingFace (medium priority)
            if HuggingFaceDetector is not None:
                self.detector_registry.register_detector(
                    "huggingface", 
                    HuggingFaceDetector, 
                    priority=80,
                    config=detector_config.get("huggingface")
                )
                
            # OpenCV (lowest priority)
            if OpenCVDetector is not None:
                self.detector_registry.register_detector(
                    "opencv", 
                    OpenCVDetector, 
                    priority=10,
                    config=detector_config.get("opencv")
                )
                
            # Get best available detector
            self.current_detector = self.detector_registry.get_detector()
            
            if self.current_detector:
                capabilities = self.current_detector.get_capabilities()
                logger.info(f"Using {capabilities.get('name')} detector for UI element detection")
                
                # Initialize confidence calibrator
                calibration_path = os.path.join(self.memory_path, "calibration.json")
                self.confidence_calibrator = ConfidenceCalibrator(calibration_path)
                
                # Initialize visualizer
                self.visualizer = DetectionVisualizer()
            else:
                logger.warning("No UI element detector available")
                
        except Exception as e:
            logger.error(f"Error initializing UI detectors: {e}")
            self.use_ml_detection = False
            
    def visualize_detection(self, screenshot: np.ndarray, elements: List[Dict], 
                           output_path: Optional[str] = None,
                           show_confidence: bool = True,
                           show_detector: bool = True) -> Optional[np.ndarray]:
        """
        Visualize UI element detection results
        
        Args:
            screenshot: Screenshot image
            elements: Detected UI elements
            output_path: Optional path to save visualization
            show_confidence: Whether to show confidence scores
            show_detector: Whether to show detector name
            
        Returns:
            Visualization image if visualizer is available, None otherwise
        """
        if not self.visualizer:
            logger.warning("Visualizer not available")
            return None
            
        # Create visualization
        vis_image = self.visualizer.visualize_detections(
            screenshot, elements,
            color_by="type",
            show_confidence=show_confidence,
            show_detector=show_detector
        )
        
        # Save if path provided
        if output_path and vis_image is not None:
            self.visualizer.save_visualization(vis_image, output_path)
            
        return vis_image
