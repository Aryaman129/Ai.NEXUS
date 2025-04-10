"""
Adaptive UI Element Detector

This module provides an implementation of the UIDetectorInterface that adapts
the standalone UI intelligence system for use with the enhanced detector registry.
"""

import time
import logging
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# Import the detector interface
from ..ui_detection.detector_interface import UIDetectorInterface

logger = logging.getLogger(__name__)

class AdaptiveDetector(UIDetectorInterface):
    """
    Adaptive UI element detector that combines multiple detection strategies
    and provides semantic understanding of detected elements.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the adaptive detector
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.initialized = False
        
        # Detection parameters
        self.edge_threshold = self.config.get("edge_threshold", 100)
        self.min_area = self.config.get("min_area", 100)
        self.max_area = self.config.get("max_area", 100000)
        self.color_threshold = self.config.get("color_threshold", 30)
        
        # OCR integration
        self.use_ocr = self.config.get("use_ocr", True)
        self.ocr_engine = None
        
        # LLM integration for semantic understanding
        self.use_llm = self.config.get("use_llm", True)
        self.llm_client = None
        
        # Detection cache for performance
        self.detection_cache = {}
        
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
        
        try:
            # Initialize OCR if enabled
            if self.use_ocr:
                self._initialize_ocr()
            
            # Initialize LLM if enabled
            if self.use_llm:
                self._initialize_llm()
            
            self.initialized = True
            logger.info("Adaptive detector initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing adaptive detector: {e}")
            return False
    
    def _initialize_ocr(self):
        """
        Initialize OCR engine
        """
        try:
            import pytesseract
            self.ocr_engine = pytesseract
            logger.info("OCR engine initialized")
        except ImportError:
            logger.warning("Tesseract OCR not available. Install with: pip install pytesseract")
            self.use_ocr = False
    
    def _initialize_llm(self):
        """
        Initialize LLM client for semantic understanding
        """
        try:
            import together
            
            # Set API key if provided
            api_key = self.config.get("together_api_key")
            if api_key:
                together.api_key = api_key
            
            self.llm_client = together
            logger.info("LLM client initialized")
        except ImportError:
            logger.warning("Together AI SDK not available. Install with: pip install together")
            self.use_llm = False
            
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get detector capabilities
        
        Returns:
            Dictionary of capabilities
        """
        return {
            "name": "adaptive_detector",
            "version": "1.0.0",
            "supports_ocr": self.use_ocr,
            "supports_semantic": self.use_llm,
            "detection_strategies": ["edge", "color", "feature"],
            "element_types": [
                "button", "text_field", "checkbox", "dropdown", "radio_button",
                "toggle", "slider", "icon", "menu", "toolbar", "dialog"
            ]
        }
        
    def detect_elements(self, screenshot: np.ndarray, context: Optional[Dict] = None) -> List[Dict]:
        """
        Detect UI elements in the screenshot
        
        Args:
            screenshot: Screenshot as numpy array
            context: Optional context information
            
        Returns:
            List of detected elements
        """
        if not self.initialized:
            logger.warning("Adaptive detector not initialized")
            return []
        
        # Preprocessing
        if len(screenshot.shape) == 3 and screenshot.shape[2] == 3:
            # Convert to grayscale for edge detection
            grayscale = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        else:
            grayscale = screenshot
        
        # Apply different detection strategies
        elements = []
        
        # Edge-based detection
        edge_elements = self._detect_with_edges(grayscale, screenshot)
        elements.extend(edge_elements)
        
        # Color-based detection
        color_elements = self._detect_with_color(screenshot)
        elements.extend(color_elements)
        
        # Feature-based detection (if available)
        if self.config.get("use_feature_detection", True):
            feature_elements = self._detect_with_features(screenshot)
            elements.extend(feature_elements)
        
        # Filter overlapping elements
        filtered_elements = self._filter_overlapping(elements)
        
        # Extract text with OCR if enabled
        if self.use_ocr:
            filtered_elements = self._extract_text(screenshot, filtered_elements)
        
        # Add semantic understanding if enabled
        if self.use_llm and context:
            filtered_elements = self._add_semantic_understanding(filtered_elements, context)
        
        # Sort by confidence
        filtered_elements.sort(key=lambda e: e.get("confidence", 0), reverse=True)
        
        return filtered_elements
    
    def _detect_with_edges(self, grayscale: np.ndarray, original: np.ndarray) -> List[Dict]:
        """
        Detect UI elements using edge detection
        
        Args:
            grayscale: Grayscale image
            original: Original color image
            
        Returns:
            List of detected elements
        """
        # Apply Canny edge detection
        edges = cv2.Canny(grayscale, self.edge_threshold, self.edge_threshold * 2)
        
        # Dilate edges to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process contours into elements
        elements = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area <= area <= self.max_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Skip elements that are too small or too large
                if w < 10 or h < 10 or w > original.shape[1] * 0.9 or h > original.shape[0] * 0.9:
                    continue
                
                # Calculate confidence based on contour properties
                # Higher confidence for more rectangular shapes
                rect_area = w * h
                rect_ratio = area / rect_area if rect_area > 0 else 0
                aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0
                confidence = rect_ratio * 0.5 + aspect_ratio * 0.3 + 0.2
                
                # Determine element type based on shape and size
                element_type = self._determine_element_type(w, h, rect_ratio, aspect_ratio)
                
                elements.append({
                    "type": element_type,
                    "bbox": (x, y, x + w, y + h),
                    "center": (x + w // 2, y + h // 2),
                    "confidence": confidence,
                    "detection_method": "edge",
                    "area": area,
                    "rect_ratio": rect_ratio,
                    "aspect_ratio": aspect_ratio
                })
        
        return elements
    
    def _detect_with_color(self, image: np.ndarray) -> List[Dict]:
        """
        Detect UI elements using color segmentation
        
        Args:
            image: Color image
            
        Returns:
            List of detected elements
        """
        # Convert to HSV for better color separation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Extract dominant colors
        pixels = hsv.reshape(-1, 3)
        pixels = np.float32(pixels)
        
        # Use K-means to find dominant colors
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = 8  # Number of colors to extract
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Create mask for each dominant color
        elements = []
        for i in range(k):
            # Create mask with same dimensions as the flattened pixels array
            mask = np.zeros(len(labels), dtype=np.uint8)
            mask[labels.flatten() == i] = 255
            # Reshape mask to the original image dimensions
            mask = mask.reshape(image.shape[0], image.shape[1])
            
            # Apply morphological operations to clean up mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours in mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if self.min_area <= area <= self.max_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Skip elements that are too small or too large
                    if w < 10 or h < 10 or w > image.shape[1] * 0.9 or h > image.shape[0] * 0.9:
                        continue
                    
                    # Calculate confidence based on color uniformity
                    roi = image[y:y+h, x:x+w]
                    hsv_roi = hsv[y:y+h, x:x+w]
                    color_std = np.std(hsv_roi, axis=(0, 1))
                    color_uniformity = 1.0 - min(1.0, np.mean(color_std) / 50.0)
                    
                    confidence = color_uniformity * 0.7 + 0.3
                    
                    # Determine element type
                    rect_area = w * h
                    rect_ratio = area / rect_area if rect_area > 0 else 0
                    aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0
                    element_type = self._determine_element_type(w, h, rect_ratio, aspect_ratio)
                    
                    elements.append({
                        "type": element_type,
                        "bbox": (x, y, x + w, y + h),
                        "center": (x + w // 2, y + h // 2),
                        "confidence": confidence,
                        "detection_method": "color",
                        "area": area,
                        "color_uniformity": color_uniformity,
                        "dominant_color": centers[i].tolist()
                    })
        
        return elements
    
    def _detect_with_features(self, image: np.ndarray) -> List[Dict]:
        """
        Detect UI elements using feature detection
        
        Args:
            image: Color image
            
        Returns:
            List of detected elements
        """
        # This is a placeholder for more advanced feature detection
        # In a production environment, this could use a pre-trained model
        # for specific UI element detection
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Use SIFT or ORB for feature detection
        try:
            # Try SIFT first
            sift = cv2.SIFT_create()
            keypoints = sift.detect(gray, None)
        except:
            try:
                # Fall back to ORB if SIFT not available
                orb = cv2.ORB_create()
                keypoints = orb.detect(gray, None)
            except:
                # Return empty if neither is available
                logger.warning("SIFT and ORB detectors not available")
                return []
        
        # Group nearby keypoints to form potential UI elements
        if not keypoints:
            return []
            
        # Create clusters of keypoints
        points = np.array([kp.pt for kp in keypoints])
        points = np.float32(points)
        
        # Use DBSCAN for clustering
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=30, min_samples=3).fit(points)
        
        # Process clusters into elements
        elements = []
        for cluster_id in set(clustering.labels_):
            if cluster_id == -1:  # Skip noise
                continue
                
            # Get points in this cluster
            cluster_points = points[clustering.labels_ == cluster_id]
            
            # Calculate bounding box
            min_x, min_y = np.min(cluster_points, axis=0).astype(int)
            max_x, max_y = np.max(cluster_points, axis=0).astype(int)
            
            # Add some padding
            min_x = max(0, min_x - 5)
            min_y = max(0, min_y - 5)
            max_x = min(image.shape[1], max_x + 5)
            max_y = min(image.shape[0], max_y + 5)
            
            w, h = max_x - min_x, max_y - min_y
            
            # Skip if too small
            if w < 10 or h < 10:
                continue
                
            # Calculate confidence based on number of keypoints
            num_keypoints = len(cluster_points)
            confidence = min(0.9, 0.3 + 0.1 * num_keypoints)
            
            # Determine element type
            rect_ratio = 0.8  # Assume relatively rectangular
            aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0
            element_type = self._determine_element_type(w, h, rect_ratio, aspect_ratio)
            
            elements.append({
                "type": element_type,
                "bbox": (min_x, min_y, max_x, max_y),
                "center": (min_x + w // 2, min_y + h // 2),
                "confidence": confidence,
                "detection_method": "feature",
                "keypoints": num_keypoints
            })
        
        return elements
    
    def _determine_element_type(self, width: int, height: int, rect_ratio: float, aspect_ratio: float) -> str:
        """
        Determine UI element type based on shape and size
        
        Args:
            width: Element width
            height: Element height
            rect_ratio: Rectangularity ratio (area / bounding rect area)
            aspect_ratio: Aspect ratio (min_dim / max_dim)
            
        Returns:
            Element type string
        """
        # Very horizontal elements
        if aspect_ratio < 0.2:
            if width > 200:
                return "toolbar"
            else:
                return "slider"
                
        # Square-ish elements
        if aspect_ratio > 0.8:
            if width < 50 and height < 50:
                if rect_ratio > 0.8:
                    return "checkbox"
                else:
                    return "icon"
            else:
                return "button"
                
        # More horizontal elements
        if width > height:
            if width > 150:
                return "text_field"
            else:
                return "button"
                
        # More vertical elements
        if rect_ratio > 0.8:
            if width < 100:
                return "radio_button"
            else:
                return "menu"
        else:
            return "dropdown"
            
    def _filter_overlapping(self, elements: List[Dict]) -> List[Dict]:
        """
        Filter overlapping elements and establish parent-child relationships
        
        Args:
            elements: List of detected elements
            
        Returns:
            Filtered list of elements
        """
        if not elements:
            return []
            
        # Sort by area (descending)
        elements.sort(key=lambda e: (e["bbox"][2] - e["bbox"][0]) * (e["bbox"][3] - e["bbox"][1]), reverse=True)
        
        # Filter elements with high overlap and same type
        filtered = []
        for i, e1 in enumerate(elements):
            keep = True
            e1_area = (e1["bbox"][2] - e1["bbox"][0]) * (e1["bbox"][3] - e1["bbox"][1])
            
            # Check for children (smaller elements contained within this one)
            children = []
            
            for j, e2 in enumerate(elements):
                if i == j:
                    continue
                    
                # Calculate overlap
                x1 = max(e1["bbox"][0], e2["bbox"][0])
                y1 = max(e1["bbox"][1], e2["bbox"][1])
                x2 = min(e1["bbox"][2], e2["bbox"][2])
                y2 = min(e1["bbox"][3], e2["bbox"][3])
                
                if x1 < x2 and y1 < y2:
                    overlap_area = (x2 - x1) * (y2 - y1)
                    e2_area = (e2["bbox"][2] - e2["bbox"][0]) * (e2["bbox"][3] - e2["bbox"][1])
                    
                    # Check if e2 is mostly contained within e1
                    e2_overlap_ratio = overlap_area / e2_area if e2_area > 0 else 0
                    
                    # Check if e1 and e2 are the same element (high overlap and same type)
                    if e2_overlap_ratio > 0.8 and e1["type"] == e2["type"] and e2_area > 0.8 * e1_area:
                        # Keep the one with higher confidence
                        if e2["confidence"] > e1["confidence"]:
                            keep = False
                            break
                    
                    # Check if e2 is a child of e1
                    if e2_overlap_ratio > 0.9 and e2_area < 0.8 * e1_area:
                        children.append(j)
            
            if keep:
                # Add children information
                if children:
                    e1["children"] = children
                filtered.append(e1)
        
        return filtered
    
    def _extract_text(self, image: np.ndarray, elements: List[Dict]) -> List[Dict]:
        """
        Extract text from UI elements using OCR
        
        Args:
            image: Original image
            elements: List of detected elements
            
        Returns:
            Elements with extracted text
        """
        if not self.use_ocr or not self.ocr_engine:
            return elements
            
        for element in elements:
            try:
                # Extract region of interest
                x1, y1, x2, y2 = element["bbox"]
                roi = image[y1:y2, x1:x2]
                
                # Skip small regions
                if roi.shape[0] < 10 or roi.shape[1] < 10:
                    continue
                
                # Apply OCR
                text = self.ocr_engine.image_to_string(roi).strip()
                
                if text:
                    element["text"] = text
                    
                    # Adjust element type based on text content
                    if element["type"] == "button" and len(text) > 20:
                        element["type"] = "text_field"
                    elif element["type"] == "text_field" and len(text) < 10 and text.lower() in ["ok", "cancel", "submit", "save"]:
                        element["type"] = "button"
            except Exception as e:
                logger.warning(f"OCR extraction failed: {e}")
        
        return elements
    
    def _add_semantic_understanding(self, elements: List[Dict], context: Dict) -> List[Dict]:
        """
        Add semantic understanding to UI elements using LLM
        
        Args:
            elements: List of detected elements
            context: Context information
            
        Returns:
            Elements with semantic understanding
        """
        if not self.use_llm or not self.llm_client:
            return elements
            
        # Only process if we have enough elements
        if len(elements) < 3:
            return elements
            
        try:
            # Prepare element descriptions for LLM
            element_descriptions = []
            for i, elem in enumerate(elements[:10]):  # Limit to top 10 elements
                desc = f"Element {i+1}: Type={elem['type']}, "
                if "text" in elem:
                    desc += f"Text='{elem['text']}', "
                desc += f"Position={elem['center']}"
                element_descriptions.append(desc)
                
            # Prepare prompt for LLM
            app_name = context.get("app_name", "unknown application")
            prompt = f"""Analyze these UI elements from {app_name}:\n\n"""
            prompt += "\n".join(element_descriptions)
            prompt += "\n\nFor each element, provide: 1) Purpose, 2) State, 3) Hierarchy. Keep it concise."
            
            # Call LLM
            response = self.llm_client.Complete.create(
                prompt=prompt,
                model="togethercomputer/llama-2-7b",
                max_tokens=500,
                temperature=0.2
            )
            
            # Process response
            analysis = response.output.text
            
            # Parse analysis into element-specific insights
            insights = self._parse_llm_analysis(analysis, len(element_descriptions))
            
            # Add insights to elements
            for i, elem in enumerate(elements[:10]):
                if i < len(insights):
                    elem["semantic"] = insights[i]
            
        except Exception as e:
            logger.warning(f"Semantic analysis failed: {e}")
            
        return elements
    
    def _parse_llm_analysis(self, analysis: str, num_elements: int) -> List[Dict]:
        """
        Parse LLM analysis into structured insights
        
        Args:
            analysis: LLM analysis text
            num_elements: Number of elements
            
        Returns:
            List of semantic insights
        """
        insights = []
        
        # Simple parsing based on "Element X:" pattern
        import re
        elements = re.split(r"Element \d+:", analysis)
        
        # Skip first split (usually empty)
        for i, element_text in enumerate(elements[1:]):
            if i >= num_elements:
                break
                
            lines = element_text.strip().split("\n")
            insight = {}
            
            for line in lines:
                if line.startswith("Purpose:") or "purpose:" in line.lower():
                    insight["purpose"] = line.split(":", 1)[1].strip()
                elif line.startswith("State:") or "state:" in line.lower():
                    insight["state"] = line.split(":", 1)[1].strip()
                elif line.startswith("Hierarchy:") or "hierarchy:" in line.lower():
                    insight["hierarchy"] = line.split(":", 1)[1].strip()
                    
            insights.append(insight)
            
        return insights
    
    def supports_incremental_learning(self) -> bool:
        """
        Whether this detector supports incremental learning
        
        Returns:
            True if incremental learning is supported, False otherwise
        """
        return True
        
    def learn_from_feedback(self, elements: List[Dict], feedback: Dict):
        """
        Update detector based on feedback
        
        Args:
            elements: Detected elements
            feedback: Feedback information with corrections
        """
        if not feedback:
            return
            
        # Extract corrections
        corrections = feedback.get("corrections", [])
        if not corrections:
            return
            
        # Update detection parameters based on feedback
        false_positives = sum(1 for c in corrections if c.get("action") == "remove")
        false_negatives = sum(1 for c in corrections if c.get("action") == "add")
        type_corrections = sum(1 for c in corrections if c.get("action") == "change_type")
        
        # Adapt edge threshold
        if false_positives > false_negatives and self.edge_threshold < 200:
            self.edge_threshold += 5
            logger.info(f"Increased edge threshold to {self.edge_threshold}")
        elif false_negatives > false_positives and self.edge_threshold > 50:
            self.edge_threshold -= 5
            logger.info(f"Decreased edge threshold to {self.edge_threshold}")
            
        # Adapt color threshold
        if false_positives > 2 and self.color_threshold < 50:
            self.color_threshold += 5
            logger.info(f"Increased color threshold to {self.color_threshold}")
        elif false_negatives > 2 and self.color_threshold > 10:
            self.color_threshold -= 5
            logger.info(f"Decreased color threshold to {self.color_threshold}")
            
        # Update configuration
        self.config["edge_threshold"] = self.edge_threshold
        self.config["color_threshold"] = self.color_threshold
