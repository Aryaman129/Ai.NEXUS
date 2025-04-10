"""
Enhanced Vision Layer for NEXUS UI Intelligence

This module provides advanced computer vision capabilities for UI element detection
using multiple strategies, feature extraction, and machine learning techniques.
"""

import os
import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UIElement:
    """Represents a detected UI element with all its properties"""
    element_id: str
    element_type: str  
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    center: Tuple[int, int] = field(init=False)
    width: int = field(init=False)
    height: int = field(init=False)
    text: str = ""
    state: Dict[str, Any] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    detection_method: str = "unknown"
    
    def __post_init__(self):
        """Calculate derived properties after initialization"""
        x1, y1, x2, y2 = self.bbox
        self.width = x2 - x1
        self.height = y2 - y1
        self.center = (x1 + self.width // 2, y1 + self.height // 2)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "id": self.element_id,
            "type": self.element_type,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "center": self.center,
            "width": self.width,
            "height": self.height,
            "text": self.text,
            "state": self.state,
            "attributes": self.attributes,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "detection_method": self.detection_method
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UIElement':
        """Create from dictionary"""
        element = cls(
            element_id=data["id"],
            element_type=data["type"],
            bbox=data["bbox"],
            confidence=data["confidence"],
            detection_method=data.get("detection_method", "unknown")
        )
        if "text" in data:
            element.text = data["text"]
        if "state" in data:
            element.state = data["state"]
        if "attributes" in data:
            element.attributes = data["attributes"]
        if "parent_id" in data:
            element.parent_id = data["parent_id"]
        if "children_ids" in data:
            element.children_ids = data["children_ids"]
        return element


class EnhancedVisionDetector:
    """Advanced UI element detector using multiple computer vision techniques"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the enhanced vision detector with configuration"""
        self.config = config or {}
        self.enabled_detectors = self.config.get("enabled_detectors", [
            "contour", "color", "template", "feature"
        ])
        self.min_confidence = self.config.get("min_confidence", 0.5)
        self.element_counter = 0
        
        # Feature detection parameters
        self.feature_detector = cv2.SIFT_create() if hasattr(cv2, 'SIFT_create') else None
        if self.feature_detector is None:
            logger.warning("SIFT feature detector not available, falling back to ORB")
            self.feature_detector = cv2.ORB_create()
            
        # Template matching parameters
        self.template_dir = self.config.get("template_dir", os.path.join("data", "ui_templates"))
        self.templates = self._load_templates() if "template" in self.enabled_detectors else {}
        
        # Set up color profiles for common UI elements
        self.ui_color_profiles = {
            "button": {
                "light": [(200, 200, 200), (240, 240, 240)],  # Light gray buttons
                "dark": [(40, 40, 40), (80, 80, 80)],         # Dark gray buttons
                "accent": [(0, 120, 215), (0, 150, 255)]      # Blue accent buttons
            },
            "text_field": {
                "light": [(240, 240, 240), (255, 255, 255)],  # White text fields
                "dark": [(30, 30, 30), (50, 50, 50)]          # Dark text fields
            },
            "icon": {
                "standard": [(0, 0, 0), (50, 50, 50)],        # Dark icons
                "accent": [(0, 120, 215), (0, 150, 255)]      # Blue accent icons
            }
        }
        
        logger.info(f"EnhancedVisionDetector initialized with {len(self.enabled_detectors)} detectors")
    
    def _load_templates(self) -> Dict[str, List[Dict]]:
        """Load template images for template matching"""
        templates = {}
        if not os.path.exists(self.template_dir):
            os.makedirs(self.template_dir, exist_ok=True)
            logger.warning(f"Template directory {self.template_dir} created but empty")
            return templates
            
        # Process each template file
        for root, _, files in os.walk(self.template_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    # Parse filename: element_type_variant.png (e.g., button_standard.png)
                    parts = os.path.splitext(file)[0].split('_')
                    if len(parts) >= 2:
                        element_type = parts[0]
                        variant = '_'.join(parts[1:])
                        
                        # Load template
                        template_path = os.path.join(root, file)
                        template_img = cv2.imread(template_path)
                        
                        if template_img is not None:
                            if element_type not in templates:
                                templates[element_type] = []
                                
                            templates[element_type].append({
                                "image": template_img,
                                "variant": variant,
                                "path": template_path
                            })
                            logger.debug(f"Loaded template: {template_path}")
        
        logger.info(f"Loaded {sum(len(v) for v in templates.values())} templates for {len(templates)} element types")
        return templates
    
    def detect_elements(self, image: np.ndarray) -> List[UIElement]:
        """Detect UI elements using all enabled detection methods"""
        if image is None or image.size == 0:
            logger.error("Invalid image provided to detector")
            return []
        
        # Reset element counter for this detection session
        self.element_counter = 0
        
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            grayscale = image.copy()
            
        # Run all enabled detectors
        all_elements = []
        
        start_time = time.time()
        
        # 1. Contour-based detection
        if "contour" in self.enabled_detectors:
            contour_elements = self._detect_with_contours(grayscale, image)
            all_elements.extend(contour_elements)
            logger.debug(f"Contour detection found {len(contour_elements)} elements")
        
        # 2. Color-based detection
        if "color" in self.enabled_detectors:
            color_elements = self._detect_with_colors(image)
            all_elements.extend(color_elements)
            logger.debug(f"Color detection found {len(color_elements)} elements")
        
        # 3. Template matching
        if "template" in self.enabled_detectors and self.templates:
            template_elements = self._detect_with_templates(grayscale, image)
            all_elements.extend(template_elements)
            logger.debug(f"Template matching found {len(template_elements)} elements")
        
        # 4. Feature-based detection
        if "feature" in self.enabled_detectors:
            feature_elements = self._detect_with_features(grayscale, image)
            all_elements.extend(feature_elements)
            logger.debug(f"Feature detection found {len(feature_elements)} elements")
            
        # Post-processing: remove duplicates and weak detections
        filtered_elements = self._filter_elements(all_elements)
        
        # Establish parent-child relationships
        hierarchical_elements = self._build_element_hierarchy(filtered_elements)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Enhanced detection found {len(hierarchical_elements)} elements in {elapsed_time:.3f}s")
        
        return hierarchical_elements
    
    def _generate_element_id(self) -> str:
        """Generate a unique ID for a UI element"""
        self.element_counter += 1
        return f"ui_element_{self.element_counter}"
    
    def _detect_with_contours(self, grayscale: np.ndarray, original: np.ndarray) -> List[UIElement]:
        """Detect UI elements using contour analysis"""
        elements = []
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Apply edge detection
        edges = cv2.Canny(grayscale, 50, 150)
        
        # Combine binary and edges
        combined = cv2.bitwise_or(binary, edges)
        
        # Dilate to connect nearby edges
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(combined, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Filter by size
            if cv2.contourArea(contour) < 100:  # Minimum area threshold
                continue
                
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate aspect ratio
            aspect_ratio = w / h if h > 0 else 0
            
            # Calculate fill ratio (contour area / bounding box area)
            fill_ratio = cv2.contourArea(contour) / (w * h) if w * h > 0 else 0
            
            # Determine element type based on geometry
            element_type, confidence = self._classify_by_geometry(w, h, aspect_ratio, fill_ratio)
                
            # Create element
            element = UIElement(
                element_id=self._generate_element_id(),
                element_type=element_type,
                bbox=(x, y, x + w, y + h),
                confidence=confidence,
                detection_method="contour"
            )
            
            # Store additional attributes
            element.attributes["aspect_ratio"] = aspect_ratio
            element.attributes["fill_ratio"] = fill_ratio
            element.attributes["area"] = w * h
            
            elements.append(element)
            
        return elements
    
    def _detect_with_colors(self, image: np.ndarray) -> List[UIElement]:
        """Detect UI elements using color-based segmentation"""
        elements = []
        
        # Convert to HSV color space for better segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Detect elements for each UI type and color profile
        for element_type, color_profiles in self.ui_color_profiles.items():
            for profile_name, (lower_rgb, upper_rgb) in color_profiles.items():
                # Convert RGB to HSV ranges
                lower_hsv = cv2.cvtColor(np.uint8([[lower_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
                upper_hsv = cv2.cvtColor(np.uint8([[upper_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
                
                # Create mask for this color range
                mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
                
                # Apply morphological operations to clean up mask
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                
                # Find contours in the mask
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    # Filter small contours
                    if cv2.contourArea(contour) < 150:
                        continue
                        
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Base confidence on contour area and regularity
                    peri = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
                    regularity = 0.8 if len(approx) == 4 else 0.6  # Higher for rectangular
                    
                    confidence = min(0.9, 0.6 + 0.3 * min(1.0, cv2.contourArea(contour) / 5000))
                    confidence *= regularity
                    
                    # Create element
                    element = UIElement(
                        element_id=self._generate_element_id(),
                        element_type=element_type,
                        bbox=(x, y, x + w, y + h),
                        confidence=confidence,
                        detection_method=f"color_{profile_name}"
                    )
                    
                    # Store color profile as attribute
                    element.attributes["color_profile"] = profile_name
                    
                    elements.append(element)
        
        return elements
    
    def _detect_with_templates(self, grayscale: np.ndarray, original: np.ndarray) -> List[UIElement]:
        """Detect UI elements using template matching"""
        elements = []
        
        # Process each template
        for element_type, templates in self.templates.items():
            for template_data in templates:
                template_img = template_data["image"]
                
                # Convert template to grayscale
                if len(template_img.shape) == 3:
                    template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
                else:
                    template_gray = template_img
                
                # Get template dimensions
                h, w = template_gray.shape
                
                # Apply template matching
                method = cv2.TM_CCOEFF_NORMED
                res = cv2.matchTemplate(grayscale, template_gray, method)
                threshold = 0.7
                
                # Find locations above threshold
                locations = np.where(res >= threshold)
                
                for pt in zip(*locations[::-1]):
                    # Create element
                    confidence = float(res[pt[1], pt[0]])
                    
                    element = UIElement(
                        element_id=self._generate_element_id(),
                        element_type=element_type,
                        bbox=(pt[0], pt[1], pt[0] + w, pt[1] + h),
                        confidence=confidence,
                        detection_method="template"
                    )
                    
                    # Add template metadata
                    element.attributes["template_variant"] = template_data["variant"]
                    element.attributes["template_path"] = template_data["path"]
                    element.attributes["match_score"] = confidence
                    
                    elements.append(element)
        
        return elements
    
    def _detect_with_features(self, grayscale: np.ndarray, original: np.ndarray) -> List[UIElement]:
        """Detect UI elements using feature detection and analysis"""
        elements = []
        
        # Skip if no feature detector available
        if self.feature_detector is None:
            return elements
            
        # Detect keypoints and compute descriptors
        keypoints, _ = self.feature_detector.detectAndCompute(grayscale, None)
        
        if not keypoints:
            return elements
            
        # Cluster keypoints into potential UI elements
        clusters = self._cluster_keypoints(keypoints, grayscale.shape)
        
        for cluster in clusters:
            # Extract points
            points = np.array([kp.pt for kp in cluster])
            
            if len(points) < 3:
                continue
                
            # Find bounding box
            min_x = int(min(p[0] for p in points))
            min_y = int(min(p[1] for p in points))
            max_x = int(max(p[0] for p in points))
            max_y = int(max(p[1] for p in points))
            
            # Calculate width and height
            w = max_x - min_x
            h = max_y - min_y
            
            # Skip if too small
            if w < 15 or h < 15:
                continue
                
            # Calculate aspect ratio
            aspect_ratio = w / h if h > 0 else 0
            
            # Determine element type based on feature distribution
            element_type, confidence = self._classify_by_features(cluster, aspect_ratio, w, h)
            
            # Create element
            element = UIElement(
                element_id=self._generate_element_id(),
                element_type=element_type,
                bbox=(min_x, min_y, max_x, max_y),
                confidence=confidence,
                detection_method="feature"
            )
            
            element.attributes["keypoint_count"] = len(cluster)
            element.attributes["keypoint_density"] = len(cluster) / (w * h) if w * h > 0 else 0
            
            elements.append(element)
        
        return elements
    
    def _cluster_keypoints(self, keypoints, img_shape, dist_threshold=30):
        """Cluster keypoints into potential UI elements"""
        if not keypoints:
            return []
            
        # Create distance matrix
        n = len(keypoints)
        clusters = [[keypoints[0]]]
        
        # Simple clustering based on distance
        for i in range(1, n):
            kp = keypoints[i]
            added = False
            
            for cluster in clusters:
                # Check if this keypoint belongs to an existing cluster
                for existing_kp in cluster:
                    dist = np.sqrt((kp.pt[0] - existing_kp.pt[0])**2 + 
                                  (kp.pt[1] - existing_kp.pt[1])**2)
                    if dist < dist_threshold:
                        cluster.append(kp)
                        added = True
                        break
                        
                if added:
                    break
                    
            # If not added to any cluster, create a new one
            if not added:
                clusters.append([kp])
        
        # Filter out small clusters
        return [c for c in clusters if len(c) >= 3]
    
    def _classify_by_geometry(self, width, height, aspect_ratio, fill_ratio) -> Tuple[str, float]:
        """Classify UI element based on its geometry"""
        if 0.95 <= aspect_ratio <= 1.05:
            # Square elements
            if width < 30:
                return "icon", 0.75 if fill_ratio > 0.7 else 0.6
            else:
                return "button", 0.7 if fill_ratio > 0.8 else 0.6
                
        elif aspect_ratio > 3.0:
            # Wide elements
            if height < 30:
                return "menu_item", 0.65
            else:
                return "text_field", 0.7 if fill_ratio < 0.3 else 0.6
                
        elif aspect_ratio < 0.3:
            # Tall elements
            if width < 20:
                return "scrollbar", 0.7
            else:
                return "sidebar", 0.65
                
        else:
            # Default rectangles
            if width > 100 and height > 30:
                return "panel", 0.6
            elif width < 50 and height < 50:
                return "control", 0.65 if fill_ratio > 0.6 else 0.5
            else:
                return "button", 0.6
    
    def _classify_by_features(self, keypoints, aspect_ratio, width, height) -> Tuple[str, float]:
        """Classify UI element based on feature distribution"""
        # Calculate keypoint density
        num_keypoints = len(keypoints)
        density = num_keypoints / (width * height) if width * height > 0 else 0
        
        # Check if keypoints form a specific pattern
        points = np.array([kp.pt for kp in keypoints])
        x_std = np.std(points[:, 0])
        y_std = np.std(points[:, 1])
        
        # High x_std, low y_std suggests horizontal alignment (menu items)
        if x_std > 3 * y_std and width > 100:
            return "menu_item", 0.7
            
        # High y_std, low x_std suggests vertical alignment (sidebar)
        if y_std > 3 * x_std and height > 100:
            return "sidebar", 0.7
            
        # High density suggests complex UI (dropdown, combo box)
        if density > 0.01:
            if aspect_ratio > 3.0:
                return "dropdown", 0.75
            else:
                return "complex_control", 0.7
                
        # Default classification based on size and aspect ratio
        if width < 40 and height < 40:
            return "icon", 0.6
        elif aspect_ratio > 2.0:
            return "text_field", 0.65
        else:
            return "button", 0.6
    
    def _filter_elements(self, elements: List[UIElement]) -> List[UIElement]:
        """Filter out overlapping and low-confidence elements"""
        if not elements:
            return []
            
        # Filter by minimum confidence
        elements = [e for e in elements if e.confidence >= self.min_confidence]
        
        # Sort by confidence (highest first)
        elements.sort(key=lambda e: e.confidence, reverse=True)
        
        # Filter overlapping elements
        filtered = []
        for element in elements:
            should_keep = True
            
            for kept in filtered:
                iou = self._calculate_iou(element.bbox, kept.bbox)
                
                # If high overlap and same element type, keep the higher confidence one
                if iou > 0.7:
                    if element.element_type == kept.element_type:
                        should_keep = False
                        break
                        
                    # If different types but very high overlap, favor the higher confidence
                    elif iou > 0.9:
                        should_keep = False
                        break
                    
            if should_keep:
                filtered.append(element)
                
        return filtered
    
    def _build_element_hierarchy(self, elements: List[UIElement]) -> List[UIElement]:
        """Establish parent-child relationships between elements"""
        # Sort by area (largest first) for proper nesting
        elements.sort(key=lambda e: e.width * e.height, reverse=True)
        
        # Find parent-child relationships
        for i, potential_parent in enumerate(elements):
            parent_bbox = potential_parent.bbox
            
            for j, potential_child in enumerate(elements):
                if i == j:  # Skip self
                    continue
                    
                # Check if potential_child is inside potential_parent
                if self._is_contained(potential_child.bbox, parent_bbox):
                    # Set parent-child relationship
                    potential_child.parent_id = potential_parent.element_id
                    if potential_child.element_id not in potential_parent.children_ids:
                        potential_parent.children_ids.append(potential_child.element_id)
        
        return elements
    
    def _is_contained(self, child_bbox, parent_bbox) -> bool:
        """Check if child bounding box is contained within parent"""
        c_x1, c_y1, c_x2, c_y2 = child_bbox
        p_x1, p_y1, p_x2, p_y2 = parent_bbox
        
        # Child must be completely inside parent with some margin
        margin = 2  # Pixel margin for slight overlap cases
        return (p_x1 - margin <= c_x1 and c_x2 <= p_x2 + margin and 
                p_y1 - margin <= c_y1 and c_y2 <= p_y2 + margin)
    
    def _calculate_iou(self, bbox1, bbox2) -> float:
        """Calculate Intersection over Union for two bounding boxes"""
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
