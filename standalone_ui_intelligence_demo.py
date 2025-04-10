"""
NEXUS Adaptive UI Intelligence Demonstration

This standalone script demonstrates the advanced UI detection and understanding
capabilities without requiring the full NEXUS package structure.

Key features demonstrated:
1. Multi-strategy UI element detection
2. Semantic understanding of elements
3. Adaptive confidence calibration
4. Hierarchical element relationships
"""

import os
import cv2
import numpy as np
import time
import logging
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import base64
import threading
import queue
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create output directories
os.makedirs("ui_intelligence_results", exist_ok=True)
os.makedirs("test_data", exist_ok=True)

# =====================================================================
# Data Structures
# =====================================================================

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

@dataclass
class UIAnalysisResult:
    """Complete result of UI analysis"""
    elements: List[UIElement]
    structure: Dict[str, Any] = field(default_factory=dict) 
    application_type: str = ""
    confidence: float = 0.0
    processing_time: float = 0.0
    analyzer_version: str = "0.1.0"

# =====================================================================
# Core Detection Components
# =====================================================================

class AdaptiveDetector:
    """
    Advanced UI element detector using multiple strategies.
    
    This detector combines edge detection, color analysis, and feature detection
    to identify UI elements with high accuracy.
    """
    
    def __init__(self):
        # Counter for generating element IDs
        self.element_counter = 0
        
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
        
        logger.info("AdaptiveDetector initialized")
    
    def detect_elements(self, image: np.ndarray) -> List[UIElement]:
        """
        Detect UI elements using multiple strategies.
        
        Args:
            image: Screenshot to analyze
            
        Returns:
            List of detected UI elements
        """
        if image is None or image.size == 0:
            logger.error("Invalid image provided")
            return []
        
        # Reset element counter
        self.element_counter = 0
        
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            grayscale = image.copy()
        
        # Apply different detection strategies
        all_elements = []
        
        # 1. Edge-based detection
        edge_elements = self._detect_with_edges(grayscale, image)
        all_elements.extend(edge_elements)
        
        # 2. Color-based detection
        color_elements = self._detect_with_colors(image)
        all_elements.extend(color_elements)
        
        # Filter overlapping elements
        filtered_elements = self._filter_elements(all_elements)
        
        # Establish parent-child relationships
        hierarchical_elements = self._build_element_hierarchy(filtered_elements)
        
        logger.info(f"Detected {len(hierarchical_elements)} UI elements")
        return hierarchical_elements
    
    def _generate_element_id(self) -> str:
        """Generate a unique ID for an element"""
        self.element_counter += 1
        return f"ui_element_{self.element_counter}"
    
    def _detect_with_edges(self, grayscale: np.ndarray, original: np.ndarray) -> List[UIElement]:
        """Detect UI elements using edge detection"""
        elements = []
        
        # Apply Canny edge detection
        edges = cv2.Canny(grayscale, 50, 150)
        
        # Dilate to connect broken lines
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Filter small contours
            if cv2.contourArea(contour) < 100:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate aspect ratio
            aspect_ratio = w / h if h > 0 else 0
            
            # Calculate fill ratio
            contour_area = cv2.contourArea(contour)
            bbox_area = w * h
            fill_ratio = contour_area / bbox_area if bbox_area > 0 else 0
            
            # Classify element type based on geometry
            element_type, confidence = self._classify_by_geometry(w, h, aspect_ratio, fill_ratio)
            
            # Create element
            element = UIElement(
                element_id=self._generate_element_id(),
                element_type=element_type,
                bbox=(x, y, x + w, y + h),
                confidence=confidence,
                detection_method="edge"
            )
            
            elements.append(element)
        
        return elements
    
    def _detect_with_colors(self, image: np.ndarray) -> List[UIElement]:
        """Detect UI elements using color analysis"""
        elements = []
        
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Process each color profile
        for element_type, profiles in self.ui_color_profiles.items():
            for profile_name, (lower_rgb, upper_rgb) in profiles.items():
                # Convert RGB to HSV
                lower_hsv = cv2.cvtColor(np.uint8([[lower_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
                upper_hsv = cv2.cvtColor(np.uint8([[upper_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
                
                # Create mask
                mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
                
                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    # Filter small contours
                    if cv2.contourArea(contour) < 150:
                        continue
                    
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate confidence
                    peri = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
                    regularity = 0.8 if len(approx) == 4 else 0.6
                    
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
                    
                    elements.append(element)
        
        return elements
    
    def _classify_by_geometry(self, width, height, aspect_ratio, fill_ratio) -> Tuple[str, float]:
        """Classify UI element based on geometry"""
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
    
    def _filter_elements(self, elements: List[UIElement]) -> List[UIElement]:
        """Filter overlapping elements"""
        if not elements:
            return []
        
        # Sort by confidence
        elements.sort(key=lambda e: e.confidence, reverse=True)
        
        # Filter overlapping elements
        filtered = []
        for element in elements:
            should_keep = True
            
            for kept in filtered:
                iou = self._calculate_iou(element.bbox, kept.bbox)
                
                if iou > 0.7:
                    if element.element_type == kept.element_type:
                        should_keep = False
                        break
                    
                    elif iou > 0.9:
                        should_keep = False
                        break
            
            if should_keep:
                filtered.append(element)
        
        return filtered
    
    def _build_element_hierarchy(self, elements: List[UIElement]) -> List[UIElement]:
        """Establish parent-child relationships"""
        # Sort by area (largest first)
        elements.sort(key=lambda e: e.width * e.height, reverse=True)
        
        # Find containment relationships
        for i, parent in enumerate(elements):
            parent_bbox = parent.bbox
            
            for j, child in enumerate(elements):
                if i == j:
                    continue
                
                if self._is_contained(child.bbox, parent_bbox):
                    child.parent_id = parent.element_id
                    if child.element_id not in parent.children_ids:
                        parent.children_ids.append(child.element_id)
        
        return elements
    
    def _is_contained(self, child_bbox, parent_bbox) -> bool:
        """Check if child is contained within parent"""
        c_x1, c_y1, c_x2, c_y2 = child_bbox
        p_x1, p_y1, p_x2, p_y2 = parent_bbox
        
        # Allow for some margin
        margin = 2
        return (p_x1 - margin <= c_x1 and c_x2 <= p_x2 + margin and 
                p_y1 - margin <= c_y1 and c_y2 <= p_y2 + margin)
    
    def _calculate_iou(self, bbox1, bbox2) -> float:
        """Calculate Intersection over Union"""
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


# =====================================================================
# Semantic Understanding Components
# =====================================================================

class SemanticAnalyzer:
    """
    Provides semantic understanding of UI elements.
    
    This component enhances detected UI elements with semantic information
    about their purpose, state, and relationships.
    """
    
    def __init__(self):
        # Initialize OCR capabilities using the adapter
        try:
            # First try to import our new adapter
            from nexus_ocr_adapter import ocr
            self.ocr = ocr
            self.ocr_available = ocr.ocr_available
            self.pytesseract = None
            
            if self.ocr_available:
                # Also import pytesseract for direct access if needed
                import pytesseract
                self.pytesseract = pytesseract
                tesseract_path = ocr.tesseract_path
                logger.info(f"OCR capabilities available through NEXUS adapter at {tesseract_path}")
            else:
                logger.warning("OCR adapter initialized but Tesseract not found or not working")
                
        except ImportError:
            # Fallback to the original method if adapter is not available
            logger.warning("NEXUS OCR adapter not available, falling back to direct pytesseract")
            self.ocr_available = False
            try:
                import pytesseract
                self.pytesseract = pytesseract
                
                # Check OCR_TESSERACT environment variable first
                env_tesseract = os.environ.get("OCR_TESSERACT")
                if env_tesseract:
                    if os.path.exists(env_tesseract):
                        # Direct path was provided
                        if env_tesseract.lower().endswith("tesseract.exe"):
                            pytesseract.pytesseract.tesseract_cmd = env_tesseract
                        # Directory was provided
                        else:
                            tesseract_path = os.path.join(env_tesseract, "tesseract.exe")
                            if os.path.exists(tesseract_path):
                                pytesseract.pytesseract.tesseract_cmd = tesseract_path
                
                try:
                    # Test if tesseract is available
                    pytesseract.get_tesseract_version()
                    self.ocr_available = True
                    logger.info(f"OCR capabilities available at {pytesseract.pytesseract.tesseract_cmd}")
                except Exception as e:
                    logger.warning(f"Tesseract installation found but not working: {e}")
                    # Try common locations
                    for path in [
                        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                        os.path.expanduser("~\AppData\Local\Tesseract-OCR\tesseract.exe")
                    ]:
                        if os.path.exists(path):
                            pytesseract.pytesseract.tesseract_cmd = path
                            try:
                                pytesseract.get_tesseract_version()
                                self.ocr_available = True
                                logger.info(f"OCR capabilities available at {path}")
                                break
                            except:
                                pass
            except ImportError:
                logger.warning("Pytesseract not available. Install for OCR capabilities.")
            
            # Create a simplified adapter interface for backwards compatibility
            class SimpleOCRAdapter:
                def __init__(self, pytesseract_instance=None):
                    self.ocr_available = pytesseract_instance is not None
                    self.pytesseract = pytesseract_instance
                    
                def extract_text(self, image):
                    if not self.ocr_available:
                        return ""
                    try:
                        return self.pytesseract.image_to_string(image).strip()
                    except Exception as e:
                        logger.error(f"OCR error: {e}")
                        return ""
                        
                def extract_text_with_boxes(self, image):
                    if not self.ocr_available:
                        return []
                    try:
                        boxes = self.pytesseract.image_to_data(image, output_type=self.pytesseract.Output.DICT)
                        
                        results = []
                        for i in range(len(boxes['text'])):
                            if boxes['text'][i].strip():
                                results.append({
                                    'text': boxes['text'][i],
                                    'x': boxes['left'][i],
                                    'y': boxes['top'][i],
                                    'width': boxes['width'][i],
                                    'height': boxes['height'][i],
                                    'conf': boxes['conf'][i]
                                })
                        return results
                    except Exception as e:
                        logger.error(f"OCR box extraction error: {e}")
                        return []
            
            # Create the simple adapter instance
            self.ocr = SimpleOCRAdapter(self.pytesseract if self.ocr_available else None)
        
        # Initialize adaptive LLM capabilities using multi-provider connector system
        self.llm_available = False
        try:
            # First try to import our new LLM connector system
            from nexus_llm_connector import llm, AdaptiveLLMManager
            self.llm_manager = llm
            
            # Check if any providers are available
            status = self.llm_manager.get_status()
            available_providers = status['available_connectors']
            
            if available_providers > 0:
                self.llm_available = True
                logger.info(f"LLM capabilities available through {available_providers} providers")
                for connector in status['connectors']:
                    if connector['available']:
                        logger.info(f"  - {connector['name']} is available")
            else:
                logger.warning("No LLM providers available. Semantic understanding will be limited.")
                
        except ImportError:
            # Fallback to original Together AI integration if adapter is not available
            logger.warning("NEXUS LLM connector system not available, falling back to direct Together AI")
            try:
                import requests
                self.requests = requests
                
                # API key from memory or environment
                self.together_api_key = os.environ.get("TOGETHER_API_KEY") or "4ec34405c082ae11d558aabe290486bd73ae6497fb623ba0bba481df21f5ec39"
                self.together_api_key = self.together_api_key.strip() if self.together_api_key else ""
                self.llm_available = bool(self.together_api_key)
                
                # Use Mixtral model which doesn't require special permissions
                self.together_model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
                
                if self.llm_available:
                    logger.info("LLM capabilities available via Together AI fallback")
                else:
                    logger.warning("No API key for Together AI. LLM capabilities disabled.")
            except ImportError:
                logger.warning("Requests library not available. LLM capabilities disabled.")
    
    def enhance_elements(self, image: np.ndarray, elements: List[UIElement]) -> List[UIElement]:
        """
        Enhance UI elements with semantic understanding.
        
        Args:
            image: Screenshot containing UI elements
            elements: Detected UI elements
            
        Returns:
            Enhanced elements with semantic information
        """
        if not elements:
            return elements
        
        # Extract text where possible
        if self.ocr_available:
            elements = self._extract_text(image, elements)
        
        # Add semantic understanding via LLM if available
        if self.llm_available:
            elements = self._add_semantic_understanding(image, elements)
        else:
            # Fallback: use heuristic understanding
            elements = self._add_heuristic_understanding(elements)
        
        logger.info(f"Enhanced {len(elements)} elements with semantic information")
        return elements
    
    def _extract_text(self, image: np.ndarray, elements: List[UIElement]) -> List[UIElement]:
        """Extract text from UI elements using OCR with adaptive strategies"""
        if not self.ocr_available:
            # Use element type to set placeholder text when OCR not available
            for element in elements:
                element_type = element.element_type
                if element_type in ["text_field", "label", "button"]:
                    element.text = f"[{element_type.upper()}]"
            logger.warning("OCR not available - using element type as placeholder text")
            return elements
            
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Text-containing element types
        text_element_types = [
            "button", "text_field", "label", "menu_item", 
            "link", "checkbox", "radio_button"
        ]
        
        elements_with_text = 0
        
        for element in elements:
            if element.element_type in text_element_types:
                # Extract element region
                x1, y1, x2, y2 = element.bbox
                roi = gray[y1:y2, x1:x2]
                
                # Skip if element is too small
                if roi.shape[0] < 5 or roi.shape[1] < 5:
                    continue
                
                try:
                    # Try different PSM modes for better accuracy
                    psm_modes = [7, 6, 3]  # Single line, Single block, Auto
                    best_text = ""
                    
                    for psm in psm_modes:
                        try:
                            # Apply OCR with specific PSM mode
                            text = self.pytesseract.image_to_string(
                                roi, config=f'--psm {psm} --oem 3'
                            ).strip()
                            
                            # Keep best result (longest text)
                            if text and len(text) > len(best_text):
                                best_text = text
                                # If we get good text, no need to try other modes
                                if len(text) > 3:
                                    break
                        except Exception:
                            continue
                    
                    # Update element if text was found
                    if best_text:
                        element.text = best_text
                        elements_with_text += 1
                    else:
                        element.text = f"[{element.element_type}]"
                        
                except Exception as e:
                    logger.error(f"OCR error: {e}")
                    element.text = f"[{element.element_type}]"
        
        logger.info(f"Extracted text from {elements_with_text} elements")
        return elements
    
    def _add_semantic_understanding(self, image: np.ndarray, elements: List[UIElement]) -> List[UIElement]:
        """Add semantic understanding using multi-model intelligence fusion"""
        try:
            # Prepare input for LLM with enhanced context
            elements_data = [
                {
                    "id": e.element_id,
                    "type": e.element_type,
                    "bbox": e.bbox,
                    "text": e.text if hasattr(e, "text") and e.text else "",
                    "dimensions": {"width": e.width, "height": e.height},
                    "position": {"center_x": e.center[0], "center_y": e.center[1]}
                }
                for e in elements[:30]  # Limit to avoid token limits
            ]
            
            # Create an enhanced prompt with more context awareness
            prompt = f"""You are an expert in UI analysis. Analyze these UI elements and provide semantic understanding:

{json.dumps(elements_data, indent=2)}

For each element, determine:
1. Purpose - what is this element's specific function?
2. State - is it active, disabled, selected, etc.?
3. Importance - how critical is this element to the interface (high/medium/low)?
4. Interaction - what's the expected user interaction (click, type, drag, etc.)?
5. Relationship - how does this element relate to others (parent/child/group)?

Respond with a valid JSON object where keys are element IDs and values contain purpose, state, importance, interaction, and relationship."""
            
            # Determine whether to use the new LLM connector or fallback to Together AI
            if hasattr(self, 'llm_manager'):
                # Use our adaptive LLM connector system
                try:
                    # The connector automatically handles retries, fallbacks, and error handling
                    success, response, provider = self.llm_manager.get_completion(
                        prompt=prompt,
                        max_tokens=1200,
                        temperature=0.3
                    )
                    
                    if not success:
                        raise Exception(f"LLM error: {response}")
                    
                    # The response is already the text output
                    llm_output = response
                    
                    # Log which provider was used
                    logger.info(f"Semantic understanding provided by {provider}")
                    
                except Exception as e:
                    logger.error(f"All LLM providers failed: {e}")
                    raise
            else:
                # Fallback to original Together AI integration
                url = "https://api.together.xyz/v1/completions"
                payload = {
                    "model": self.together_model,
                    "prompt": prompt,
                    "max_tokens": 1200,
                    "temperature": 0.3,
                    "top_p": 0.7
                }
                headers = {
                    "accept": "application/json",
                    "content-type": "application/json",
                    "Authorization": f"Bearer {self.together_api_key}"
                }
                
                # Add retry mechanism
                max_retries = 2
                retry_count = 0
                response = None
                
                while retry_count < max_retries and not response:
                    try:
                        response = self.requests.post(url, json=payload, headers=headers, timeout=10)
                        response.raise_for_status()
                    except Exception as e:
                        logger.warning(f"API attempt {retry_count+1} failed: {e}")
                        retry_count += 1
                        time.sleep(1)  # Brief delay before retry
                        if retry_count >= max_retries:
                            raise
                
                # Extract and process response
                result = response.json()
                llm_output = result.get("choices", [{}])[0].get("text", "")
            
            # Enhanced JSON extraction with better resilience
            semantic_data = self._extract_json_from_llm_response(llm_output)
            
            if semantic_data:
                # Enhance elements with semantic data using an adaptive approach
                element_map = {e.element_id: e for e in elements}
                for element_id, semantic_info in semantic_data.items():
                    if element_id in element_map:
                        element = element_map[element_id]
                        
                        # Update element with all available semantic information
                        for key, value in semantic_info.items():
                            if key == "state" and value:
                                element.state["detected_state"] = value
                            elif value:  # Only add non-empty values
                                element.attributes[key] = value
                
                logger.info("Added LLM-based semantic understanding with enhanced properties")
            else:
                # Fallback to heuristic understanding
                logger.warning("Couldn't extract semantic data from LLM - using heuristics")
                elements = self._add_heuristic_understanding(elements)
            
        except Exception as e:
            logger.error(f"Error in semantic understanding: {e}")
            # Graceful fallback to heuristics
            elements = self._add_heuristic_understanding(elements)
            
        return elements
        
    def _extract_json_from_llm_response(self, text):
        """Extract JSON from LLM response with enhanced resilience"""
        try:
            # Try the typical JSON object format first
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(text[json_start:json_end])
                
            # Try array format if that failed
            array_start = text.find("[")
            array_end = text.rfind("]") + 1
            if array_start >= 0 and array_end > array_start:
                array_data = json.loads(text[array_start:array_end])
                # Convert array to dictionary by element ID
                result = {}
                for item in array_data:
                    if "id" in item:
                        elem_id = item.pop("id")
                        result[elem_id] = item
                return result
                
            # Look for JSON in code blocks (markdown format)
            import re
            code_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
            if code_match:
                json_text = code_match.group(1).strip()
                if json_text.startswith("{"): 
                    return json.loads(json_text)
                elif json_text.startswith("["):
                    array_data = json.loads(json_text)
                    # Convert array to dictionary by element ID
                    result = {}
                    for item in array_data:
                        if "id" in item:
                            elem_id = item.pop("id")
                            result[elem_id] = item
                    return result
            
            # All extraction attempts failed
            return None
            
        except Exception as e:
            logger.error(f"JSON extraction error: {e}")
            return None
    
    def _add_heuristic_understanding(self, elements: List[UIElement]) -> List[UIElement]:
        """Add comprehensive semantic understanding using advanced heuristics"""
        for element in elements:
            # Initialize attributes if they don't exist
            if not hasattr(element, "attributes"):
                element.attributes = {}
            if not hasattr(element, "state"):
                element.state = {}
                
            element_type = element.element_type.lower()
            element_text = element.text if hasattr(element, "text") and element.text else ""
            
            # Determine purpose with more context awareness
            if "button" in element_type:
                purpose = "interactive control for user action"
                if element_text and not element_text.startswith("["):
                    purpose = f"button to {element_text.lower()}"
                element.attributes["purpose"] = purpose
                element.attributes["importance"] = "high"
                element.attributes["interaction"] = "click"
                element.state["detected_state"] = "active"
            
            elif "text_field" in element_type or "input" in element_type:
                input_type = "general text"
                if "search" in element_text.lower():
                    input_type = "search term"
                elif "email" in element_text.lower():
                    input_type = "email address"
                elif "pass" in element_text.lower():
                    input_type = "password"
                    
                element.attributes["purpose"] = f"input area for {input_type}"
                element.attributes["importance"] = "medium"
                element.attributes["interaction"] = "type"
                element.state["detected_state"] = "empty"
            
            elif "checkbox" in element_type or "radio" in element_type:
                option_type = "option" if element_text else "setting"
                element.attributes["purpose"] = f"toggle control for {option_type}"
                element.attributes["importance"] = "medium"
                element.attributes["interaction"] = "toggle"
                element.state["detected_state"] = "unchecked"
            
            elif "icon" in element_type:
                action = "menu" if element.width < 50 else "function"
                element.attributes["purpose"] = f"visual indicator for {action}"
                element.attributes["importance"] = "medium"
                element.attributes["interaction"] = "click"
                element.state["detected_state"] = "active"
                
            elif "image" in element_type:
                element.attributes["purpose"] = "visual content display"
                element.attributes["importance"] = "medium"
                element.attributes["interaction"] = "view"
                element.state["detected_state"] = "visible"
                
            elif "menu" in element_type:
                element.attributes["purpose"] = "navigation or selection control"
                element.attributes["importance"] = "high"
                element.attributes["interaction"] = "click"
                element.state["detected_state"] = "active"
                
            elif "scroll" in element_type or "slider" in element_type:
                element.attributes["purpose"] = "content navigation control"
                element.attributes["importance"] = "medium"
                element.attributes["interaction"] = "drag"
                element.state["detected_state"] = "active"
                
            elif "label" in element_type or "text" in element_type:
                if element.width > 200:
                    text_type = "paragraph"
                else:
                    text_type = "label"
                element.attributes["purpose"] = f"informational {text_type}"
                element.attributes["importance"] = "low"
                element.attributes["interaction"] = "none"
                element.state["detected_state"] = "static"
                
            elif "menu_item" in element_type:
                element.attributes["purpose"] = "navigation or action option"
                element.attributes["importance"] = "medium"
                element.attributes["interaction"] = "click"
                element.state["detected_state"] = "active"
            
            elif "panel" in element_type or "container" in element_type:
                element.attributes["purpose"] = "container for related elements"
                element.attributes["importance"] = "low"
                element.attributes["interaction"] = "none"
                element.state["detected_state"] = "static"
                
            else:
                element.attributes["purpose"] = f"general UI {element_type}"
                element.attributes["importance"] = "medium"
                element.attributes["interaction"] = "click"
                element.state["detected_state"] = "unknown"
            
            # Add state information based on position and relationships
            if element.parent_id:
                element.state["relationship"] = "child"
            elif element.children_ids:
                element.state["relationship"] = "parent"
        
        logger.info("Added heuristic-based semantic understanding")
        return elements


class AdaptiveCalibrator:
    """
    Calibrates confidence scores based on feedback.
    
    This component learns from feedback to improve detection accuracy over time.
    """
    
    def __init__(self):
        # Calibration data structure: {detector_type: {element_type: {bucket: {correct, total}}}}
        self.calibration_data = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: {"correct": 0, "total": 0}
                )
            )
        )
        
        # Number of confidence buckets (0.0-0.1, 0.1-0.2, ..., 0.9-1.0)
        self.num_buckets = 10
        
        # Minimum samples needed for calibration
        self.min_samples = 5
        
        logger.info("AdaptiveCalibrator initialized")
    
    def calibrate_confidence(self, 
                          detector_name: str, 
                          element_type: str, 
                          raw_confidence: float) -> float:
        """
        Calibrate raw confidence score based on historical accuracy.
        
        Args:
            detector_name: Detection method
            element_type: UI element type
            raw_confidence: Raw confidence score
            
        Returns:
            Calibrated confidence score
        """
        # If not enough data, return raw confidence
        if not self._has_sufficient_data(detector_name, element_type):
            return raw_confidence
        
        # Get bucket for this confidence
        bucket = min(self.num_buckets - 1, int(raw_confidence * self.num_buckets))
        
        # Get historical accuracy
        bucket_data = self.calibration_data[detector_name][element_type][bucket]
        historical_accuracy = bucket_data["correct"] / bucket_data["total"] if bucket_data["total"] > 0 else 0.5
        
        # Blend raw confidence with historical accuracy
        weight = min(0.8, bucket_data["total"] / 100)
        calibrated_confidence = (1 - weight) * raw_confidence + weight * historical_accuracy
        
        return calibrated_confidence
    
    def update_from_feedback(self, 
                           detector_name: str, 
                           element_type: str, 
                           raw_confidence: float, 
                           was_correct: bool) -> None:
        """
        Update calibration data based on feedback.
        
        Args:
            detector_name: Detection method
            element_type: UI element type
            raw_confidence: Raw confidence score
            was_correct: Whether detection was correct
        """
        # Get bucket for this confidence
        bucket = min(self.num_buckets - 1, int(raw_confidence * self.num_buckets))
        
        # Update data
        self.calibration_data[detector_name][element_type][bucket]["total"] += 1
        if was_correct:
            self.calibration_data[detector_name][element_type][bucket]["correct"] += 1
        
        logger.info(f"Updated calibration for {detector_name} {element_type}")
    
    def _has_sufficient_data(self, detector_name: str, element_type: str) -> bool:
        """Check if we have enough data for calibration"""
        if detector_name not in self.calibration_data:
            return False
        
        if element_type not in self.calibration_data[detector_name]:
            return False
        
        total_samples = sum(
            self.calibration_data[detector_name][element_type][bucket]["total"]
            for bucket in range(self.num_buckets)
        )
        
        return total_samples >= self.min_samples


# =====================================================================
# Main UI Intelligence System
# =====================================================================

class NexusUIIntelligence:
    """
    Main NEXUS UI Intelligence System.
    
    This class integrates detection, semantic understanding, and calibration
    into a unified system that learns and adapts over time.
    """
    
    def __init__(self):
        # Initialize components
        self.detector = AdaptiveDetector()
        self.semantic_analyzer = SemanticAnalyzer()
        self.calibrator = AdaptiveCalibrator()
        
        # Analysis history
        self.current_analysis = None
        self.analysis_history = []
        self.max_history_size = 10
        
        # Create output directory
        os.makedirs("ui_intelligence_results", exist_ok=True)
        
        logger.info("NEXUS UI Intelligence initialized")
    
    def analyze_screenshot(self, 
                        screenshot: np.ndarray,
                        min_confidence: float = 0.5) -> UIAnalysisResult:
        """
        Analyze a screenshot to detect and understand UI elements.
        
        Args:
            screenshot: Screenshot to analyze
            min_confidence: Minimum confidence threshold
            
        Returns:
            UI analysis result
        """
        start_time = time.time()
        
        # 1. Detect UI elements
        elements = self.detector.detect_elements(screenshot)
        
        # 2. Enhance with semantic understanding
        elements = self.semantic_analyzer.enhance_elements(screenshot, elements)
        
        # 3. Calibrate confidence scores
        for element in elements:
            calibrated_confidence = self.calibrator.calibrate_confidence(
                detector_name=element.detection_method,
                element_type=element.element_type,
                raw_confidence=element.confidence
            )
            element.confidence = calibrated_confidence
        
        # 4. Filter by confidence
        filtered_elements = [e for e in elements if e.confidence >= min_confidence]
        
        # 5. Determine application type
        app_type = self._determine_application_type(filtered_elements)
        
        # 6. Create analysis result
        analysis = UIAnalysisResult(
            elements=filtered_elements,
            application_type=app_type,
            confidence=self._calculate_overall_confidence(filtered_elements),
            processing_time=time.time() - start_time
        )
        
        # Update current analysis and history
        self.current_analysis = analysis
        self._update_history(analysis)
        
        # Create visualization
        self._visualize_analysis(screenshot, analysis)
        
        logger.info(f"Screenshot analysis completed with {len(filtered_elements)} elements")
        return analysis
    
    def provide_feedback(self, 
                       element_id: str, 
                       was_correct: bool,
                       correct_type: Optional[str] = None) -> None:
        """
        Provide feedback on element detection for adaptive learning.
        
        Args:
            element_id: ID of the element
            was_correct: Whether detection was correct
            correct_type: Correct element type if detection was wrong
        """
        # Find element
        element = None
        for analysis in reversed(self.analysis_history):
            for e in analysis.elements:
                if e.element_id == element_id:
                    element = e
                    break
            if element:
                break
        
        if not element:
            logger.warning(f"Element {element_id} not found for feedback")
            return
        
        # Update calibrator
        self.calibrator.update_from_feedback(
            detector_name=element.detection_method,
            element_type=element.element_type,
            raw_confidence=element.confidence,
            was_correct=was_correct
        )
        
        logger.info(f"Feedback recorded for element {element_id}")
    
    def get_element_by_id(self, element_id: str) -> Optional[UIElement]:
        """Get an element by its ID from the current analysis"""
        if not self.current_analysis:
            return None
        
        for element in self.current_analysis.elements:
            if element.element_id == element_id:
                return element
        
        return None
    
    def suggest_interactions(self, 
                          element_id: Optional[str] = None) -> List[Dict]:
        """
        Suggest possible interactions with UI elements.
        
        Args:
            element_id: Optional specific element ID
            
        Returns:
            List of interaction suggestions
        """
        suggestions = []
        
        if not self.current_analysis:
            return suggestions
        
        if element_id:
            # Specific element
            element = self.get_element_by_id(element_id)
            if not element:
                return suggestions
            
            # Generate suggestions based on element type
            if element.element_type in ["button", "link", "menu_item"]:
                suggestions.append({
                    "action": "click",
                    "description": f"Click the {element.element_type}",
                    "confidence": 0.9
                })
            
            elif element.element_type in ["text_field", "input"]:
                suggestions.append({
                    "action": "type",
                    "description": f"Type text into the {element.element_type}",
                    "confidence": 0.9
                })
            
            elif element.element_type in ["checkbox", "radio_button"]:
                suggestions.append({
                    "action": "toggle",
                    "description": f"Toggle the {element.element_type}",
                    "confidence": 0.9
                })
        else:
            # Get suggestions for all interactive elements
            interactive_types = ["button", "text_field", "checkbox", "link", "menu_item"]
            
            for element in self.current_analysis.elements:
                if element.element_type in interactive_types:
                    # Basic suggestion
                    action = "click"
                    if element.element_type in ["text_field", "input"]:
                        action = "type"
                    elif element.element_type in ["checkbox", "radio_button"]:
                        action = "toggle"
                    
                    suggestion = {
                        "element_id": element.element_id,
                        "element_type": element.element_type,
                        "action": action,
                        "description": f"{action.capitalize()} the {element.element_type}"
                    }
                    
                    # Add text if available
                    if hasattr(element, "text") and element.text:
                        suggestion["element_text"] = element.text
                        suggestion["description"] += f" '{element.text}'"
                    
                    suggestions.append(suggestion)
        
        return suggestions
    
    def _determine_application_type(self, elements: List[UIElement]) -> str:
        """Determine the application type from UI elements"""
        # Count element types
        type_counts = {}
        for element in elements:
            if element.element_type not in type_counts:
                type_counts[element.element_type] = 0
            type_counts[element.element_type] += 1
        
        # Extract text content
        text_content = " ".join([e.text for e in elements if hasattr(e, "text") and e.text])
        
        # Simple heuristics
        if "menu_item" in type_counts and type_counts["menu_item"] > 5:
            return "desktop_application"
        
        if "table_cell" in type_counts and type_counts["table_cell"] > 10:
            return "data_application"
        
        if "link" in type_counts and type_counts["link"] > 5:
            return "web_application"
        
        # Text-based detection
        text_lower = text_content.lower()
        if any(word in text_lower for word in ["login", "sign in", "password"]):
            return "authentication_screen"
        
        if any(word in text_lower for word in ["search", "find", "filter"]):
            return "search_interface"
        
        return "generic_interface"
    
    def _calculate_overall_confidence(self, elements: List[UIElement]) -> float:
        """Calculate overall confidence for the analysis"""
        if not elements:
            return 0.0
        
        # Weight by element area
        weighted_confidences = []
        total_area = 0
        
        for element in elements:
            area = element.width * element.height
            weighted_confidences.append((element.confidence, area))
            total_area += area
        
        if total_area == 0:
            return sum(c for c, _ in weighted_confidences) / len(weighted_confidences)
        
        return sum(c * a for c, a in weighted_confidences) / total_area
    
    def _update_history(self, analysis: UIAnalysisResult) -> None:
        """Update analysis history"""
        self.analysis_history.append(analysis)
        
        # Trim history if needed
        if len(self.analysis_history) > self.max_history_size:
            self.analysis_history = self.analysis_history[-self.max_history_size:]
    
    def _visualize_analysis(self, 
                         screenshot: np.ndarray, 
                         analysis: UIAnalysisResult) -> None:
        """Create visualization of analysis results"""
        try:
            # Create a copy of the screenshot
            visualization = screenshot.copy()
            
            # Draw each element
            for element in analysis.elements:
                x1, y1, x2, y2 = element.bbox
                
                # Determine color based on element type
                color_map = {
                    "button": (0, 165, 255),     # Orange
                    "text_field": (0, 255, 0),  # Green
                    "icon": (255, 0, 0),        # Blue
                    "control": (255, 0, 255),   # Magenta
                    "menu_item": (255, 255, 0), # Cyan
                    "checkbox": (0, 255, 255),  # Yellow
                    "panel": (128, 0, 128),     # Purple
                    "sidebar": (0, 128, 255)    # Light blue
                }
                
                color = color_map.get(element.element_type, (128, 128, 128))
                
                # Draw rectangle
                cv2.rectangle(visualization, (x1, y1), (x2, y2), color, 2)
                
                # Draw element type and confidence
                label = f"{element.element_type} ({element.confidence:.2f})"
                cv2.putText(visualization, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Draw text content if available
                if hasattr(element, "text") and element.text:
                    text_label = element.text[:20] + "..." if len(element.text) > 20 else element.text
                    cv2.putText(visualization, text_label, (x1, y2 + 15), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Save visualization
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join("ui_intelligence_results", f"analysis_{timestamp}.png")
            cv2.imwrite(output_path, visualization)
            
            logger.info(f"Visualization saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")


# =====================================================================
# Demo and Test Functions
# =====================================================================

def create_test_ui_image() -> np.ndarray:
    """Create a test UI image with various controls"""
    # Create a blank image (white background)
    image = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Draw a header bar
    cv2.rectangle(image, (0, 0), (800, 50), (50, 50, 50), -1)
    cv2.putText(image, "NEXUS Dashboard", (20, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Draw a search box
    cv2.rectangle(image, (600, 10), (780, 40), (255, 255, 255), -1)
    cv2.rectangle(image, (600, 10), (780, 40), (200, 200, 200), 1)
    cv2.putText(image, "Search...", (610, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    
    # Draw a sidebar
    cv2.rectangle(image, (0, 50), (180, 600), (240, 240, 240), -1)
    
    # Draw menu items
    menu_items = ["Dashboard", "Analytics", "Reports", "Settings", "Help"]
    for i, item in enumerate(menu_items):
        y = 90 + i * 40
        cv2.putText(image, item, (20, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)
    
    # Draw main content area title
    cv2.putText(image, "System Overview", (200, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 50), 2)
    
    # Draw stat cards
    card_positions = [
        ((200, 100), (350, 200), "CPU", "32%"),
        ((370, 100), (520, 200), "Memory", "64%"),
        ((540, 100), (690, 200), "Disk", "45%")
    ]
    
    for (x1, y1), (x2, y2), title, value in card_positions:
        # Card background
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), -1)
        cv2.rectangle(image, (x1, y1), (x2, y2), (200, 200, 200), 1)
        # Card title
        cv2.putText(image, title, (x1 + 10, y1 + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
        # Card value
        cv2.putText(image, value, (x1 + 10, y1 + 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 50, 50), 2)
    
    # Draw buttons
    button_positions = [
        ((200, 220), (300, 260), "Refresh"),
        ((320, 220), (450, 260), "Export Data"),
        ((470, 220), (570, 260), "Settings")
    ]
    
    for (x1, y1), (x2, y2), text in button_positions:
        # Button background
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 120, 215), -1)
        # Button text
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x1 + (x2 - x1 - text_size[0]) // 2
        text_y = y1 + (y2 - y1 + text_size[1]) // 2
        cv2.putText(image, text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw a table
    table_y = 300
    # Table header
    cv2.rectangle(image, (200, table_y), (700, table_y + 40), (240, 240, 240), -1)
    cv2.rectangle(image, (200, table_y), (700, table_y + 40), (200, 200, 200), 1)
    headers = ["Process", "Status", "CPU", "Memory"]
    header_x = 200
    for header in headers:
        cv2.putText(image, header, (header_x + 10, table_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 2)
        header_x += 125
    
    # Table rows
    processes = [
        ("NEXUS Core", "Running", "12%", "256MB"),
        ("UI Service", "Running", "8%", "128MB"),
        ("Database", "Running", "15%", "512MB"),
        ("API Server", "Stopped", "0%", "0MB")
    ]
    
    for i, proc in enumerate(processes):
        row_y = table_y + 40 + i * 40
        cv2.rectangle(image, (200, row_y), (700, row_y + 40), (255, 255, 255), -1)
        cv2.rectangle(image, (200, row_y), (700, row_y + 40), (220, 220, 220), 1)
        
        cell_x = 200
        for j, cell in enumerate(proc):
            # Color for status column
            if j == 1:
                text_color = (0, 180, 0) if cell == "Running" else (180, 0, 0)  # Green or Red
            else:
                text_color = (80, 80, 80)
                
            cv2.putText(image, cell, (cell_x + 10, row_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            cell_x += 125
    
    # Save the image
    output_path = os.path.join("test_data", "test_ui.png")
    cv2.imwrite(output_path, image)
    logger.info(f"Created test UI image: {output_path}")
    
    return image


def capture_real_screenshot() -> Optional[np.ndarray]:
    """Capture current screen content"""
    try:
        from PIL import ImageGrab
        screenshot = ImageGrab.grab()
        screenshot_np = np.array(screenshot)
        return cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.error(f"Error capturing screenshot: {e}")
        return None


def run_demo():
    """Run the UI Intelligence demonstration"""
    print("NEXUS Adaptive UI Intelligence Demo")
    print("===================================\n")
    
    # Initialize the system
    print("Initializing NEXUS UI Intelligence system...")
    ui_intelligence = NexusUIIntelligence()
    
    # Create test image
    print("\nCreating test UI image...")
    test_image = create_test_ui_image()
    
    # Analyze test image
    print("\nAnalyzing test UI image...")
    analysis = ui_intelligence.analyze_screenshot(test_image)
    
    # Print results
    print("\nAnalysis Results:")
    print(f"  - Detected {len(analysis.elements)} UI elements")
    print(f"  - Application type: {analysis.application_type}")
    print(f"  - Overall confidence: {analysis.confidence:.2f}")
    print(f"  - Processing time: {analysis.processing_time:.3f} seconds")
    
    # Count by element type
    element_types = {}
    for element in analysis.elements:
        element_type = element.element_type
        if element_type not in element_types:
            element_types[element_type] = 0
        element_types[element_type] += 1
    
    print("\nElement Types:")
    for element_type, count in sorted(element_types.items()):
        print(f"  - {element_type}: {count}")
    
    # Show elements with text
    text_elements = [e for e in analysis.elements if hasattr(e, "text") and e.text]
    if text_elements:
        print("\nElements with Text:")
        for i, element in enumerate(text_elements[:5]):  # Show top 5
            print(f"  {i+1}. {element.element_type}: '{element.text}'")
    
    # Get interaction suggestions
    suggestions = ui_intelligence.suggest_interactions()
    print("\nInteraction Suggestions:")
    for i, suggestion in enumerate(suggestions[:5]):  # Show top 5
        element_type = suggestion.get("element_type", "unknown")
        element_text = suggestion.get("element_text", "")
        action = suggestion.get("action", "")
        description = suggestion.get("description", "")
        
        print(f"  {i+1}. {action.upper()} {element_type}{' ' + element_text if element_text else ''}: {description}")
    
    # Capture and analyze real screenshot if available
    print("\nCapturing real screenshot...")
    real_screenshot = capture_real_screenshot()
    if real_screenshot is not None:
        # Save screenshot
        screenshot_path = os.path.join("test_data", "real_screenshot.png")
        cv2.imwrite(screenshot_path, real_screenshot)
        print(f"  - Screenshot saved to: {screenshot_path}")
        
        # Analyze real screenshot
        print("\nAnalyzing real screenshot...")
        real_analysis = ui_intelligence.analyze_screenshot(real_screenshot)
        
        # Print results
        print("\nReal Screenshot Analysis:")
        print(f"  - Detected {len(real_analysis.elements)} UI elements")
        print(f"  - Application type: {real_analysis.application_type}")
        print(f"  - Overall confidence: {real_analysis.confidence:.2f}")
        print(f"  - Processing time: {real_analysis.processing_time:.3f} seconds")
        
        # Elements by type
        real_element_types = {}
        for element in real_analysis.elements:
            element_type = element.element_type
            if element_type not in real_element_types:
                real_element_types[element_type] = 0
            real_element_types[element_type] += 1
        
        print("\nElement Types:")
        for element_type, count in sorted(real_element_types.items()):
            print(f"  - {element_type}: {count}")
    
    print("\nDemo completed! Results saved in ui_intelligence_results directory.")


if __name__ == "__main__":
    run_demo()
