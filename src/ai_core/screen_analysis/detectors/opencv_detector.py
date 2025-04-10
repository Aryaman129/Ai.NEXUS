"""
UI element detector using OpenCV.

This detector uses traditional computer vision techniques from OpenCV
to detect UI elements. It serves as a reliable fallback when ML-based
detectors are not available or when processing needs to be fast and lightweight.
"""

import logging
import time
import os
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import cv2

from .base import UIDetectorInterface

logger = logging.getLogger(__name__)

class OpenCVDetector(UIDetectorInterface):
    """UI element detector using OpenCV computer vision techniques"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the OpenCV detector
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.initialized = False
        
        # Default configuration
        self.min_button_width = self.config.get("min_button_width", 30)
        self.min_button_height = self.config.get("min_button_height", 20)
        self.canny_threshold1 = self.config.get("canny_threshold1", 50)
        self.canny_threshold2 = self.config.get("canny_threshold2", 150)
        self.use_ocr = self.config.get("use_ocr", False)
        self.ocr_engine = None
            
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
            self.min_button_width = self.config.get("min_button_width", self.min_button_width)
            self.min_button_height = self.config.get("min_button_height", self.min_button_height)
            self.canny_threshold1 = self.config.get("canny_threshold1", self.canny_threshold1)
            self.canny_threshold2 = self.config.get("canny_threshold2", self.canny_threshold2)
            self.use_ocr = self.config.get("use_ocr", self.use_ocr)
            
        # Initialize OCR if needed
        if self.use_ocr:
            try:
                import pytesseract
                pytesseract.get_tesseract_version()
                # Check if custom path is provided
                tesseract_path = self.config.get("tesseract_path")
                if tesseract_path:
                    pytesseract.pytesseract.tesseract_cmd = tesseract_path
                self.ocr_engine = pytesseract
                logger.info("Initialized Tesseract OCR for OpenCV detector")
            except (ImportError, Exception) as e:
                logger.warning(f"Failed to initialize OCR: {e}")
                self.use_ocr = False
                
        self.initialized = True
        return True
            
    def detect_elements(self, screenshot: np.ndarray, context: Optional[Dict] = None) -> List[Dict]:
        """
        Detect UI elements in the screenshot
        
        Args:
            screenshot: Image as numpy array (RGB format)
            context: Optional context information
            
        Returns:
            List of detected elements
        """
        if not self.initialized:
            self.initialize()
            
        try:
            # Start timing
            start_time = time.time()
            
            # Convert to grayscale if needed
            if len(screenshot.shape) == 3:
                grayscale = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            else:
                grayscale = screenshot
                
            # Apply edge detection
            edges = cv2.Canny(grayscale, self.canny_threshold1, self.canny_threshold2)
            
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
                if w < self.min_button_width or h < self.min_button_height:
                    continue
                    
                # Calculate aspect ratio to help classify the element
                aspect_ratio = w / h if h > 0 else 0
                
                # Determine element type based on aspect ratio and size
                element_type = self._classify_element(grayscale, aspect_ratio, x, y, w, h)
                
                # Create element dict
                element = {
                    "type": element_type,
                    "bbox": (x, y, x + w, y + h),
                    "center": (x + w // 2, y + h // 2),
                    "confidence": self._calculate_confidence(element_type, aspect_ratio, w, h),
                    "width": w,
                    "height": h,
                    "detector": "opencv"
                }
                
                # Extract text if OCR is enabled
                if self.use_ocr and self.ocr_engine:
                    try:
                        roi = grayscale[y:y+h, x:x+w]
                        text = self.ocr_engine.image_to_string(roi)
                        element["text"] = text.strip()
                    except Exception as e:
                        logger.debug(f"OCR failed: {e}")
                        element["text"] = ""
                else:
                    element["text"] = ""
                
                ui_elements.append(element)
                
            # Apply non-maximum suppression to remove overlapping detections
            ui_elements = self._apply_nms(ui_elements)
                
            # Calculate timing
            elapsed_time = time.time() - start_time
            
            # Add context if provided
            if context:
                for element in ui_elements:
                    element["context"] = context
                    
            logger.debug(f"OpenCV detected {len(ui_elements)} elements in {elapsed_time:.3f}s")
            
            return ui_elements
            
        except Exception as e:
            logger.error(f"Error in OpenCV detection: {e}")
            return []
            
    def _classify_element(self, image, aspect_ratio, x, y, w, h):
        """
        Classify UI element based on its characteristics
        
        Args:
            image: Grayscale image
            aspect_ratio: Width to height ratio
            x, y, w, h: Element coordinates and dimensions
            
        Returns:
            Element type
        """
        # Extract ROI
        roi = image[y:y+h, x:x+w]
        
        # Calculate variance (buttons often have more uniform intensity)
        variance = np.var(roi) if roi.size > 0 else 0
        
        # Calculate fill ratio (percentage of non-background pixels)
        # Assuming background is lighter, threshold to separate foreground
        _, thresholded = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY_INV)
        fill_ratio = np.count_nonzero(thresholded) / (w * h) if (w * h) > 0 else 0
        
        # Classify based on metrics
        if aspect_ratio > 3.0:
            # Very wide elements are likely text fields
            return "text_field"
        elif 0.9 < aspect_ratio < 1.1 and fill_ratio < 0.4:
            # Square elements with low fill are likely checkboxes
            return "checkbox"
        elif aspect_ratio < 0.3 and h > 100:
            # Tall, narrow elements might be scrollbars
            return "scrollbar"
        elif variance < 500 and fill_ratio > 0.6:
            # Uniform intensity with high fill ratio often indicates buttons
            return "button"
        elif variance > 1000:
            # High variance could indicate icons or images
            return "icon"
        else:
            # Default to button for other cases
            return "button"
            
    def _calculate_confidence(self, element_type, aspect_ratio, width, height):
        """
        Calculate confidence score for the detected element
        
        Args:
            element_type: Detected element type
            aspect_ratio: Width to height ratio
            width, height: Element dimensions
            
        Returns:
            Confidence score (0-1)
        """
        # Base confidence
        confidence = 0.6
        
        # Adjust based on element type and characteristics
        if element_type == "button":
            # Buttons typically have aspect ratios between 1.5 and 4
            if 1.5 <= aspect_ratio <= 4.0:
                confidence += 0.2
            # Buttons are usually reasonably sized
            if width >= 50 and height >= 30:
                confidence += 0.1
        elif element_type == "text_field":
            # Text fields are usually wider than tall
            if aspect_ratio > 3.0:
                confidence += 0.2
        elif element_type == "checkbox":
            # Checkboxes are usually square
            if 0.9 <= aspect_ratio <= 1.1:
                confidence += 0.3
            # And fairly small
            if width < 40 and height < 40:
                confidence += 0.1
                
        # Cap at 0.9 since CV-based detection is less reliable than ML
        return min(0.9, confidence)
        
    def _apply_nms(self, elements, iou_threshold=0.5):
        """
        Apply non-maximum suppression to remove overlapping detections
        
        Args:
            elements: List of detected elements
            iou_threshold: IoU threshold for suppression
            
        Returns:
            Filtered elements
        """
        if not elements:
            return []
            
        # Sort by confidence
        elements.sort(key=lambda e: e["confidence"], reverse=True)
        
        # NMS
        keep = []
        for i, element in enumerate(elements):
            keep_element = True
            
            for j in range(len(keep)):
                iou = self._calculate_iou(element["bbox"], elements[keep[j]]["bbox"])
                if iou > iou_threshold:
                    keep_element = False
                    break
                    
            if keep_element:
                keep.append(i)
                
        # Return kept elements
        return [elements[i] for i in keep]
        
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
            
    def supports_incremental_learning(self) -> bool:
        """Whether this detector supports incremental learning"""
        return False
        
    def get_capabilities(self) -> Dict[str, Any]:
        """Get detector capabilities and metadata"""
        capabilities = super().get_capabilities()
        
        # Add OpenCV-specific capabilities
        capabilities.update({
            "name": "opencv",
            "requires_gpu": False,
            "version": "0.1.0",
            "supports_batch_processing": False,
            "uses_ml": False,
            "low_latency": True
        })
        
        return capabilities
