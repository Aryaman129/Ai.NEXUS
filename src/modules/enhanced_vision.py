"""
Enhanced Vision Module for NEXUS
Provides comprehensive vision capabilities for analyzing screen content
"""
import logging
import asyncio
import os
import time
import numpy as np
from PIL import Image
from typing import Dict, List, Any, Optional, Tuple, Union

from .advanced_screen_capture import AdvancedScreenCapture

logger = logging.getLogger(__name__)

class EnhancedVision:
    """
    Enhanced vision capabilities for NEXUS.
    
    This module provides comprehensive screen analysis including:
    - Advanced screen capture 
    - OCR (text recognition)
    - UI element detection
    - Object recognition
    - Semantic image understanding
    
    It dynamically utilizes the best available tools based on what's installed.
    """
    
    def __init__(self):
        """Initialize enhanced vision capabilities"""
        # Screen capture
        self.screen_capture = AdvancedScreenCapture()
        
        # OCR capabilities
        self.has_tesseract = self._check_tesseract()
        self.has_easyocr = self._check_easyocr()
        
        # Advanced ML capabilities
        self.has_yolo = False
        self.yolo_detector = None
        self._initialize_ml_capabilities()
        
        logger.info(f"Enhanced Vision initialized with: Tesseract={self.has_tesseract}, EasyOCR={self.has_easyocr}, YOLO={self.has_yolo}")
    
    def _check_tesseract(self) -> bool:
        """Check if Tesseract OCR is available"""
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            return True
        except (ImportError, Exception):
            logger.info("Tesseract OCR not available")
            return False
    
    def _check_easyocr(self) -> bool:
        """Check if EasyOCR is available"""
        try:
            import easyocr
            return True
        except ImportError:
            logger.info("EasyOCR not available")
            return False
    
    def _initialize_ml_capabilities(self) -> None:
        """Initialize advanced ML capabilities if available"""
        try:
            # Import here to avoid hard dependency
            import sys
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
            from ai_core.ml_models import YOLODetector
            
            # Initialize YOLO detector
            self.yolo_detector = YOLODetector(model_name="yolov8n.pt")
            self.has_yolo = True
            logger.info("YOLO detector initialized for vision")
            
        except ImportError as e:
            logger.info(f"YOLO detector not available: {e}")
    
    async def take_screenshot(self, region: Optional[Tuple[int, int, int, int]] = None) -> Image.Image:
        """
        Take a screenshot of the entire screen or a specific region
        
        Args:
            region: Optional tuple (left, top, width, height) for a specific region
            
        Returns:
            PIL Image of the screenshot
        """
        return self.screen_capture.capture_screen(region)
    
    async def recognize_text(self, image: Optional[Image.Image] = None, 
                          region: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """
        Recognize text in an image or screen region
        
        Args:
            image: Optional PIL Image to analyze
            region: Optional screen region to capture and analyze
            
        Returns:
            Dictionary with recognized text and positions
        """
        if image is None:
            image = await self.take_screenshot(region)
        
        # Try using EasyOCR first (better quality)
        if self.has_easyocr:
            try:
                return await self._recognize_text_easyocr(image)
            except Exception as e:
                logger.error(f"EasyOCR failed: {e}, falling back to Tesseract")
        
        # Fall back to Tesseract
        if self.has_tesseract:
            return await self._recognize_text_tesseract(image)
        
        # Last resort - ask for YOLO to identify text regions
        if self.has_yolo and self.yolo_detector:
            try:
                text_regions = await self.yolo_detector.detect_text_regions(image)
                if text_regions.get("success", False) and text_regions.get("text_regions", []):
                    return {
                        "success": True,
                        "text": "Text regions detected, but OCR is not available",
                        "text_regions": text_regions.get("text_regions", []),
                        "method": "YOLO text region detection"
                    }
            except Exception as e:
                logger.error(f"YOLO text region detection failed: {e}")
        
        return {
            "success": False,
            "error": "No OCR capabilities available",
            "text": ""
        }
    
    async def _recognize_text_tesseract(self, image: Image.Image) -> Dict[str, Any]:
        """Recognize text using Tesseract OCR"""
        try:
            import pytesseract
            
            # Perform OCR
            text = pytesseract.image_to_string(image)
            
            # Get text locations
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            # Build structured result
            results = []
            for i in range(len(data["text"])):
                if data["text"][i].strip():
                    results.append({
                        "text": data["text"][i],
                        "conf": data["conf"][i],
                        "box": {
                            "x": data["left"][i],
                            "y": data["top"][i],
                            "width": data["width"][i],
                            "height": data["height"][i]
                        }
                    })
            
            return {
                "success": True,
                "text": text,
                "text_regions": results,
                "method": "Tesseract OCR"
            }
            
        except Exception as e:
            logger.error(f"Tesseract OCR error: {e}")
            return {
                "success": False,
                "error": str(e),
                "text": ""
            }
    
    async def _recognize_text_easyocr(self, image: Image.Image) -> Dict[str, Any]:
        """Recognize text using EasyOCR"""
        try:
            import easyocr
            
            # Initialize reader if not already done
            if not hasattr(self, 'reader'):
                self.reader = easyocr.Reader(['en'])
            
            # Convert PIL to numpy if needed
            if isinstance(image, Image.Image):
                image_np = np.array(image)
            else:
                image_np = image
            
            # Perform OCR
            results = self.reader.readtext(image_np)
            
            # Structure the results
            text_regions = []
            full_text = []
            
            for bbox, text, prob in results:
                # Extract bounding box coordinates
                top_left, top_right, bottom_right, bottom_left = bbox
                x = int(top_left[0])
                y = int(top_left[1])
                width = int(bottom_right[0] - top_left[0])
                height = int(bottom_right[1] - top_left[1])
                
                # Add to results
                text_regions.append({
                    "text": text,
                    "conf": float(prob),
                    "box": {
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height
                    }
                })
                
                full_text.append(text)
            
            return {
                "success": True,
                "text": " ".join(full_text),
                "text_regions": text_regions,
                "method": "EasyOCR"
            }
            
        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
            raise e
    
    async def detect_ui_elements(self, screenshot: Optional[Image.Image] = None) -> Dict[str, Any]:
        """
        Detect UI elements in an image
        
        Args:
            screenshot: Optional screenshot to analyze, if None, a new screenshot is taken
            
        Returns:
            Dictionary with detected UI elements
        """
        if screenshot is None:
            screenshot = await self.take_screenshot()
        
        # Use YOLO detector if available (preferred method)
        if self.has_yolo and self.yolo_detector:
            try:
                ui_results = await self.yolo_detector.detect_ui_elements(screenshot)
                if ui_results.get("success", False):
                    return ui_results
            except Exception as e:
                logger.error(f"YOLO UI detection error: {e}")
        
        # Fallback to template matching or other methods
        # For now, return a basic response indicating no advanced detection is available
        return {
            "success": False,
            "message": "Advanced UI element detection requires YOLO capabilities",
            "ui_elements": []
        }
    
    async def detect_objects(self, screenshot: Optional[Image.Image] = None) -> Dict[str, Any]:
        """
        Detect objects in an image
        
        Args:
            screenshot: Optional screenshot to analyze, if None, a new screenshot is taken
            
        Returns:
            Dictionary with detected objects
        """
        if screenshot is None:
            screenshot = await self.take_screenshot()
        
        # Use YOLO detector if available
        if self.has_yolo and self.yolo_detector:
            try:
                return await self.yolo_detector.detect_objects(screenshot)
            except Exception as e:
                logger.error(f"YOLO object detection error: {e}")
        
        # Fallback - no advanced object detection available
        return {
            "success": False,
            "message": "Object detection requires YOLO capabilities",
            "detections": []
        }
    
    async def analyze_screen_content(self, screenshot: Optional[Image.Image] = None) -> Dict[str, Any]:
        """
        Comprehensive analysis of screen content
        
        Args:
            screenshot: Optional screenshot to analyze, if None, a new screenshot is taken
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        if screenshot is None:
            screenshot = await self.take_screenshot()
        
        # Perform multiple analyses in parallel
        text_task = asyncio.create_task(self.recognize_text(screenshot))
        ui_task = asyncio.create_task(self.detect_ui_elements(screenshot))
        objects_task = asyncio.create_task(self.detect_objects(screenshot))
        
        # Wait for all analyses to complete
        text_results = await text_task
        ui_results = await ui_task
        object_results = await objects_task
        
        # Combine results
        return {
            "success": True,
            "text_analysis": text_results,
            "ui_analysis": ui_results,
            "object_analysis": object_results,
            "timestamp": time.time()
        }
    
    def get_monitor_info(self) -> List[Dict[str, Any]]:
        """Get information about available monitors"""
        return self.screen_capture.get_monitor_info()
