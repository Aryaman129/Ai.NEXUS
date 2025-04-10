"""
Adaptive Vision Integration for NEXUS
Learns and adapts to use whatever capabilities are available
"""
import os
import sys
import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import base64
import io
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

# Import AsyncVectorWrapper
from .async_vector_wrapper import AsyncVectorWrapper

# Optional imports - will be used if available
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Gemini API not available. Please install with: pip install google-generativeai")
    GEMINI_AVAILABLE = False

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("OpenCV not available. Please install with: pip install opencv-python")
    OPENCV_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("EasyOCR not available. Please install with: pip install easyocr")
    EASYOCR_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)

class AdaptiveVision:
    """
    Adaptive Vision for NEXUS
    
    Combines multiple vision capabilities and automatically adapts to what's available:
    1. Gemini Vision API (best option when available)
    2. Local computer vision (OpenCV, EasyOCR as fallback)
    
    This creates a system that learns and adapts based on what's accessible,
    prioritizing the best tools but gracefully falling back when needed.
    """
    
    def __init__(self, 
                gemini_api_key: str = None, 
                vector_storage = None):
        """
        Initialize the adaptive vision system
        
        Args:
            gemini_api_key: Gemini API key
            vector_storage: Vector storage for storing results
        """
        self.gemini_api_key = gemini_api_key
        
        # Wrap vector storage with async-friendly wrapper if provided
        if vector_storage:
            self.vector_storage = AsyncVectorWrapper(vector_storage)
        else:
            self.vector_storage = None
        
        # Keep track of available capabilities
        self.available_capabilities = {
            "gemini": False,
            "opencv": OPENCV_AVAILABLE,
            "easyocr": EASYOCR_AVAILABLE
        }
        
        # Initialize clients
        self.gemini_models = []
        self.vision_model = None # Preferred model for vision tasks
        if GEMINI_AVAILABLE and self.gemini_api_key:
            try:
                genai.configure(api_key=self.gemini_api_key)
                self.gemini_models = [m.name for m in genai.list_models()]
                
                # Select preferred vision model (prioritize 1.5 flash/pro)
                preferred = ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest", "gemini-1.5-flash", "gemini-1.5-pro"]
                for pref_model in preferred:
                    if f"models/{pref_model}" in self.gemini_models:
                        self.vision_model = f"models/{pref_model}"
                        logger.info(f"Selected preferred Gemini model: {self.vision_model}")
                        break
                
                # Fallback if no preferred model found
                if not self.vision_model:
                    for model_name in self.gemini_models:
                         # Check if model supports generateContent method and vision modality
                         model_info = genai.get_model(model_name)
                         if 'generateContent' in model_info.supported_generation_methods: 
                            # We assume if it supports generateContent it might support vision
                            # This is an approximation, ideally check modality support if API allows
                            self.vision_model = model_name
                            logger.info(f"Using fallback Gemini model: {self.vision_model}")
                            break
                
                if self.vision_model:        
                    self.available_capabilities["gemini"] = True
                    logger.info(f"Gemini initialized with API key. {len(self.gemini_models)} models available.")
                else:
                    logger.warning("No suitable Gemini model found.")
                    
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")
                self.available_capabilities["gemini"] = False
        
        # Initialize EasyOCR if available
        self.ocr_reader = None
        if EASYOCR_AVAILABLE:
            try:
                self.ocr_reader = easyocr.Reader(['en']) # Initialize for English
                self.available_capabilities["easyocr"] = True
                logger.info("EasyOCR initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR: {e}")
                self.available_capabilities["easyocr"] = False

        # Log final capabilities
        logger.info(f"Adaptive Vision initialized with capabilities: {self.available_capabilities}")
    
    async def analyze_image(self, 
                           image_path: str = None, 
                           image: Image.Image = None) -> Dict[str, Any]:
        """
        Analyze an image using whatever capabilities are available
        
        Args:
            image_path: Path to image file
            image: PIL Image object
            
        Returns:
            Dictionary with analysis results
        """
        if not image_path and not image:
            return {"success": False, "error": "No image provided"}
        
        # Load image if path provided
        if image_path and not image:
            try:
                image = Image.open(image_path)
            except Exception as e:
                return {"success": False, "error": f"Could not open image: {e}"}
        
        # Get image bytes if needed for some methods
        image_bytes = None
        if image_path:
            try:
                image_bytes = self._get_image_bytes(image_path)
            except Exception as e:
                logger.warning(f"Could not get image bytes: {e}")
        
        # Store results
        results = {
            "success": False,
            "sources": [],
            "analyze_timestamp": datetime.now().isoformat()
        }
        
        # Try Gemini first (best quality when available)
        if self.available_capabilities["gemini"]:
            try:
                gemini_results = await self._analyze_with_gemini(image)
                if gemini_results.get("success", False):
                    # Merge Gemini results
                    for key, value in gemini_results.items():
                        if key != "success":  # Don't overwrite success yet
                            results[key] = value
                    
                    if "sources" in results:
                        results["sources"].append("gemini")
                    else:
                        results["sources"] = ["gemini"]
            except Exception as e:
                logger.error(f"Error in Gemini analysis: {e}")
        
        # Finally try local vision as fallback
        should_use_local = not results.get("success", False) or \
                           ("description" not in results and "labels" not in results)
                           
        if self.available_capabilities["opencv"] and should_use_local:
            try:
                local_results = await self._analyze_with_local_vision(image, image_bytes)
                if local_results.get("success", False):
                    # Only add fields that don't exist yet, as local vision is lower quality
                    for key, value in local_results.items():
                        if key != "success" and key not in results:
                            results[key] = value
                    
                    if "sources" in results:
                        results["sources"].append("local_vision")
                    else:
                        results["sources"] = ["local_vision"]
            except Exception as e:
                logger.error(f"Error in local vision analysis: {e}")
        
        # Mark as successful if we got any results
        results["success"] = len(results.get("sources", [])) > 0
        
        # Store results if we have vector storage
        if results["success"] and self.vector_storage:
            try:
                await self._store_analysis_results(results, image_path)
            except Exception as e:
                logger.error(f"Error storing analysis results: {e}")
        
        return results
    
    async def detect_ui_elements(self, 
                               image_path: str = None, 
                               image: Image.Image = None) -> Dict[str, Any]:
        """
        Detect UI elements in a screenshot
        
        Args:
            image_path: Path to screenshot file
            image: PIL Image object
            
        Returns:
            Dictionary with UI detection results
        """
        if not image_path and not image:
            return {"success": False, "error": "No screenshot provided"}
        
        # Load image if path provided
        if image_path and not image:
            try:
                image = Image.open(image_path)
            except Exception as e:
                return {"success": False, "error": f"Could not open image: {e}"}
        
        # Store results
        results = {
            "success": False,
            "sources": [],
            "ui_elements": [],
            "analyze_timestamp": datetime.now().isoformat()
        }
        
        # Try Gemini first for UI analysis
        if self.available_capabilities["gemini"]:
            try:
                # Use Gemini to analyze UI with specialized prompt
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_bytes = img_byte_arr.getvalue()
                img = {"mime_type": "image/png", "data": img_bytes}
                
                # Get model
                if not hasattr(self, 'vision_model') or not self.vision_model:
                    # Default to 1.5-flash if no model selected yet
                    self.vision_model = "gemini-1.5-flash"
                
                model = genai.GenerativeModel(self.vision_model)
                
                # Specific UI-specific prompt
                ui_prompt = """
                Analyze this UI screenshot and identify all UI elements. For each element provide:
                1. Element type (button, input field, menu, icon, etc.)
                2. Text content (if any)
                3. Approximate position
                4. Purpose or function
                
                Format your response as a clear list of UI elements.
                """
                
                # Generate analysis
                response = model.generate_content([ui_prompt, img])
                
                if hasattr(response, 'text') and response.text:
                    results["ui_description"] = response.text
                    
                    # Try to extract structured UI elements from description
                    if "sources" in results:
                        results["sources"].append("gemini")
                    else:
                        results["sources"] = ["gemini"]
                    
                    # Flag as successful
                    results["success"] = True
            except Exception as e:
                logger.error(f"UI-specific Gemini analysis failed: {e}")
        
        # Try OpenCV if available as fallback or if Gemini didn't find elements
        should_use_opencv = self.available_capabilities["opencv"] and \
                            (not results.get("success", False) or not results.get("ui_elements"))
                            
        if should_use_opencv:
            try:
                # Convert PIL to OpenCV format
                cv_image = np.array(image)
                cv_image = cv_image[:, :, ::-1].copy()  # RGB to BGR
                
                # Simple edge detection to find UI element boundaries
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                
                # Find contours which could be UI elements
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Filter for reasonable UI element sizes
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Filter by size (too small might be noise, too large might be background)
                    if 20 < w < 300 and 10 < h < 100:
                        # Check if it's rectangular (most UI elements are)
                        rect_area = w * h
                        contour_area = cv2.contourArea(contour)
                        if contour_area > 0 and rect_area / contour_area < 2.0:
                            results["ui_elements"].append({
                                "type": "unknown_element",
                                "text": "",
                                "position": {
                                    "x": int(x),
                                    "y": int(y),
                                    "width": int(w),
                                    "height": int(h)
                                }
                            })
                
                if "sources" in results:
                    results["sources"].append("local_vision")
                else:
                    results["sources"] = ["local_vision"]
                
                # Flag as successful if we found any elements
                if len(results["ui_elements"]) > 0:
                    results["success"] = True
            except Exception as e:
                logger.error(f"OpenCV UI analysis failed: {e}")
        
        # Store results if we have vector storage
        if results["success"] and self.vector_storage:
            try:
                await self._store_analysis_results(results, image_path)
            except Exception as e:
                logger.error(f"Error storing analysis results: {e}")
        
        return results
    
    async def extract_text_from_image(self, 
                                    image_path: str = None, 
                                    image: Image.Image = None) -> Dict[str, Any]:
        """
        Extract text from an image
        
        Args:
            image_path: Path to image file
            image: PIL Image object
            
        Returns:
            Dictionary with extracted text
        """
        if not image_path and not image:
            return {"success": False, "error": "No image provided"}
        
        # Load image if path provided
        if image_path and not image:
            try:
                image = Image.open(image_path)
            except Exception as e:
                return {"success": False, "error": f"Could not open image: {e}"}
        
        # Store results
        results = {
            "success": False,
            "sources": [],
            "analyze_timestamp": datetime.now().isoformat()
        }
        
        # Try Gemini first (best quality when available)
        if self.available_capabilities["gemini"]:
            try:
                # Get model
                if not hasattr(self, 'vision_model') or not self.vision_model:
                    # Default to 1.5-flash if no model selected yet
                    self.vision_model = "gemini-1.5-flash"
                
                # Convert PIL image to format suitable for Gemini
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_bytes = img_byte_arr.getvalue()
                img = {"mime_type": "image/png", "data": img_bytes}
                
                # Create model
                model = genai.GenerativeModel(self.vision_model)
                
                # Specific text extraction prompt
                text_prompt = """
                Extract all text visible in this image. 
                Format it preserving the original layout as much as possible.
                Include all text, no matter how small.
                """
                
                # Generate result
                response = model.generate_content([text_prompt, img])
                
                if hasattr(response, 'text') and response.text:
                    results["text"] = {"full_text": response.text}
                    
                    if "sources" in results:
                        results["sources"].append("gemini")
                    else:
                        results["sources"] = ["gemini"]
                    
                    # Flag as successful
                    results["success"] = True
            except Exception as e:
                logger.error(f"Gemini text extraction failed: {e}")
        
        # Try EasyOCR if available and Gemini failed
        should_use_easyocr = self.available_capabilities["easyocr"] and \
                             self.ocr_reader and \
                             not results.get("success", False)
                             
        if should_use_easyocr:
            try:
                if image_path:
                    results_list = self.ocr_reader.readtext(image_path)
                else:
                    # Convert PIL to OpenCV format
                    img_array = np.array(image.convert('RGB'))
                    results_list = self.ocr_reader.readtext(img_array)
                
                if results_list:
                    full_text = "\n".join([text for _, text, _ in results_list])
                    
                    # Extract individual text blocks too
                    text_items = []
                    for box, text, conf in results_list:
                        # Box is a list of 4 points (corners)
                        x_coords = [p[0] for p in box]
                        y_coords = [p[1] for p in box]
                        
                        text_items.append({
                            "text": text,
                            "confidence": conf,
                            "position": {
                                "x": min(x_coords),
                                "y": min(y_coords),
                                "width": max(x_coords) - min(x_coords),
                                "height": max(y_coords) - min(y_coords)
                            }
                        })
                    
                    # Only use EasyOCR if we don't already have text results
                    if "text" not in results:
                        results["text"] = {
                            "full_text": full_text,
                            "items": text_items
                        }
                        
                        if "sources" in results:
                            results["sources"].append("easyocr")
                        else:
                            results["sources"] = ["easyocr"]
                        
                        # Flag as successful
                        results["success"] = True
            except Exception as e:
                logger.error(f"EasyOCR text extraction failed: {e}")
        
        # Try basic OpenCV OCR as last resort
        if not results["success"] and self.available_capabilities["opencv"]:
            try:
                # Convert PIL to OpenCV format
                cv_image = np.array(image)
                cv_image = cv_image[:, :, ::-1].copy()  # RGB to BGR
                
                # Basic preprocessing for OCR
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                
                # This is a placeholder - actual OCR would require tesseract or other OCR engine
                # We just note that OpenCV was used for image processing
                if "sources" in results:
                    results["sources"].append("local_vision")
                else:
                    results["sources"] = ["local_vision"]
            except Exception as e:
                logger.error(f"OpenCV basic OCR failed: {e}")
        
        # Store results if we have vector storage and succeeded
        if results["success"] and self.vector_storage:
            try:
                await self._store_analysis_results(results, image_path)
            except Exception as e:
                logger.error(f"Error storing analysis results: {e}")
        
        return results
    
    async def _store_analysis_results(self, results, image_path=None):
        """Store analysis results in vector storage"""
        if not self.vector_storage:
            return
        
        try:
            # Create a text representation of the analysis
            text_parts = []
            
            # Add image path
            if image_path:
                text_parts.append(f"Image: {os.path.basename(image_path)}")
            
            # Add description if we have one
            if "description" in results:
                text_parts.append(f"Description: {results['description']}")
            
            # Add labels
            if results.get("labels"):
                labels_text = ", ".join([label["description"] for label in results["labels"][:10]])
                text_parts.append(f"Labels: {labels_text}")
            
            # Add text content
            if results.get("text", {}).get("full_text"):
                text_parts.append(f"Text content: {results['text']['full_text']}")
            
            # Add objects
            if results.get("objects"):
                objects_text = ", ".join([obj["name"] for obj in results["objects"][:10]])
                text_parts.append(f"Objects: {objects_text}")
            
            # Create the document to store
            document_text = "\n".join(text_parts)
            
            # Store in vector database - fix the async/await by using proper list handling
            await self.vector_storage.add_texts(
                texts=[document_text],
                metadatas=[{
                    "source": "adaptive_vision",
                    "image_path": image_path,
                    "timestamp": results.get("analyze_timestamp", datetime.now().isoformat()),
                    "analysis_sources": ",".join(results.get("sources", []))
                }],
                ids=[f"img_{datetime.now().strftime('%Y%m%d%H%M%S')}_{os.urandom(4).hex()}"]
            )
            
            logger.info(f"Stored analysis results in vector database")
        except Exception as e:
            logger.error(f"Error storing analysis results: {e}")
    
    def _get_image_bytes(self, image_path):
        """Get image bytes from file path"""
        with open(image_path, 'rb') as f:
            return f.read()
    
    def get_capabilities(self):
        """Get available capabilities"""
        return self.available_capabilities
