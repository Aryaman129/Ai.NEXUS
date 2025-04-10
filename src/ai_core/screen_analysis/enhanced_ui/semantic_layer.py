"""
Semantic Understanding Layer for UI Intelligence

This module provides advanced semantic understanding of UI elements
by integrating LLMs and OCR to derive meaning, purpose, and relationships.
"""

import os
import logging
import json
import base64
import io
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import time
import threading
import queue
from PIL import Image
import traceback

# Import vision_layer for UIElement
from .vision_layer import UIElement

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import OCR capabilities
try:
    import pytesseract
    OCR_AVAILABLE = True
    logger.info("Tesseract OCR available")
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("Tesseract OCR not available. Install pytesseract for text recognition.")

class APIConnector:
    """
    Base class for connecting to AI APIs for semantic understanding.
    
    This provides a common interface for different LLM services
    that can be used for UI understanding.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.name = "base_connector"
        
    def analyze_ui_elements(self, 
                          image: np.ndarray, 
                          elements: List[UIElement]) -> List[UIElement]:
        """Analyze UI elements using this API service"""
        # Base implementation does nothing
        logger.warning(f"{self.name} analyze_ui_elements not implemented")
        return elements
    
    def extract_text(self, 
                  image: np.ndarray, 
                  element: UIElement) -> str:
        """Extract text from an element"""
        # Base implementation does nothing
        return ""
    
    def understand_ui_structure(self, 
                             image: np.ndarray, 
                             elements: List[UIElement]) -> Dict:
        """Understand the overall UI structure and relationships"""
        # Base implementation does nothing
        return {}
    
    def _prepare_image(self, image: np.ndarray) -> str:
        """Convert an image to base64 for API transmission"""
        _, buffer = cv2.imencode('.png', image)
        return base64.b64encode(buffer).decode('utf-8')


class TogetherAIConnector(APIConnector):
    """
    Connector for Together AI API for UI understanding.
    
    This leverages Together AI's powerful language models
    for deeper UI understanding and analysis.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.name = "together_ai"
        # NOTE: Securely handle API key retrieval
        self.api_key = self.config.get("api_key") or os.environ.get("TOGETHER_AI_API_KEY") or "4ec34405c082ae11d558aabe290486bd73ae6497fb623ba0bba481df21f5ec39"
        
        # Ensure API key is properly formatted
        self.api_key = self.api_key.strip()
        
        # Use a model that doesn't require special permissions
        self.model = self.config.get("model", "mistralai/Mixtral-8x7B-Instruct-v0.1")
        
        # Import requests lazily to avoid dependency issues
        try:
            import requests
            self.requests = requests
            self.session = requests.Session()
            self._available = True
            logger.info(f"Together AI connector initialized with model: {self.model}")
        except ImportError:
            self._available = False
            logger.warning("Requests library not available. Together AI connector disabled.")
    
    def analyze_ui_elements(self, 
                          image: np.ndarray, 
                          elements: List[UIElement]) -> List[UIElement]:
        """
        Analyze UI elements using Together AI to enrich with semantic understanding.
        
        Args:
            image: Screenshot containing UI elements
            elements: List of detected UI elements
            
        Returns:
            Enhanced list of UI elements with semantic information
        """
        if not self._available or not elements:
            return elements
        
        try:
            # Group elements by type for more efficient analysis
            elements_by_type = {}
            for element in elements:
                if element.element_type not in elements_by_type:
                    elements_by_type[element.element_type] = []
                elements_by_type[element.element_type].append(element)
            
            # Create content prompt for the LLM
            prompt = self._create_ui_analysis_prompt(image, elements_by_type)
            
            # Call Together AI API
            response = self._call_together_api(prompt)
            
            # Parse response and enhance elements
            enhanced_elements = self._parse_ui_analysis(elements, response)
            
            logger.info(f"Together AI enhanced {len(enhanced_elements)} UI elements with semantic understanding")
            return enhanced_elements
            
        except Exception as e:
            logger.error(f"Error in Together AI analysis: {e}")
            logger.debug(traceback.format_exc())
            return elements
    
    def extract_text(self, 
                  image: np.ndarray, 
                  element: UIElement) -> str:
        """
        Extract text from a UI element using Together AI's vision capabilities.
        
        Args:
            image: Screenshot containing UI element
            element: UI element to extract text from
            
        Returns:
            Extracted text
        """
        if not self._available:
            return ""
            
        try:
            # Crop the element from the image
            x1, y1, x2, y2 = element.bbox
            element_image = image[y1:y2, x1:x2]
            
            # Skip if element is too small
            if element_image.shape[0] < 5 or element_image.shape[1] < 5:
                return ""
            
            # Create prompt for text extraction
            prompt = self._create_text_extraction_prompt(element_image, element)
            
            # Call Together AI API
            response = self._call_together_api(prompt)
            
            # Extract text from response
            text = self._parse_text_extraction(response)
            
            logger.debug(f"Extracted text from {element.element_type}: '{text}'")
            return text
            
        except Exception as e:
            logger.error(f"Error in text extraction: {e}")
            return ""
    
    def understand_ui_structure(self, 
                             image: np.ndarray, 
                             elements: List[UIElement]) -> Dict:
        """
        Understand the overall UI structure using Together AI.
        
        Args:
            image: Screenshot of the UI
            elements: Detected UI elements
            
        Returns:
            Dictionary containing structural understanding
        """
        if not self._available or not elements:
            return {}
            
        try:
            # Create prompt for structure analysis
            prompt = self._create_structure_analysis_prompt(image, elements)
            
            # Call Together AI API
            response = self._call_together_api(prompt)
            
            # Parse structure from response
            structure = self._parse_structure_analysis(response)
            
            logger.info(f"Generated UI structure understanding with {len(structure)} components")
            return structure
            
        except Exception as e:
            logger.error(f"Error in UI structure analysis: {e}")
            return {}
    
    def _create_ui_analysis_prompt(self, 
                                image: np.ndarray, 
                                elements_by_type: Dict[str, List[UIElement]]) -> str:
        """Create a prompt for analyzing UI elements"""
        # Convert elements to simplified format for the prompt
        simplified_elements = []
        for element_type, elements in elements_by_type.items():
            type_elements = [
                {
                    "id": e.element_id,
                    "type": e.element_type,
                    "bbox": e.bbox,
                    "confidence": round(e.confidence, 2)
                }
                for e in elements[:10]  # Limit to 10 elements per type
            ]
            simplified_elements.extend(type_elements)
        
        # Create the prompt
        prompt = """You are an expert in UI analysis and understanding. I'm providing you with detected UI elements from a screenshot.
For each element, analyze its likely purpose, function, and role in the interface.

Here are the detected elements:
"""
        prompt += json.dumps(simplified_elements, indent=2)
        
        prompt += """
Please analyze these elements and provide the following information:
1. Enhanced classification - is the detected element type correct? If not, what would be better?
2. Purpose - what is the likely function of this element in the interface?
3. State - is the element active, disabled, selected, etc.?
4. Content - what content might this element contain or represent?
5. Relationships - how might this element relate to others?

Return your analysis in JSON format:
{
  "elements": [
    {
      "id": "element_id",
      "enhanced_type": "more_specific_type",
      "purpose": "description of purpose",
      "state": "active/inactive/etc",
      "content": "content description",
      "relationships": ["related_to_id1", "related_to_id2"]
    },
    ...
  ]
}
"""
        return prompt
    
    def _create_text_extraction_prompt(self, 
                                    element_image: np.ndarray, 
                                    element: UIElement) -> str:
        """Create a prompt for extracting text from an element"""
        # Convert element image to base64
        element_base64 = self._prepare_image(element_image)
        
        # Create prompt
        prompt = f"""You are an expert in OCR and UI text extraction. I'm providing an image of a UI element.
Element type: {element.element_type}
Confidence: {element.confidence}

Extract ONLY the text content from this UI element image.
If there's no text, respond with "NO_TEXT".
Be precise and concise. Return ONLY the text, nothing else.

[Image: data:image/png;base64,{element_base64}]
"""
        return prompt
    
    def _create_structure_analysis_prompt(self, 
                                       image: np.ndarray, 
                                       elements: List[UIElement]) -> str:
        """Create a prompt for analyzing overall UI structure"""
        # Convert to base64
        image_base64 = self._prepare_image(image)
        
        # Simplify elements for the prompt
        simplified_elements = [
            {
                "id": e.element_id,
                "type": e.element_type,
                "bbox": e.bbox,
                "text": e.text if hasattr(e, "text") and e.text else ""
            }
            for e in elements[:50]  # Limit to 50 elements to avoid token limits
        ]
        
        # Create prompt
        prompt = """You are an expert in UI analysis. I'm providing a screenshot with detected UI elements.
Analyze the overall structure and organization of this interface.

Here are the detected elements:
"""
        prompt += json.dumps(simplified_elements, indent=2)
        
        prompt += """
[Image: data:image/png;base64,""" + image_base64 + """]

Please provide an analysis of the UI structure including:
1. Application type - what kind of application is this likely to be?
2. Layout structure - how is the interface organized (e.g., navigation, content areas)?
3. Key components - what are the main functional areas?
4. Interaction flow - how would a user typically interact with this interface?
5. Hierarchy - what is the visual and functional hierarchy?

Return your analysis in JSON format:
{
  "application_type": "type of application",
  "layout_structure": {
    "description": "overall layout description",
    "components": [
      {
        "name": "component name",
        "purpose": "component purpose",
        "elements": ["element_id1", "element_id2"]
      },
      ...
    ]
  },
  "interaction_flow": ["step1", "step2", ...],
  "hierarchy": {
    "primary": ["element_id1", "element_id2"],
    "secondary": ["element_id3", "element_id4"],
    "tertiary": ["element_id5", "element_id6"]
  }
}
"""
        return prompt
    
    def _call_together_api(self, prompt: str) -> str:
        """Call the Together AI API with the given prompt"""
        if not self._available:
            return ""
            
        try:
            url = "https://api.together.xyz/v1/completions"
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": 1024,
                "temperature": 0.2,
                "top_p": 0.7,
                "top_k": 50,
                "repetition_penalty": 1,
                "stop": ["</answer>", "Human:", "USER:"]
            }
            
            headers = {
                "accept": "application/json",
                "content-type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            response = self.session.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            return result.get("choices", [{}])[0].get("text", "")
            
        except Exception as e:
            logger.error(f"Error calling Together AI API: {e}")
            return ""
    
    def _parse_ui_analysis(self, 
                         elements: List[UIElement], 
                         response: str) -> List[UIElement]:
        """Parse the LLM response and enhance the UI elements"""
        if not response:
            return elements
            
        try:
            # Extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                analysis = json.loads(json_str)
            else:
                return elements
                
            # Extract element analysis
            element_analysis = analysis.get("elements", [])
            
            # Create mapping from element_id to original element
            element_map = {e.element_id: e for e in elements}
            
            # Update elements with enhanced information
            for analysis_item in element_analysis:
                element_id = analysis_item.get("id")
                if element_id in element_map:
                    element = element_map[element_id]
                    
                    # Update element type if enhanced type is provided
                    enhanced_type = analysis_item.get("enhanced_type")
                    if enhanced_type and enhanced_type != element.element_type:
                        logger.debug(f"Enhanced type for {element_id}: {element.element_type} -> {enhanced_type}")
                        element.element_type = enhanced_type
                    
                    # Add purpose to attributes
                    purpose = analysis_item.get("purpose")
                    if purpose:
                        element.attributes["purpose"] = purpose
                    
                    # Add state information
                    state = analysis_item.get("state")
                    if state:
                        element.state["detected_state"] = state
                    
                    # Add content description
                    content = analysis_item.get("content")
                    if content:
                        element.attributes["content_description"] = content
                    
                    # Add relationships
                    relationships = analysis_item.get("relationships", [])
                    if relationships:
                        element.attributes["semantic_relationships"] = relationships
            
            return elements
            
        except Exception as e:
            logger.error(f"Error parsing UI analysis: {e}")
            return elements
    
    def _parse_text_extraction(self, response: str) -> str:
        """Parse text extraction response"""
        if not response:
            return ""
            
        # Clean up and normalize text
        text = response.strip()
        if text == "NO_TEXT":
            return ""
            
        return text
    
    def _parse_structure_analysis(self, response: str) -> Dict:
        """Parse the structure analysis response"""
        if not response:
            return {}
            
        try:
            # Extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                structure = json.loads(json_str)
                return structure
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Error parsing structure analysis: {e}")
            return {}


class OCRService:
    """
    Provides OCR capabilities for extracting text from UI elements.
    
    This service uses Tesseract OCR when available and falls back
    to alternative methods when needed.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the OCR service"""
        self.config = config or {}
        
        # Tesseract configuration
        self.tesseract_config = self.config.get("tesseract_config", {
            "lang": "eng",
            "config": "--psm 6"  # Assume single line of text
        })
        
        # Check if OCR is available
        self.ocr_available = OCR_AVAILABLE
        if OCR_AVAILABLE:
            # Check if tesseract path is configured
            tesseract_path = self.config.get("tesseract_path")
            if tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
                
            logger.info("OCR service initialized with Tesseract")
        else:
            logger.warning("OCR service initialized without Tesseract. Text extraction will be limited.")
    
    def extract_text(self, 
                  image: np.ndarray, 
                  element: UIElement) -> str:
        """
        Extract text from a UI element.
        
        Args:
            image: Screenshot containing the UI element
            element: UI element to extract text from
            
        Returns:
            Extracted text
        """
        # Extract element from image
        x1, y1, x2, y2 = element.bbox
        element_image = image[y1:y2, x1:x2]
        
        # Skip if element is too small
        if element_image.shape[0] < 5 or element_image.shape[1] < 5:
            return ""
        
        # Extract text using Tesseract if available
        if self.ocr_available:
            try:
                # Preprocess image for better OCR
                processed_image = self._preprocess_for_ocr(element_image, element.element_type)
                
                # Extract text
                text = pytesseract.image_to_string(
                    processed_image,
                    lang=self.tesseract_config["lang"],
                    config=self.tesseract_config["config"]
                )
                
                # Clean up text
                text = self._clean_text(text)
                
                logger.debug(f"OCR extracted text: '{text}'")
                return text
            except Exception as e:
                logger.error(f"OCR error: {e}")
                return ""
        else:
            return ""
    
    def _preprocess_for_ocr(self, 
                         image: np.ndarray, 
                         element_type: str) -> np.ndarray:
        """
        Preprocess an image for OCR based on element type.
        
        Different element types may require different preprocessing
        for optimal OCR results.
        """
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply different preprocessing based on element type
        if element_type in ["text_field", "label", "text"]:
            # For text elements, apply mild enhancement
            # Adjust brightness and contrast
            alpha = 1.2  # Contrast control
            beta = 10    # Brightness control
            gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
            
            # Apply Gaussian blur to reduce noise
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Apply adaptive thresholding
            gray = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
        elif element_type in ["button", "menu_item"]:
            # For buttons and menu items, apply stronger enhancement
            # Sharpen the image
            kernel = np.array([[-1, -1, -1], 
                             [-1, 9, -1], 
                             [-1, -1, -1]])
            gray = cv2.filter2D(gray, -1, kernel)
            
            # Apply Otsu's thresholding
            _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
        else:
            # Default preprocessing for other elements
            # Apply basic thresholding
            _, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        return gray
    
    def _clean_text(self, text: str) -> str:
        """Clean up OCR text results"""
        if not text:
            return ""
            
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Replace multiple spaces with a single space
        text = ' '.join(text.split())
        
        # Remove special characters often misinterpreted by OCR
        text = text.replace('|', 'I').replace('1', 'I', )
        
        return text


class SemanticAnalyzer:
    """
    Provides semantic understanding of UI elements by integrating
    visual detection, OCR, and AI-based analysis.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the semantic analyzer"""
        self.config = config or {}
        
        # Initialize OCR service
        self.ocr = OCRService(self.config.get("ocr_config"))
        
        # Initialize AI connectors
        self.connector_configs = self.config.get("connectors", {})
        self.connectors = []
        
        # Add Together AI connector if available
        together_config = self.connector_configs.get("together_ai")
        if together_config is not None:
            self.connectors.append(TogetherAIConnector(together_config))
        
        # Initialize processing queue and thread
        self.processing_queue = queue.Queue()
        self.processing_thread = None
        self.processing_active = False
        
        logger.info(f"SemanticAnalyzer initialized with {len(self.connectors)} AI connectors")
    
    def analyze_elements(self, 
                       image: np.ndarray, 
                       elements: List[UIElement],
                       async_mode: bool = False) -> List[UIElement]:
        """
        Analyze UI elements to add semantic understanding.
        
        Args:
            image: Screenshot containing UI elements
            elements: Detected UI elements to analyze
            async_mode: If True, process in background thread
            
        Returns:
            Enhanced list of UI elements with semantic information
        """
        if async_mode:
            # Process asynchronously
            self.processing_queue.put((image, elements))
            
            # Start processing thread if not already running
            if not self.processing_active:
                self._start_processing_thread()
                
            # Return original elements immediately
            return elements
        else:
            # Process synchronously
            return self._process_elements(image, elements)
    
    def extract_text_from_elements(self, 
                                image: np.ndarray, 
                                elements: List[UIElement]) -> List[UIElement]:
        """
        Extract text from UI elements.
        
        Args:
            image: Screenshot containing UI elements
            elements: UI elements to extract text from
            
        Returns:
            UI elements with extracted text
        """
        text_elements = []
        
        # Process elements that are likely to contain text
        text_element_types = [
            "text_field", "label", "button", "menu_item",
            "link", "checkbox", "radio_button", "text"
        ]
        
        for element in elements:
            if element.element_type in text_element_types:
                # Extract text using OCR
                text = self.ocr.extract_text(image, element)
                
                # Update element with extracted text
                if text:
                    element.text = text
                    text_elements.append(element)
        
        logger.info(f"Extracted text from {len(text_elements)} elements")
        return elements
    
    def understand_ui_structure(self, 
                             image: np.ndarray, 
                             elements: List[UIElement]) -> Dict:
        """
        Understand the overall UI structure.
        
        Args:
            image: Screenshot of the UI
            elements: Detected UI elements
            
        Returns:
            Dictionary containing structural understanding
        """
        # Use the first available connector for structure analysis
        for connector in self.connectors:
            structure = connector.understand_ui_structure(image, elements)
            if structure:
                return structure
        
        # If no connector provided a structure, return empty
        return {}
    
    def _process_elements(self, 
                       image: np.ndarray, 
                       elements: List[UIElement]) -> List[UIElement]:
        """Process elements to add semantic understanding"""
        if not elements:
            return elements
            
        start_time = time.time()
        
        # Extract text first
        elements = self.extract_text_from_elements(image, elements)
        
        # Use AI connectors for deeper analysis
        enhanced_elements = elements
        for connector in self.connectors:
            enhanced_elements = connector.analyze_ui_elements(image, enhanced_elements)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Semantic analysis completed in {elapsed_time:.3f}s")
        
        return enhanced_elements
    
    def _start_processing_thread(self):
        """Start the background processing thread"""
        self.processing_active = True
        self.processing_thread = threading.Thread(
            target=self._processing_worker,
            daemon=True
        )
        self.processing_thread.start()
    
    def _processing_worker(self):
        """Background worker for processing queue"""
        logger.info("Semantic analysis background worker started")
        
        while self.processing_active:
            try:
                # Get item from queue with timeout
                try:
                    image, elements = self.processing_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # Process elements
                self._process_elements(image, elements)
                
                # Mark task as done
                self.processing_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in semantic analysis worker: {e}")
        
        logger.info("Semantic analysis background worker stopped")
    
    def stop(self):
        """Stop the processing thread"""
        self.processing_active = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
            self.processing_thread = None
