#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-API Vision Detector for UI element detection.
Supports multiple vision APIs including Gemini, Hugging Face, OpenAI, and others.
"""

import os
import sys
import json
import logging
import base64
import requests
import numpy as np
import cv2
from io import BytesIO
from typing import Dict, List, Any, Optional, Tuple, Union
from PIL import Image
from pathlib import Path
import re

# Configure logging
logger = logging.getLogger("adaptive_ui_detection")

class MultiAPIVisionDetector:
    """Vision detector that integrates multiple vision APIs for UI element detection."""
    
    def __init__(self, config_path: str = None):
        """Initialize the multi-API vision detector.
        
        Args:
            config_path: Path to the API configuration file
        """
        self.name = "MultiAPIVisionDetector"
        self.description = "Detects UI elements using multiple vision APIs"
        self.config = self._load_config(config_path)
        self.api_preferences = self.config.get('api_preferences', ['gemini', 'huggingface', 'openai', 'groq'])
        self.active_api = None
        self.initialize_apis()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load API configuration from file."""
        default_config = {
            'api_preferences': ['gemini', 'huggingface', 'openai', 'groq'],
            'gemini': {
                'api_key': os.environ.get('GEMINI_API_KEY', ''),
                'model': 'gemini-1.5-flash'
            },
            'huggingface': {
                'api_key': os.environ.get('HUGGINGFACE_API_KEY', ''),
                'model': 'Salesforce/blip-image-captioning-large'
            },
            'openai': {
                'api_key': os.environ.get('OPENAI_API_KEY', ''),
                'model': 'gpt-4-vision-preview'
            },
            'groq': {
                'api_key': os.environ.get('GROQ_API_KEY', ''),
                'model': 'llama-3-8b-8192'
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    # Merge with default config
                    for key, value in user_config.items():
                        if key in default_config and isinstance(default_config[key], dict):
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error loading API config: {e}")
                
        return default_config
    
    def initialize_apis(self) -> None:
        """Initialize APIs based on available keys and preferences."""
        for api_name in self.api_preferences:
            if api_name in self.config and self.config[api_name].get('api_key'):
                self.active_api = api_name
                logger.info(f"Initialized {api_name.capitalize()} API for vision detection")
                return
                
        logger.warning("No valid API keys found. MultiAPIVisionDetector will not function.")
        self.active_api = None
    
    def detect_ui_elements(self, image: np.ndarray, context: Dict = None) -> List[Dict]:
        """Detect UI elements in the image using the active API.
        
        Args:
            image: Image as numpy array (BGR format)
            context: Optional context information about the image
            
        Returns:
            List of detected UI elements with bounding boxes and metadata
        """
        if not self.active_api:
            logger.warning("No active API configured. Cannot detect UI elements.")
            return []
            
        # Convert image from BGR to RGB if necessary
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
            
        # Dispatch to appropriate API method
        if self.active_api == 'gemini':
            return self._detect_with_gemini(rgb_image, context)
        elif self.active_api == 'huggingface':
            return self._detect_with_huggingface(rgb_image, context)
        elif self.active_api == 'openai':
            return self._detect_with_openai(rgb_image, context)
        elif self.active_api == 'groq':
            return self._detect_with_groq(rgb_image, context)
        else:
            logger.warning(f"Unknown API: {self.active_api}")
            return []
    
    def _encode_image(self, image: np.ndarray) -> str:
        """Encode image as base64 string."""
        pil_image = Image.fromarray(image)
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def _detect_with_gemini(self, image: np.ndarray, context: Dict = None) -> List[Dict]:
        """Detect UI elements using Google's Gemini API."""
        try:
            import google.generativeai as genai
            
            # Configure the Gemini API
            genai.configure(api_key=self.config['gemini']['api_key'])
            model = genai.GenerativeModel(self.config['gemini']['model'])
            
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image)
            
            # Prepare prompt based on context
            prompt = """
            Analyze this screenshot and identify all UI elements present.
            For each element, provide the following information:
            1. Element type (button, text_input, checkbox, dropdown, radio_button, toggle, icon)
            2. Bounding box coordinates (x, y, width, height) as percentages of the image dimensions
            3. Text content (if any)
            4. Function or purpose of the element (if obvious)
            
            Format your response as a JSON array with one object per element, like this:
            [
              {
                "type": "button",
                "x": 0.1,
                "y": 0.2,
                "width": 0.15,
                "height": 0.05,
                "text": "Submit",
                "purpose": "Submit form data"
              },
              ...
            ]
            
            Only output the JSON array, nothing else. Be precise with coordinates.
            """
            
            try:
                # Create a content part for the image
                image_bytes = cv2.imencode(".jpg", image)[1].tobytes()
                image_part = {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(image_bytes).decode('utf-8')
                }
                
                # Make the API call with the appropriate structure for Gemini 1.5
                response = model.generate_content(
                    contents=[prompt, image_part],
                    generation_config={
                        "temperature": 0.2,
                        "max_output_tokens": 4096
                    }
                )
                
                # Extract JSON from the response
                response_text = response.text
                
                # Try to extract JSON from the response
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
                if json_match:
                    json_text = json_match.group(1)
                else:
                    # If no code block, try to extract the JSON directly
                    json_text = response_text.strip()
                    
                # Additional cleanup to handle potential Gemini 1.5 response format differences
                json_text = re.sub(r'^[^[{]*', '', json_text)  # Remove any text before the start of JSON
                json_text = re.sub(r'[^}\]]*$', '', json_text)  # Remove any text after the end of JSON
                    
                # Parse the JSON response
                elements = json.loads(json_text)
                
                # Convert coordinates from percentages to pixels
                image_height, image_width = image.shape[:2]
                for element in elements:
                    if all(k in element for k in ['x', 'y', 'width', 'height']):
                        element['x'] = int(element['x'] * image_width)
                        element['y'] = int(element['y'] * image_height)
                        element['width'] = int(element['width'] * image_width)
                        element['height'] = int(element['height'] * image_height)
                        element['confidence'] = element.get('confidence', 0.8)  # Default confidence
                        element['detector'] = 'gemini'
                        
                return elements
                
            except Exception as e:
                logger.error(f"Error detecting with Gemini API: {e}")
                return []
            
        except Exception as e:
            logger.error(f"Error detecting with Gemini API: {e}")
            return []
    
    def _detect_with_huggingface(self, image: np.ndarray, context: Dict = None) -> List[Dict]:
        """Detect UI elements using Hugging Face API."""
        try:
            # Prepare the image
            encoded_image = self._encode_image(image)
            
            # Prepare request headers
            headers = {
                "Authorization": f"Bearer {self.config['huggingface']['api_key']}"
            }
            
            # Prepare prompt based on context
            prompt = "Analyze this UI screenshot and identify all UI elements. "
            prompt += "For each element, provide: element type (button, text_input, checkbox, etc.), "
            prompt += "bounding box coordinates (x, y, width, height), and any text content. "
            prompt += "Return the result as a JSON array of elements."
            
            if context and 'description' in context:
                prompt += f" Context: {context['description']}"
            
            # API endpoint and payload depend on the model
            if 'layout' in self.config['huggingface']['model'].lower():
                # Use a layout analysis model
                url = f"https://api-inference.huggingface.co/models/{self.config['huggingface']['model']}"
                payload = {"inputs": {"image": encoded_image}}
            else:
                # Use a vision-language model
                url = f"https://api-inference.huggingface.co/models/{self.config['huggingface']['model']}"
                payload = {
                    "inputs": {
                        "image": encoded_image,
                        "prompt": prompt
                    }
                }
            
            # Make the API call
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            
            # Extract elements from the response (format depends on model)
            if isinstance(result, list) and all(isinstance(item, dict) for item in result):
                # Direct JSON output
                elements = result
            elif isinstance(result, dict) and 'generated_text' in result:
                # Text response to parse
                elements = self._extract_json_from_response(result['generated_text'])
            else:
                # Attempt to extract from any text in the response
                elements = self._extract_json_from_response(str(result))
            
            # Process and normalize the elements
            return self._normalize_elements(elements, 'huggingface')
            
        except Exception as e:
            logger.error(f"Error detecting with Hugging Face API: {e}")
            return []
    
    def _detect_with_openai(self, image: np.ndarray, context: Dict = None) -> List[Dict]:
        """Detect UI elements using OpenAI's API."""
        try:
            from openai import OpenAI
            
            # Prepare the client
            client = OpenAI(api_key=self.config['openai']['api_key'])
            
            # Prepare the image
            encoded_image = self._encode_image(image)
            
            # Prepare prompt based on context
            system_prompt = "You are an expert UI element detector. "
            system_prompt += "Analyze the image and identify all UI elements. "
            system_prompt += "For each element, determine its type (button, text_input, checkbox, etc.), "
            system_prompt += "exact bounding box coordinates (x, y, width, height), and any text content."
            
            user_prompt = "Identify all UI elements in this image and return them as a JSON array of elements. "
            user_prompt += "Each element should have: 'type', 'rect' (with x, y, width, height), and 'text' properties."
            
            if context and 'description' in context:
                user_prompt += f" Context: {context['description']}"
            
            # Call the API
            response = client.chat.completions.create(
                model=self.config['openai']['model'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }}
                    ]}
                ],
                response_format={"type": "json_object"}
            )
            
            # Extract elements from the response
            response_content = response.choices[0].message.content
            response_json = json.loads(response_content)
            
            if 'elements' in response_json:
                elements = response_json['elements']
            else:
                elements = response_json.get('ui_elements', [])
                if not elements and isinstance(response_json, list):
                    elements = response_json
            
            # Process and normalize the elements
            return self._normalize_elements(elements, 'openai')
            
        except Exception as e:
            logger.error(f"Error detecting with OpenAI API: {e}")
            return []
    
    def _detect_with_groq(self, image: np.ndarray, context: Dict = None) -> List[Dict]:
        """Detect UI elements using Groq's API."""
        try:
            from groq import Groq
            
            # Prepare the client
            client = Groq(api_key=self.config['groq']['api_key'])
            
            # Prepare the image
            encoded_image = self._encode_image(image)
            
            # Prepare prompt based on context
            system_prompt = "You are an expert UI element detector. "
            system_prompt += "Analyze the image and identify all UI elements. "
            system_prompt += "For each element, determine its type (button, text_input, checkbox, etc.), "
            system_prompt += "exact bounding box coordinates (x, y, width, height), and any text content."
            
            user_prompt = "Identify all UI elements in this image and return them as a JSON array of elements. "
            user_prompt += "Each element should have: 'type', 'rect' (with x, y, width, height), and 'text' properties."
            user_prompt += "\n\nHere is the image base64 encoded: data:image/jpeg;base64," + encoded_image
            
            if context and 'description' in context:
                user_prompt += f"\n\nContext: {context['description']}"
            
            # Call the API
            response = client.chat.completions.create(
                model=self.config['groq']['model'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            # Extract elements from the response
            response_content = response.choices[0].message.content
            elements = self._extract_json_from_response(response_content)
            
            # Process and normalize the elements
            return self._normalize_elements(elements, 'groq')
            
        except Exception as e:
            logger.error(f"Error detecting with Groq API: {e}")
            return []
    
    def _extract_json_from_response(self, response_text: str) -> List[Dict]:
        """Extract JSON data from a text response."""
        try:
            # Find JSON array in the response
            import re
            json_pattern = r'\[\s*{[^\[\]]*}\s*(,\s*{[^\[\]]*}\s*)*\]'
            json_matches = re.findall(json_pattern, response_text)
            
            if json_matches:
                return json.loads(json_matches[0])
            
            # Look for JSON object with elements array
            json_obj_pattern = r'{[^{}]*"elements"\s*:\s*\[[^{}]*\][^{}]*}'
            obj_matches = re.findall(json_obj_pattern, response_text)
            
            if obj_matches:
                obj = json.loads(obj_matches[0])
                if 'elements' in obj:
                    return obj['elements']
            
            # Try parsing the entire response as JSON
            try:
                parsed = json.loads(response_text)
                if isinstance(parsed, list):
                    return parsed
                elif isinstance(parsed, dict) and 'elements' in parsed:
                    return parsed['elements']
                elif isinstance(parsed, dict) and 'ui_elements' in parsed:
                    return parsed['ui_elements']
            except:
                pass
                
            # Fallback: look for anything that might be a UI element description
            elements = []
            element_types = ['button', 'text_input', 'checkbox', 'radio', 'dropdown', 'toggle', 'icon']
            
            for element_type in element_types:
                pattern = f'["\']?type["\']?\\s*:\\s*["\']?{element_type}["\']?'
                if re.search(pattern, response_text, re.IGNORECASE):
                    # Extract individual element JSON objects
                    element_pattern = r'{[^{}]*' + f'["\']?type["\']?\\s*:\\s*["\']?{element_type}["\']?' + r'[^{}]*}'
                    element_matches = re.findall(element_pattern, response_text, re.IGNORECASE)
                    
                    for match in element_matches:
                        try:
                            element = json.loads(match)
                            elements.append(element)
                        except:
                            continue
            
            return elements
            
        except Exception as e:
            logger.error(f"Error extracting JSON from response: {e}")
            return []
    
    def _normalize_elements(self, elements: List[Dict], api_source: str) -> List[Dict]:
        """Normalize element data to a consistent format."""
        normalized = []
        
        for i, elem in enumerate(elements):
            try:
                element_type = elem.get('type', 'unknown').lower()
                
                # Validate and normalize element type
                valid_types = ['button', 'text_input', 'checkbox', 'radio', 'dropdown', 'toggle', 'icon', 'label']
                if element_type not in valid_types:
                    # Try to map to valid type
                    if 'button' in element_type or 'btn' in element_type:
                        element_type = 'button'
                    elif 'input' in element_type or 'field' in element_type or 'text' in element_type:
                        element_type = 'text_input'
                    elif 'check' in element_type:
                        element_type = 'checkbox'
                    elif 'radio' in element_type:
                        element_type = 'radio'
                    elif 'select' in element_type or 'dropdown' in element_type or 'combo' in element_type:
                        element_type = 'dropdown'
                    elif 'toggle' in element_type or 'switch' in element_type:
                        element_type = 'toggle'
                    elif 'icon' in element_type or 'image' in element_type:
                        element_type = 'icon'
                    else:
                        element_type = 'unknown'
                
                # Extract bounding box
                rect = {}
                if 'rect' in elem and isinstance(elem['rect'], dict):
                    rect = elem['rect']
                elif 'bbox' in elem and isinstance(elem['bbox'], dict):
                    rect = {
                        'x': elem['bbox'].get('x', 0),
                        'y': elem['bbox'].get('y', 0),
                        'width': elem['bbox'].get('width', 0),
                        'height': elem['bbox'].get('height', 0)
                    }
                elif all(k in elem for k in ['x', 'y', 'width', 'height']):
                    rect = {
                        'x': elem.get('x', 0),
                        'y': elem.get('y', 0),
                        'width': elem.get('width', 0),
                        'height': elem.get('height', 0)
                    }
                elif all(k in elem for k in ['x', 'y', 'w', 'h']):
                    rect = {
                        'x': elem.get('x', 0),
                        'y': elem.get('y', 0),
                        'width': elem.get('w', 0),
                        'height': elem.get('h', 0)
                    }
                else:
                    # Default to empty values if no coordinates found
                    rect = {'x': 0, 'y': 0, 'width': 0, 'height': 0}
                
                # Ensure values are integers
                for k in rect:
                    rect[k] = int(float(rect[k]))
                
                # Extract text content
                text = elem.get('text', '')
                if not text and 'content' in elem:
                    text = elem.get('content', '')
                    
                # Calculate confidence
                confidence = elem.get('confidence', 0.8)
                if not isinstance(confidence, (int, float)):
                    try:
                        confidence = float(confidence)
                    except:
                        confidence = 0.8
                
                # Create normalized element
                normalized_elem = {
                    'id': f"{api_source}_{element_type}_{i}",
                    'type': element_type,
                    'rect': rect,
                    'text': text,
                    'confidence': confidence,
                    'source_detector': f"{api_source}"
                }
                
                # Add any additional properties
                for k, v in elem.items():
                    if k not in ['id', 'type', 'rect', 'text', 'confidence', 'source_detector']:
                        normalized_elem[k] = v
                
                normalized.append(normalized_elem)
                
            except Exception as e:
                logger.warning(f"Error normalizing element: {e}")
                continue
                
        return normalized
    
    def supports_incremental_learning(self) -> bool:
        """Whether this detector supports incremental learning."""
        return True
    
    def provide_feedback(self, detection_results: List[Dict], correct_results: List[Dict]) -> None:
        """Provide feedback to improve detection in future iterations."""
        # Currently only logging feedback, but this could be extended to fine-tune models
        logger.info(f"Received feedback for {len(detection_results)} detections")
        
    def detect_elements(self, image: np.ndarray, context: Dict = None) -> List[Dict]:
        """Detect UI elements in the image.
        
        Args:
            image: Image as numpy array
            context: Optional context information
            
        Returns:
            List of detected UI elements
        """
        return self.detect_ui_elements(image, context)
