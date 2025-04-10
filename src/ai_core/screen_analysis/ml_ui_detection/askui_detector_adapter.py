"""
AskUI Detector Adapter

This module provides an integration with AskUI's vision agent capabilities
for advanced UI element detection and automation.
"""

import logging
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import os
import tempfile
from PIL import Image

# Import necessary interfaces
from ..ui_detection.detector_interface import UIDetectorInterface

logger = logging.getLogger(__name__)

class AskUIDetectorAdapter(UIDetectorInterface):
    """
    Adapter for AskUI's vision-based UI detection capabilities.
    This detector leverages AskUI's advanced computer vision models
    specifically designed for UI element recognition.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the AskUI detector adapter
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.initialized = False
        self.askui_client = None
        self.element_type_map = {
            'button': 'button',
            'checkbox': 'checkbox',
            'radio': 'radio_button',
            'dropdown': 'dropdown',
            'textbox': 'text_field',
            'input': 'text_field',
            'toggle': 'toggle',
            'slider': 'slider',
            'icon': 'icon',
            'menu': 'menu',
            'dialog': 'dialog',
            'link': 'link',
            'image': 'image'
        }
        
        try:
            import askui
            self.has_askui = True
            logger.info("AskUI package is available for UI detection")
            
            # Initialize the AskUI client
            try:
                self.initialize_askui()
                logger.info("AskUI client initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing AskUI client: {e}")
                self.initialized = False
                
        except ImportError:
            self.has_askui = False
            logger.warning("AskUI package not available. To enable advanced UI detection, install with: pip install askui")
            self.initialized = False
    
    def initialize_askui(self):
        """Initialize the AskUI client for UI detection"""
        if self.initialized:
            return True
            
        if not self.has_askui:
            return False
            
        try:
            import askui
            from askui import VisionAgent
            
            # Configure AskUI based on provided config
            api_key = self.config.get('askui_api_key', os.environ.get('ASKUI_API_KEY', ''))
            
            if not api_key:
                logger.warning("No AskUI API key provided. Using free local mode with limited capabilities.")
                # In free mode, we'll use local detection capabilities
                
            # Initialize AskUI client
            self.askui_client = VisionAgent()
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize AskUI client: {e}")
            self.initialized = False
            return False
    
    def detect_elements(self, image: np.ndarray, context: Dict = None) -> List[Dict]:
        """
        Detect UI elements in the image using AskUI
        
        Args:
            image: Input image as a numpy array (BGR format)
            context: Optional context information
            
        Returns:
            List of detected UI elements
        """
        start_time = time.time()
        elements = []
        
        if not self.initialized and not self.initialize_askui():
            logger.error("AskUI detection failed - client not initialized")
            return elements
            
        try:
            # Convert image to RGB (AskUI expects RGB format)
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Convert BGR to RGB if necessary
                rgb_image = image[..., ::-1] if context.get('bgr_format', True) else image
            else:
                logger.error(f"Unsupported image format with shape {image.shape}")
                return elements
                
            # Save image to temporary file for AskUI processing
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
                Image.fromarray(rgb_image).save(temp_path)
            
            try:
                # Use AskUI to analyze the UI elements in the image
                import askui
                
                # Take a screenshot to establish context, then use the saved image
                self.askui_client.capture_screen()
                
                # Use AskUI's vision capabilities to detect UI elements
                results = self.askui_client.detect_ui_elements(image_path=temp_path)
                
                # Process detected elements
                for idx, elem in enumerate(results):
                    element_type = elem.get('type', 'unknown')
                    mapped_type = self.element_type_map.get(element_type, 'unknown')
                    
                    # Extract bounding box in the format (x1, y1, x2, y2)
                    bbox = elem.get('bbox', [0, 0, 0, 0])
                    if len(bbox) == 4:
                        x1, y1, width, height = bbox
                        bbox = (int(x1), int(y1), int(x1 + width), int(y1 + height))
                    
                    # Create element dictionary
                    element = {
                        'id': f'askui_{idx}',
                        'type': mapped_type,
                        'bbox': bbox,
                        'confidence': elem.get('confidence', 0.8),
                        'text': elem.get('text', ''),
                        'attributes': elem.get('attributes', {}),
                        'detection_method': 'askui',
                        'source_detector': 'askui'
                    }
                    
                    elements.append(element)
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            logger.error(f"Error in AskUI detection: {e}")
        
        detection_time = time.time() - start_time
        logger.info(f"AskUI detection completed in {detection_time:.2f}s, found {len(elements)} elements")
        
        return elements
    
    def get_capabilities(self) -> Dict:
        """
        Get detector capabilities
        
        Returns:
            Dictionary of capabilities
        """
        return {
            'name': 'askui_detector',
            'version': '1.0.0',
            'element_types': list(set(self.element_type_map.values())),
            'supports_ocr': True,
            'supports_semantic': True,
            'supports_action': True
        }
    
    def supports_incremental_learning(self) -> bool:
        """Whether this detector supports incremental learning"""
        return False
    
    def get_name(self) -> str:
        """
        Get detector name
        
        Returns:
            Detector name
        """
        return "askui_detector"
