"""
Multimodal Detection Coordinator

This module provides a coordinator that leverages multiple UI detection models 
and combines them with advanced vision-language models for semantic understanding.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import cv2
from dataclasses import dataclass, field

# Import the detector interface
from ..ui_detection.detector_interface import UIDetectorInterface
from .enhanced_detector_registry import EnhancedDetectorRegistry

logger = logging.getLogger(__name__)

@dataclass
class DetectionRequest:
    """Represents a UI element detection request"""
    image: np.ndarray
    context: Dict = field(default_factory=dict)
    priority_detectors: List[str] = field(default_factory=list)
    min_confidence: float = 0.3
    max_detections: int = 100
    semantic_analysis: bool = True

@dataclass
class DetectionResult:
    """Represents the result of a detection operation"""
    elements: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    performance: Dict = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

class MultimodalDetectionCoordinator:
    """
    Coordinates multiple detection models, combines their results,
    and enhances them with semantic understanding using vision-language models.
    """
    
    def __init__(self, registry: Optional[EnhancedDetectorRegistry] = None, config: Optional[Dict] = None):
        """
        Initialize the multimodal detection coordinator
        
        Args:
            registry: Optional enhanced detector registry
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.registry = registry
        self.vlm_initialized = False
        self.vlm_model = None
        self.confidence_threshold = self.config.get("confidence_threshold", 0.3)
        self.max_elements = self.config.get("max_elements", 100)
        self.semantic_fusion_enabled = self.config.get("semantic_fusion_enabled", True)
        
        # Initialize performance tracking
        self.performance_history = []
        
        # Try to initialize the registry if not provided
        if not self.registry:
            try:
                self.registry = EnhancedDetectorRegistry()
                logger.info("Created new EnhancedDetectorRegistry")
            except Exception as e:
                logger.error(f"Failed to create EnhancedDetectorRegistry: {e}")
    
    def initialize_vlm(self) -> bool:
        """
        Initialize the vision-language model for semantic understanding
        with adaptive selection based on available resources
        
        Returns:
            Success status
        """
        if self.vlm_initialized and self.vlm_model:
            return True
            
        try:
            # Try to import vision-language models
            try:
                from transformers import AutoProcessor, AutoModelForVision2Seq
                import torch
                import os
                import psutil
                
                # Adaptive model selection based on available resources
                available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
                gpu_available = torch.cuda.is_available()
                gpu_memory_gb = 0
                
                if gpu_available:
                    try:
                        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                    except:
                        gpu_memory_gb = 0
                
                logger.info(f"Available system memory: {available_memory_gb:.2f} GB, GPU: {gpu_available}, GPU memory: {gpu_memory_gb:.2f} GB")
                
                # Model selection tiers based on available resources - verified model identifiers
                large_models = ["Salesforce/blip2-opt-2.7b"]
                medium_models = ["Salesforce/blip-image-captioning-large", "microsoft/git-large"]
                small_models = ["Salesforce/blip-image-captioning-base", "microsoft/git-base"]
                tiny_models = ["nlpconnect/vit-gpt2-image-captioning"]
                
                # Select appropriate model tier
                selected_models = tiny_models
                if gpu_available and gpu_memory_gb > 12 or available_memory_gb > 16:
                    selected_models = large_models
                elif gpu_available and gpu_memory_gb > 6 or available_memory_gb > 8:
                    selected_models = medium_models
                elif gpu_available and gpu_memory_gb > 4 or available_memory_gb > 4:
                    selected_models = small_models
                
                # Override with user config if specified
                model_name = self.config.get("vlm_model", None)
                if not model_name:
                    model_name = selected_models[0]
                    logger.info(f"Auto-selected model based on resources: {model_name}")
                
                vlm_device = "cuda" if gpu_available else "cpu"
                
                # Configure memory optimization options based on available resources
                memory_config = {}
                if vlm_device == "cuda" and gpu_memory_gb < 8:
                    memory_config = {"device_map": "auto", "load_in_8bit": True}
                    logger.info("Using 8-bit quantization for memory optimization")
                elif vlm_device == "cpu" and available_memory_gb < 8:
                    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
                    logger.info("Using memory optimization settings for CPU")
                
                logger.info(f"Loading vision-language model: {model_name} on {vlm_device}")
                
                # Try loading with memory optimizations if specified
                if memory_config:
                    self.vlm_processor = AutoProcessor.from_pretrained(model_name)
                    self.vlm_model = AutoModelForVision2Seq.from_pretrained(model_name, **memory_config)
                else:
                    self.vlm_processor = AutoProcessor.from_pretrained(model_name)
                    self.vlm_model = AutoModelForVision2Seq.from_pretrained(model_name).to(vlm_device)
                
                self.vlm_initialized = True
                return True
                
            except ImportError:
                # Try alternate models if transformers not available
                try:
                    import requests
                    from PIL import Image
                    import base64
                    import json
                    import io
                    
                    # Set up API-based VLM using available services
                    # Try multiple APIs in sequence for fallback
                    self.vlm_apis = [
                        {
                            "name": "together_ai",
                            "endpoint": self.config.get("together_ai_endpoint", "https://api.together.xyz/v1/vision/chat/completions"),
                            "api_key": self.config.get("together_ai_key", ""),
                            "enabled": self.config.get("use_together_ai", True)
                        },
                        {
                            "name": "gemini",
                            "endpoint": self.config.get("gemini_endpoint", "https://generativelanguage.googleapis.com/v1/models/gemini-pro-vision:generateContent"),
                            "api_key": self.config.get("gemini_api_key", ""),
                            "enabled": self.config.get("use_gemini", True)
                        },
                        {
                            "name": "custom_api",
                            "endpoint": self.config.get("vlm_api_endpoint", ""),
                            "api_key": self.config.get("vlm_api_key", ""),
                            "enabled": self.config.get("use_custom_api", False)
                        }
                    ]
                    
                    # Check if any APIs are available
                    self.available_apis = [api for api in self.vlm_apis if api['enabled']]
                    apis_with_keys = [api for api in self.available_apis if api['api_key']]
                    
                    if apis_with_keys:
                        logger.info(f"Using cloud APIs for vision-language processing: {[api['name'] for api in apis_with_keys]}")
                        self.using_cloud_api = True
                        self.vlm_initialized = True
                        return True
                    elif self.available_apis:
                        # We have APIs configured but no keys
                        logger.warning(f"Cloud APIs are configured but missing API keys: {[api['name'] for api in self.available_apis]}")
                        logger.warning("Consider adding API keys in configuration to enable cloud vision-language processing")
                        self.using_cloud_api = False
                        return False
                    else:
                        logger.warning("No VLM API endpoints configured for semantic analysis")
                        return False
                        
                    self.vlm_initialized = True
                    logger.info(f"Using API-based vision-language model")
                    return True
                    
                except ImportError:
                    logger.warning("No vision-language model dependencies available")
                    return False
        except Exception as e:
            logger.error(f"Error initializing vision-language model: {e}")
            return False
    
    def detect(self, request: DetectionRequest) -> DetectionResult:
        """
        Perform UI element detection using multiple models and semantic understanding
        with adaptive learning to improve over time
        
        Args:
            request: Detection request object
            
        Returns:
            Detection result object
        """
        result = DetectionResult()
        start_time = time.time()
        
        # Ensure we have a registry
        if not self.registry:
            error_msg = "No detector registry available"
            logger.error(error_msg)
            result.errors.append(error_msg)
            return result
        
        # Get available detectors with adaptive selection
        # Check if we have context information to use for context-aware selection
        if request.context and 'element_type' in request.context:
            # Context-aware detector selection
            target_type = request.context['element_type']
            try:
                best_detector_name = self.registry.select_detector_for_context(target_type)
                if best_detector_name:
                    logger.info(f"Adaptively selected {best_detector_name} for {target_type} detection")
                    request.priority_detectors = [best_detector_name] + request.priority_detectors
            except Exception as e:
                logger.warning(f"Error in adaptive detector selection: {e}")
                
        # Get detectors with priorities applied
        detectors = self._get_detectors(request.priority_detectors)
        if not detectors:
            error_msg = "No UI element detectors available"
            logger.error(error_msg)
            result.errors.append(error_msg)
            return result
        
        # Run detection with adaptive strategy
        all_elements = []
        detector_performance = {}
        failed_detectors = []
        success_level = 0  # Used to track overall detection success
        
        # First try primary detectors 
        for detector_name, detector in detectors.items():
            detector_start = time.time()
            try:
                # Detect elements
                elements = detector.detect_elements(request.image, request.context)
                detector_end = time.time()
                
                # Filter by confidence
                confident_elements = [e for e in elements if e.get("confidence", 0) >= request.min_confidence]
                
                # Track performance including the success level
                detector_time = detector_end - detector_start
                success = len(confident_elements) > 0
                success_level = max(success_level, 1 if success else 0)
                
                # Add detector info to elements
                for element in confident_elements:
                    if "source_detector" not in element:
                        element["source_detector"] = detector_name
                
                # Add to all elements
                all_elements.extend(confident_elements)
                
                # Track performance with extended metrics for learning
                detector_performance[detector_name] = {
                    "time": detector_time,
                    "element_count": len(confident_elements),
                    "avg_confidence": sum(e.get("confidence", 0) for e in confident_elements) / len(confident_elements) if confident_elements else 0,
                    "success": success,
                    "failure_mode": "none" if success else "no_elements_detected"
                }
                
                logger.debug(f"Detector {detector_name} found {len(confident_elements)} elements in {detector_time:.2f}s")
                
            except Exception as e:
                failed_detectors.append(detector_name)
                error_msg = f"Error with detector {detector_name}: {e}"
                logger.error(error_msg)
                result.errors.append(error_msg)
                detector_performance[detector_name] = {
                    "time": time.time() - detector_start,
                    "element_count": 0,
                    "avg_confidence": 0,
                    "success": False,
                    "failure_mode": str(e)
                }

        # Fallback strategy: If primary detectors failed to find elements, try AskUI if available
        if len(all_elements) == 0 and "askui_detector" not in detectors and success_level == 0:
            try:
                # Get AskUI detector if available but not already used
                askui_detector = self.registry.get_detector("askui_detector")
                if askui_detector:
                    logger.info("Primary detectors failed, trying AskUI fallback")
                    detector_start = time.time()
                    
                    elements = askui_detector.detect_elements(request.image, request.context)
                    detector_end = time.time()
                    
                    # Filter by confidence
                    confident_elements = [e for e in elements if e.get("confidence", 0) >= request.min_confidence]
                    
                    # Add detector info
                    for element in confident_elements:
                        element["source_detector"] = "askui_detector"
                        element["is_fallback"] = True
                    
                    all_elements.extend(confident_elements)
                    success = len(confident_elements) > 0
                    success_level = max(success_level, 2 if success else 0)  # Higher success level for fallback
                    
                    # Track performance
                    detector_performance["askui_detector"] = {
                        "time": detector_end - detector_start,
                        "element_count": len(confident_elements),
                        "avg_confidence": sum(e.get("confidence", 0) for e in confident_elements) / len(confident_elements) if confident_elements else 0,
                        "success": success,
                        "is_fallback": True
                    }
                    
                    logger.info(f"AskUI fallback found {len(confident_elements)} elements")
            except Exception as e:
                logger.error(f"Error using AskUI fallback: {e}")
        
        # Merge overlapping elements and resolve conflicts
        merged_elements = self._merge_elements(all_elements)
        
        # Limit number of elements
        merged_elements = self._select_top_elements(merged_elements, request.max_detections)
        
        # Add semantic understanding if requested
        if request.semantic_analysis and self.semantic_fusion_enabled:
            try:
                self._enhance_with_semantic_understanding(merged_elements, request.image, request.context)
            except Exception as e:
                error_msg = f"Error enhancing with semantic understanding: {e}"
                logger.error(error_msg)
                result.errors.append(error_msg)
        
        # Calculate final performance metrics
        end_time = time.time()
        total_time = end_time - start_time
        
        # Set result
        result.elements = merged_elements
        result.performance = {
            "total_time": total_time,
            "detector_performance": detector_performance,
            "element_count": len(merged_elements)
        }
        result.metadata = {
            "detectors_used": list(detectors.keys()),
            "semantic_analysis": request.semantic_analysis and self.semantic_fusion_enabled
        }
        
        # Update performance history
        self.performance_history.append(result.performance)
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
        
        return result
    
    def _get_detectors(self, priority_detectors: List[str] = None) -> Dict[str, UIDetectorInterface]:
        """
        Get available detectors, prioritizing requested ones
        
        Args:
            priority_detectors: List of detector names to prioritize
            
        Returns:
            Dictionary of detector names to detector instances
        """
        # Get all available detectors
        all_detectors = self.registry.get_all_detectors() if self.registry else {}
        
        # If no priority detectors specified, return all
        if not priority_detectors:
            return all_detectors
        
        # Prioritize requested detectors
        prioritized = {}
        
        # First add priority detectors in requested order
        for detector_name in priority_detectors:
            if detector_name in all_detectors:
                prioritized[detector_name] = all_detectors[detector_name]
        
        # Then add remaining detectors
        for detector_name, detector in all_detectors.items():
            if detector_name not in prioritized:
                prioritized[detector_name] = detector
        
        return prioritized
    
    def _merge_elements(self, elements: List[Dict]) -> List[Dict]:
        """
        Merge overlapping elements and resolve conflicts
        
        Args:
            elements: List of detected elements
            
        Returns:
            Merged list of elements
        """
        if not elements:
            return []
        
        # Sort by confidence
        sorted_elements = sorted(elements, key=lambda e: e.get("confidence", 0), reverse=True)
        
        # Track merged elements
        merged = []
        merged_indices = set()
        
        def calculate_iou(box1, box2):
            # Extract coordinates
            x1_1, y1_1, x2_1, y2_1 = box1
            x1_2, y1_2, x2_2, y2_2 = box2
            
            # Calculate intersection area
            x_left = max(x1_1, x1_2)
            y_top = max(y1_1, y1_2)
            x_right = min(x2_1, x2_2)
            y_bottom = min(y2_1, y2_2)
            
            if x_right < x_left or y_bottom < y_top:
                return 0.0
            
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            
            # Calculate union area
            box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
            box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
            union_area = box1_area + box2_area - intersection_area
            
            return intersection_area / union_area if union_area > 0 else 0.0
        
        # For each element
        for i, element in enumerate(sorted_elements):
            if i in merged_indices:
                continue
            
            # Get bounding box
            bbox1 = element.get("bbox")
            if not bbox1:
                merged.append(element)
                merged_indices.add(i)
                continue
            
            # Find overlapping elements
            overlaps = []
            for j, other in enumerate(sorted_elements):
                if i == j or j in merged_indices:
                    continue
                
                # Get other bounding box
                bbox2 = other.get("bbox")
                if not bbox2:
                    continue
                
                # Calculate IoU
                iou = calculate_iou(bbox1, bbox2)
                
                # If overlapping
                if iou > 0.5:  # IoU threshold
                    overlaps.append((j, other, iou))
            
            # If no overlaps, add as is
            if not overlaps:
                merged.append(element)
                merged_indices.add(i)
                continue
            
            # Merge with overlapping elements
            merged_element = element.copy()
            merged_element["merged_from"] = [element.get("source_detector", "unknown")]
            merged_element["merged_count"] = 1
            
            # For each overlapping element
            for j, other, iou in overlaps:
                merged_indices.add(j)
                
                # Update merged element
                merged_element["merged_count"] += 1
                merged_element["merged_from"].append(other.get("source_detector", "unknown"))
                
                # Boost confidence for elements detected by multiple detectors
                confidence_boost = 0.05 * min(5, merged_element["merged_count"])  # Max 25% boost
                merged_element["confidence"] = min(0.99, merged_element.get("confidence", 0.5) + confidence_boost)
                
                # Use the most specific element type if available
                other_type = other.get("type")
                if other_type and other_type != "unknown" and other_type != merged_element.get("type", "unknown"):
                    # Prioritize certain element types
                    priority_types = ["button", "checkbox", "radio_button", "dropdown", "text_field"]
                    if other_type in priority_types and merged_element.get("type") not in priority_types:
                        merged_element["type"] = other_type
            
            merged.append(merged_element)
        
        return merged
    
    def _select_top_elements(self, elements: List[Dict], max_count: int) -> List[Dict]:
        """
        Select top elements by confidence
        
        Args:
            elements: List of detected elements
            max_count: Maximum number of elements to select
            
        Returns:
            Selected elements
        """
        if not elements or len(elements) <= max_count:
            return elements
        
        # Sort by confidence
        sorted_elements = sorted(elements, key=lambda e: e.get("confidence", 0), reverse=True)
        
        return sorted_elements[:max_count]
        
    def _enhance_with_semantic_understanding(self, elements: List[Dict], image: np.ndarray, context: Dict) -> None:
        """
        Enhance elements with semantic understanding using vision-language models
        with adaptive fallback to cloud APIs if local resources are insufficient
        
        Args:
            elements: List of elements to enhance
            image: Original image
            context: Context information
        """
        if not elements:
            return
        
        # Initialize VLM if needed
        if not self.vlm_initialized:
            success = self.initialize_vlm()
            if not success:
                logger.warning("Failed to initialize vision-language model, skipping semantic enhancement")
                return
        
        # Only analyze elements with high confidence or unknown type
        elements_to_analyze = []
        for element in elements:
            if element.get("confidence", 0) > 0.7 or element.get("type") == "unknown":
                elements_to_analyze.append(element)
        
        # If no elements to analyze
        if not elements_to_analyze:
            return
        
        try:
            # If using local transformers model
            if hasattr(self, "vlm_model") and hasattr(self, "vlm_processor") and self.vlm_model and self.vlm_processor:
                import torch
                from PIL import Image
                
                for element in elements_to_analyze:
                    # Extract element region
                    bbox = element.get("bbox")
                    if not bbox:
                        continue
                    
                    x1, y1, x2, y2 = bbox
                    element_img = image[y1:y2, x1:x2]
                    
                    # Skip if element region is empty
                    if element_img.size == 0 or element_img.shape[0] == 0 or element_img.shape[1] == 0:
                        continue
                    
                    # Convert to PIL Image
                    element_pil = Image.fromarray(cv2.cvtColor(element_img, cv2.COLOR_BGR2RGB))
                    
                    # Prepare inputs
                    prompt = "What type of UI element is this? Describe its function."
                    inputs = self.vlm_processor(element_pil, text=prompt, return_tensors="pt").to(self.vlm_model.device)
                    
                    # Generate response
                    with torch.no_grad():
                        outputs = self.vlm_model.generate(
                            **inputs,
                            max_new_tokens=50,
                            num_beams=5
                        )
                    
                    # Decode response
                    response = self.vlm_processor.decode(outputs[0], skip_special_tokens=True)
                    
                    # Process response to extract relevant information
                    element["semantic_description"] = response
                    
                    # Extract element type from response if needed
                    if element.get("type") == "unknown":
                        type_map = {
                            "button": ["button", "submit", "clickable button"],
                            "checkbox": ["checkbox", "check box", "tick box"],
                            "radio_button": ["radio", "radio button", "option button"],
                            "dropdown": ["dropdown", "select", "combobox", "combo box"],
                            "text_field": ["text field", "input field", "textbox", "text box", "text input"],
                            "image": ["image", "picture", "photo", "graphic"],
                            "icon": ["icon", "symbol"],
                            "link": ["link", "hyperlink", "url"],
                            "menu": ["menu", "navigation", "nav"]
                        }
                        
                        for type_name, keywords in type_map.items():
                            if any(keyword.lower() in response.lower() for keyword in keywords):
                                element["type"] = type_name
                                element["confidence"] = min(0.99, element.get("confidence", 0.5) + 0.1)
                                break
            
            # If using cloud APIs for vision understanding
            elif hasattr(self, 'using_cloud_api') and self.using_cloud_api and hasattr(self, 'available_apis') and self.available_apis:
                import requests
                import base64
                import io
                import json
                from PIL import Image
                
                # Prepare batch of elements for API request
                batch_size = 5  # Process in batches to avoid large requests
                for i in range(0, len(elements_to_analyze), batch_size):
                    batch = elements_to_analyze[i:i+batch_size]
                    element_images = []
                    valid_indices = []
                    
                    # Extract element regions
                    for j, element in enumerate(batch):
                        bbox = element.get("bbox")
                        if not bbox:
                            continue
                        
                        x1, y1, x2, y2 = bbox
                        element_img = image[y1:y2, x1:x2]
                        
                        # Skip if element region is empty
                        if element_img.size == 0 or element_img.shape[0] == 0 or element_img.shape[1] == 0:
                            continue
                        
                        # Convert to base64
                        element_pil = Image.fromarray(cv2.cvtColor(element_img, cv2.COLOR_BGR2RGB))
                        buffer = io.BytesIO()
                        element_pil.save(buffer, format="JPEG")
                        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                        
                        element_images.append({
                            "base64": image_base64,
                            "format": "JPEG"
                        })
                        valid_indices.append(j)
                    
                    # Skip if no valid elements
                    if not element_images:
                        continue

                    # Try each available API in sequence until one succeeds
                    descriptions = []
                    success = False
                    
                    for api in self.available_apis:
                        try:
                            # Skip if API is not enabled or missing key
                            if not api.get("enabled") or not api.get("api_key"):
                                continue
                                
                            api_name = api.get("name")
                            endpoint = api.get("endpoint")
                            api_key = api.get("api_key")
                            
                            # Prepare headers
                            headers = {
                                "Content-Type": "application/json"
                            }
                            
                            if api_name == "together_ai":
                                headers["Authorization"] = f"Bearer {api_key}"
                                
                                # Format payload for Together AI
                                payload = {
                                    "model": "llama-3-8b-vision",
                                    "messages": [
                                        {
                                            "role": "user",
                                            "content": [
                                                {
                                                    "type": "text",
                                                    "text": "What type of UI element is this? Describe its function briefly."
                                                },
                                                {
                                                    "type": "image_url",
                                                    "image_url": {"url": f"data:image/jpeg;base64,{element_images[0]['base64']}"}
                                                }
                                            ]
                                        }
                                    ]
                                }
                                
                                # Make API request
                                response = requests.post(endpoint, headers=headers, json=payload)
                                
                                if response.status_code == 200:
                                    data = response.json()
                                    text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                                    descriptions = [text] * len(element_images)
                                    success = True
                                    logger.info(f"Successfully used {api_name} for vision analysis")
                                    break
                            
                            elif api_name == "gemini":
                                # Add API key as URL parameter
                                request_url = f"{endpoint}?key={api_key}"
                                
                                # Format payload for Gemini
                                payload = {
                                    "contents": [
                                        {
                                            "parts": [
                                                {
                                                    "text": "What type of UI element is this? Describe its function briefly."
                                                },
                                                {
                                                    "inline_data": {
                                                        "mime_type": "image/jpeg",
                                                        "data": element_images[0]['base64']
                                                    }
                                                }
                                            ]
                                        }
                                    ]
                                }
                                
                                # Make API request
                                response = requests.post(request_url, json=payload)
                                
                                if response.status_code == 200:
                                    data = response.json()
                                    text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                                    descriptions = [text] * len(element_images)
                                    success = True
                                    logger.info(f"Successfully used {api_name} for vision analysis")
                                    break
                            
                            elif api_name == "custom_api":
                                headers["Authorization"] = f"Bearer {api_key}"
                                
                                # Format payload for custom API
                                payload = {
                                    "images": [img["base64"] for img in element_images],
                                    "prompt": "What type of UI element is this? Describe its function."
                                }
                                
                                # Make API request
                                response = requests.post(endpoint, headers=headers, json=payload)
                                
                                if response.status_code == 200:
                                    data = response.json()
                                    descriptions = data.get("descriptions", [])
                                    if descriptions:
                                        success = True
                                        logger.info(f"Successfully used {api_name} for vision analysis")
                                        break
                        
                        except Exception as e:
                            logger.error(f"Error using {api.get('name', 'unknown')} API: {e}")
                    
                    # If we couldn't get descriptions from any API, continue to next batch
                    if not success or not descriptions:
                        logger.warning("Failed to get descriptions from any cloud API")
                        continue
                    
                    # Update elements with descriptions
                    for j, description in enumerate(descriptions):
                        if j < len(valid_indices):
                            element_idx = valid_indices[j]
                            if element_idx < len(batch):
                                element = batch[element_idx]
                                element["semantic_description"] = description
                                
                                # Extract element type from description if needed
                                if element.get("type") == "unknown":
                                    type_map = {
                                        "button": ["button", "submit", "clickable button"],
                                        "checkbox": ["checkbox", "check box", "tick box"],
                                        "radio_button": ["radio", "radio button", "option button"],
                                        "dropdown": ["dropdown", "select", "combobox", "combo box"],
                                        "text_field": ["text field", "input field", "textbox", "text box", "text input"],
                                        "image": ["image", "picture", "photo", "graphic"],
                                        "icon": ["icon", "symbol"],
                                        "link": ["link", "hyperlink", "url"],
                                        "menu": ["menu", "navigation", "nav"]
                                    }
                                    
                                    for type_name, keywords in type_map.items():
                                        if any(keyword.lower() in description.lower() for keyword in keywords):
                                            element["type"] = type_name
                                            element["confidence"] = min(0.99, element.get("confidence", 0.5) + 0.1)
                                            # Record learning data for this element to improve future detections
                                            element["learned_from_semantic"] = True
                                            break
        
        except Exception as e:
            logger.error(f"Error enhancing with semantic understanding: {e}")
    
    def get_performance_statistics(self) -> Dict:
        """
        Get statistics about detection performance
        
        Returns:
            Dictionary of performance statistics
        """
        if not self.performance_history:
            return {}
        
        # Calculate average performance metrics
        avg_total_time = sum(p["total_time"] for p in self.performance_history) / len(self.performance_history)
        
        # Get detector-specific metrics
        detector_stats = {}
        for performance in self.performance_history:
            detector_performance = performance.get("detector_performance", {})
            for detector_name, metrics in detector_performance.items():
                if detector_name not in detector_stats:
                    detector_stats[detector_name] = {
                        "count": 0,
                        "total_time": 0,
                        "total_elements": 0,
                        "total_confidence": 0
                    }
                
                detector_stats[detector_name]["count"] += 1
                detector_stats[detector_name]["total_time"] += metrics.get("time", 0)
                detector_stats[detector_name]["total_elements"] += metrics.get("element_count", 0)
                detector_stats[detector_name]["total_confidence"] += metrics.get("avg_confidence", 0)
        
        # Calculate averages
        for detector_name, stats in detector_stats.items():
            if stats["count"] > 0:
                stats["avg_time"] = stats["total_time"] / stats["count"]
                stats["avg_elements"] = stats["total_elements"] / stats["count"]
                stats["avg_confidence"] = stats["total_confidence"] / stats["count"]
        
        return {
            "avg_total_time": avg_total_time,
            "detector_stats": detector_stats,
            "history_count": len(self.performance_history)
        }
    
    def _analyze_performance_patterns(self, detector_feedback: Dict) -> None:
        """
        Analyze detector performance patterns for adaptive learning
        
        Args:
            detector_feedback: Detector feedback data
        """
        # Identify performance patterns across detectors
        element_type_performance = {}
        
        # Collect performance by element type across all detectors
        for detector_name, feedback in detector_feedback.items():
            for elem_type, stats in feedback.get("element_types", {}).items():
                if elem_type not in element_type_performance:
                    element_type_performance[elem_type] = []
                
                # Calculate success rate for this detector on this element type
                total = stats.get("total", 0)
                correct = stats.get("correct", 0)
                success_rate = correct / total if total > 0 else 0
                
                element_type_performance[elem_type].append({
                    "detector": detector_name,
                    "success_rate": success_rate,
                    "sample_size": total
                })
        
        # Log insights about detector specializations
        specialization_threshold = 0.8  # 80% success rate indicates specialization
        for elem_type, performances in element_type_performance.items():
            # Sort by success rate
            sorted_performance = sorted(performances, key=lambda x: x["success_rate"], reverse=True)
            
            # Identify specialists for this element type
            specialists = [p for p in sorted_performance if p["success_rate"] >= specialization_threshold and p["sample_size"] >= 3]
            
            if specialists:
                logger.info(f"Detected element type specialization: {elem_type} - Best detectors: {[s['detector'] for s in specialists]}")
                
                # Update registry with specialization data
                if self.registry:
                    for specialist in specialists:
                        try:
                            self.registry.update_specialization(
                                specialist["detector"],
                                elem_type,
                                specialist["success_rate"]
                            )
                        except Exception as e:
                            logger.warning(f"Error updating specialization data: {e}")
    
    def provide_feedback(self, detection_result: DetectionResult, feedback: Dict) -> None:
        """
        Provide feedback on detection results to improve future performance
        through adaptive learning and continuous optimization
        
        Args:
            detection_result: Detection result
            feedback: Feedback information
        """
        if not self.registry:
            logger.warning("No detector registry available for feedback")
            return
        
        # Extract detector feedback with enhanced metadata
        detector_feedback = {}
        context_data = {}
        
        # Extract context information for adaptive learning
        if 'context' in feedback:
            context_data = feedback['context']
            # Capture the task type/context for better specialization
            if 'task_type' in context_data:
                logger.info(f"Learning from feedback for task type: {context_data['task_type']}")
        
        # Analyze elements and collect detailed feedback
        for element in detection_result.elements:
            detector_name = element.get("source_detector")
            if not detector_name:
                continue
            
            # Get element feedback if available
            element_id = element.get("id")
            if not element_id:
                continue
                
            element_feedback = feedback.get("elements", {}).get(element_id, {})
            if not element_feedback:
                continue
                
            # Add to detector feedback with enhanced tracking
            if detector_name not in detector_feedback:
                detector_feedback[detector_name] = {
                    "correct_detections": 0,
                    "incorrect_detections": 0,
                    "elements": [],
                    "context": context_data,
                    "element_types": {},
                    "confidence_analysis": {
                        "true_positives": [],
                        "false_positives": [],
                        "missed_elements": []
                    }
                }
                
            # Update counts with enhanced metrics
            is_correct = element_feedback.get("is_correct", False)
            element_type = element.get("type", "unknown")
            confidence = element.get("confidence", 0)
            
            # Track element type success rates for specialized learning
            if element_type not in detector_feedback[detector_name]["element_types"]:
                detector_feedback[detector_name]["element_types"][element_type] = {
                    "correct": 0,
                    "incorrect": 0,
                    "total": 0
                }
            
            # Update statistics
            if is_correct:
                detector_feedback[detector_name]["correct_detections"] += 1
                detector_feedback[detector_name]["element_types"][element_type]["correct"] += 1
                detector_feedback[detector_name]["confidence_analysis"]["true_positives"].append(confidence)
            else:
                detector_feedback[detector_name]["incorrect_detections"] += 1
                detector_feedback[detector_name]["element_types"][element_type]["incorrect"] += 1
                detector_feedback[detector_name]["confidence_analysis"]["false_positives"].append(confidence)
                
            detector_feedback[detector_name]["element_types"][element_type]["total"] += 1
                
            # Add detailed element feedback
            detector_feedback[detector_name]["elements"].append({
                "element": element,
                "feedback": element_feedback,
                "corrections": element_feedback.get("corrections", {})
            })
        
        # Add missed elements for training
        if "missed_elements" in feedback:
            for missed in feedback["missed_elements"]:
                element_type = missed.get("type", "unknown")
                suggested_detector = missed.get("suggested_detector")
                
                if suggested_detector and suggested_detector in detector_feedback:
                    # Track missed elements for this detector
                    detector_feedback[suggested_detector]["confidence_analysis"]["missed_elements"].append(missed)
                    
                    # Update element type stats
                    if element_type not in detector_feedback[suggested_detector]["element_types"]:
                        detector_feedback[suggested_detector]["element_types"][element_type] = {
                            "correct": 0,
                            "incorrect": 0,
                            "total": 0,
                            "missed": 1
                        }
                    else:
                        missed_count = detector_feedback[suggested_detector]["element_types"][element_type].get("missed", 0)
                        detector_feedback[suggested_detector]["element_types"][element_type]["missed"] = missed_count + 1
        
        # Enhanced learning: Analyze detector performance patterns
        self._analyze_performance_patterns(detector_feedback)
        
        # Provide adaptive feedback to each detector
        for detector_name, feedback_data in detector_feedback.items():
            try:
                detector = self.registry.get_detector(detector_name)
                if not detector:
                    logger.warning(f"Detector {detector_name} not found for feedback")
                    continue
                    
                # Enhanced feedback capabilities
                if hasattr(detector, "supports_incremental_learning") and detector.supports_incremental_learning():
                    logger.info(f"Providing adaptive learning feedback to {detector_name}")
                    
                    # Prepare enhanced feedback packet
                    feedback_packet = {
                        "correct_detections": feedback_data["correct_detections"],
                        "incorrect_detections": feedback_data["incorrect_detections"],
                        "elements": {ef["element"].get("id"): ef["feedback"] for ef in feedback_data["elements"]},
                        "element_type_performance": feedback_data["element_types"],
                        "context": feedback_data["context"],
                        "confidence_analysis": feedback_data["confidence_analysis"]
                    }
                    
                    # Provide enhanced feedback if supported
                    if hasattr(detector, "provide_enhanced_feedback") and callable(getattr(detector, "provide_enhanced_feedback")):
                        detector.provide_enhanced_feedback(feedback_packet)
                    # Fallback to standard feedback
                    elif hasattr(detector, "provide_feedback") and callable(getattr(detector, "provide_feedback")):
                        detector.provide_feedback(feedback_packet)
                        
                    logger.info(f"Provided adaptive learning feedback to detector {detector_name}")
            except Exception as e:
                logger.error(f"Error providing feedback to detector {detector_name}: {e}")
        
        # Update registry metrics with enhanced learning data
        try:
            # Update basic detector metrics
            for detector_name, feedback_data in detector_feedback.items():
                total = feedback_data["correct_detections"] + feedback_data["incorrect_detections"]
                if total > 0:
                    success_rate = feedback_data["correct_detections"] / total
                    metrics_update = {
                        "success_rate": success_rate,
                        "total_calls": 1,  # Increment call count
                        "successful_calls": 1 if success_rate > 0.5 else 0,
                        "total_elements_detected": len(feedback_data["elements"])
                    }
                    try:
                        self.registry.update_detector_metric(detector_name, metrics_update)
                    except Exception as e:
                        logger.warning(f"Error updating basic metrics for {detector_name}: {e}")
            
            # Update specialized context associations in the registry
            if context_data and 'task_type' in context_data:
                task_type = context_data['task_type']
                for detector_name, feedback_data in detector_feedback.items():
                    success_rate = 0
                    if feedback_data["correct_detections"] + feedback_data["incorrect_detections"] > 0:
                        success_rate = feedback_data["correct_detections"] / (feedback_data["correct_detections"] + feedback_data["incorrect_detections"])
                    
                    if success_rate > 0.7:  # Good performance threshold
                        try:
                            if hasattr(self.registry, "update_context_association"):
                                self.registry.update_context_association(detector_name, task_type, success_rate)
                        except Exception as e:
                            logger.warning(f"Error updating context association: {e}")
            
            logger.info("Updated detector metrics in registry with adaptive learning data")
        except Exception as e:
            logger.error(f"Error updating detector metrics: {e}")
