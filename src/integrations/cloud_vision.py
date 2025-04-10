"""
Google Cloud Vision API Integration for NEXUS
Provides advanced image analysis capabilities with vector database storage
"""
import os
import logging
import base64
import json
import time
import asyncio
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import requests
from PIL import Image
import io
import google.auth
from google.oauth2 import service_account
from google.cloud import vision

logger = logging.getLogger(__name__)

class CloudVisionService:
    """
    Google Cloud Vision API integration for NEXUS.
    
    This component provides advanced image analysis capabilities:
    - Object detection and labeling
    - Optical Character Recognition (OCR)
    - Landmark detection
    - Logo detection
    - Similar image search
    - Face detection
    - Safe search detection
    
    Results are stored in the vector database for future reference.
    """
    
    def __init__(self, 
                 service_account_path: str = None, 
                 api_key: str = None,
                 vector_storage=None):
        """Initialize the Cloud Vision service"""
        self.client = None
        self.available = False
        
        # Store vector storage reference if provided
        self.vector_storage = vector_storage
        
        # Cache directory for API results
        self.cache_dir = Path("cache/vision_api")
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Try to initialize with service account
        if service_account_path:
            try:
                credentials = service_account.Credentials.from_service_account_file(
                    service_account_path,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                self.client = vision.ImageAnnotatorClient(credentials=credentials)
                self.available = True
                logger.info("Cloud Vision API initialized with service account")
            except Exception as e:
                logger.error(f"Error initializing Cloud Vision with service account: {e}")
                
        # If service account didn't work and API key is provided, use REST API
        if not self.available and api_key:
            # Store API key for REST calls
            self.api_key = api_key
            self.vision_api_url = "https://vision.googleapis.com/v1/images:annotate"
            self.available = True
            logger.info("Cloud Vision API initialized with API key (REST mode)")
        
        if not self.available:
            logger.warning("Cloud Vision API not available. Neither service account nor API key provided.")
            
        # Configure default features
        self.default_features = [
            {"type": "LABEL_DETECTION", "maxResults": 10},
            {"type": "TEXT_DETECTION", "maxResults": 10},
            {"type": "OBJECT_LOCALIZATION", "maxResults": 10},
            {"type": "WEB_DETECTION", "maxResults": 10}
        ]
        
        logger.info(f"Cloud Vision Service initialized (available: {self.available})")
    
    async def analyze_image(self, 
                         image_path: str = None, 
                         image: Image.Image = None,
                         features: List[Dict[str, Any]] = None,
                         store_results: bool = True) -> Dict[str, Any]:
        """
        Analyze an image using Google Cloud Vision API
        
        Args:
            image_path: Path to image file
            image: PIL Image object (used if image_path not provided)
            features: List of features to analyze, defaults to standard set
            store_results: Whether to store results in vector database
            
        Returns:
            Dictionary with analysis results
        """
        if not self.available:
            return {
                "success": False,
                "error": "Cloud Vision API not available",
                "timestamp": time.time()
            }
            
        try:
            # Check cache first if image_path is provided
            cache_result = None
            if image_path:
                cache_key = f"{Path(image_path).name}_{hash(str(features))}"
                cache_result = self._check_cache(cache_key)
                if cache_result:
                    logger.info(f"Returning cached Cloud Vision results for: {image_path}")
                    return cache_result
            
            # Use provided features or default
            if features is None:
                features = self.default_features
            
            # Get API result
            if self.client:  # Using client library
                api_result = await self._analyze_with_client(image_path, image, features)
            else:  # Using REST API
                api_result = await self._analyze_with_rest(image_path, image, features)
            
            if not api_result:
                return {
                    "success": False,
                    "error": "Failed to get API result",
                    "timestamp": time.time()
                }
            
            # Process response
            result = self._process_api_response(api_result)
            result["timestamp"] = time.time()
            result["success"] = True
            
            # Cache results if image_path was provided
            if image_path:
                cache_key = f"{Path(image_path).name}_{hash(str(features))}"
                self._cache_results(cache_key, result)
            
            # Store in vector database if requested
            if store_results and self.vector_storage:
                await self._store_in_vector_db(result, image_path)
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing image with Cloud Vision API: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def _analyze_with_client(self, image_path, image, features) -> Dict[str, Any]:
        """Analyze image using the client library"""
        # Prepare image content
        if image_path:
            with open(image_path, "rb") as image_file:
                content = image_file.read()
            vision_image = vision.Image(content=content)
        elif image:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            content = buffer.getvalue()
            vision_image = vision.Image(content=content)
        else:
            return None
        
        # Prepare feature requests
        feature_requests = []
        for feature in features:
            feature_type = getattr(vision.Feature.Type, feature["type"])
            max_results = feature.get("maxResults", 10)
            feature_requests.append(vision.Feature(type_=feature_type, max_results=max_results))
        
        # Make API request
        response = self.client.annotate_image({
            'image': vision_image,
            'features': feature_requests
        })
        
        # Convert response to dict for consistent processing
        return self._client_response_to_dict(response)
    
    def _client_response_to_dict(self, response) -> Dict[str, Any]:
        """Convert client library response to dictionary format"""
        # Create a structured response similar to the REST API format
        result = {"responses": [{}]}
        
        # Process labels
        if response.label_annotations:
            result["responses"][0]["labelAnnotations"] = [
                {
                    "description": label.description,
                    "score": label.score,
                    "topicality": label.topicality
                }
                for label in response.label_annotations
            ]
        
        # Process text
        if response.text_annotations:
            result["responses"][0]["textAnnotations"] = [
                {
                    "description": text.description,
                    "boundingPoly": {
                        "vertices": [
                            {"x": vertex.x, "y": vertex.y}
                            for vertex in text.bounding_poly.vertices
                        ]
                    }
                }
                for text in response.text_annotations
            ]
        
        # Process objects
        if response.localized_object_annotations:
            result["responses"][0]["localizedObjectAnnotations"] = [
                {
                    "name": obj.name,
                    "score": obj.score,
                    "boundingPoly": {
                        "normalizedVertices": [
                            {"x": vertex.x, "y": vertex.y}
                            for vertex in obj.bounding_poly.normalized_vertices
                        ]
                    }
                }
                for obj in response.localized_object_annotations
            ]
        
        # Process web detection
        if response.web_detection:
            web = response.web_detection
            web_result = {}
            
            if web.visually_similar_images:
                web_result["visuallySimilarImages"] = [
                    {"url": img.url}
                    for img in web.visually_similar_images
                ]
            
            if web.pages_with_matching_images:
                web_result["pagesWithMatchingImages"] = [
                    {"url": page.url, "pageTitle": page.page_title}
                    for page in web.pages_with_matching_images
                ]
            
            if web.web_entities:
                web_result["webEntities"] = [
                    {
                        "entityId": entity.entity_id,
                        "description": entity.description,
                        "score": entity.score
                    }
                    for entity in web.web_entities
                ]
            
            result["responses"][0]["webDetection"] = web_result
        
        # Process landmarks
        if response.landmark_annotations:
            result["responses"][0]["landmarkAnnotations"] = [
                {
                    "description": landmark.description,
                    "score": landmark.score,
                    "locations": [
                        {
                            "latLng": {
                                "latitude": location.lat_lng.latitude,
                                "longitude": location.lat_lng.longitude
                            }
                        }
                        for location in landmark.locations
                    ]
                }
                for landmark in response.landmark_annotations
            ]
        
        # Process logos
        if response.logo_annotations:
            result["responses"][0]["logoAnnotations"] = [
                {
                    "description": logo.description,
                    "score": logo.score
                }
                for logo in response.logo_annotations
            ]
        
        # Process faces
        if response.face_annotations:
            result["responses"][0]["faceAnnotations"] = [
                {
                    "joyLikelihood": vision.Likelihood(face.joy_likelihood).name,
                    "sorrowLikelihood": vision.Likelihood(face.sorrow_likelihood).name,
                    "angerLikelihood": vision.Likelihood(face.anger_likelihood).name,
                    "surpriseLikelihood": vision.Likelihood(face.surprise_likelihood).name,
                    "detectionConfidence": face.detection_confidence,
                    "landmarks": [
                        {
                            "type": landmark.type_.name,
                            "position": {
                                "x": landmark.position.x,
                                "y": landmark.position.y,
                                "z": landmark.position.z
                            }
                        }
                        for landmark in face.landmarks
                    ]
                }
                for face in response.face_annotations
            ]
        
        # Process safe search
        if response.safe_search_annotation:
            safe_search = response.safe_search_annotation
            result["responses"][0]["safeSearchAnnotation"] = {
                "adult": vision.Likelihood(safe_search.adult).name,
                "spoof": vision.Likelihood(safe_search.spoof).name,
                "medical": vision.Likelihood(safe_search.medical).name,
                "violence": vision.Likelihood(safe_search.violence).name,
                "racy": vision.Likelihood(safe_search.racy).name
            }
        
        return result
    
    async def _analyze_with_rest(self, image_path, image, features) -> Dict[str, Any]:
        """Analyze image using REST API"""
        # Prepare image content
        if image_path:
            with open(image_path, "rb") as image_file:
                content = base64.b64encode(image_file.read()).decode()
        elif image:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            content = base64.b64encode(buffer.getvalue()).decode()
        else:
            return None
        
        # Prepare API request
        request_data = {
            "requests": [
                {
                    "image": {
                        "content": content
                    },
                    "features": features
                }
            ]
        }
        
        # Make API request
        response = requests.post(
            f"{self.vision_api_url}?key={self.api_key}",
            json=request_data
        )
        response.raise_for_status()
        
        # Return response
        return response.json()
    
    def _process_api_response(self, api_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process and structure Cloud Vision API response"""
        # Initialize structured result
        result = {
            "labels": [],
            "text": {
                "full_text": "",
                "text_blocks": []
            },
            "objects": [],
            "web": {
                "similar_images": [],
                "matching_pages": [],
                "entities": []
            },
            "landmarks": [],
            "logos": [],
            "faces": [],
            "safe_search": {}
        }
        
        # Extract responses
        if "responses" in api_result and api_result["responses"]:
            response = api_result["responses"][0]
            
            # Process labels
            if "labelAnnotations" in response:
                for label in response["labelAnnotations"]:
                    result["labels"].append({
                        "description": label.get("description", ""),
                        "score": label.get("score", 0.0),
                        "topicality": label.get("topicality", 0.0)
                    })
            
            # Process text
            if "textAnnotations" in response:
                if response["textAnnotations"]:
                    # Full text is usually the first annotation
                    result["text"]["full_text"] = response["textAnnotations"][0].get("description", "")
                    
                    # Process individual text blocks
                    for text_block in response["textAnnotations"][1:]:  # Skip the first one which is full text
                        # Extract vertices
                        vertices = []
                        if "boundingPoly" in text_block and "vertices" in text_block["boundingPoly"]:
                            for vertex in text_block["boundingPoly"]["vertices"]:
                                vertices.append({
                                    "x": vertex.get("x", 0),
                                    "y": vertex.get("y", 0)
                                })
                        
                        # Add text block
                        result["text"]["text_blocks"].append({
                            "text": text_block.get("description", ""),
                            "confidence": text_block.get("score", 1.0),  # Text detection doesn't always have a score
                            "vertices": vertices
                        })
            
            # Process objects
            if "localizedObjectAnnotations" in response:
                for obj in response["localizedObjectAnnotations"]:
                    # Extract bounding polygon
                    vertices = []
                    if "boundingPoly" in obj and "normalizedVertices" in obj["boundingPoly"]:
                        for vertex in obj["boundingPoly"]["normalizedVertices"]:
                            vertices.append({
                                "x": vertex.get("x", 0),
                                "y": vertex.get("y", 0)
                            })
                    
                    # Add object
                    result["objects"].append({
                        "name": obj.get("name", ""),
                        "score": obj.get("score", 0.0),
                        "vertices": vertices
                    })
            
            # Process web detection
            if "webDetection" in response:
                web = response["webDetection"]
                
                # Extract similar images
                if "visuallySimilarImages" in web:
                    for similar in web["visuallySimilarImages"]:
                        result["web"]["similar_images"].append({
                            "url": similar.get("url", "")
                        })
                
                # Extract pages with matching images
                if "pagesWithMatchingImages" in web:
                    for page in web["pagesWithMatchingImages"]:
                        result["web"]["matching_pages"].append({
                            "url": page.get("url", ""),
                            "page_title": page.get("pageTitle", "")
                        })
                
                # Extract web entities
                if "webEntities" in web:
                    for entity in web["webEntities"]:
                        result["web"]["entities"].append({
                            "entity_id": entity.get("entityId", ""),
                            "description": entity.get("description", ""),
                            "score": entity.get("score", 0.0)
                        })
            
            # Process landmarks
            if "landmarkAnnotations" in response:
                for landmark in response["landmarkAnnotations"]:
                    result["landmarks"].append({
                        "name": landmark.get("description", ""),
                        "score": landmark.get("score", 0.0),
                        "latitude": landmark.get("locations", [{}])[0].get("latLng", {}).get("latitude", 0.0) if landmark.get("locations") else 0.0,
                        "longitude": landmark.get("locations", [{}])[0].get("latLng", {}).get("longitude", 0.0) if landmark.get("locations") else 0.0
                    })
            
            # Process logos
            if "logoAnnotations" in response:
                for logo in response["logoAnnotations"]:
                    result["logos"].append({
                        "name": logo.get("description", ""),
                        "score": logo.get("score", 0.0)
                    })
            
            # Process faces
            if "faceAnnotations" in response:
                for face in response["faceAnnotations"]:
                    result["faces"].append({
                        "joy": face.get("joyLikelihood", ""),
                        "sorrow": face.get("sorrowLikelihood", ""),
                        "anger": face.get("angerLikelihood", ""),
                        "surprise": face.get("surpriseLikelihood", ""),
                        "detection_confidence": face.get("detectionConfidence", 0.0),
                        "landmarks": [
                            {"type": landmark.get("type", ""), "position": landmark.get("position", {})} 
                            for landmark in face.get("landmarks", [])
                        ]
                    })
            
            # Process safe search
            if "safeSearchAnnotation" in response:
                result["safe_search"] = {
                    "adult": response["safeSearchAnnotation"].get("adult", ""),
                    "spoof": response["safeSearchAnnotation"].get("spoof", ""),
                    "medical": response["safeSearchAnnotation"].get("medical", ""),
                    "violence": response["safeSearchAnnotation"].get("violence", ""),
                    "racy": response["safeSearchAnnotation"].get("racy", "")
                }
        
        return result
    
    async def detect_ui_elements(self, 
                               image_path: str = None, 
                               image: Image.Image = None,
                               store_results: bool = True) -> Dict[str, Any]:
        """
        Specialized method to detect UI elements using Cloud Vision
        
        Args:
            image_path: Path to image file
            image: PIL Image object (used if image_path not provided)
            store_results: Whether to store results in vector database
            
        Returns:
            Dictionary with UI element detection results
        """
        # Use object detection + OCR for UI elements
        features = [
            {"type": "OBJECT_LOCALIZATION", "maxResults": 20},
            {"type": "TEXT_DETECTION", "maxResults": 100},
            {"type": "WEB_DETECTION", "maxResults": 10}
        ]
        
        # Analyze image
        result = await self.analyze_image(
            image_path=image_path,
            image=image,
            features=features,
            store_results=False  # We'll store processed results later
        )
        
        if not result.get("success", False):
            return result
        
        # Extract UI elements from results
        ui_elements = []
        
        # Check objects for potential UI elements
        for obj in result.get("objects", []):
            name = obj.get("name", "").lower()
            
            # Look for UI-related objects
            ui_related = any(term in name for term in [
                "button", "icon", "screen", "menu", "window", "dialog", 
                "interface", "display", "panel", "control"
            ])
            
            if ui_related:
                ui_elements.append({
                    "type": "ui_component",
                    "component_type": name,
                    "confidence": obj.get("score", 0.0),
                    "vertices": obj.get("vertices", []),
                    "source": "object_detection"
                })
        
        # Check text blocks for buttons, labels, etc.
        for text_block in result.get("text", {}).get("text_blocks", []):
            text = text_block.get("text", "").lower()
            
            # Simple heuristic to identify buttons, fields, etc.
            if len(text) < 30:  # Short text likely UI element
                if any(term in text for term in ["submit", "cancel", "ok", "yes", "no", "next", "back", "login", "sign"]):
                    ui_type = "button"
                elif any(term in text for term in ["search", "find"]):
                    ui_type = "search_box"
                elif any(term in text for term in ["menu", "options"]):
                    ui_type = "menu_item"
                elif text.endswith(":"):
                    ui_type = "label"
                else:
                    ui_type = "text_element"
                
                ui_elements.append({
                    "type": "ui_text",
                    "component_type": ui_type,
                    "text": text_block.get("text", ""),
                    "confidence": text_block.get("confidence", 0.0),
                    "vertices": text_block.get("vertices", []),
                    "source": "text_detection"
                })
        
        # Use web detection to identify similar UI components
        for entity in result.get("web", {}).get("entities", []):
            description = entity.get("description", "").lower()
            
            # Look for UI-related entities
            ui_related = any(term in description for term in [
                "button", "icon", "ui", "interface", "app", "website", "page", 
                "design", "layout"
            ])
            
            if ui_related:
                ui_elements.append({
                    "type": "ui_concept",
                    "description": entity.get("description", ""),
                    "confidence": entity.get("score", 0.0),
                    "source": "web_detection"
                })
        
        # Create structured UI analysis result
        ui_result = {
            "success": True,
            "ui_elements": ui_elements,
            "count": len(ui_elements),
            "raw_analysis": result,
            "timestamp": time.time()
        }
        
        # Store in vector database if requested
        if store_results and self.vector_storage and ui_elements:
            await self._store_ui_elements_in_vector_db(ui_result, image_path)
        
        return ui_result
    
    async def ocr_document(self,
                         image_path: str = None,
                         image: Image.Image = None,
                         store_results: bool = True) -> Dict[str, Any]:
        """
        Extract text from a document using Cloud Vision OCR
        
        Args:
            image_path: Path to image file
            image: PIL Image object (used if image_path not provided)
            store_results: Whether to store results in vector database
            
        Returns:
            Dictionary with OCR results
        """
        # Use document text detection for better OCR results
        features = [
            {"type": "DOCUMENT_TEXT_DETECTION"}
        ]
        
        # Analyze image
        result = await self.analyze_image(
            image_path=image_path,
            image=image,
            features=features,
            store_results=False  # We'll store processed results later
        )
        
        if not result.get("success", False):
            return result
        
        # Create structured OCR result
        ocr_result = {
            "success": True,
            "text": result.get("text", {}).get("full_text", ""),
            "text_blocks": result.get("text", {}).get("text_blocks", []),
            "timestamp": time.time()
        }
        
        # Store in vector database if requested
        if store_results and self.vector_storage and ocr_result["text"]:
            metadata = {
                "source": "cloud_vision_ocr",
                "file_path": image_path if image_path else "unknown",
                "content_type": "document_text"
            }
            
            # Add to vector storage
            if self.vector_storage:
                self.vector_storage.add_texts(
                    texts=[ocr_result["text"]],
                    metadatas=[metadata]
                )
        
        return ocr_result
    
    async def _store_in_vector_db(self, result: Dict[str, Any], image_path: str = None) -> None:
        """Store analysis results in vector database"""
        if not self.vector_storage:
            return
        
        try:
            # Create text representation of the analysis
            texts = []
            metadatas = []
            
            # Add labels
            if result["labels"]:
                label_text = "Image contains: " + ", ".join([label["description"] for label in result["labels"]])
                texts.append(label_text)
                metadatas.append({
                    "image_path": image_path,
                    "content_type": "image_labels",
                    "source": "cloud_vision"
                })
            
            # Add extracted text
            if result["text"]["full_text"]:
                texts.append(result["text"]["full_text"])
                metadatas.append({
                    "image_path": image_path,
                    "content_type": "image_text",
                    "source": "cloud_vision"
                })
            
            # Add objects
            if result["objects"]:
                object_text = "Objects detected: " + ", ".join([obj["name"] for obj in result["objects"]])
                texts.append(object_text)
                metadatas.append({
                    "image_path": image_path,
                    "content_type": "image_objects",
                    "source": "cloud_vision"
                })
            
            # Add web entities
            if result["web"]["entities"]:
                entity_text = "Related entities: " + ", ".join([entity["description"] for entity in result["web"]["entities"] if "description" in entity])
                texts.append(entity_text)
                metadatas.append({
                    "image_path": image_path,
                    "content_type": "web_entities",
                    "source": "cloud_vision"
                })
            
            # Store in vector database
            if texts and metadatas:
                self.vector_storage.add_texts(
                    texts=texts,
                    metadatas=metadatas
                )
                
        except Exception as e:
            logger.error(f"Error storing Cloud Vision results in vector DB: {e}")
    
    async def _store_ui_elements_in_vector_db(self, ui_result: Dict[str, Any], image_path: str = None) -> None:
        """Store UI element detection results in vector database"""
        if not self.vector_storage:
            return
        
        try:
            # Create text representation of UI elements
            texts = []
            metadatas = []
            
            # Add general UI description
            ui_types = set()
            for element in ui_result["ui_elements"]:
                if "component_type" in element:
                    ui_types.add(element["component_type"])
            
            if ui_types:
                ui_text = f"UI contains: {', '.join(ui_types)}"
                texts.append(ui_text)
                metadatas.append({
                    "image_path": image_path,
                    "content_type": "ui_elements",
                    "source": "cloud_vision"
                })
            
            # Add each UI element
            for element in ui_result["ui_elements"]:
                element_text = ""
                if element["type"] == "ui_component":
                    element_text = f"UI Component: {element['component_type']}"
                elif element["type"] == "ui_text":
                    element_text = f"UI Text ({element['component_type']}): {element.get('text', '')}"
                elif element["type"] == "ui_concept":
                    element_text = f"UI Concept: {element.get('description', '')}"
                
                if element_text:
                    texts.append(element_text)
                    metadatas.append({
                        "image_path": image_path,
                        "content_type": "ui_element",
                        "element_type": element["type"],
                        "component_type": element.get("component_type", ""),
                        "source": "cloud_vision"
                    })
            
            # Store in vector database
            if texts and metadatas:
                self.vector_storage.add_texts(
                    texts=texts,
                    metadatas=metadatas
                )
                
        except Exception as e:
            logger.error(f"Error storing UI elements in vector DB: {e}")
    
    def _check_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Check if results exist in cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
            
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading cache: {e}")
            return None
    
    def _cache_results(self, cache_key: str, results: Dict[str, Any]) -> None:
        """Cache results to disk"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            with open(cache_file, "w") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            logger.error(f"Error caching results: {e}")
            
    async def analyze_screenshot_for_ui(self, screenshot_path: str = None, screenshot: Image.Image = None) -> Dict[str, Any]:
        """
        Analyze a screenshot specifically for UI elements
        
        Args:
            screenshot_path: Path to screenshot file
            screenshot: PIL Image object (used if screenshot_path not provided)
            
        Returns:
            Dictionary with UI analysis results
        """
        return await self.detect_ui_elements(
            image_path=screenshot_path,
            image=screenshot,
            store_results=True
        )
