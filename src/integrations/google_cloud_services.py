"""
Google Cloud Services Integration Module for NEXUS
Provides unified access to Google Cloud APIs (Vision, Search, Generative AI)
"""
import os
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import base64
import io
from PIL import Image

# Google Cloud Libraries
from google.oauth2 import service_account
from google.cloud import vision
from googleapiclient.discovery import build
import google.cloud.aiplatform as aiplatform

logger = logging.getLogger(__name__)

class GoogleCloudServices:
    """
    Unified Google Cloud Services integration for NEXUS
    
    Provides access to multiple Google Cloud APIs through a single interface:
    - Cloud Vision API: For advanced image analysis
    - Custom Search API: For web search with more features than DuckDuckGo
    - Generative Language API: For using Gemini and other generative models
    """
    
    def __init__(self, service_account_path: str = None, api_key: str = None):
        """
        Initialize Google Cloud Services with appropriate credentials
        
        Args:
            service_account_path: Path to service account JSON file
            api_key: API key for Google Cloud APIs (alternative to service account)
        """
        self.service_account_path = service_account_path
        self.api_key = api_key
        
        # Initialize available APIs
        self.vision_client = None
        self.search_client = None
        self.genai_configured = False
        
        # Try to initialize with service account
        self._initialize_with_service_account()
        
        # Fall back to API key if needed and available
        if not self.vision_client and self.api_key:
            self._initialize_with_api_key()
        
        # Log availability
        logger.info(f"Google Cloud Services initialized (Vision: {self.vision_client is not None}, "
                   f"Search: {self.search_client is not None}, "
                   f"GenAI: {self.genai_configured})")
    
    def _initialize_with_service_account(self):
        """Initialize services using a service account"""
        if not self.service_account_path:
            # Try standard locations
            potential_paths = [
                "autonomus-1743898709312-8589efbc502d.json",  # Current directory
                os.path.expanduser("~/autonomus-1743898709312-8589efbc502d.json"),  # Home dir
                "d:/Ai/Nexus/autonomus-1743898709312-8589efbc502d.json",  # Project dir
                os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")  # Environment variable
            ]
            
            for path in potential_paths:
                if path and os.path.exists(path):
                    self.service_account_path = path
                    break
        
        if not self.service_account_path or not os.path.exists(self.service_account_path):
            logger.warning("No valid service account credentials found")
            return
        
        try:
            # Load credentials
            credentials = service_account.Credentials.from_service_account_file(
                self.service_account_path
            )
            
            # Extract API key from service account if needed
            try:
                with open(self.service_account_path, 'r') as f:
                    sa_info = json.load(f)
                    if not self.api_key and 'private_key' in sa_info:
                        # Not the actual API key, but we'll use project id as a fallback identifier
                        self.api_key = sa_info.get('project_id')
            except Exception as e:
                logger.warning(f"Could not extract project info from service account: {e}")
            
            # Initialize Cloud Vision
            try:
                self.vision_client = vision.ImageAnnotatorClient(credentials=credentials)
                logger.info("Cloud Vision API initialized with service account")
            except Exception as e:
                logger.error(f"Failed to initialize Cloud Vision API: {e}")
            
            # Initialize Custom Search API
            try:
                self.search_client = build(
                    'customsearch', 'v1',
                    credentials=credentials,
                    cache_discovery=False
                )
                logger.info("Custom Search API initialized with service account")
            except Exception as e:
                logger.error(f"Failed to initialize Custom Search API: {e}")
            
            # Initialize Generative AI
            try:
                # Configure genai with service account
                aiplatform.init(project=self.api_key, credentials=credentials)
                self.genai_configured = True
                logger.info("Generative AI API initialized with service account")
            except Exception as e:
                logger.error(f"Failed to initialize Generative AI API: {e}")
                
        except Exception as e:
            logger.error(f"Error initializing with service account: {e}")
    
    def _initialize_with_api_key(self):
        """Initialize services using an API key as fallback"""
        if not self.api_key:
            logger.warning("No API key provided for fallback authentication")
            return
        
        # We can't initialize Vision API with just an API key
        # But we can initialize Custom Search
        try:
            self.search_client = build(
                'customsearch', 'v1',
                developerKey=self.api_key,
                cache_discovery=False
            )
            logger.info("Custom Search API initialized with API key")
        except Exception as e:
            logger.error(f"Failed to initialize Custom Search API with API key: {e}")
        
        # Initialize Generative AI with API key
        try:
            aiplatform.init(project=self.api_key, api_endpoint='https://us-central1-aiplatform.googleapis.com')
            self.genai_configured = True
            logger.info("Generative AI API initialized with API key")
        except Exception as e:
            logger.error(f"Failed to initialize Generative AI API with API key: {e}")
    
    @property
    def available_services(self) -> List[str]:
        """Get list of available services"""
        services = []
        if self.vision_client:
            services.append("vision")
        if self.search_client:
            services.append("search")
        if self.genai_configured:
            services.append("generative_ai")
        return services
    
    async def analyze_image(self, image_path=None, image=None) -> Dict[str, Any]:
        """
        Analyze an image using Cloud Vision API
        
        Args:
            image_path: Path to image file
            image: PIL Image object
            
        Returns:
            Dictionary with analysis results
        """
        if not self.vision_client:
            return {
                "success": False,
                "error": "Cloud Vision API not available",
                "service": "vision"
            }
        
        try:
            # Prepare the image
            if image_path and os.path.exists(image_path):
                with open(image_path, 'rb') as image_file:
                    content = image_file.read()
            elif image:
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                content = img_byte_arr.getvalue()
            else:
                return {
                    "success": False, 
                    "error": "No valid image provided",
                    "service": "vision"
                }
            
            vision_image = vision.Image(content=content)
            
            # Run multiple feature detections in parallel tasks
            tasks = [
                self._detect_labels(vision_image),
                self._detect_text(vision_image),
                self._detect_objects(vision_image),
                self._detect_landmarks(vision_image),
                self._detect_faces(vision_image),
                self._detect_web_entities(vision_image)
            ]
            
            # Gather results
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            analysis_results = {
                "success": True,
                "service": "vision",
                "image_size": len(content) if content else 0
            }
            
            # Add individual feature results
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Error in Vision API feature detection: {result}")
                    continue
                    
                if result and isinstance(result, dict):
                    analysis_results.update(result)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing image with Cloud Vision: {e}")
            return {
                "success": False,
                "error": str(e),
                "service": "vision"
            }
    
    async def _detect_labels(self, vision_image):
        """Detect labels in image"""
        try:
            response = self.vision_client.label_detection(image=vision_image)
            labels = response.label_annotations
            
            return {
                "labels": [
                    {
                        "description": label.description,
                        "score": label.score,
                        "topicality": label.topicality
                    }
                    for label in labels
                ]
            }
        except Exception as e:
            logger.error(f"Error detecting labels: {e}")
            return None
    
    async def _detect_text(self, vision_image):
        """Detect text in image"""
        try:
            response = self.vision_client.text_detection(image=vision_image)
            texts = response.text_annotations
            
            # Get full text
            full_text = texts[0].description if texts else ""
            
            # Get individual text elements
            text_elements = []
            if len(texts) > 1:  # First element is the entire text
                for text in texts[1:]:
                    vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
                    text_elements.append({
                        "text": text.description,
                        "position": vertices
                    })
            
            return {
                "text": {
                    "full_text": full_text,
                    "elements": text_elements
                }
            }
        except Exception as e:
            logger.error(f"Error detecting text: {e}")
            return None
    
    async def _detect_objects(self, vision_image):
        """Detect objects in image"""
        try:
            response = self.vision_client.object_localization(image=vision_image)
            objects = response.localized_object_annotations
            
            return {
                "objects": [
                    {
                        "name": obj.name,
                        "score": obj.score,
                        "bounding_box": {
                            "x": obj.bounding_poly.normalized_vertices[0].x,
                            "y": obj.bounding_poly.normalized_vertices[0].y,
                            "width": (obj.bounding_poly.normalized_vertices[1].x - 
                                     obj.bounding_poly.normalized_vertices[0].x),
                            "height": (obj.bounding_poly.normalized_vertices[2].y - 
                                      obj.bounding_poly.normalized_vertices[0].y)
                        }
                    }
                    for obj in objects
                ]
            }
        except Exception as e:
            logger.error(f"Error detecting objects: {e}")
            return None
    
    async def _detect_landmarks(self, vision_image):
        """Detect landmarks in image"""
        try:
            response = self.vision_client.landmark_detection(image=vision_image)
            landmarks = response.landmark_annotations
            
            return {
                "landmarks": [
                    {
                        "description": landmark.description,
                        "score": landmark.score,
                        "locations": [
                            {
                                "latitude": location.lat_lng.latitude,
                                "longitude": location.lat_lng.longitude
                            }
                            for location in landmark.locations
                        ]
                    }
                    for landmark in landmarks
                ]
            }
        except Exception as e:
            logger.error(f"Error detecting landmarks: {e}")
            return None
    
    async def _detect_faces(self, vision_image):
        """Detect faces in image (without identifying individuals)"""
        try:
            response = self.vision_client.face_detection(image=vision_image)
            faces = response.face_annotations
            
            return {
                "faces": [
                    {
                        "confidence": face.detection_confidence,
                        "expressions": {
                            "joy": face.joy_likelihood,
                            "sorrow": face.sorrow_likelihood,
                            "anger": face.anger_likelihood,
                            "surprise": face.surprise_likelihood
                        },
                        "headwear": face.headwear_likelihood
                    }
                    for face in faces
                ]
            }
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return None
    
    async def _detect_web_entities(self, vision_image):
        """Detect web entities in image"""
        try:
            response = self.vision_client.web_detection(image=vision_image)
            web = response.web_detection
            
            web_entities = []
            if web.web_entities:
                web_entities = [
                    {
                        "entity_id": entity.entity_id,
                        "description": entity.description,
                        "score": entity.score
                    }
                    for entity in web.web_entities
                ]
            
            similar_images = []
            if web.visually_similar_images:
                similar_images = [image.url for image in web.visually_similar_images]
            
            return {
                "web_detection": {
                    "entities": web_entities,
                    "similar_images": similar_images[:5],
                    "pages_with_matching_images": [
                        page.url for page in web.pages_with_matching_images
                    ][:5] if web.pages_with_matching_images else []
                }
            }
        except Exception as e:
            logger.error(f"Error detecting web entities: {e}")
            return None
    
    async def detect_ui_elements(self, image_path=None, image=None) -> Dict[str, Any]:
        """
        Detect UI elements in a screenshot
        
        Args:
            image_path: Path to image file
            image: PIL Image object
            
        Returns:
            Dictionary with UI analysis results
        """
        # Use Cloud Vision to detect text and objects, then post-process for UI elements
        vision_results = await self.analyze_image(image_path, image)
        
        if not vision_results.get("success", False):
            return vision_results
        
        ui_elements = []
        
        # Extract text blocks as potential UI elements
        if "text" in vision_results and vision_results["text"].get("elements"):
            for text_element in vision_results["text"]["elements"]:
                # Detect if this might be a button, label, input field, etc.
                element_type = "text"
                text = text_element["text"].lower()
                
                # Check if it's a button (buttons usually have action verbs)
                if text in ["ok", "cancel", "submit", "login", "sign in", "sign up", 
                           "send", "save", "delete", "edit", "update", "search"]:
                    element_type = "button"
                
                # Check if it's a menu item
                elif "menu" in text or text in ["file", "edit", "view", "help", "tools", "settings"]:
                    element_type = "menu_item"
                
                # Check if it's a link (may have URL patterns or common link phrases)
                elif "http" in text or "www." in text or any(x in text for x in ["click here", "learn more"]):
                    element_type = "link"
                
                ui_elements.append({
                    "type": element_type,
                    "text": text_element["text"],
                    "position": text_element["position"],
                    "confidence": 0.8  # Placeholder confidence
                })
        
        # Extract objects that might be UI elements (icons, images, etc.)
        if "objects" in vision_results:
            for obj in vision_results["objects"]:
                # Some objects might be UI elements like icons
                if obj["name"].lower() in ["icon", "button", "checkbox", "toggle", "arrow"]:
                    ui_elements.append({
                        "type": obj["name"].lower(),
                        "text": "",  # No text for these elements
                        "position": [
                            (obj["bounding_box"]["x"], obj["bounding_box"]["y"]),
                            (obj["bounding_box"]["x"] + obj["bounding_box"]["width"], 
                             obj["bounding_box"]["y"]),
                            (obj["bounding_box"]["x"] + obj["bounding_box"]["width"], 
                             obj["bounding_box"]["y"] + obj["bounding_box"]["height"]),
                            (obj["bounding_box"]["x"], 
                             obj["bounding_box"]["y"] + obj["bounding_box"]["height"])
                        ],
                        "confidence": obj["score"]
                    })
        
        return {
            "success": True,
            "service": "vision",
            "ui_elements": ui_elements,
            "text_blocks": vision_results.get("text", {}).get("elements", []),
            "raw_objects": vision_results.get("objects", [])
        }
    
    async def analyze_image_with_genai(self, image_path=None, image=None, prompt=None) -> Dict[str, Any]:
        """
        Analyze an image using Gemini multimodal capabilities
        
        Args:
            image_path: Path to image file
            image: PIL Image object
            prompt: Text prompt to guide the analysis
            
        Returns:
            Dictionary with Gemini analysis results
        """
        if not self.genai_configured:
            return {
                "success": False,
                "error": "Generative AI (Gemini) not available",
                "service": "generative_ai"
            }
        
        try:
            # Prepare the image
            if image_path and os.path.exists(image_path):
                img = Image.open(image_path)
            elif image:
                img = image
            else:
                return {
                    "success": False, 
                    "error": "No valid image provided",
                    "service": "generative_ai"
                }
            
            # Set up Gemini Pro Vision model
            model = aiplatform.Model('projects/your-project/models/your-model-id')
            
            # Default prompt if none provided
            if not prompt:
                prompt = ("Analyze this image and describe what you see. "
                         "Identify key elements, objects, text, and the overall context.")
            
            # Generate response
            response = model.predict([prompt, img])
            
            return {
                "success": True,
                "service": "generative_ai",
                "model": "gemini-pro-vision",
                "prompt": prompt,
                "response": response,
                "analysis": response  # For consistency with vision API
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image with Gemini: {e}")
            return {
                "success": False,
                "error": str(e),
                "service": "generative_ai"
            }
    
    async def search_web(self, query: str, num_results: int = 10, 
                        search_type: str = "web") -> Dict[str, Any]:
        """
        Search the web using Google Custom Search API
        
        Args:
            query: Search query
            num_results: Number of results to return
            search_type: Type of search (web, image)
            
        Returns:
            Dictionary with search results
        """
        if not self.search_client:
            return {
                "success": False,
                "error": "Custom Search API not available",
                "service": "search"
            }
        
        try:
            # Default to 10 results, max supported is 10 per page
            num_results = min(max(1, num_results), 10)
            
            # Set up search parameters
            params = {
                'q': query,
                'num': num_results
            }
            
            # Add search type if not web
            if search_type == "image":
                params['searchType'] = 'image'
            
            # Execute search
            search_results = self.search_client.cse().list(
                cx='<YOUR_CSE_ID>',  # Need custom search engine ID
                **params
            ).execute()
            
            # Process results
            items = []
            if 'items' in search_results:
                for item in search_results['items']:
                    result_item = {
                        'title': item.get('title', ''),
                        'link': item.get('link', ''),
                        'snippet': item.get('snippet', '')
                    }
                    
                    # Add image information if available
                    if search_type == 'image' and 'image' in item:
                        result_item['image'] = {
                            'url': item['image'].get('thumbnailLink', ''),
                            'height': item['image'].get('thumbnailHeight', 0),
                            'width': item['image'].get('thumbnailWidth', 0)
                        }
                    
                    items.append(result_item)
            
            return {
                "success": True,
                "service": "search",
                "query": query,
                "search_type": search_type,
                "results": items,
                "total_results": search_results.get('searchInformation', {}).get('totalResults', 0)
            }
            
        except Exception as e:
            logger.error(f"Error searching web with Custom Search API: {e}")
            return {
                "success": False,
                "error": str(e),
                "service": "search"
            }
    
    async def generate_text(self, prompt: str, max_tokens: int = 1024, 
                           temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate text using Google's Generative AI (Gemini)
        
        Args:
            prompt: Text prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            
        Returns:
            Dictionary with generated text
        """
        if not self.genai_configured:
            return {
                "success": False,
                "error": "Generative AI not available",
                "service": "generative_ai"
            }
        
        try:
            # Set up model and generation config
            model = aiplatform.Model('projects/your-project/models/your-model-id')
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "top_p": 0.95,
                "top_k": 40
            }
            
            # Generate content
            response = model.predict(prompt, generation_config=generation_config)
            
            return {
                "success": True,
                "service": "generative_ai",
                "model": "gemini-pro",
                "prompt": prompt,
                "response": response,
                "generated_text": response  # For consistency
            }
            
        except Exception as e:
            logger.error(f"Error generating text with Gemini: {e}")
            return {
                "success": False,
                "error": str(e),
                "service": "generative_ai"
            }
    
    async def analyze_text_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze text sentiment using Language API
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis
        """
        if not self.vision_client:  # We use same credentials pattern as vision
            return {
                "success": False,
                "error": "Language API not available",
                "service": "language"
            }
        
        try:
            # This is a placeholder. In a real implementation, we would
            # initialize the Language API client separately.
            return {
                "success": False,
                "error": "Language API implementation pending",
                "service": "language"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing text sentiment: {e}")
            return {
                "success": False,
                "error": str(e),
                "service": "language"
            }
    
    def service_status(self) -> Dict[str, bool]:
        """Get the status of all services"""
        return {
            "vision": self.vision_client is not None,
            "search": self.search_client is not None,
            "generative_ai": self.genai_configured
        }
