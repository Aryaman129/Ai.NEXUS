"""
Visual Intelligence Module for NEXUS RAG Engine
Combines multimodal search, Cloud Vision, and vector storage for visual understanding
"""
import logging
import asyncio
import os
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from PIL import Image

from .knowledge_manager import KnowledgeManager
from ...integrations.cloud_vision import CloudVisionService
from ...integrations.multimodal_search import MultimodalSearch

logger = logging.getLogger(__name__)

class VisualIntelligence:
    """
    Visual Intelligence for NEXUS that combines multiple visual understanding systems:
    
    1. Google Cloud Vision API - for advanced image analysis
    2. Multimodal Search - for discovering visual content from the web
    3. Vector storage - for storing and retrieving visual knowledge
    
    This creates a continuously learning visual system that can:
    - Understand UI elements and screens
    - Locate visual information on the web
    - Store and retrieve visual patterns and concepts
    - Learn from past visual experiences
    
    The system dynamically adapts to what's available and learns continuously.
    """
    
    def __init__(self, vector_storage, ollama_client=None, knowledge_base_dir="memory/visual_intelligence"):
        """Initialize the visual intelligence system"""
        # Store references
        self.vector_storage = vector_storage
        self.ollama = ollama_client
        
        # Create base directory
        self.base_dir = Path(knowledge_base_dir)
        self.base_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize multimodal search
        self.multimodal_search = MultimodalSearch(vector_storage=vector_storage)
        
        # Try to initialize Cloud Vision
        self.cloud_vision = None
        self._initialize_cloud_vision()
        
        # Initialize knowledge manager for text knowledge
        self.knowledge_manager = KnowledgeManager(
            ollama_client=ollama_client,
            vector_storage=vector_storage,
            knowledge_base_dir=str(self.base_dir / "knowledge")
        )
        
        logger.info(f"Visual Intelligence initialized (Cloud Vision available: {self.cloud_vision is not None})")
    
    def _initialize_cloud_vision(self):
        """Initialize Cloud Vision API if credentials are available"""
        # Standard locations to look for credentials
        service_account_paths = [
            "autonomus-1743898709312-8589efbc502d.json",  # Same directory
            os.path.expanduser("~/autonomus-1743898709312-8589efbc502d.json"),  # Home directory
            "d:/Ai/Nexus/autonomus-1743898709312-8589efbc502d.json",  # Project directory
            os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")  # Environment variable
        ]
        
        # Try each path
        for path in service_account_paths:
            if path and os.path.exists(path):
                try:
                    self.cloud_vision = CloudVisionService(
                        service_account_path=path,
                        vector_storage=self.vector_storage
                    )
                    logger.info(f"Cloud Vision initialized with service account: {path}")
                    break
                except Exception as e:
                    logger.error(f"Failed to initialize Cloud Vision with {path}: {e}")
        
        if not self.cloud_vision:
            logger.warning("Cloud Vision not available. Visual analysis will be limited.")
    
    async def analyze_image(self, image_path=None, image=None, store_results=True):
        """
        Perform comprehensive image analysis using all available methods
        
        Args:
            image_path: Path to image file
            image: PIL Image object
            store_results: Whether to store results in vector database
            
        Returns:
            Dictionary with analysis results
        """
        results = {
            "cloud_vision": None,
            "search_results": None,
            "local_analysis": None,
            "learned_items": []
        }
        
        # Use Cloud Vision if available
        if self.cloud_vision and self.cloud_vision.available:
            try:
                cloud_vision_results = await self.cloud_vision.analyze_image(
                    image_path=image_path,
                    image=image,
                    store_results=store_results
                )
                
                results["cloud_vision"] = cloud_vision_results
                
                # If we found labels, try to enrich with web search
                if cloud_vision_results.get("success", False) and cloud_vision_results.get("labels"):
                    # Get top labels
                    top_labels = [label["description"] for label in 
                                cloud_vision_results.get("labels", [])[:3]]
                    
                    # Search for additional information
                    if top_labels:
                        search_query = f"information about {', '.join(top_labels)}"
                        search_results = await self.multimodal_search.search_web(
                            query=search_query,
                            max_results=3,
                            include_images=True
                        )
                        
                        results["search_results"] = search_results
                
                # Store findings as knowledge
                if store_results and cloud_vision_results.get("success", False):
                    await self._store_vision_knowledge(cloud_vision_results, image_path)
                    
            except Exception as e:
                logger.error(f"Error during Cloud Vision analysis: {e}")
        
        return results
    
    async def analyze_ui(self, screenshot_path=None, screenshot=None):
        """
        Analyze UI elements in a screenshot
        
        Args:
            screenshot_path: Path to screenshot file
            screenshot: PIL Image object
            
        Returns:
            Dictionary with UI analysis results
        """
        if self.cloud_vision and self.cloud_vision.available:
            try:
                return await self.cloud_vision.detect_ui_elements(
                    image_path=screenshot_path,
                    image=screenshot,
                    store_results=True
                )
            except Exception as e:
                logger.error(f"Error during UI analysis with Cloud Vision: {e}")
        
        # Fallback to basic analysis or return empty result
        return {
            "success": False,
            "message": "Cloud Vision not available for UI analysis",
            "ui_elements": []
        }
    
    async def search_ui_reference(self, element_type, description=""):
        """
        Search for reference UI elements to assist with recognition
        
        Args:
            element_type: Type of UI element (button, icon, menu, etc.)
            description: Description of the element
            
        Returns:
            Dictionary with search results and learning outcomes
        """
        try:
            # Search for UI elements on the web
            search_results = await self.multimodal_search.search_ui_elements(
                element_type=element_type,
                description=description
            )
            
            # Extract and store knowledge
            learned_items = []
            
            if search_results.get("success", False) and search_results.get("results"):
                for result in search_results.get("results", [])[:3]:  # Limit to top 3
                    image_url = result.get("url", "")
                    
                    if not image_url:
                        continue
                    
                    # Download and analyze with Cloud Vision if available
                    download_result = await self.multimodal_search.download_and_store_image(
                        image_url=image_url,
                        metadata={
                            "element_type": element_type,
                            "description": description,
                            "title": result.get("title", ""),
                            "source": "web_search"
                        }
                    )
                    
                    if download_result.get("success", False):
                        local_path = download_result.get("local_path")
                        learned_items.append({
                            "type": "image",
                            "element_type": element_type,
                            "path": local_path
                        })
                        
                        # If Cloud Vision is available, analyze the downloaded image
                        if self.cloud_vision and self.cloud_vision.available:
                            await self.cloud_vision.analyze_image(
                                image_path=local_path,
                                store_results=True
                            )
            
            return {
                "search_results": search_results,
                "learned_items": learned_items,
                "element_type": element_type,
                "description": description
            }
            
        except Exception as e:
            logger.error(f"Error searching for UI reference: {e}")
            return {
                "success": False,
                "error": str(e),
                "element_type": element_type,
                "description": description
            }
    
    async def find_similar_visuals(self, image_path=None, image=None, query=None, max_results=5):
        """
        Find visually similar items in the knowledge base
        
        Args:
            image_path: Path to image file
            image: PIL Image object
            query: Text query to search for
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary with similar visuals
        """
        results = {
            "query_text": query,
            "similar_items": []
        }
        
        # If we have an image, analyze it with Cloud Vision to get labels
        if (image_path or image) and self.cloud_vision and self.cloud_vision.available:
            analysis = await self.cloud_vision.analyze_image(
                image_path=image_path,
                image=image,
                store_results=False
            )
            
            if analysis.get("success", False) and analysis.get("labels"):
                # Create query from labels
                labels = [label["description"] for label in analysis.get("labels", [])[:5]]
                if not query:
                    query = " ".join(labels)
                else:
                    query = f"{query} {' '.join(labels)}"
        
        # If we have a text query, search the vector database
        if query:
            vector_results = self.vector_storage.similarity_search(
                query=query,
                k=max_results
            )
            
            results["similar_items"] = vector_results
        
        return results
    
    async def _store_vision_knowledge(self, vision_results, image_path=None):
        """Store vision analysis results as knowledge"""
        # Extract key information to store
        if not vision_results.get("success", False):
            return []
        
        stored_items = []
        
        # Store label information
        if vision_results.get("labels"):
            labels_text = "Image contains: " + ", ".join(
                [label["description"] for label in vision_results.get("labels", [])]
            )
            
            knowledge_id = await self.knowledge_manager.add_knowledge(
                text=labels_text,
                metadata={
                    "source": "cloud_vision",
                    "image_path": image_path,
                    "content_type": "image_labels"
                },
                category="visual_elements"
            )
            
            stored_items.append({
                "id": knowledge_id,
                "type": "labels",
                "text": labels_text
            })
        
        # Store text found in image
        if vision_results.get("text", {}).get("full_text"):
            text = vision_results.get("text", {}).get("full_text", "")
            
            knowledge_id = await self.knowledge_manager.add_knowledge(
                text=f"Text in image: {text}",
                metadata={
                    "source": "cloud_vision",
                    "image_path": image_path,
                    "content_type": "image_text"
                },
                category="visual_elements"
            )
            
            stored_items.append({
                "id": knowledge_id,
                "type": "text",
                "text": text
            })
        
        # Store objects found
        if vision_results.get("objects"):
            objects_text = "Objects detected: " + ", ".join(
                [obj["name"] for obj in vision_results.get("objects", [])]
            )
            
            knowledge_id = await self.knowledge_manager.add_knowledge(
                text=objects_text,
                metadata={
                    "source": "cloud_vision",
                    "image_path": image_path,
                    "content_type": "image_objects"
                },
                category="visual_elements"
            )
            
            stored_items.append({
                "id": knowledge_id,
                "type": "objects",
                "text": objects_text
            })
            
        return stored_items
