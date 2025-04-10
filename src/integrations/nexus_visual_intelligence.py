"""
NEXUS Visual Intelligence - Integrated Multimodal Vision System
Combines Cloud Vision, Gemini, and local vision capabilities in an adaptive system
"""
import os
import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import base64
import io
from datetime import datetime
from PIL import Image, ImageDraw

# Import the Adaptive Vision system
from .adaptive_vision import AdaptiveVision
from .async_vector_wrapper import AsyncVectorWrapper

logger = logging.getLogger(__name__)

class NexusVisualIntelligence:
    """
    NEXUS Visual Intelligence
    
    A self-adapting vision system that:
    1. Combines multiple capabilities (Cloud Vision, Gemini, local vision)
    2. Learns from each image it processes
    3. Adapts to use whatever services are available
    4. Stores acquired knowledge for future reference
    
    This implementation follows NEXUS's philosophy of continuous learning
    and adaptation without being limited by hard-coded rules.
    """
    
    def __init__(self, 
                gemini_api_key: str = None, 
                service_account_path: str = None,
                vector_storage = None):
        """
        Initialize NEXUS Visual Intelligence
        
        Args:
            gemini_api_key: Optional Gemini API key
            service_account_path: Optional path to Google Cloud service account
            vector_storage: Optional vector storage for knowledge
        """
        # Config values
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        self.service_account_path = service_account_path
        
        # Wrap vector storage with async-friendly wrapper if provided
        if vector_storage:
            self.vector_storage = AsyncVectorWrapper(vector_storage)
        else:
            self.vector_storage = None
        
        # Find service account if not provided
        if not self.service_account_path:
            potential_paths = [
                "autonomus-1743898709312-8589efbc502d.json",  # Current directory
                os.path.expanduser("~/autonomus-1743898709312-8589efbc502d.json"),  # Home
                "d:/Ai/Nexus/autonomus-1743898709312-8589efbc502d.json",  # Project dir
                os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")  # Environment variable
            ]
            
            for path in potential_paths:
                if path and os.path.exists(path):
                    self.service_account_path = path
                    logger.info(f"Found service account at: {path}")
                    break
        
        # Initialize the core adaptive vision system
        self.vision = AdaptiveVision(
            gemini_api_key=self.gemini_api_key,
            cloud_vision_credentials=self.service_account_path,
            vector_storage=self.vector_storage
        )
        
        # Track learning progress
        self.analyzed_images = 0
        self.knowledge_items = 0
        self.capabilities = self.vision.get_capabilities()
        
        logger.info(f"NEXUS Visual Intelligence initialized with capabilities: {self.capabilities}")
    
    async def analyze_image(self, 
                           image_path: str = None, 
                           image: Image.Image = None, 
                           analysis_type: str = "full") -> Dict[str, Any]:
        """
        Analyze an image using all available capabilities
        
        Args:
            image_path: Path to image file
            image: PIL Image object
            analysis_type: Type of analysis ("full", "ui", "text", "objects")
            
        Returns:
            Dictionary with analysis results
        """
        if not image_path and not image:
            return {"success": False, "error": "No image provided"}
        
        # Track that we're analyzing a new image
        self.analyzed_images += 1
        
        # Choose analysis method based on type
        if analysis_type == "ui":
            results = await self.vision.detect_ui_elements(image_path, image)
        elif analysis_type == "text":
            results = await self.vision.extract_text_from_image(image_path, image)
        else:  # Default to full analysis
            results = await self.vision.analyze_image(image_path, image)
        
        # Augment results with timestamp and type
        if isinstance(results, dict):
            results["timestamp"] = datetime.now().isoformat()
            results["analysis_type"] = analysis_type
            
            # Record what we learned if successful
            if results.get("success", False):
                try:
                    await self._record_learning(results, image_path)
                except Exception as e:
                    logger.error(f"Error recording learning: {e}")
        
        return results
    
    async def analyze_ui(self, 
                        screenshot_path: str = None, 
                        screenshot: Image.Image = None) -> Dict[str, Any]:
        """
        Analyze UI elements in a screenshot
        
        Args:
            screenshot_path: Path to screenshot file
            screenshot: PIL Image object
            
        Returns:
            Dictionary with UI analysis results
        """
        return await self.analyze_image(
            image_path=screenshot_path,
            image=screenshot,
            analysis_type="ui"
        )
    
    async def extract_text(self, 
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
        return await self.analyze_image(
            image_path=image_path,
            image=image,
            analysis_type="text"
        )
    
    async def find_similar_images(self, 
                                 query: str = None, 
                                 image_path: str = None,
                                 image: Image.Image = None,
                                 max_results: int = 5) -> Dict[str, Any]:
        """
        Find similar images in the knowledge base
        
        Args:
            query: Text query to search for
            image_path: Path to image to use as reference
            image: PIL Image to use as reference
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary with search results
        """
        if not self.vector_storage:
            return {"success": False, "error": "No vector storage available for search"}
        
        combined_query = query
        
        # If an image is provided, analyze it first to use as part of the search
        if image_path or image:
            try:
                # Quick analysis to extract features for search
                analysis = await self.vision.analyze_image(image_path, image)
                
                # Start with description if available
                if "description" in analysis:
                    combined_query = analysis["description"]
                
                # Add labels if available
                if "labels" in analysis and analysis["labels"]:
                    labels_text = " ".join([label["description"] for label in analysis["labels"][:5]])
                    if combined_query:
                        combined_query += " " + labels_text
                    else:
                        combined_query = labels_text
            except Exception as e:
                logger.warning(f"Error analyzing reference image for search: {e}")
        
        if not combined_query:
            return {"success": False, "error": "No query or valid image provided"}
        
        # Search the vector database
        try:
            # Properly handle the async/await by ensuring the vector_storage.similarity_search
            # is properly awaited and the result comes back as a list, not a coroutine
            results = await self.vector_storage.similarity_search(
                query=combined_query,
                k=max_results
            )
            
            # Make sure results is actually a list
            if not isinstance(results, list):
                results = [{"text": "Result format error", "metadata": {}}]
            
            return {
                "success": True,
                "query": combined_query,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error searching for similar images: {e}")
            return {"success": False, "error": str(e)}
    
    async def _record_learning(self, results, image_path=None):
        """Record what we've learned from this image"""
        if not self.vector_storage:
            return
        
        try:
            # Create a summary of what we learned
            learned_text = []
            
            if image_path:
                learned_text.append(f"Image: {os.path.basename(image_path)}")
            
            # Add description
            if "description" in results:
                learned_text.append(f"Understanding: {results['description']}")
            
            # Add labels
            if "labels" in results and results["labels"]:
                labels_text = ", ".join([label["description"] for label in results["labels"][:10]])
                learned_text.append(f"Recognized: {labels_text}")
            
            # Add text found
            if "text" in results and isinstance(results["text"], dict) and results["text"].get("full_text"):
                learned_text.append(f"Text content: {results['text']['full_text']}")
            elif "text" in results and isinstance(results["text"], str):
                learned_text.append(f"Text content: {results['text']}")
            
            # Add UI elements
            if "ui_elements" in results and results["ui_elements"]:
                ui_elements = ", ".join([f"{e.get('type', 'element')}: {e.get('text', 'unnamed')}" 
                                       for e in results["ui_elements"][:5]])
                learned_text.append(f"UI elements: {ui_elements}")
            
            # Store in vector database
            if learned_text:
                knowledge_text = "\n".join(learned_text)
                
                knowledge_id = f"vision_knowledge_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                
                # Ensure we pass valid lists not futures/coroutines 
                await self.vector_storage.add_texts(
                    texts=[knowledge_text],
                    metadatas=[{
                        "source": "nexus_visual_intelligence",
                        "image_path": image_path if image_path else "uploaded_image",
                        "timestamp": results.get("timestamp", datetime.now().isoformat()),
                        "analysis_type": results.get("analysis_type", "unknown"),
                        # Convert sources to a string with join if it's a list, or use a default
                        "capabilities_used": ",".join(results.get("sources", ["local_vision"]))
                    }],
                    ids=[knowledge_id]
                )
                
                self.knowledge_items += 1
                logger.info(f"Stored new visual knowledge: {knowledge_id}")
        
        except Exception as e:
            logger.error(f"Error recording learning: {e}")
    
    def get_capabilities(self) -> Dict[str, bool]:
        """Get available capabilities"""
        return self.capabilities
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about learning progress"""
        return {
            "analyzed_images": self.analyzed_images,
            "knowledge_items": self.knowledge_items,
            "available_capabilities": self.capabilities
        }
