"""
Multimodal Knowledge Manager for NEXUS RAG Engine
Extends knowledge capabilities to include visual data and web-discovered information
"""
import logging
import asyncio
import json
import time
from datetime import datetime
import os
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import numpy as np
from PIL import Image

from .knowledge_manager import KnowledgeManager
from .vector_storage import VectorStorage
from ...integrations.multimodal_search import MultimodalSearch

logger = logging.getLogger(__name__)

class MultimodalKnowledgeManager(KnowledgeManager):
    """
    Multimodal Knowledge Manager that extends standard KnowledgeManager
    with capabilities for handling visual data and web-discovered information.
    
    This component:
    1. Can search and retrieve information from the web
    2. Stores both textual and visual knowledge
    3. Allows for similarity search across modalities
    4. Continuously learns and improves from discovered data
    
    This is a learning-focused system that adapts over time based on what
    it discovers through interaction with the web and search results.
    """
    
    def __init__(self, 
                 ollama_client=None, 
                 vector_storage: Optional[VectorStorage] = None,
                 knowledge_base_dir: str = "memory/multimodal_knowledge"):
        """Initialize the multimodal knowledge manager"""
        # Initialize the base KnowledgeManager
        super().__init__(
            ollama_client=ollama_client,
            vector_storage=vector_storage,
            knowledge_base_dir=knowledge_base_dir
        )
        
        # Additional knowledge categories for multimodal content
        self.knowledge_categories.update({
            "visual_elements": "UI elements, icons, and visual components",
            "navigation_patterns": "Navigation flows and UI interaction patterns",
            "discovered_content": "Content discovered through web search"
        })
        
        # Initialize multimodal search
        self.multimodal_search = MultimodalSearch(vector_storage=self.vector_storage)
        
        # Visual knowledge base directory
        self.visual_kb_dir = Path(knowledge_base_dir) / "visual_elements"
        self.visual_kb_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info("NEXUS Multimodal Knowledge Manager initialized")
    
    async def search_and_learn(self, query: str, max_results: int = 5, 
                              include_images: bool = True) -> Dict[str, Any]:
        """
        Search the web for information and learn from the results
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            include_images: Whether to include images in results
            
        Returns:
            Dictionary with search results and learning outcomes
        """
        # Perform web search
        search_results = await self.multimodal_search.search_web(
            query=query,
            max_results=max_results,
            include_images=include_images
        )
        
        # Extract and store knowledge from results
        learned_items = []
        
        if search_results.get("success", False):
            # Learn from text results
            for result in search_results.get("results", []):
                # Extract text content and metadata
                content = result.get("snippet", "")
                title = result.get("title", "")
                url = result.get("url", "")
                
                if not content or not url:
                    continue
                
                # Store in knowledge base
                knowledge_id = await self.add_knowledge(
                    text=f"{title}: {content}",
                    metadata={
                        "source": "web_search",
                        "url": url,
                        "title": title,
                        "query": query
                    },
                    category="discovered_content"
                )
                
                if knowledge_id:
                    learned_items.append({
                        "id": knowledge_id,
                        "type": "text",
                        "title": title,
                        "url": url
                    })
            
            # Learn from image results if present
            if include_images and search_results.get("image_results"):
                for image in search_results.get("image_results", []):
                    # Download and store image
                    image_url = image.get("url", "")
                    title = image.get("title", "")
                    description = image.get("description", "")
                    
                    if not image_url:
                        continue
                    
                    # Store image with metadata
                    result = await self.multimodal_search.download_and_store_image(
                        image_url=image_url,
                        metadata={
                            "title": title,
                            "description": description,
                            "query": query,
                            "source": "web_search",
                            "source_url": image.get("source_url", "")
                        }
                    )
                    
                    if result.get("success", False):
                        learned_items.append({
                            "id": result.get("local_path", ""),
                            "type": "image",
                            "title": title,
                            "path": result.get("local_path", "")
                        })
        
        # Return combined results
        return {
            "search_results": search_results,
            "learned_items": learned_items,
            "query": query,
            "timestamp": time.time()
        }
    
    async def search_ui_elements(self, element_type: str, description: str = "") -> Dict[str, Any]:
        """
        Search for UI elements and learn them for future use
        
        Args:
            element_type: Type of UI element (button, icon, menu, etc.)
            description: Description of the element
            
        Returns:
            Dictionary with search results and learning outcomes
        """
        # Search for UI elements
        search_results = await self.multimodal_search.search_ui_elements(
            element_type=element_type,
            description=description
        )
        
        # Extract and store knowledge from results
        learned_items = []
        
        if search_results.get("success", False):
            for element in search_results.get("results", []):
                # Download and store element
                element_url = element.get("url", "")
                title = element.get("title", "") or f"{element_type}: {description}"
                
                if not element_url:
                    continue
                
                # Store image with metadata
                result = await self.multimodal_search.download_and_store_image(
                    image_url=element_url,
                    metadata={
                        "element_type": element_type,
                        "description": description,
                        "title": title,
                        "source": "ui_element_search",
                        "category": "visual_elements"
                    }
                )
                
                if result.get("success", False):
                    # Add text knowledge about this element
                    knowledge_id = await self.add_knowledge(
                        text=f"UI Element: {element_type} - {description}",
                        metadata={
                            "element_type": element_type,
                            "description": description,
                            "image_path": result.get("local_path", ""),
                            "category": "visual_elements"
                        },
                        category="visual_elements"
                    )
                    
                    learned_items.append({
                        "id": result.get("local_path", ""),
                        "knowledge_id": knowledge_id,
                        "type": "ui_element",
                        "element_type": element_type,
                        "path": result.get("local_path", "")
                    })
        
        # Return combined results
        return {
            "search_results": search_results,
            "learned_items": learned_items,
            "element_type": element_type,
            "description": description,
            "timestamp": time.time()
        }
    
    async def find_similar_ui_elements(self, image: Image.Image = None, 
                                     image_path: str = None, 
                                     element_type: str = None,
                                     description: str = None) -> List[Dict[str, Any]]:
        """
        Find UI elements similar to an image or description
        
        Args:
            image: Optional PIL Image to find similar elements for
            image_path: Optional path to image file
            element_type: Optional type of element to filter by
            description: Optional description to search by
            
        Returns:
            List of similar UI elements
        """
        if not image and not image_path and not description:
            return []
        
        # If we have an image path but no image, load it
        if not image and image_path:
            try:
                image = Image.open(image_path)
            except Exception as e:
                logger.error(f"Error loading image: {e}")
                image = None
        
        # If we have an image, convert to a query
        query = ""
        if description:
            query = description
        elif element_type:
            query = f"UI Element: {element_type}"
        else:
            query = "UI Element"
        
        # Retrieve from vector database
        filter_dict = {"category": "visual_elements"}
        if element_type:
            filter_dict["element_type"] = element_type
        
        # Get similar elements from vector storage
        results = self.vector_storage.similarity_search(
            query=query,
            k=10,
            filter_dict=filter_dict
        )
        
        return results
    
    async def learn_navigation_pattern(self, pattern_name: str, 
                                     steps: List[Dict[str, Any]],
                                     description: str = "",
                                     example_image: Optional[Image.Image] = None) -> str:
        """
        Learn a navigation pattern for future reference
        
        Args:
            pattern_name: Name of the navigation pattern
            steps: List of steps in the pattern
            description: Description of the pattern
            example_image: Optional example image of the pattern
            
        Returns:
            ID of the stored knowledge
        """
        # Format the pattern as text
        pattern_text = f"Navigation Pattern: {pattern_name}\n\n"
        pattern_text += f"Description: {description}\n\n"
        pattern_text += "Steps:\n"
        
        for i, step in enumerate(steps, 1):
            pattern_text += f"{i}. {step.get('action', '')} - {step.get('description', '')}\n"
        
        # Store in knowledge base
        pattern_id = await self.add_knowledge(
            text=pattern_text,
            metadata={
                "pattern_name": pattern_name,
                "steps": steps,
                "description": description
            },
            category="navigation_patterns"
        )
        
        # If we have an example image, store it too
        if example_image:
            # Save image
            image_path = self.visual_kb_dir / f"pattern_{pattern_name.replace(' ', '_')}.png"
            example_image.save(image_path)
            
            # Update knowledge with image path
            await self.vector_storage.add_texts(
                texts=[f"Navigation Pattern Image: {pattern_name}"],
                metadatas=[{
                    "pattern_name": pattern_name,
                    "image_path": str(image_path),
                    "description": description,
                    "category": "navigation_patterns",
                    "pattern_id": pattern_id
                }]
            )
        
        return pattern_id
    
    async def get_navigation_patterns(self, query: str = None) -> List[Dict[str, Any]]:
        """
        Get navigation patterns that match a query
        
        Args:
            query: Optional search query
            
        Returns:
            List of matching navigation patterns
        """
        # Use a default query if none provided
        if not query:
            query = "Navigation Pattern"
        
        # Retrieve from vector database
        filter_dict = {"category": "navigation_patterns"}
        
        # Get patterns from vector storage
        results = self.vector_storage.similarity_search(
            query=query,
            k=10,
            filter_dict=filter_dict
        )
        
        return results
    
    async def extract_and_learn_from_url(self, url: str) -> Dict[str, Any]:
        """
        Extract content from a URL and learn from it
        
        Args:
            url: URL to extract content from
            
        Returns:
            Dictionary with extraction results and learning outcomes
        """
        # Extract metadata from URL
        metadata = await self.multimodal_search.extract_metadata_from_url(url)
        
        learned_items = []
        
        if "error" not in metadata:
            # Learn from page content
            title = metadata.get("title", "")
            description = metadata.get("description", "")
            
            if title or description:
                # Store in knowledge base
                content = f"{title}: {description}"
                knowledge_id = await self.add_knowledge(
                    text=content,
                    metadata={
                        "source": "web_extraction",
                        "url": url,
                        "title": title
                    },
                    category="discovered_content"
                )
                
                if knowledge_id:
                    learned_items.append({
                        "id": knowledge_id,
                        "type": "text",
                        "title": title,
                        "url": url
                    })
            
            # Learn from images
            for img in metadata.get("images", []):
                image_url = img.get("url", "")
                alt = img.get("alt", "")
                
                if not image_url:
                    continue
                
                # Determine if this might be a UI element
                is_ui_element = False
                element_type = "unknown"
                
                # Simple heuristic based on alt text and url
                if alt:
                    ui_indicators = ["button", "icon", "menu", "nav", "logo", "ui", "interface"]
                    for indicator in ui_indicators:
                        if indicator in alt.lower():
                            is_ui_element = True
                            element_type = indicator
                            break
                
                # Store image with appropriate category
                category = "visual_elements" if is_ui_element else "discovered_content"
                
                result = await self.multimodal_search.download_and_store_image(
                    image_url=image_url,
                    metadata={
                        "alt_text": alt,
                        "source_url": url,
                        "element_type": element_type if is_ui_element else None,
                        "is_ui_element": is_ui_element,
                        "category": category
                    }
                )
                
                if result.get("success", False):
                    learned_items.append({
                        "id": result.get("local_path", ""),
                        "type": "image",
                        "is_ui_element": is_ui_element,
                        "alt": alt,
                        "path": result.get("local_path", "")
                    })
            
            # Learn from icons
            for icon in metadata.get("icons", []):
                icon_url = icon.get("url", "")
                icon_type = icon.get("type", "")
                
                if not icon_url:
                    continue
                
                # Icons are UI elements by definition
                result = await self.multimodal_search.download_and_store_image(
                    image_url=icon_url,
                    metadata={
                        "element_type": "icon",
                        "icon_type": icon_type,
                        "source_url": url,
                        "is_ui_element": True,
                        "category": "visual_elements"
                    }
                )
                
                if result.get("success", False):
                    learned_items.append({
                        "id": result.get("local_path", ""),
                        "type": "icon",
                        "icon_type": icon_type,
                        "path": result.get("local_path", "")
                    })
        
        # Return results
        return {
            "metadata": metadata,
            "learned_items": learned_items,
            "url": url,
            "timestamp": time.time()
        }
    
    async def get_visual_elements_by_type(self, element_type: str) -> List[Dict[str, Any]]:
        """
        Get visual elements by type
        
        Args:
            element_type: Type of element to retrieve
            
        Returns:
            List of matching visual elements
        """
        # Retrieve from vector database
        filter_dict = {
            "category": "visual_elements",
            "element_type": element_type
        }
        
        # Get elements from vector storage
        results = self.vector_storage.similarity_search(
            query=f"UI Element: {element_type}",
            k=10,
            filter_dict=filter_dict
        )
        
        return results
