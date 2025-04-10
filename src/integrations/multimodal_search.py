"""
Multimodal Search Integration for NEXUS
Provides capabilities to search and extract both text and visual data from the web
"""
import os
import logging
import asyncio
import json
import time
import base64
from io import BytesIO
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import requests
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

class MultimodalSearch:
    """
    Multimodal search capabilities for NEXUS.
    
    This component enables searching for and extracting both text and visual content:
    - Web search for information
    - Image search for UI elements, icons, visual references
    - Extraction and storage of visual data for future reference
    
    It supports multiple search providers while respecting API rate limits.
    """
    
    def __init__(self, vector_storage=None):
        """Initialize the multimodal search capabilities"""
        self.config = {
            "search_providers": {
                "duckduckgo": {
                    "enabled": True,
                    "requires_api_key": False,
                    "supports_images": False
                },
                "google": {
                    "enabled": False,
                    "requires_api_key": True,
                    "api_key": os.environ.get("GOOGLE_API_KEY", ""),
                    "cx": os.environ.get("GOOGLE_CX", ""),
                    "supports_images": True,
                    "daily_limit": 100,  # Free tier daily limit
                    "usage_count": 0
                },
                "bing": {
                    "enabled": False,
                    "requires_api_key": True,
                    "api_key": os.environ.get("BING_API_KEY", ""),
                    "supports_images": True,
                    "daily_limit": 1000,  # Free tier daily limit
                    "usage_count": 0
                }
            },
            "cache_dir": "cache/search",
            "image_cache_dir": "cache/images",
            "cache_expiry": 86400,  # 24 hours in seconds
            "max_results": 10,
            "timeout": 10  # seconds
        }
        
        # Create cache directories
        self.cache_dir = Path(self.config["cache_dir"])
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        self.image_cache_dir = Path(self.config["image_cache_dir"])
        self.image_cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Store vector storage reference if provided
        self.vector_storage = vector_storage
        
        # Initialize search providers
        self._initialize_search_providers()
        
        logger.info("Multimodal Search initialized")
    
    def _initialize_search_providers(self):
        """Initialize search providers based on available capabilities"""
        # Check for DuckDuckGo
        try:
            import duckduckgo_search
            self.config["search_providers"]["duckduckgo"]["enabled"] = True
            logger.info("DuckDuckGo search enabled")
        except ImportError:
            self.config["search_providers"]["duckduckgo"]["enabled"] = False
            logger.info("DuckDuckGo search not available (duckduckgo_search not installed)")
        
        # Check for Google API key
        if self.config["search_providers"]["google"]["api_key"] and self.config["search_providers"]["google"]["cx"]:
            self.config["search_providers"]["google"]["enabled"] = True
            logger.info("Google search API enabled")
        else:
            logger.info("Google search API not enabled (missing API key or CX)")
        
        # Check for Bing API key
        if self.config["search_providers"]["bing"]["api_key"]:
            self.config["search_providers"]["bing"]["enabled"] = True
            logger.info("Bing search API enabled")
        else:
            logger.info("Bing search API not enabled (missing API key)")
    
    async def search_web(self, query: str, max_results: int = 10, include_images: bool = False) -> Dict[str, Any]:
        """
        Search the web for information on a topic
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            include_images: Whether to include images in results
            
        Returns:
            Dictionary with search results
        """
        # Check cache first
        cache_key = f"web_{query.replace(' ', '_')}_{max_results}_{include_images}"
        cached_result = self._check_cache(cache_key)
        if cached_result:
            logger.info(f"Returning cached web search results for: {query}")
            return cached_result
        
        # Initialize results
        results = {
            "query": query,
            "timestamp": time.time(),
            "success": False,
            "results": [],
            "provider": None,
            "image_results": [] if include_images else None
        }
        
        # Try search providers in priority order
        providers = ["duckduckgo", "google", "bing"]
        
        for provider in providers:
            provider_config = self.config["search_providers"][provider]
            
            if not provider_config["enabled"]:
                continue
                
            try:
                # Call the appropriate search method
                if provider == "duckduckgo":
                    search_results = await self._search_duckduckgo(query, max_results)
                elif provider == "google":
                    search_results = await self._search_google(query, max_results)
                elif provider == "bing":
                    search_results = await self._search_bing(query, max_results)
                else:
                    continue
                
                # If successful, store provider and results
                if search_results.get("success", False):
                    results["success"] = True
                    results["results"] = search_results.get("results", [])
                    results["provider"] = provider
                    
                    # If we need images and this provider doesn't support them,
                    # we'll get them separately
                    if include_images and not provider_config["supports_images"]:
                        image_results = await self.search_images(query, max_results=max(5, max_results // 2))
                        results["image_results"] = image_results.get("results", [])
                    elif include_images:
                        results["image_results"] = search_results.get("image_results", [])
                    
                    # Cache results
                    self._cache_results(cache_key, results)
                    
                    return results
                    
            except Exception as e:
                logger.error(f"Error searching with {provider}: {e}")
        
        # If all search providers failed
        logger.warning(f"All search providers failed for query: {query}")
        return results
    
    async def search_images(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Search for images related to a query
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary with image search results
        """
        # Check cache first
        cache_key = f"img_{query.replace(' ', '_')}_{max_results}"
        cached_result = self._check_cache(cache_key)
        if cached_result:
            logger.info(f"Returning cached image search results for: {query}")
            return cached_result
        
        # Initialize results
        results = {
            "query": query,
            "timestamp": time.time(),
            "success": False,
            "results": [],
            "provider": None
        }
        
        # Try search providers that support images in priority order
        providers = ["google", "bing"]
        
        for provider in providers:
            provider_config = self.config["search_providers"][provider]
            
            if not provider_config["enabled"] or not provider_config["supports_images"]:
                continue
                
            try:
                # Call the appropriate search method
                if provider == "google":
                    search_results = await self._search_google_images(query, max_results)
                elif provider == "bing":
                    search_results = await self._search_bing_images(query, max_results)
                else:
                    continue
                
                # If successful, store provider and results
                if search_results.get("success", False):
                    results["success"] = True
                    results["results"] = search_results.get("results", [])
                    results["provider"] = provider
                    
                    # Cache results
                    self._cache_results(cache_key, results)
                    
                    return results
                    
            except Exception as e:
                logger.error(f"Error searching images with {provider}: {e}")
        
        # If all search providers failed, try a fallback method
        try:
            fallback_results = await self._search_images_fallback(query, max_results)
            if fallback_results.get("success", False):
                results["success"] = True
                results["results"] = fallback_results.get("results", [])
                results["provider"] = "fallback"
                
                # Cache results
                self._cache_results(cache_key, results)
                
                return results
        except Exception as e:
            logger.error(f"Error with fallback image search: {e}")
        
        # If everything failed
        logger.warning(f"All image search methods failed for query: {query}")
        return results
    
    async def search_ui_elements(self, element_type: str, description: str = "") -> Dict[str, Any]:
        """
        Search specifically for UI elements like buttons, icons, etc.
        
        Args:
            element_type: Type of UI element (button, icon, menu, etc.)
            description: Optional description of the element
            
        Returns:
            Dictionary with UI element search results
        """
        # Formulate a more specific query for UI elements
        query = f"{element_type} UI element"
        if description:
            query += f" {description}"
        
        # Search with UI-specific terms
        results = await self.search_images(f"{query} transparent png")
        
        # Add element type metadata
        if results.get("success", False):
            for result in results.get("results", []):
                result["element_type"] = element_type
                result["description"] = description
        
        return results
    
    async def download_and_store_image(self, image_url: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Download an image and store it locally
        
        Args:
            image_url: URL of the image to download
            metadata: Optional metadata to associate with the image
            
        Returns:
            Dictionary with download results and local path
        """
        try:
            # Create a unique filename based on URL
            import hashlib
            url_hash = hashlib.md5(image_url.encode()).hexdigest()
            
            # Extract extension from URL or default to png
            import os
            from urllib.parse import urlparse
            path = urlparse(image_url).path
            ext = os.path.splitext(path)[1]
            if not ext or len(ext) > 5:  # Sanity check on extension
                ext = ".png"
            
            # Create local path
            local_path = self.image_cache_dir / f"{url_hash}{ext}"
            
            # Check if already downloaded
            if local_path.exists():
                logger.info(f"Image already downloaded: {local_path}")
                
                # If we have vector storage, ensure it's stored there too
                if self.vector_storage and metadata:
                    # Load the image
                    image = Image.open(local_path)
                    
                    # Store in vector storage if not already there
                    await self._store_image_in_vector_db(image, str(local_path), metadata)
                
                return {
                    "success": True,
                    "local_path": str(local_path),
                    "metadata": metadata,
                    "already_downloaded": True
                }
            
            # Download the image
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            response = requests.get(image_url, headers=headers, timeout=self.config["timeout"], stream=True)
            response.raise_for_status()
            
            # Save the image
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Create JSON metadata file if metadata provided
            if metadata:
                metadata_path = self.image_cache_dir / f"{url_hash}.json"
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                
                # If we have vector storage, store it there too
                if self.vector_storage:
                    # Load the image
                    image = Image.open(local_path)
                    
                    # Store in vector storage
                    await self._store_image_in_vector_db(image, str(local_path), metadata)
            
            logger.info(f"Downloaded image to {local_path}")
            
            return {
                "success": True,
                "local_path": str(local_path),
                "metadata": metadata,
                "already_downloaded": False
            }
                
        except Exception as e:
            logger.error(f"Error downloading image from {image_url}: {e}")
            return {
                "success": False,
                "error": str(e),
                "url": image_url
            }
    
    async def _store_image_in_vector_db(self, image: Image.Image, local_path: str, metadata: Dict[str, Any]) -> str:
        """Store an image in the vector database with its metadata"""
        if not self.vector_storage:
            return ""
        
        try:
            # Create a textual description of the image
            image_description = f"Image: {metadata.get('description', 'No description')}"
            if "element_type" in metadata:
                image_description += f" - Type: {metadata['element_type']}"
            if "alt_text" in metadata:
                image_description += f" - Alt text: {metadata['alt_text']}"
            
            # Add file path to metadata
            metadata["local_path"] = local_path
            
            # Add to vector storage
            ids = self.vector_storage.add_texts(
                texts=[image_description],
                metadatas=[metadata]
            )
            
            return ids[0] if ids else ""
            
        except Exception as e:
            logger.error(f"Error storing image in vector DB: {e}")
            return ""
    
    async def _search_duckduckgo(self, query: str, max_results: int) -> Dict[str, Any]:
        """Search using DuckDuckGo"""
        try:
            from duckduckgo_search import DDGS
            
            # Initialize results
            results = {
                "success": False,
                "results": []
            }
            
            # Create DuckDuckGo search object
            ddgs = DDGS()
            
            # Perform search
            search_results = list(ddgs.text(query, max_results=max_results))
            
            # Format results
            formatted_results = []
            for result in search_results:
                formatted_results.append({
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", ""),
                    "source": "DuckDuckGo"
                })
            
            results["success"] = len(formatted_results) > 0
            results["results"] = formatted_results
            
            return results
            
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return {"success": False, "error": str(e), "results": []}
    
    async def _search_google(self, query: str, max_results: int) -> Dict[str, Any]:
        """Search using Google Custom Search API"""
        try:
            provider_config = self.config["search_providers"]["google"]
            
            # Check rate limits
            if provider_config["usage_count"] >= provider_config["daily_limit"]:
                logger.warning("Google API daily limit reached")
                return {"success": False, "error": "API limit reached", "results": []}
            
            # Initialize results
            results = {
                "success": False,
                "results": []
            }
            
            # Prepare API request
            api_key = provider_config["api_key"]
            cx = provider_config["cx"]
            
            url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cx}&q={query}&num={max_results}"
            
            # Make request
            response = requests.get(url)
            response.raise_for_status()
            search_data = response.json()
            
            # Increment usage counter
            self.config["search_providers"]["google"]["usage_count"] += 1
            
            # Format results
            formatted_results = []
            if "items" in search_data:
                for item in search_data["items"]:
                    formatted_results.append({
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                        "source": "Google"
                    })
            
            results["success"] = len(formatted_results) > 0
            results["results"] = formatted_results
            
            return results
            
        except Exception as e:
            logger.error(f"Google search error: {e}")
            return {"success": False, "error": str(e), "results": []}
    
    async def _search_bing(self, query: str, max_results: int) -> Dict[str, Any]:
        """Search using Bing API"""
        try:
            provider_config = self.config["search_providers"]["bing"]
            
            # Check rate limits
            if provider_config["usage_count"] >= provider_config["daily_limit"]:
                logger.warning("Bing API daily limit reached")
                return {"success": False, "error": "API limit reached", "results": []}
            
            # Initialize results
            results = {
                "success": False,
                "results": []
            }
            
            # Prepare API request
            api_key = provider_config["api_key"]
            
            url = f"https://api.bing.microsoft.com/v7.0/search?q={query}&count={max_results}"
            headers = {"Ocp-Apim-Subscription-Key": api_key}
            
            # Make request
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            search_data = response.json()
            
            # Increment usage counter
            self.config["search_providers"]["bing"]["usage_count"] += 1
            
            # Format results
            formatted_results = []
            if "webPages" in search_data and "value" in search_data["webPages"]:
                for item in search_data["webPages"]["value"]:
                    formatted_results.append({
                        "title": item.get("name", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("snippet", ""),
                        "source": "Bing"
                    })
            
            results["success"] = len(formatted_results) > 0
            results["results"] = formatted_results
            
            return results
            
        except Exception as e:
            logger.error(f"Bing search error: {e}")
            return {"success": False, "error": str(e), "results": []}
    
    async def _search_google_images(self, query: str, max_results: int) -> Dict[str, Any]:
        """Search for images using Google Custom Search API"""
        try:
            provider_config = self.config["search_providers"]["google"]
            
            # Check rate limits
            if provider_config["usage_count"] >= provider_config["daily_limit"]:
                logger.warning("Google API daily limit reached")
                return {"success": False, "error": "API limit reached", "results": []}
            
            # Initialize results
            results = {
                "success": False,
                "results": []
            }
            
            # Prepare API request
            api_key = provider_config["api_key"]
            cx = provider_config["cx"]
            
            url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cx}&q={query}&num={max_results}&searchType=image"
            
            # Make request
            response = requests.get(url)
            response.raise_for_status()
            search_data = response.json()
            
            # Increment usage counter
            self.config["search_providers"]["google"]["usage_count"] += 1
            
            # Format results
            formatted_results = []
            if "items" in search_data:
                for item in search_data["items"]:
                    formatted_results.append({
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "thumbnail": item.get("image", {}).get("thumbnailLink", ""),
                        "source": "Google Images",
                        "size": {
                            "width": item.get("image", {}).get("width", 0),
                            "height": item.get("image", {}).get("height", 0)
                        },
                        "description": item.get("snippet", "")
                    })
            
            results["success"] = len(formatted_results) > 0
            results["results"] = formatted_results
            
            return results
            
        except Exception as e:
            logger.error(f"Google image search error: {e}")
            return {"success": False, "error": str(e), "results": []}
    
    async def _search_bing_images(self, query: str, max_results: int) -> Dict[str, Any]:
        """Search for images using Bing Image Search API"""
        try:
            provider_config = self.config["search_providers"]["bing"]
            
            # Check rate limits
            if provider_config["usage_count"] >= provider_config["daily_limit"]:
                logger.warning("Bing API daily limit reached")
                return {"success": False, "error": "API limit reached", "results": []}
            
            # Initialize results
            results = {
                "success": False,
                "results": []
            }
            
            # Prepare API request
            api_key = provider_config["api_key"]
            
            url = f"https://api.bing.microsoft.com/v7.0/images/search?q={query}&count={max_results}"
            headers = {"Ocp-Apim-Subscription-Key": api_key}
            
            # Make request
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            search_data = response.json()
            
            # Increment usage counter
            self.config["search_providers"]["bing"]["usage_count"] += 1
            
            # Format results
            formatted_results = []
            if "value" in search_data:
                for item in search_data["value"]:
                    formatted_results.append({
                        "title": item.get("name", ""),
                        "url": item.get("contentUrl", ""),
                        "thumbnail": item.get("thumbnailUrl", ""),
                        "source": "Bing Images",
                        "size": {
                            "width": item.get("width", 0),
                            "height": item.get("height", 0)
                        },
                        "description": item.get("name", "")
                    })
            
            results["success"] = len(formatted_results) > 0
            results["results"] = formatted_results
            
            return results
            
        except Exception as e:
            logger.error(f"Bing image search error: {e}")
            return {"success": False, "error": str(e), "results": []}
    
    async def _search_images_fallback(self, query: str, max_results: int) -> Dict[str, Any]:
        """Fallback method for image search using DuckDuckGo"""
        try:
            # Use DuckDuckGo images search
            from duckduckgo_search import DDGS
            
            # Initialize results
            results = {
                "success": False,
                "results": []
            }
            
            # Create DuckDuckGo search object
            ddgs = DDGS()
            
            # Perform image search
            search_results = list(ddgs.images(query, max_results=max_results))
            
            # Format results
            formatted_results = []
            for result in search_results:
                formatted_results.append({
                    "title": result.get("title", ""),
                    "url": result.get("image", ""),
                    "thumbnail": result.get("thumbnail", ""),
                    "source": "DuckDuckGo Images",
                    "size": {
                        "width": result.get("width", 0),
                        "height": result.get("height", 0)
                    },
                    "description": result.get("title", "")
                })
            
            results["success"] = len(formatted_results) > 0
            results["results"] = formatted_results
            
            return results
            
        except Exception as e:
            logger.error(f"DuckDuckGo image search error: {e}")
            return {"success": False, "error": str(e), "results": []}
    
    def _check_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Check if a result exists in cache and is not expired"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
            
        try:
            with open(cache_file, "r") as f:
                cached_data = json.load(f)
                
            # Check if cache is expired
            if time.time() - cached_data.get("timestamp", 0) > self.config["cache_expiry"]:
                return None
                
            return cached_data
            
        except Exception as e:
            logger.error(f"Error reading cache: {e}")
            return None
    
    def _cache_results(self, cache_key: str, results: Dict[str, Any]) -> bool:
        """Cache search results to disk"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            with open(cache_file, "w") as f:
                json.dump(results, f, indent=2)
                
            return True
            
        except Exception as e:
            logger.error(f"Error caching results: {e}")
            return False
    
    async def extract_metadata_from_url(self, url: str) -> Dict[str, Any]:
        """
        Extract metadata from a URL (title, description, images, etc.)
        
        Args:
            url: URL to extract metadata from
            
        Returns:
            Dictionary with extracted metadata
        """
        try:
            # Use requests to get the page content
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            response = requests.get(url, headers=headers, timeout=self.config["timeout"])
            response.raise_for_status()
            
            # Use BeautifulSoup to parse HTML
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract metadata
            metadata = {
                "url": url,
                "title": soup.title.string if soup.title else "",
                "description": "",
                "images": [],
                "icons": [],
                "timestamp": time.time()
            }
            
            # Extract description
            description_meta = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
            if description_meta:
                metadata["description"] = description_meta.get("content", "")
            
            # Extract images
            for img in soup.find_all("img", limit=10):
                src = img.get("src", "")
                alt = img.get("alt", "")
                
                if src and (src.startswith("http") or src.startswith("/")):
                    # Convert relative URLs to absolute
                    if src.startswith("/"):
                        from urllib.parse import urlparse
                        parsed_url = urlparse(url)
                        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
                        src = base_url + src
                    
                    # Add to images list
                    metadata["images"].append({
                        "url": src,
                        "alt": alt
                    })
            
            # Extract icons
            for link in soup.find_all("link"):
                rel = link.get("rel", [])
                if isinstance(rel, list):
                    rel = " ".join(rel)
                
                if "icon" in rel.lower() and link.get("href"):
                    href = link.get("href")
                    
                    # Convert relative URLs to absolute
                    if href.startswith("/"):
                        from urllib.parse import urlparse
                        parsed_url = urlparse(url)
                        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
                        href = base_url + href
                    
                    # Add to icons list
                    metadata["icons"].append({
                        "url": href,
                        "type": link.get("type", "")
                    })
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata from {url}: {e}")
            return {
                "url": url,
                "error": str(e),
                "timestamp": time.time()
            }
