"""
DuckDuckGo Search Integration for NEXUS
Uses the duckduckgo-search package if available, with fallback to direct HTML scraping
"""
import logging
import aiohttp
import asyncio
import re
from typing import List, Dict, Any, Optional, Union
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class DuckDuckGoSearch:
    """DuckDuckGo search integration that provides web search capabilities without API keys"""
    
    def __init__(self):
        """Initialize the DuckDuckGo search integration"""
        self.session = None
        self.use_package = self._check_package_available()
        self.available = True  # Assume available until proven otherwise
        self.success_count = 0
        self.failure_count = 0
        self.last_successful_method = None  # Track which method works best
        logger.info(f"DuckDuckGo search initialized, using package: {self.use_package}")
    
    def _check_package_available(self) -> bool:
        """Check if the duckduckgo-search package is available"""
        try:
            import duckduckgo_search
            return True
        except ImportError:
            logger.info("duckduckgo-search package not available, using fallback method")
            return False
            
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            })
        return self.session
    
    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Execute a search on DuckDuckGo
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            
        Returns:
            List of search results, each containing title, snippet, and URL
        """
        start_time = asyncio.get_event_loop().time()
        
        # Adaptively choose the method based on past performance
        if self.last_successful_method == "package" and self.use_package:
            # Try package first if it succeeded last time
            results = await self._search_with_package(query, max_results)
            if results:
                return results
            # Fall back to direct method if package fails
            results = await self._search_with_fallback(query, max_results)
        elif self.last_successful_method == "fallback" or not self.use_package:
            # Try fallback first if it succeeded last time or package isn't available
            results = await self._search_with_fallback(query, max_results)
            if results and self.use_package:
                # Only try package if fallback fails and package is available
                if not results:
                    results = await self._search_with_package(query, max_results)
        else:
            # No past success data, follow original logic
            if self.use_package:
                results = await self._search_with_package(query, max_results)
                if not results:
                    results = await self._search_with_fallback(query, max_results)
            else:
                results = await self._search_with_fallback(query, max_results)
        
        # Track success/failure for adaptive behavior
        end_time = asyncio.get_event_loop().time()
        execution_time = end_time - start_time
        
        if results:
            self.success_count += 1
            # Determine which method succeeded
            if self.last_successful_method is None:
                # First success
                self.last_successful_method = "package" if self.use_package and len(results) > 0 else "fallback"
            logger.info(f"Search succeeded in {execution_time:.2f}s using {self.last_successful_method} method")
            self.available = True
        else:
            self.failure_count += 1
            logger.warning(f"Search failed after {execution_time:.2f}s")
            if self.failure_count > 3 and self.success_count == 0:
                # Consider the service unavailable after 3 consecutive failures
                self.available = False
                logger.warning("DuckDuckGo search marked as unavailable after multiple failures")
        
        return results
            
    async def _search_with_package(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """Search using the duckduckgo-search package"""
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            logger.warning("duckduckgo-search package not available despite earlier check")
            self.use_package = False
            return []
            
        results = []
        try:
            # This package is synchronous, so run it in a thread pool
            loop = asyncio.get_event_loop()
            
            def search_sync():
                try:
                    ddgs = DDGS()
                    return list(ddgs.text(query, max_results=max_results))
                except Exception as e:
                    logger.error(f"Error inside ddgs.text search: {e}")
                    return []
                    
            raw_results = await loop.run_in_executor(None, search_sync)
            
            for r in raw_results:
                results.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href", "")
                })
            
            if results:
                self.last_successful_method = "package"
                
        except Exception as e:
            logger.error(f"Error using duckduckgo-search package: {e}")
                
        return results
            
    async def _search_with_fallback(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """Search using direct HTML scraping as a fallback method"""
        results = []
        
        try:
            session = await self._get_session()
            
            # Format search URL
            search_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
            
            async with session.get(search_url, timeout=10.0) as response:
                if response.status != 200:
                    logger.error(f"Error fetching search results: HTTP {response.status}")
                    return results
                    
                html = await response.text()
                
                # Parse HTML 
                soup = BeautifulSoup(html, "html.parser")
                
                # Extract search results
                for result in soup.select(".result"):
                    title_element = result.select_one(".result__title")
                    snippet_element = result.select_one(".result__snippet")
                    
                    if not title_element:
                        continue
                        
                    title = title_element.get_text().strip()
                    snippet = snippet_element.get_text().strip() if snippet_element else ""
                    
                    # Extract URL - it's in a href attribute
                    url = ""
                    if title_element.a and title_element.a.has_attr("href"):
                        url = title_element.a["href"]
                        # Parse out the actual URL from DuckDuckGo redirect
                        match = re.search(r'uddg=([^&]+)', url)
                        if match:
                            import urllib.parse
                            url = urllib.parse.unquote(match.group(1))
                    
                    results.append({
                        "title": title,
                        "snippet": snippet,
                        "url": url
                    })
                    
                    if len(results) >= max_results:
                        break
        
        except asyncio.TimeoutError:
            logger.error("Timeout while fetching DuckDuckGo search results")
        except Exception as e:
            logger.error(f"Error in fallback search: {e}")
        
        if results:
            self.last_successful_method = "fallback"
                    
        return results
        
    async def get_webpage_content(self, url: str, max_length: int = 5000) -> str:
        """
        Fetch and extract main content from a webpage
        
        Args:
            url: URL to fetch content from
            max_length: Maximum content length to return
            
        Returns:
            Extracted text content
        """
        try:
            session = await self._get_session()
            
            async with session.get(url) as response:
                if response.status != 200:
                    return f"Error: HTTP {response.status}"
                    
                html = await response.text()
                
                # Parse HTML 
                soup = BeautifulSoup(html, "html.parser")
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.extract()
                    
                # Get text
                text = soup.get_text()
                
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = '\n'.join(chunk for chunk in chunks if chunk)
                
                # Truncate if needed
                if len(text) > max_length:
                    return text[:max_length] + "..."
                else:
                    return text
                    
        except Exception as e:
            logger.error(f"Error fetching webpage content: {e}")
            return f"Error: {str(e)}"
            
    async def close(self):
        """Close the session"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("DuckDuckGo search session closed")
