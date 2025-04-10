"""
Research Service for NEXUS
Provides web search capabilities with advanced content extraction and summarization
"""
import logging
import asyncio
import os
import sys
from typing import Dict, List, Any, Optional, Union

# Import the DuckDuckGo integration and content extractor
from ..integrations.duckduckgo_search import DuckDuckGoSearch
from ..integrations.content_extractor import ContentExtractor, SimpleSummarizer

# Check if advanced summarization is available
try:
    from ..integrations.content_extractor import AIModelSummarizer
    ADVANCED_SUMMARIZATION = True
except (ImportError, TypeError):
    ADVANCED_SUMMARIZATION = False

logger = logging.getLogger(__name__)

class ResearchService:
    """
    Service for web search and information retrieval with content extraction
    
    This service enhances NEXUS with the ability to search the web for information,
    extract detailed content from web pages, and provide intelligent summarization
    of found information.
    """
    
    def __init__(self):
        """Initialize the research service"""
        self.logger = logging.getLogger(__name__)
        self.duckduckgo = DuckDuckGoSearch()
        self.content_extractor = ContentExtractor()
        
        # Initialize the appropriate summarizer
        if ADVANCED_SUMMARIZATION:
            self.summarizer = AIModelSummarizer()
            self.logger.info("Using AI-powered summarization")
        else:
            self.summarizer = SimpleSummarizer()
            self.logger.info("Using simple extractive summarization")
            
        self.name = "ResearchService"
        self.description = "Web search and information retrieval capabilities with content extraction and summarization"
        self.logger.info("NEXUS Research Service initialized")
    
    async def search(self, query: str, max_results: int = 5, extract_content: bool = False) -> Dict[str, Any]:
        """
        Search the web for information
        
        Args:
            query: The search query or research question
            max_results: Maximum number of results to return
            extract_content: Whether to extract full content from the top results
            
        Returns:
            Dict containing search results and metadata
        """
        try:
            self.logger.info(f"Processing research query: {query}")
            
            # Refine the search query for better results
            search_query = self._refine_search_query(query)
            
            # Perform the search
            results = await self.duckduckgo.search(search_query, max_results=max_results)
            
            # Format the response for simple results
            response = self._format_search_results(results, query)
            
            # If content extraction is requested
            detailed_results = {}
            if extract_content and results:
                # Extract content from top results
                urls = [r.get('url', '') for r in results]
                titles = [r.get('title', 'Untitled') for r in results]
                
                # Extract content from each URL
                content_results = await self.content_extractor.extract_content_batch(urls[:3], titles[:3])
                
                # Generate summaries
                if ADVANCED_SUMMARIZATION:
                    summaries = await self.summarizer.summarize_batch(content_results, query=query)
                else:
                    summaries = await self.summarizer.summarize_batch(content_results, query=query)
                
                # Add detailed content and summaries
                detailed_results = {
                    "extracted_content": content_results,
                    "summaries": summaries
                }
                
                # Add a summary section to the response
                response += "\n\n## Detailed Summary\n\n"
                for summary in summaries:
                    response += f"### {summary['title']}\n{summary['summary']}\n\n"
            
            return {
                "content": response,
                "results": results,
                "detailed_results": detailed_results,
                "metadata": {
                    "query": search_query,
                    "original_query": query,
                    "result_count": len(results),
                    "urls": [r.get('url', '') for r in results]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in research service: {e}")
            return {
                "content": f"Error while searching for information: {str(e)}",
                "results": [],
                "metadata": {"error": str(e)}
            }
    
    async def extract_content_from_url(self, url: str, title: str = "Web Page") -> Dict[str, Any]:
        """
        Extract and summarize content from a specific URL
        
        Args:
            url: URL to extract content from
            title: Title of the page
            
        Returns:
            Dictionary with extracted content and summary
        """
        try:
            self.logger.info(f"Extracting content from URL: {url}")
            
            # Extract content
            content = await self.content_extractor.extract_content(url, title)
            
            # Generate summary
            if ADVANCED_SUMMARIZATION:
                summary = await self.summarizer.summarize(content)
            else:
                summary = self.summarizer.summarize(content)
            
            return {
                "url": url,
                "title": title,
                "content": content,
                "summary": summary
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting content from URL: {e}")
            return {
                "url": url,
                "title": title,
                "content": f"Error: {str(e)}",
                "summary": f"Failed to extract content from {url}: {str(e)}"
            }
    
    def _refine_search_query(self, prompt: str) -> str:
        """
        Refine a natural language query into a search-optimized query
        
        This method removes question words and fillers to create a more
        effective search query from natural language.
        """
        # Remove question words and other fillers
        query = prompt.lower()
        question_words = [
            "what is", "how to", "why does", "where can", "who is", 
            "can you", "please", "find", "search for", "tell me about",
            "i want to know", "do you know", "could you find"
        ]
        
        for word in question_words:
            if query.startswith(word):
                query = query.replace(word, "", 1).strip()
        
        # Remove punctuation at the end
        if query and query[-1] in "?!.,;:":
            query = query[:-1]
            
        return query.strip()
    
    def _format_search_results(self, results: List[Dict[str, str]], original_query: str) -> str:
        """Format search results into a readable response"""
        if not results:
            return f"I searched for information about '{original_query}' but couldn't find any relevant results."
        
        response = f"## Search Results for: '{original_query}'\n\n"
        
        for i, result in enumerate(results, 1):
            response += f"**{i}. {result.get('title', 'Untitled')}**\n"
            response += f"{result.get('snippet', 'No description available')}\n"
            response += f"Source: {result.get('url', 'No URL available')}\n\n"
        
        return response
    
    async def register_with_nexus(self, tool_registry):
        """
        Register this service as a tool with the NEXUS tool registry
        
        Args:
            tool_registry: The NEXUS tool registry to register with
        """
        self.logger.info("Registering Research Service with NEXUS")
        
        # Register the search function
        tool_registry.register_tool(
            "web_search",
            self.search,
            categories=["web_search", "information_retrieval", "research"],
            description="Search the web for information on any topic"
        )
        
        # Register content extraction function
        tool_registry.register_tool(
            "extract_web_content",
            self.extract_content_from_url,
            categories=["web_content", "information_retrieval", "research"],
            description="Extract and summarize content from a specific webpage"
        )
        
        return True
    
    async def cleanup(self):
        """Clean up resources"""
        self.logger.info("Cleaning up NEXUS Research Service")
        await self.duckduckgo.close()
        await self.content_extractor.close()
