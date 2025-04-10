"""
Content Extraction and Summarization for NEXUS
Provides advanced web page content extraction and AI-powered summarization
"""
import logging
import asyncio
import re
import aiohttp
from typing import List, Dict, Any, Optional, Tuple
from bs4 import BeautifulSoup
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class ContentExtractor:
    """Advanced content extraction from web pages with intelligent parsing"""
    
    def __init__(self):
        """Initialize the content extractor"""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        self.timeout = 15  # seconds
        self.session = None
        logger.info("Content Extractor initialized")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(headers=self.headers)
        return self.session
        
    async def extract_content_batch(self, urls: List[str], titles: List[str]) -> List[Tuple[str, str, str]]:
        """Extract content from multiple URLs concurrently
        
        Args:
            urls: List of URLs to extract content from
            titles: List of titles corresponding to the URLs
            
        Returns:
            List of tuples (url, title, content)
        """
        logger.info(f"Extracting content from {len(urls)} URLs")
        tasks = [self.extract_content(url, title) for url, title in zip(urls, titles)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error extracting content from {urls[i]}: {result}")
                processed_results.append((urls[i], titles[i], f"Error: {str(result)}"))
            else:
                processed_results.append((urls[i], titles[i], result))
                
        return processed_results
    
    async def extract_content(self, url: str, title: str) -> str:
        """Extract useful content from a webpage with intelligent parsing
        
        Args:
            url: URL to extract content from
            title: Title of the webpage
            
        Returns:
            Extracted text content
        """
        try:
            # Skip URLs that are likely to cause issues
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                return f"Invalid URL: {url}"
            
            session = await self._get_session()
            async with session.get(url, timeout=self.timeout) as response:
                if response.status != 200:
                    return f"Error: HTTP {response.status}"
                
                html = await response.text()
                
                # Parse HTML
                soup = BeautifulSoup(html, 'html.parser')
                
                # Remove non-content elements
                for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe"]):
                    element.decompose()
                
                # Identify main content (heuristic approach)
                main_content = self._find_main_content(soup)
                
                # Get text and clean it up
                if main_content:
                    text = main_content.get_text(separator='\n')
                else:
                    text = soup.get_text(separator='\n')
                
                # Clean up text
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = '\n'.join(chunk for chunk in chunks if chunk)
                
                # Remove excessive newlines
                text = re.sub(r'\n{3,}', '\n\n', text)
                
                # Add title and URL
                result = f"Title: {title}\nURL: {url}\n\n{text}"
                
                logger.info(f"Extracted {len(result)} characters from {url}")
                return result
                
        except Exception as e:
            logger.error(f"Error in extract_content for {url}: {e}")
            return f"Failed to extract content from {url}: {str(e)}"
    
    def _find_main_content(self, soup: BeautifulSoup) -> Optional[Any]:
        """Try to identify the main content area of the page using intelligent heuristics"""
        # Try common content containers
        for container in ['article', 'main', '#content', '.content', '.post', '.entry', '.post-content', '.article-content', '.main-content']:
            if container.startswith(('#', '.')):
                # Handle CSS selectors
                element = soup.select_one(container)
            else:
                # Handle HTML tags
                element = soup.find(container)
            
            if element and len(element.get_text(strip=True)) > 200:
                return element
        
        # Find the div with the most paragraph text
        paragraphs = soup.find_all('p')
        if paragraphs:
            # Find parent elements of paragraphs
            parents = {}
            for p in paragraphs:
                parent = p.parent
                if parent:
                    if parent not in parents:
                        parents[parent] = 0
                    parents[parent] += len(p.get_text(strip=True))
            
            if parents:
                # Get the parent with the most text
                main_parent = max(parents.items(), key=lambda x: x[1])[0]
                if parents[main_parent] > 200:
                    return main_parent
        
        # Fallback: return the body
        return soup.body
    
    async def close(self):
        """Close the session"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("Content Extractor session closed")


class SimpleSummarizer:
    """Simple extractive summarization for text content"""
    
    def summarize(self, text: str, query: str = None, max_sentences: int = 10) -> str:
        """Create an extractive summary by selecting important sentences
        
        Args:
            text: Text to summarize
            query: Optional query to focus the summary on
            max_sentences: Maximum number of sentences to include
            
        Returns:
            Summarized text
        """
        if not text:
            return "No content to summarize."
            
        # Split into sentences
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        
        # Score sentences (simple heuristic based on length and keyword presence)
        sentence_scores = {}
        keywords = query.lower().split() if query else []
        
        for i, sentence in enumerate(sentences):
            if len(sentence.split()) < 4:  # Skip very short sentences
                continue
            
            score = 0
            # Prefer sentences with query terms
            for keyword in keywords:
                if keyword.lower() in sentence.lower():
                    score += 2
            
            # Prefer medium-length sentences
            words = len(sentence.split())
            if 10 <= words <= 25:
                score += 1
            
            # Prefer sentences with information-rich indicators
            info_indicators = ["is", "are", "was", "were", "has", "had", "can", "will", 
                             "should", "because", "therefore", "thus", "also", 
                             "additionally", "enables", "allows", "for example"]
            for indicator in info_indicators:
                if f" {indicator} " in f" {sentence.lower()} ":
                    score += 0.5
                    
            # Store score
            sentence_scores[i] = score
            
        # Get top sentences
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:max_sentences]
        
        # Sort by position in text
        top_sentences.sort(key=lambda x: x[0])
        
        # Extract the sentences
        summary_sentences = [sentences[i] for i, _ in top_sentences]
        
        # Join into a summary
        summary = " ".join(summary_sentences)
        
        return summary
        
    async def summarize_batch(self, content_list: List[Tuple[str, str, str]], query: str = None) -> List[Dict[str, str]]:
        """Summarize a batch of content
        
        Args:
            content_list: List of tuples (url, title, content)
            query: Optional query to focus the summaries on
            
        Returns:
            List of dictionaries with url, title, content, and summary
        """
        results = []
        
        for url, title, content in content_list:
            # Skip errors
            if content.startswith("Error:") or content.startswith("Failed to extract"):
                results.append({
                    "url": url,
                    "title": title,
                    "content": "Content extraction failed",
                    "summary": content
                })
                continue
                
            # Summarize the content
            summary = self.summarize(content, query=query)
            
            results.append({
                "url": url,
                "title": title,
                "content": content,
                "summary": summary
            })
            
        return results


# Check for availability of better summarization technologies
try:
    import torch
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
    
    class AIModelSummarizer:
        """Advanced summarizer using transformer models"""
        
        def __init__(self):
            """Initialize the model-based summarizer"""
            self.summarizer = None
            self.model_name = "facebook/bart-large-cnn"  # Default model
            self.fallback_model = "sshleifer/distilbart-cnn-12-6"  # Smaller fallback model
            
            try:
                logger.info(f"Loading summarization model: {self.model_name}")
                self.summarizer = pipeline("summarization", model=self.model_name)
                logger.info("Summarization model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load summarization model: {e}")
                try:
                    # Try a smaller model as fallback
                    logger.info(f"Trying smaller summarization model: {self.fallback_model}")
                    self.summarizer = pipeline("summarization", model=self.fallback_model)
                    logger.info("Fallback summarization model loaded successfully")
                except Exception as e2:
                    logger.error(f"Failed to load fallback model: {e2}")
                    self.summarizer = None
        
        def is_available(self) -> bool:
            """Check if the model is available"""
            return self.summarizer is not None
            
        async def summarize(self, text: str, query: str = None, max_length: int = 250) -> str:
            """Summarize text using a transformer model
            
            Args:
                text: Text to summarize
                query: Optional query (not used in the model directly but could prepend)
                max_length: Maximum length of the summary
                
            Returns:
                Summarized text
            """
            if not self.is_available():
                # Fall back to simple summarizer
                simple = SimpleSummarizer()
                return simple.summarize(text, query)
                
            try:
                # Truncate text if too long (most models have context limits)
                max_input_length = 1024  # Most BART models have ~1024 token limit
                words = text.split()
                if len(words) > max_input_length:
                    text = " ".join(words[:max_input_length])
                
                # Generate summary
                result = self.summarizer(text, max_length=max_length, min_length=min(100, max_length//2), 
                                        do_sample=False)
                
                if result and len(result) > 0:
                    return result[0]['summary_text']
                else:
                    return "Model produced no summary."
                    
            except Exception as e:
                logger.error(f"Error in model summarization: {e}")
                # Fall back to simple summarizer
                simple = SimpleSummarizer()
                return simple.summarize(text, query)
                
        async def summarize_batch(self, content_list: List[Tuple[str, str, str]], query: str = None) -> List[Dict[str, str]]:
            """Summarize a batch of content
            
            Args:
                content_list: List of tuples (url, title, content)
                query: Optional query to focus the summaries on
                
            Returns:
                List of dictionaries with url, title, content, and summary
            """
            results = []
            
            for url, title, content in content_list:
                # Skip errors
                if content.startswith("Error:") or content.startswith("Failed to extract"):
                    results.append({
                        "url": url,
                        "title": title,
                        "content": "Content extraction failed",
                        "summary": content
                    })
                    continue
                    
                # Summarize the content
                summary = await self.summarize(content, query=query)
                
                results.append({
                    "url": url,
                    "title": title,
                    "content": content,
                    "summary": summary
                })
                
            return results
            
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AIModelSummarizer = None  # Set to None if not available
    logger.info("HuggingFace transformers not installed. Using simple summarization only.")
