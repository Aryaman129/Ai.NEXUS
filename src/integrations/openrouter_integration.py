"""
OpenRouter API Integration for NEXUS

This module provides integration with OpenRouter's API, giving access to various
powerful LLM models like Llama 4, Gemini Pro, and more.

Key features:
- Multi-model access through a single API
- Adaptive model selection based on task requirements
- Fallback mechanisms for reliability
- Learning from API responses to improve local capabilities
"""
import os
import time
import logging
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenRouterIntegration:
    """
    Integration with OpenRouter's API services
    
    Provides access to top-tier models like Meta's Llama 4, Google's Gemini,
    and many others through a single API interface.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenRouter integration
        
        Args:
            api_key: OpenRouter API key (optional, will use env var if not provided)
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.available = False
        self.models = []
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 request per second limit
        
        # Task-specific model recommendations
        self.task_models = {
            "general": ["meta-llama/llama-4-scout", "meta-llama/llama-3-70b-instruct"],
            "creative": ["meta-llama/llama-4-maverick", "anthropic/claude-3-opus"],
            "technical": ["meta-llama/llama-4-maverick", "deepseek-ai/deepseek-coder"],
            "factual": ["google/gemini-2.5-pro-preview", "meta-llama/llama-4-maverick"],
            "vision": ["google/gemini-2.5-pro-preview", "anthropic/claude-3-opus"]
        }
        
    async def initialize(self) -> bool:
        """
        Initialize the OpenRouter integration and check its availability
        
        Returns:
            True if successfully initialized, False otherwise
        """
        if not self.api_key:
            logger.warning("OpenRouter API key not provided")
            return False
        
        try:
            models = await self.list_models()
            self.models = models
            self.available = len(models) > 0
            
            if self.available:
                logger.info(f"OpenRouter integration initialized with {len(models)} available models")
                # Log a few top models
                for model in models[:5]:
                    logger.info(f"Available model: {model.get('id')} - {model.get('name')}")
            else:
                logger.warning("OpenRouter initialized but no models available")
                
            return self.available
            
        except Exception as e:
            logger.error(f"OpenRouter initialization failed: {e}")
            return False
    
    async def get_capabilities(self) -> List[str]:
        """
        Get the capabilities provided by OpenRouter
        
        Returns:
            List of supported capabilities
        """
        capabilities = [
            "text_generation", 
            "code_generation",
            "creative_writing",
            "summarization",
            "translation"
        ]
        
        # Check if any multimodal models are available
        if any("gemini" in model.get('id', '').lower() for model in self.models):
            capabilities.append("image_understanding")
            
        return capabilities
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models from OpenRouter
        
        Returns:
            List of model data dictionaries
        """
        await self._respect_rate_limit()
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/models", headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", [])
                logger.error(f"Failed to list models: {response.status} - {await response.text()}")
                return []
    
    async def get_recommended_model(self, task_type: str = "general") -> str:
        """
        Get the recommended model for a specific task type
        
        Args:
            task_type: The type of task (general, creative, technical, etc.)
            
        Returns:
            Recommended model ID for the task
        """
        # Get models recommended for this task
        recommended_models = self.task_models.get(task_type, self.task_models["general"])
        
        # Find the first recommended model that's available
        available_model_ids = [model.get('id') for model in self.models]
        
        for model_id in recommended_models:
            if model_id in available_model_ids:
                return model_id
                
        # Fallback to first available model
        if available_model_ids:
            return available_model_ids[0]
            
        # Last resort fallback
        return "meta-llama/llama-4-maverick"
    
    async def execute(self, capability: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a capability using the appropriate method
        
        Args:
            capability: The capability to execute
            **kwargs: Parameters for the capability
            
        Returns:
            Result dictionary
        """
        # Map capabilities to methods
        capability_map = {
            "text_generation": self.generate_text,
            "code_generation": self.generate_code,
            "creative_writing": self.generate_creative_text,
            "summarization": self.generate_summary,
            "translation": self.translate_text,
            "image_understanding": self.analyze_image
        }
        
        if capability in capability_map:
            # Get the appropriate method
            method = capability_map[capability]
            # Execute the method with the provided parameters
            return await method(**kwargs)
        else:
            # Default to text generation for unknown capabilities
            return await self.generate_text(**kwargs)
    
    async def generate_text(self, 
                           prompt: str, 
                           model: Optional[str] = None,
                           task_type: str = "general",
                           **kwargs) -> Dict[str, Any]:
        """
        Generate text using OpenRouter's API
        
        Args:
            prompt: The input prompt
            model: Specific model to use (optional)
            task_type: Type of task for model recommendation
            **kwargs: Additional parameters
            
        Returns:
            Response dictionary with generated text
        """
        # If no specific model provided, get recommended model for task
        if not model:
            model = await self.get_recommended_model(task_type)
        
        await self._respect_rate_limit()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://nexus.ai"  # Required by OpenRouter
        }
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1024)
        }
        
        logger.info(f"Generating text with model: {model}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions", 
                    headers=headers, 
                    json=data,
                    timeout=30  # 30 second timeout
                ) as response:
                    self.last_request_time = time.time()
                    
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "text": result["choices"][0]["message"]["content"],
                            "model": model,
                            "provider": "openrouter"
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"OpenRouter generation failed: {response.status} - {error_text}")
                        return {
                            "text": f"Error: {error_text}", 
                            "error": True,
                            "error_code": response.status
                        }
        except Exception as e:
            logger.error(f"Exception during OpenRouter API call: {str(e)}")
            return {"text": f"Error: {str(e)}", "error": True}
    
    async def generate_code(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate code using a coding-specialized model"""
        return await self.generate_text(
            prompt=prompt, 
            task_type="technical",
            temperature=kwargs.get("temperature", 0.2),  # Lower temperature for code
            **kwargs
        )
    
    async def generate_creative_text(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate creative text using creativity-optimized models"""
        return await self.generate_text(
            prompt=prompt, 
            task_type="creative",
            temperature=kwargs.get("temperature", 0.8),  # Higher temperature for creativity
            **kwargs
        )
    
    async def generate_summary(self, text: str, **kwargs) -> Dict[str, Any]:
        """Generate a summary of the provided text"""
        prompt = f"Please summarize the following text:\n\n{text}"
        return await self.generate_text(prompt=prompt, task_type="general", **kwargs)
    
    async def translate_text(self, text: str, target_language: str, **kwargs) -> Dict[str, Any]:
        """Translate text to the target language"""
        prompt = f"Translate the following text to {target_language}:\n\n{text}"
        return await self.generate_text(prompt=prompt, task_type="general", **kwargs)
    
    async def analyze_image(self, image_url: str, prompt: str = "Describe this image in detail", **kwargs) -> Dict[str, Any]:
        """
        Analyze an image using a multimodal model
        
        Args:
            image_url: URL of the image to analyze
            prompt: Prompt for image analysis
            
        Returns:
            Analysis result
        """
        # This requires a multimodal model like Gemini
        model = "google/gemini-2.5-pro-preview"
        
        await self._respect_rate_limit()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://nexus.ai"
        }
        
        data = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1024)
        }
        
        logger.info(f"Analyzing image with model: {model}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions", 
                    headers=headers, 
                    json=data,
                    timeout=30
                ) as response:
                    self.last_request_time = time.time()
                    
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "text": result["choices"][0]["message"]["content"],
                            "model": model,
                            "provider": "openrouter"
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"Image analysis failed: {response.status} - {error_text}")
                        return {
                            "text": f"Error: {error_text}", 
                            "error": True,
                            "error_code": response.status
                        }
        except Exception as e:
            logger.error(f"Exception during image analysis: {str(e)}")
            return {"text": f"Error: {str(e)}", "error": True}
    
    async def _respect_rate_limit(self) -> None:
        """
        Ensure we respect OpenRouter's rate limits
        
        Waits if necessary to maintain the rate limit
        """
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        # If we've made a request less than min_interval seconds ago, wait
        if elapsed < self.min_request_interval:
            wait_time = self.min_request_interval - elapsed
            logger.debug(f"Rate limit: waiting {wait_time:.2f}s before next request")
            await asyncio.sleep(wait_time)
