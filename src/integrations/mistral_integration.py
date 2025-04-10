"""
Mistral AI Integration for NEXUS

This module provides integration with Mistral AI's API services, offering access to
powerful specialized models including:
- Mistral Small (multimodal with image capabilities)
- Codestral (specialized for code generation)
- Pixtral (image understanding)
- Open source models like Open-Mistral-Nemo

Key features:
- Task-specific model selection
- Multimodal capabilities
- Specialized code generation
- Adaptive rate limit management
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

class MistralAIIntegration:
    """
    Integration with Mistral AI's API services
    
    Provides access to specialized models like Codestral for code,
    Pixtral for images, and other task-specific models.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Mistral AI integration
        
        Args:
            api_key: Mistral API key (optional, will use env var if not provided)
        """
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        self.base_url = "https://api.mistral.ai/v1"
        self.available = False
        self.models = []
        self.last_request_time = 0
        self.min_request_interval = 2.0  # 0.5 requests per second (2s interval)
        
        # Task-specific model recommendations
        self.task_models = {
            "general": ["mistral-small-latest", "mistral-medium-latest"],
            "creative": ["mistral-large-latest", "mistral-medium-latest"],
            "technical": ["codestral-latest", "open-codestral-mamba"],
            "factual": ["mistral-large-latest", "mistral-medium-latest"],
            "vision": ["pixtral-12b-latest", "mistral-small-latest"]
        }
        
    async def initialize(self) -> bool:
        """
        Initialize the Mistral AI integration and check its availability
        
        Returns:
            True if successfully initialized, False otherwise
        """
        if not self.api_key:
            logger.warning("Mistral API key not provided")
            return False
        
        try:
            models = await self.list_models()
            self.models = models
            self.available = len(models) > 0
            
            if self.available:
                logger.info(f"Mistral AI integration initialized with {len(models)} available models")
                # Log some top models
                top_models = [
                    model for model in models 
                    if any(keyword in model.get('id', '') for keyword in 
                          ['small', 'large', 'codestral', 'pixtral'])
                ]
                for model in top_models[:5]:
                    logger.info(f"Available model: {model.get('id')}")
            else:
                logger.warning("Mistral AI initialized but no models available")
                
            return self.available
            
        except Exception as e:
            logger.error(f"Mistral AI initialization failed: {e}")
            return False
    
    async def get_capabilities(self) -> List[str]:
        """
        Get the capabilities provided by Mistral AI
        
        Returns:
            List of supported capabilities
        """
        capabilities = [
            "text_generation", 
            "code_generation"
        ]
        
        # Check if specialized models are available
        model_ids = [model.get('id', '') for model in self.models]
        
        if any("pixtral" in model_id for model_id in model_ids):
            capabilities.append("image_understanding")
            
        if any("codestral" in model_id for model_id in model_ids):
            capabilities.extend(["code_completion", "code_explanation"])
            
        if any("small" in model_id or "large" in model_id for model_id in model_ids):
            capabilities.extend(["summarization", "translation"])
            
        return capabilities
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List available Mistral models
        
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
                
        # Fallback to mistral-small if available
        if "mistral-small-latest" in available_model_ids:
            return "mistral-small-latest"
            
        # Last resort fallback to first available model
        if available_model_ids:
            return available_model_ids[0]
            
        # Ultimate fallback
        return "mistral-small-latest"
    
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
            "code_completion": self.generate_code,
            "code_explanation": self.explain_code,
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
        Generate text using Mistral AI API
        
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
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1024)
        }
        
        logger.info(f"Generating text with Mistral model: {model}")
        
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
                            "provider": "mistral"
                        }
                    elif response.status == 429:  # Rate limit
                        # Increase our rate limit delay
                        self.min_request_interval *= 1.5
                        logger.warning(f"Rate limit hit. Increased delay to {self.min_request_interval}s")
                        error_text = await response.text()
                        return {
                            "text": f"Rate limit exceeded. Please try again later.", 
                            "error": True,
                            "error_code": 429
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"Mistral generation failed: {response.status} - {error_text}")
                        return {
                            "text": f"Error: {error_text}", 
                            "error": True,
                            "error_code": response.status
                        }
        except Exception as e:
            logger.error(f"Exception during Mistral API call: {str(e)}")
            return {"text": f"Error: {str(e)}", "error": True}
    
    async def generate_code(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate code using Codestral or another code-specialized model
        
        Args:
            prompt: Code generation prompt
            **kwargs: Additional parameters
            
        Returns:
            Generated code response
        """
        # Always try to use Codestral for code generation if available
        model_ids = [model.get('id') for model in self.models]
        model = "codestral-latest" if "codestral-latest" in model_ids else None
        
        return await self.generate_text(
            prompt=prompt, 
            model=model,
            task_type="technical",
            temperature=kwargs.get("temperature", 0.2),  # Lower temperature for code
            **kwargs
        )
    
    async def explain_code(self, code: str, **kwargs) -> Dict[str, Any]:
        """
        Explain a code snippet
        
        Args:
            code: Code to explain
            **kwargs: Additional parameters
            
        Returns:
            Code explanation
        """
        prompt = f"Explain the following code in detail:\n\n```\n{code}\n```"
        return await self.generate_code(prompt=prompt, **kwargs)
    
    async def generate_summary(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a summary of the provided text
        
        Args:
            text: Text to summarize
            **kwargs: Additional parameters
            
        Returns:
            Summarization result
        """
        prompt = f"Please summarize the following text:\n\n{text}"
        return await self.generate_text(prompt=prompt, task_type="general", **kwargs)
    
    async def translate_text(self, text: str, target_language: str, **kwargs) -> Dict[str, Any]:
        """
        Translate text to the target language
        
        Args:
            text: Text to translate
            target_language: Target language
            **kwargs: Additional parameters
            
        Returns:
            Translation result
        """
        prompt = f"Translate the following text to {target_language}:\n\n{text}"
        return await self.generate_text(prompt=prompt, task_type="general", **kwargs)
    
    async def analyze_image(self, image_url: str, prompt: str = "Describe this image in detail", **kwargs) -> Dict[str, Any]:
        """
        Analyze an image using Pixtral or other image-capable model
        
        Args:
            image_url: URL of the image to analyze
            prompt: Prompt for image analysis
            
        Returns:
            Image analysis result
        """
        # Try to use Pixtral if available, otherwise mistral-small
        model_ids = [model.get('id') for model in self.models]
        if "pixtral-12b-latest" in model_ids:
            model = "pixtral-12b-latest"
        elif "mistral-small-latest" in model_ids:
            model = "mistral-small-latest"
        else:
            # Find any pixtral model
            pixtral_models = [m for m in model_ids if "pixtral" in m]
            model = pixtral_models[0] if pixtral_models else "mistral-small-latest"
        
        await self._respect_rate_limit()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
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
        
        logger.info(f"Analyzing image with Mistral model: {model}")
        
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
                            "provider": "mistral"
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
        Ensure we respect Mistral's rate limits
        
        Waits if necessary to maintain the rate limit
        """
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        # If we've made a request less than min_interval seconds ago, wait
        if elapsed < self.min_request_interval:
            wait_time = self.min_request_interval - elapsed
            logger.debug(f"Rate limit: waiting {wait_time:.2f}s before next request")
            await asyncio.sleep(wait_time)
