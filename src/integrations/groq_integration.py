"""
Groq Integration for NEXUS
Provides access to fast LLM inference via Groq API
"""
import os
import logging
from typing import Dict, List, Any, Optional, Union
import json
import asyncio

logger = logging.getLogger(__name__)

class GroqIntegration:
    """Integration with Groq for fast LLM inference"""
    
    def __init__(self, api_key: str = None):
        """Initialize the Groq integration
        
        Args:
            api_key: Groq API key (can be set via env var GROQ_API_KEY)
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "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")
        self.available = False
        self.client = None
        
        if self.api_key:
            try:
                import groq
                self.client = groq.Client(api_key=self.api_key)
                self.available = True
                self.models_cache = None
                logger.info("Groq integration initialized successfully")
            except ImportError:
                logger.warning("Groq package not installed. Run 'pip install groq' to use Groq integration.")
        else:
            logger.warning("Groq API key not provided. This integration will not be available.")
    
    async def generate_text(self, 
                     prompt: str, 
                     model: str = "llama3-8b-8192", 
                     max_tokens: int = 1024,
                     temperature: float = 0.7,
                     system_prompt: str = None) -> Dict[str, Any]:
        """Generate text using Groq's LLM API
        
        Args:
            prompt: The text prompt to generate from
            model: The model to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            system_prompt: Optional system prompt
            
        Returns:
            The generated text and related metadata
        """
        if not self.available:
            return {"error": "Groq integration not available", "text": ""}
            
        try:
            messages = []
            
            # Add system message if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
                
            # Add user message
            messages.append({"role": "user", "content": prompt})
            
            # Run in executor to prevent blocking the event loop
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            )
            
            return {
                "text": response.choices[0].message.content,
                "model": model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        except Exception as e:
            logger.error(f"Error generating text with Groq: {e}")
            return {"error": str(e), "text": ""}
    
    async def list_available_models(self) -> List[Dict[str, str]]:
        """List available models on Groq
        
        Returns:
            List of available models with their details
        """
        if not self.available:
            return []
            
        # If we already cached the models, return them
        if self.models_cache:
            return self.models_cache
            
        try:
            loop = asyncio.get_event_loop()
            models_response = await loop.run_in_executor(
                None,
                lambda: self.client.models.list()
            )
            
            models = [{"id": model.id, "description": f"Groq model: {model.id}"} 
                     for model in models_response.data]
            
            # Cache the results
            self.models_cache = models
            return models
            
        except Exception as e:
            logger.error(f"Error listing Groq models: {e}")
            # Fallback to known models from user's account
            fallback_models = [
                {"id": "llama3-8b-8192", "description": "Llama 3 8B (context: 8192)"},
                {"id": "llama3-70b-8192", "description": "Llama 3 70B (context: 8192)"},
                {"id": "gemma2-9b-it", "description": "Gemma 2 9B instruction-tuned model"},
                {"id": "meta-llama/llama-4-scout-17b-16e-instruct", "description": "Llama 4 Scout 17B"}
            ]
            
            # Cache the fallback models
            self.models_cache = fallback_models
            return fallback_models
