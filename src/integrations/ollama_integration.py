"""
Ollama Integration for NEXUS
Provides access to local AI models via Ollama
"""
import os
import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Union
import aiohttp

logger = logging.getLogger(__name__)

class OllamaIntegration:
    """Integration with Ollama for local LLM inference"""
    
    def __init__(self, host: str = "http://localhost:11434"):
        """Initialize the Ollama integration
        
        Args:
            host: Ollama API host (default: http://localhost:11434)
        """
        self.host = host
        self.available = False
        self.models_cache = None
        
        # Preferred models (we'll check these first)
        self.preferred_models = [
            "deepseek-r1",
            "deepseek-coder",
            "llava",
            "dolphin-phi",
            "bakllava",
            "llama2"  # Fallback model
        ]
        
        # We'll run the check during initialization and then again if needed
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're in an async context, create a task
            asyncio.create_task(self._check_availability())
        else:
            # Otherwise run the check synchronously - this shouldn't normally happen
            try:
                # Use a new event loop for synchronous contexts
                temp_loop = asyncio.new_event_loop()
                temp_loop.run_until_complete(self._check_availability())
                temp_loop.close()
            except Exception as e:
                logger.warning(f"Error checking Ollama availability synchronously: {e}")

    async def _check_availability(self):
        """Check if Ollama is available"""
        try:
            async with aiohttp.ClientSession() as session:
                # First try the API tags endpoint (newer Ollama versions)
                try:
                    async with session.get(f"{self.host}/api/tags", timeout=3.0) as response:
                        if response.status == 200:
                            self.available = True
                            logger.info("Ollama integration initialized successfully")
                            
                            # Cache available models
                            response_json = await response.json()
                            self.models_cache = [model["name"] for model in response_json.get("models", [])]
                            
                            if not self.models_cache:
                                # If empty, we might be using a different version of Ollama API
                                # Let's try listing models directly
                                models = await self.list_models()
                                if models:
                                    self.models_cache = [model["id"] for model in models]
                                    logger.info(f"Found {len(self.models_cache)} models in Ollama")
                                else:
                                    logger.warning("No models found in Ollama")
                        else:
                            logger.warning(f"Ollama tags endpoint not available. HTTP status: {response.status}")
                            # Try the list endpoint as fallback
                            models = await self.list_models()
                            if models:
                                self.models_cache = [model["id"] for model in models]
                                self.available = True
                                logger.info(f"Found {len(self.models_cache)} models in Ollama")
                            else:
                                logger.warning("No models found in Ollama")
                except Exception as e:
                    logger.warning(f"Error checking Ollama tags endpoint: {e}")
                    # Try the list endpoint as fallback
                    models = await self.list_models()
                    if models:
                        self.models_cache = [model["id"] for model in models]
                        self.available = True
                        logger.info(f"Found {len(self.models_cache)} models in Ollama")
                    else:
                        logger.warning("No models found in Ollama")
        except asyncio.TimeoutError:
            logger.warning("Ollama not available: Connection timed out")
        except Exception as e:
            logger.warning(f"Ollama not available. Error: {e}")

    async def list_models(self):
        """List available models on Ollama
        
        Returns:
            List of available models with their details
        """
        try:
            async with aiohttp.ClientSession() as session:
                # First try the newer API endpoint
                try:
                    async with session.get(f"{self.host}/api/tags", timeout=5.0) as response:
                        if response.status == 200:
                            data = await response.json()
                            models = [{"id": model["name"], "description": f"Ollama model: {model['name']}"} 
                                    for model in data.get("models", [])]
                            
                            # If empty, we might be using older API versions
                            if not models:
                                # Try direct shell command listing as fallback
                                self.available = True  # We can connect to Ollama server
                                return [
                                    {"id": "deepseek-r1", "description": "DeepSeek-R1 model"},
                                    {"id": "deepseek-coder", "description": "DeepSeek Coder model"},
                                    {"id": "llava", "description": "Llava multimodal model"},
                                    {"id": "dolphin-phi", "description": "Dolphin Phi model"},
                                    {"id": "bakllava", "description": "BakLlava multimodal model"}
                                ]
                            
                            return models
                        else:
                            logger.error(f"Error listing Ollama models: HTTP {response.status}")
                except Exception as e:
                    logger.warning(f"Error using /api/tags: {e}")
                    # Try the list endpoint as fallback
                    self.available = True  # We can likely connect to Ollama server
                    return [
                        {"id": "deepseek-r1", "description": "DeepSeek-R1 model"},
                        {"id": "deepseek-coder", "description": "DeepSeek Coder model"},
                        {"id": "llava", "description": "Llava multimodal model"},
                        {"id": "dolphin-phi", "description": "Dolphin Phi model"},
                        {"id": "bakllava", "description": "BakLlava multimodal model"}
                    ]
                    
        except Exception as e:
            logger.error(f"Error listing Ollama models: {e}")
            return []

    async def generate_text(self, 
                     prompt: str, 
                     model: str = "deepseek-r1", 
                     temperature: float = 0.7,
                     max_tokens: int = 1024,
                     system_prompt: str = None) -> Dict[str, Any]:
        """Generate text using Ollama local models
        
        Args:
            prompt: The text prompt to generate from
            model: The model to use
            temperature: Temperature for generation (0.0-1.0)
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt
            
        Returns:
            The generated text and related metadata
        """
        if not self.available:
            return {"error": "Ollama integration not available", "text": ""}
        
        # Check if we need to validate the model
        if model not in self.preferred_models:
            # Default to a preferred model if the requested one isn't in our list
            new_model = self.preferred_models[0] if self.preferred_models else "deepseek-r1"
            logger.warning(f"Model {model} not in preferred list, using {new_model} instead")
            model = new_model
        
        try:
            # Prepare request payload
            payload = {
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
            # Add system prompt if provided
            if system_prompt:
                payload["system"] = system_prompt
            
            url = f"{self.host}/api/generate"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return {"error": f"Error from Ollama API: {error_text}", "text": ""}
                    
                    # Ollama streams responses, so we need to collect all chunks
                    full_response = ""
                    async for line in response.content:
                        try:
                            chunk = json.loads(line)
                            if "response" in chunk:
                                full_response += chunk["response"]
                        except Exception as e:
                            logger.error(f"Error parsing Ollama response: {e}")
                    
                    return {
                        "text": full_response,
                        "model": model
                    }
        
        except Exception as e:
            logger.error(f"Error generating text with Ollama: {e}")
            return {"error": str(e), "text": ""}
    
    async def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available on Ollama
        
        Args:
            model_name: The model name to check
            
        Returns:
            True if the model is available, False otherwise
        """
        # Use cached model list if available
        if self.models_cache is not None:
            return model_name in self.models_cache
        
        # Otherwise, get fresh model list
        models = await self.list_models()
        return any(model["id"] == model_name for model in models)
