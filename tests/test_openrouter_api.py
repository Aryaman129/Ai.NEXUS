"""
Test script for OpenRouter API integration
"""
import os
import sys
import asyncio
import aiohttp
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# OpenRouter API key
OPENROUTER_API_KEY = "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"

class OpenRouterIntegration:
    """Integration with OpenRouter's API services"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or OPENROUTER_API_KEY
        self.base_url = "https://openrouter.ai/api/v1"
        self.available = False
        
    async def initialize(self):
        """Initialize and check availability"""
        if not self.api_key:
            logger.warning("OpenRouter API key not provided")
            return False
        
        try:
            models = await self.list_models()
            self.available = len(models) > 0
            return self.available
        except Exception as e:
            logger.error(f"OpenRouter initialization failed: {e}")
            return False
            
    async def list_models(self):
        """List available models from OpenRouter"""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/models", headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", [])
                logger.error(f"Failed to list models: {response.status} - {await response.text()}")
                return []
                
    async def generate_text(self, prompt, model="meta-llama/llama-4-maverick", **kwargs):
        """Generate text using OpenRouter's API"""
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
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/chat/completions", 
                                   headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "text": result["choices"][0]["message"]["content"],
                        "model": model
                    }
                else:
                    error_text = await response.text()
                    logger.error(f"Generation failed: {response.status} - {error_text}")
                    return {"text": f"Error: {error_text}", "error": True}


async def test_openrouter():
    """Test OpenRouter integration"""
    logger.info("Testing OpenRouter integration...")
    
    integration = OpenRouterIntegration()
    
    # Test initialization
    available = await integration.initialize()
    logger.info(f"OpenRouter available: {available}")
    
    if not available:
        logger.error("OpenRouter integration not available, API key may be invalid")
        return False
    
    # List available models
    models = await integration.list_models()
    logger.info(f"Found {len(models)} models")
    
    # Show model details
    for i, model in enumerate(models[:5]):  # Show first 5 models
        logger.info(f"Model {i+1}: {model.get('id')} - {model.get('name')}")
    
    # Test generation with a simple prompt
    test_prompt = "Write a short paragraph about artificial intelligence."
    logger.info(f"Testing text generation with prompt: '{test_prompt}'")
    
    result = await integration.generate_text(test_prompt)
    
    if "error" in result:
        logger.error(f"Generation failed: {result.get('text')}")
        return False
    
    logger.info(f"Generated text: {result.get('text')}")
    return True


if __name__ == "__main__":
    success = asyncio.run(test_openrouter())
    if success:
        logger.info("OpenRouter API test completed successfully")
    else:
        logger.error("OpenRouter API test failed")
