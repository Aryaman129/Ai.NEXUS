"""
Test script for Mistral AI API integration
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

# Mistral API key
MISTRAL_API_KEY = "j0bCYW3zNDTbRj5wTTYlgjlqMpOxcVOI"

class MistralAIIntegration:
    """Integration with Mistral AI's API services"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or MISTRAL_API_KEY
        self.base_url = "https://api.mistral.ai/v1"
        self.available = False
        
    async def initialize(self):
        """Initialize and check availability"""
        if not self.api_key:
            logger.warning("Mistral API key not provided")
            return False
        
        try:
            models = await self.list_models()
            self.available = len(models) > 0
            return self.available
        except Exception as e:
            logger.error(f"Mistral initialization failed: {e}")
            return False
            
    async def list_models(self):
        """List available Mistral models"""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/models", headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", [])
                logger.error(f"Failed to list models: {response.status} - {await response.text()}")
                return []
                
    async def generate_text(self, prompt, model="mistral-small-latest", **kwargs):
        """Generate text using Mistral AI API"""
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


async def test_mistral():
    """Test Mistral AI integration"""
    logger.info("Testing Mistral AI integration...")
    
    integration = MistralAIIntegration()
    
    # Test initialization
    available = await integration.initialize()
    logger.info(f"Mistral AI available: {available}")
    
    if not available:
        logger.error("Mistral AI integration not available, API key may be invalid")
        return False
    
    # List available models
    models = await integration.list_models()
    logger.info(f"Found {len(models)} models")
    
    # Show model details
    for i, model in enumerate(models):
        logger.info(f"Model {i+1}: {model.get('id')}")
    
    # Test generation with a simple prompt
    test_prompt = "Write a short paragraph about artificial intelligence."
    logger.info(f"Testing text generation with prompt: '{test_prompt}'")
    
    # Test with mistral-small model
    result = await integration.generate_text(test_prompt, model="mistral-small-latest")
    
    if "error" in result:
        logger.error(f"Generation failed: {result.get('text')}")
        return False
    
    logger.info(f"Generated text: {result.get('text')}")
    
    # Test a specialized model if available
    if any("codestral" in model.get('id', '') for model in models):
        code_prompt = "Write a Python function to calculate the Fibonacci sequence."
        logger.info(f"Testing code generation with specialized model...")
        
        code_result = await integration.generate_text(code_prompt, model="codestral-latest")
        
        if "error" in code_result:
            logger.warning(f"Code generation failed: {code_result.get('text')}")
        else:
            logger.info(f"Generated code: {code_result.get('text')[:200]}...")
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_mistral())
    if success:
        logger.info("Mistral AI API test completed successfully")
    else:
        logger.error("Mistral AI API test failed")
