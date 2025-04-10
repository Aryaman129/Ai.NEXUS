"""
Hugging Face Integration for NEXUS
Provides access to models hosted on the Hugging Face Hub
"""
import os
import logging
from typing import Dict, List, Any, Optional, Union
import requests
import json
import asyncio

logger = logging.getLogger(__name__)

class HuggingFaceIntegration:
    """Integration with Hugging Face Inference API for model access"""
    
    def __init__(self, api_key: str = None):
        """Initialize the Hugging Face integration
        
        Args:
            api_key: Hugging Face API token (can be set via env var HF_API_KEY)
        """
        self.api_key = api_key or os.environ.get("HF_API_KEY")
        self.api_url = "https://api-inference.huggingface.co/models/"
        self.available = self.api_key is not None
        
        if not self.available:
            logger.warning("Hugging Face API key not provided. Some features may be limited.")
        else:
            logger.info("Hugging Face integration initialized successfully")
    
    async def query_model(self, model_id: str, inputs: Any, parameters: Dict = None) -> Dict:
        """Query a model on Hugging Face
        
        Args:
            model_id: The model ID on Hugging Face (e.g., "facebook/bart-large-cnn")
            inputs: The inputs to the model
            parameters: Optional parameters for the model
            
        Returns:
            The model's response
        """
        if not self.available:
            logger.error("Hugging Face API key not available")
            return {"error": "API key not configured"}
            
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"inputs": inputs}
        
        if parameters:
            payload["parameters"] = parameters
            
        try:
            # Use asyncio to run the request in a non-blocking way
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    f"{self.api_url}{model_id}", 
                    headers=headers, 
                    json=payload
                )
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error from Hugging Face API: {response.text}")
                return {"error": f"API error: {response.status_code}", "details": response.text}
                
        except Exception as e:
            logger.error(f"Error querying Hugging Face model: {e}")
            return {"error": str(e)}
    
    async def generate_text(self, 
                     prompt: str, 
                     model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
                     max_length: int = 1024,
                     temperature: float = 0.7) -> Dict[str, Any]:
        """Generate text using Hugging Face text generation models
        
        Args:
            prompt: The text prompt to generate from
            model_id: The model ID on Hugging Face
            max_length: Maximum length of generated text
            temperature: Sampling temperature (0.0 to 1.0)
            
        Returns:
            The generated text and related metadata
        """
        parameters = {
            "max_length": max_length,
            "temperature": temperature,
            "return_full_text": False
        }
        
        response = await self.query_model(
            model_id=model_id,
            inputs=prompt,
            parameters=parameters
        )
        
        if "error" in response:
            return {"text": "", "error": response["error"]}
        
        # Handle different response formats based on the model
        if isinstance(response, list) and len(response) > 0:
            if isinstance(response[0], dict) and "generated_text" in response[0]:
                return {"text": response[0]["generated_text"], "model": model_id}
            elif isinstance(response[0], str):
                return {"text": response[0], "model": model_id}
        
        # If we can't determine the format, return the raw response and an empty text
        return {"text": str(response), "model": model_id, "raw_response": response}
    
    async def get_embeddings(self, texts: List[str], model_id: str = "sentence-transformers/all-MiniLM-L6-v2") -> Dict:
        """Get embeddings for a list of texts
        
        Args:
            texts: List of texts to get embeddings for
            model_id: ID of the embedding model to use
            
        Returns:
            Dictionary with embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
            
        response = await self.query_model(model_id=model_id, inputs=texts)
        
        if "error" in response:
            return {"error": response["error"], "embeddings": []}
            
        return {"embeddings": response}
    
    async def analyze_image(self, 
                     image_path: str = None, 
                     image_url: str = None,
                     model_id: str = "google/vit-base-patch16-224") -> Dict:
        """Analyze an image using vision models
        
        Args:
            image_path: Path to local image file
            image_url: URL of image to analyze
            model_id: Model ID to use for analysis
            
        Returns:
            Analysis results
        """
        import base64
        
        if image_path:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
                image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                inputs = {"image": image_b64}
        elif image_url:
            inputs = {"url": image_url}
        else:
            return {"error": "Either image_path or image_url must be provided"}
            
        response = await self.query_model(model_id=model_id, inputs=inputs)
        return response
    
    def list_recommended_models(self, task: str = None) -> List[Dict]:
        """List recommended models for a specific task
        
        Args:
            task: The task to find models for (e.g., "text-generation", "image-classification")
            
        Returns:
            List of recommended models with their details
        """
        # These are curated recommendations for different tasks
        task_models = {
            "text-generation": [
                {"id": "mistralai/Mistral-7B-Instruct-v0.2", "description": "Mistral 7B instruction-tuned model"},
                {"id": "meta-llama/Llama-2-7b-chat-hf", "description": "Meta's Llama 2 7B chat model"},
                {"id": "gpt2", "description": "OpenAI GPT-2 language model"}
            ],
            "image-classification": [
                {"id": "google/vit-base-patch16-224", "description": "Vision Transformer (ViT) model"},
                {"id": "microsoft/resnet-50", "description": "ResNet-50 model"},
                {"id": "facebook/convnext-tiny-224", "description": "ConvNeXt model for image classification"}
            ],
            "text-to-image": [
                {"id": "stabilityai/stable-diffusion-xl-base-1.0", "description": "Stable Diffusion XL 1.0"},
                {"id": "runwayml/stable-diffusion-v1-5", "description": "Stable Diffusion v1.5"}
            ],
            "embeddings": [
                {"id": "sentence-transformers/all-MiniLM-L6-v2", "description": "Efficient text embeddings"},
                {"id": "sentence-transformers/all-mpnet-base-v2", "description": "High-quality text embeddings"}
            ],
            "image-to-text": [
                {"id": "nlpconnect/vit-gpt2-image-captioning", "description": "Vision-language model for image captioning"},
                {"id": "Salesforce/blip-image-captioning-base", "description": "BLIP model for image captioning"}
            ],
            "object-detection": [
                {"id": "facebook/detr-resnet-50", "description": "DETR object detection model"},
                {"id": "hustvl/yolos-tiny", "description": "YOLOS object detection model"}
            ]
        }
        
        if task and task in task_models:
            return task_models[task]
        elif task:
            return []
        else:
            # Return all models if no task specified
            all_models = []
            for task_name, models in task_models.items():
                for model in models:
                    model["task"] = task_name
                all_models.extend(models)
            return all_models
    
    async def is_model_available(self, model_id: str) -> bool:
        """Check if a model is available on Hugging Face
        
        Args:
            model_id: The model ID to check
            
        Returns:
            True if the model is available, False otherwise
        """
        try:
            url = f"https://huggingface.co/api/models/{model_id}"
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.get(url)
            )
            
            return response.status_code == 200
        except Exception:
            return False
