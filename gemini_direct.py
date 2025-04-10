"""
Direct Gemini API implementation using the latest available models
"""
import os
import sys
import asyncio
import logging
from pathlib import Path
import base64
from PIL import Image, ImageDraw
import requests
import json

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NexusGemini:
    """
    Direct implementation of Gemini API access using REST API
    This approach bypasses the Python SDK limitations and adapts to what's available
    """
    
    def __init__(self, service_account_path=None):
        """Initialize with service account"""
        self.service_account_path = service_account_path
        self.project_id = None
        self.token = None
        
        # Find service account if not provided
        if not self.service_account_path:
            for path in ["autonomus-1743898709312-8589efbc502d.json", 
                         "d:/Ai/Nexus/autonomus-1743898709312-8589efbc502d.json"]:
                if os.path.exists(path):
                    self.service_account_path = path
                    break
        
        # Load service account details
        if self.service_account_path and os.path.exists(self.service_account_path):
            try:
                with open(self.service_account_path, 'r') as f:
                    sa_info = json.load(f)
                    self.project_id = sa_info.get('project_id')
                    logger.info(f"Loaded project ID: {self.project_id}")
            except Exception as e:
                logger.error(f"Error loading service account: {e}")
    
    async def authenticate(self):
        """Get authentication token from service account"""
        if not self.service_account_path:
            logger.error("No service account file found")
            return False
        
        try:
            # Generate token using Google's API
            from google.oauth2 import service_account
            from google.auth.transport.requests import Request
            
            credentials = service_account.Credentials.from_service_account_file(
                self.service_account_path,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            
            # Force token refresh
            request = Request()
            credentials.refresh(request)
            
            self.token = credentials.token
            logger.info("Successfully obtained authentication token")
            return True
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False
    
    def encode_image_base64(self, image_path=None, image=None):
        """Encode an image to base64 for API requests"""
        try:
            # Get the image
            if image_path and os.path.exists(image_path):
                img = Image.open(image_path)
            elif image:
                img = image
            else:
                return None
            
            # Convert to JPEG format in memory
            import io
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            image_bytes = buffer.getvalue()
            
            # Encode to base64
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            return base64_image
            
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            return None
    
    async def generate_text(self, prompt, model="gemini-1.5-flash"):
        """Generate text using the latest Gemini models"""
        if not self.token:
            success = await self.authenticate()
            if not success:
                return {"success": False, "error": "Authentication failed"}
        
        try:
            # Prepare request
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            }
            
            data = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": prompt}
                        ]
                    }
                ],
                "generation_config": {
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 1024
                }
            }
            
            # Make request
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract the text from the response
                generated_text = ""
                if "candidates" in result and len(result["candidates"]) > 0:
                    for part in result["candidates"][0]["content"]["parts"]:
                        if "text" in part:
                            generated_text += part["text"]
                
                return {
                    "success": True,
                    "text": generated_text,
                    "model": model,
                    "full_response": result
                }
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "error": f"API request failed: {response.status_code}",
                    "details": response.text
                }
                
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return {"success": False, "error": str(e)}
    
    async def analyze_image(self, image_path=None, image=None, prompt=None, model="gemini-1.5-flash"):
        """Analyze an image using Gemini multimodal capabilities"""
        if not self.token:
            success = await self.authenticate()
            if not success:
                return {"success": False, "error": "Authentication failed"}
        
        # Default prompt
        if not prompt:
            prompt = "Analyze this image in detail. Describe what you see, including any text, objects, and UI elements."
        
        # Encode image
        base64_image = self.encode_image_base64(image_path, image)
        if not base64_image:
            return {"success": False, "error": "Failed to encode image"}
        
        try:
            # Prepare request
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            }
            
            data = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": prompt},
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": base64_image
                                }
                            }
                        ]
                    }
                ]
            }
            
            # Make request
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract the text from the response
                analysis_text = ""
                if "candidates" in result and len(result["candidates"]) > 0:
                    for part in result["candidates"][0]["content"]["parts"]:
                        if "text" in part:
                            analysis_text += part["text"]
                
                return {
                    "success": True,
                    "analysis": analysis_text,
                    "model": model,
                    "full_response": result
                }
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "error": f"API request failed: {response.status_code}",
                    "details": response.text
                }
                
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return {"success": False, "error": str(e)}

async def test_gemini_direct():
    """Test the direct Gemini implementation"""
    logger.info("=== TESTING DIRECT GEMINI IMPLEMENTATION ===")
    
    # Initialize
    gemini = NexusGemini()
    
    # Create test directory and image
    test_dir = "demos/test_images"
    os.makedirs(test_dir, exist_ok=True)
    test_image_path = os.path.join(test_dir, "direct_test.jpg")
    
    # Create a test image if it doesn't exist
    if not os.path.exists(test_image_path):
        img = Image.new('RGB', (400, 200), color=(73, 109, 137))
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "NEXUS AI Test Image", fill=(255, 255, 255))
        draw.text((10, 50), "Testing Gemini Vision API", fill=(255, 255, 255))
        draw.rectangle([(50, 100), (350, 150)], outline=(255, 255, 255), width=2)
        draw.text((100, 115), "Click Me", fill=(255, 255, 255))
        img.save(test_image_path)
        logger.info(f"Created test image: {test_image_path}")
    
    # Test text generation
    logger.info("\n=== TESTING TEXT GENERATION ===")
    text_result = await gemini.generate_text(
        "Explain how NEXUS can learn and adapt to new information in 3 sentences."
    )
    
    if text_result.get("success", False):
        logger.info("TEXT GENERATION SUCCESS!")
        logger.info(f"Response: {text_result['text']}")
    else:
        logger.error(f"Text generation failed: {text_result.get('error')}")
        if 'details' in text_result:
            logger.error(f"Details: {text_result['details']}")
    
    # Test image analysis
    logger.info("\n=== TESTING IMAGE ANALYSIS ===")
    image_result = await gemini.analyze_image(image_path=test_image_path)
    
    if image_result.get("success", False):
        logger.info("IMAGE ANALYSIS SUCCESS!")
        logger.info(f"Analysis: {image_result['analysis']}")
    else:
        logger.error(f"Image analysis failed: {image_result.get('error')}")
        if 'details' in image_result:
            logger.error(f"Details: {image_result['details']}")
    
    # Print summary
    logger.info("\n=== TEST SUMMARY ===")
    logger.info(f"Text Generation: {'✅ Success' if text_result.get('success', False) else '❌ Failed'}")
    logger.info(f"Image Analysis: {'✅ Success' if image_result.get('success', False) else '❌ Failed'}")

if __name__ == "__main__":
    asyncio.run(test_gemini_direct())
