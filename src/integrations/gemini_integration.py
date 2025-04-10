"""
Gemini AI Integration for NEXUS
Provides multimodal AI capabilities using Google's Gemini models
"""
import os
import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import base64
import io
from PIL import Image
import google.generativeai as genai

logger = logging.getLogger(__name__)

class GeminiAI:
    """
    Gemini AI Integration for NEXUS
    
    Provides advanced AI capabilities using Google's Gemini models:
    - Text generation and reasoning (Gemini Pro)
    - Image analysis and understanding (Gemini Pro Vision)
    - Visual knowledge extraction and learning
    
    This integration adapts to what's available and learns continuously,
    following NEXUS's philosophy of dynamic tool integration.
    """
    
    def __init__(self, api_key: str = None, service_account_path: str = None):
        """
        Initialize the Gemini AI integration
        
        Args:
            api_key: Google API key for Gemini (preferred method)
            service_account_path: Path to service account JSON file (alternative)
        """
        self.api_key = api_key
        self.service_account_path = service_account_path
        self.available = False
        self.available_models = []
        
        # Try to initialize
        self._initialize()
    
    def _initialize(self):
        """Initialize the Gemini AI integration"""
        # First try API key if provided
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.available = True
                logger.info("Gemini AI initialized with API key")
                self._get_available_models()
                return
            except Exception as e:
                logger.error(f"Failed to initialize Gemini with API key: {e}")
        
        # Try service account if provided
        if self.service_account_path:
            try:
                from google.oauth2 import service_account
                credentials = service_account.Credentials.from_service_account_file(
                    self.service_account_path
                )
                genai.configure(credentials=credentials)
                self.available = True
                logger.info("Gemini AI initialized with service account")
                self._get_available_models()
                return
            except Exception as e:
                logger.error(f"Failed to initialize Gemini with service account: {e}")
        
        # If we got here, try to find service account in standard locations
        if not self.service_account_path:
            potential_paths = [
                "autonomus-1743898709312-8589efbc502d.json",  # Current directory
                os.path.expanduser("~/autonomus-1743898709312-8589efbc502d.json"),  # Home dir
                "d:/Ai/Nexus/autonomus-1743898709312-8589efbc502d.json",  # Project dir
                os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")  # Environment variable
            ]
            
            for path in potential_paths:
                if path and os.path.exists(path):
                    try:
                        from google.oauth2 import service_account
                        credentials = service_account.Credentials.from_service_account_file(path)
                        genai.configure(credentials=credentials)
                        self.service_account_path = path
                        self.available = True
                        logger.info(f"Gemini AI initialized with service account from {path}")
                        self._get_available_models()
                        return
                    except Exception as e:
                        logger.error(f"Failed to initialize Gemini with service account from {path}: {e}")
        
        logger.warning("Gemini AI initialization failed. No valid credentials found.")
    
    def _get_available_models(self):
        """Get available Gemini models"""
        try:
            models = genai.list_models()
            self.available_models = [model.name for model in models if "gemini" in model.name]
            logger.info(f"Available Gemini models: {self.available_models}")
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            self.available_models = ["gemini-pro", "gemini-pro-vision"]  # Fallback to common models
    
    def is_available(self) -> bool:
        """Check if Gemini AI is available"""
        return self.available
    
    async def generate_text(self, 
                           prompt: str, 
                           temperature: float = 0.7,
                           max_tokens: int = 1024) -> Dict[str, Any]:
        """
        Generate text using Gemini Pro
        
        Args:
            prompt: Text prompt
            temperature: Temperature for generation (0.0-1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with generation results
        """
        if not self.available:
            return self._error_response("Gemini AI not available")
        
        try:
            # Set up model
            model = genai.GenerativeModel('gemini-pro')
            
            # Configure generation parameters
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "top_p": 0.95,
                "top_k": 40
            }
            
            # Generate content
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            # Return results
            return {
                "success": True,
                "text": response.text,
                "prompt": prompt,
                "model": "gemini-pro",
                "finish_reason": "stop"  # We don't get this from Gemini API, assuming normal completion
            }
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return self._error_response(f"Text generation error: {str(e)}")
    
    async def analyze_image(self, 
                           image_path: str = None, 
                           image: Image.Image = None,
                           prompt: str = None) -> Dict[str, Any]:
        """
        Analyze an image using Gemini Pro Vision
        
        Args:
            image_path: Path to image file
            image: PIL Image object
            prompt: Optional prompt to guide analysis
            
        Returns:
            Dictionary with analysis results
        """
        if not self.available:
            return self._error_response("Gemini AI not available")
        
        # Default prompt if none provided
        if not prompt:
            prompt = ("Analyze this image in detail. Describe what you see, including objects, "
                    "text, people, actions, and the overall context. "
                    "If you see any UI elements like buttons or menus, describe those too.")
        
        try:
            # Load image
            if image_path and os.path.exists(image_path):
                img = Image.open(image_path)
            elif image:
                img = image
            else:
                return self._error_response("No valid image provided")
            
            # Set up model
            model = genai.GenerativeModel('gemini-pro-vision')
            
            # Generate content
            response = model.generate_content([prompt, img])
            
            # Return results
            return {
                "success": True,
                "analysis": response.text,
                "prompt": prompt,
                "model": "gemini-pro-vision"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return self._error_response(f"Image analysis error: {str(e)}")
    
    async def detect_ui_elements(self,
                               image_path: str = None,
                               image: Image.Image = None) -> Dict[str, Any]:
        """
        Detect UI elements in a screenshot using Gemini
        
        Args:
            image_path: Path to screenshot file
            image: PIL Image object
            
        Returns:
            Dictionary with UI detection results
        """
        if not self.available:
            return self._error_response("Gemini AI not available")
        
        ui_prompt = (
            "Analyze this screenshot and identify all UI elements. "
            "For each element, provide its type (button, textbox, link, menu, etc.), "
            "any text it contains, and its approximate position. "
            "Format your response as a structured list of elements. "
            "Be very specific and detailed."
        )
        
        try:
            # Use general image analysis with specific UI prompt
            result = await self.analyze_image(
                image_path=image_path,
                image=image,
                prompt=ui_prompt
            )
            
            # If successful, enhance the result with UI-specific metadata
            if result.get("success", False):
                result["ui_detection"] = True
                
                # We could parse the text response to extract structured UI elements,
                # but for now we'll just return the raw analysis
                
            return result
            
        except Exception as e:
            logger.error(f"Error detecting UI elements: {e}")
            return self._error_response(f"UI detection error: {str(e)}")
    
    async def extract_text_from_image(self,
                                    image_path: str = None,
                                    image: Image.Image = None) -> Dict[str, Any]:
        """
        Extract text from an image using Gemini
        
        Args:
            image_path: Path to image file
            image: PIL Image object
            
        Returns:
            Dictionary with extracted text
        """
        if not self.available:
            return self._error_response("Gemini AI not available")
        
        text_prompt = (
            "Extract ALL text from this image. "
            "Include everything you can see, such as headings, paragraphs, labels, button text, etc. "
            "Preserve the formatting as much as possible. "
            "Only output the extracted text, nothing else."
        )
        
        try:
            # Use general image analysis with specific text extraction prompt
            result = await self.analyze_image(
                image_path=image_path,
                image=image,
                prompt=text_prompt
            )
            
            # If successful, enhance the result with text-specific metadata
            if result.get("success", False):
                result["text_extraction"] = True
                result["extracted_text"] = result["analysis"]
                
            return result
            
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return self._error_response(f"Text extraction error: {str(e)}")
    
    async def chat(self, 
                  messages: List[Dict[str, str]],
                  temperature: float = 0.7) -> Dict[str, Any]:
        """
        Chat with Gemini
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Temperature for generation
            
        Returns:
            Dictionary with chat response
        """
        if not self.available:
            return self._error_response("Gemini AI not available")
        
        try:
            # Set up model
            model = genai.GenerativeModel('gemini-pro')
            
            # Configure chat
            chat = model.start_chat(history=[])
            
            # Process messages
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                
                if role == "system" or role == "assistant":
                    # Add as a model response
                    chat.history.append({"role": "model", "parts": [content]})
                else:
                    # Add as a user message
                    chat.history.append({"role": "user", "parts": [content]})
            
            # Generate response to the last message if it's from the user
            response = None
            if messages and messages[-1]["role"] == "user":
                response = chat.send_message(messages[-1]["content"], temperature=temperature)
            
            # If no response (e.g., last message was from assistant), generate from a blank prompt
            if not response:
                response = chat.send_message("Continue the conversation.", temperature=temperature)
            
            # Return results
            return {
                "success": True,
                "text": response.text,
                "chat_history": messages,
                "model": "gemini-pro"
            }
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return self._error_response(f"Chat error: {str(e)}")
    
    def _error_response(self, message: str) -> Dict[str, Any]:
        """Create an error response"""
        return {
            "success": False,
            "error": message
        }


# Simple usage example
async def example_usage():
    """Example of how to use the GeminiAI class"""
    # Initialize
    gemini = GeminiAI()
    
    if not gemini.is_available():
        print("Gemini AI not available. Please check your credentials.")
        return
    
    # Generate text
    text_result = await gemini.generate_text(
        prompt="Explain how NEXUS AI can use Gemini for visual understanding."
    )
    
    if text_result.get("success", False):
        print("\nText Generation Result:")
        print(text_result["text"])
    
    # Analyze an image (if a test image is available)
    test_image = "demos/test_images/example.jpg"
    if os.path.exists(test_image):
        print(f"\nAnalyzing image: {test_image}")
        image_result = await gemini.analyze_image(image_path=test_image)
        
        if image_result.get("success", False):
            print("\nImage Analysis Result:")
            print(image_result["analysis"])

# Run the example if executed directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_usage())
