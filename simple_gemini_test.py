"""
Simple Gemini Test for NEXUS
Tests the Gemini API with various models and capabilities
"""
import os
import sys
import logging
import json
from pathlib import Path
from PIL import Image
import google.generativeai as genai

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Your Gemini API key
GEMINI_API_KEY = "AIzaSyCF3NT8VEpeqhP2gM7to6d4J7W96NIyIrU"

def test_gemini_api():
    """Test the Gemini API with various capabilities"""
    # Configure the Gemini API
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Get available models
    logger.info("Listing available Gemini models...")
    models = genai.list_models()
    
    # Show all models
    logger.info(f"Found {len(models)} models")
    for model in models:
        logger.info(f"Model: {model.name}")
        logger.info(f"  Display name: {model.display_name}")
        logger.info(f"  Description: {model.description}")
        logger.info(f"  Generation methods: {model.supported_generation_methods}")
        logger.info("---")
    
    # Find Gemini 1.5 models
    gemini_15_models = [m for m in models if "gemini-1.5" in m.name]
    if gemini_15_models:
        logger.info(f"\nFound {len(gemini_15_models)} Gemini 1.5 models:")
        for model in gemini_15_models:
            logger.info(f"- {model.name}")
    
    # Choose a model to test (prioritize 1.5 models)
    preferred_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.5-flash-latest"]
    test_model = None
    
    for model_name in preferred_models:
        for available_model in models:
            if model_name in available_model.name:
                test_model = available_model.name
                break
        if test_model:
            break
    
    if not test_model:
        # Fallback to first vision-capable model
        for model in models:
            if "vision" in model.supported_generation_methods:
                test_model = model.name
                break
    
    logger.info(f"\nTesting with model: {test_model}")
    
    # Basic text generation test
    try:
        logger.info("\n=== Testing text generation ===")
        model = genai.GenerativeModel(test_model)
        response = model.generate_content("Explain how NEXUS, an adaptive AI system, can learn and integrate vision capabilities without following strict rules")
        
        if hasattr(response, 'text'):
            logger.info("Text generation successful:")
            logger.info(response.text)
        else:
            logger.error("Text generation failed")
    except Exception as e:
        logger.error(f"Error in text generation test: {e}")
    
    # Test with an image if the model supports vision
    test_image_path = "demos/test_images/general_test.jpg"
    
    # Create a simple test image if it doesn't exist
    create_test_image_if_needed(test_image_path)
    
    try:
        logger.info("\n=== Testing image analysis ===")
        model = genai.GenerativeModel(test_model)
        
        # Load image
        image = Image.open(test_image_path)
        
        # Vision prompt
        vision_prompt = """
        Analyze this image comprehensively and provide:
        1. A detailed description of the content
        2. Any text visible in the image
        3. Key objects and elements present
        
        Format your response in clear sections.
        """
        
        # Generate response
        response = model.generate_content([vision_prompt, image])
        
        if hasattr(response, 'text'):
            logger.info("Image analysis successful:")
            logger.info(response.text)
        else:
            logger.error("Image analysis failed")
            
    except Exception as e:
        logger.error(f"Error in image analysis test: {e}")

def create_test_image_if_needed(path):
    """Create a simple test image if it doesn't exist"""
    # Create directory if needed
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    if os.path.exists(path):
        return
    
    # Create a simple test image
    logger.info(f"Creating test image: {path}")
    img = Image.new('RGB', (500, 300), color=(240, 240, 240))
    
    # Use PIL to draw some shapes and text
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    
    # Add a title
    draw.text((20, 20), "NEXUS Test Image", fill=(0, 0, 0))
    
    # Add some shapes
    draw.rectangle([(50, 70), (200, 150)], outline=(0, 0, 255), width=2)
    draw.text((70, 100), "NEXUS Vision", fill=(0, 0, 255))
    
    draw.ellipse([(250, 70), (350, 150)], outline=(255, 0, 0), width=2)
    draw.text((260, 100), "Adaptive AI", fill=(255, 0, 0))
    
    # Add some text
    draw.text((50, 180), "This is a test image for Gemini API", fill=(0, 0, 0))
    draw.text((50, 210), "Integration with NEXUS", fill=(0, 0, 0))
    
    # Save the image
    img.save(path)
    logger.info(f"Created test image at: {path}")

if __name__ == "__main__":
    test_gemini_api()
