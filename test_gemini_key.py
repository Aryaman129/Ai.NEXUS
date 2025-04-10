"""
Test script for Gemini with the provided API key
"""
import os
import asyncio
import logging
from PIL import Image, ImageDraw
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Use the provided API key
API_KEY = "AIzaSyCF3NT8VEpeqhP2gM7to6d4J7W96NIyIrU"

# Test directory
TEST_DIR = "demos/test_images"

async def test_gemini_with_key():
    """Test Gemini API with the provided key"""
    logger.info("=== TESTING GEMINI WITH API KEY ===")
    
    # Configure Gemini with API key
    genai.configure(api_key=API_KEY)
    
    # Check available models
    try:
        models = genai.list_models()
        logger.info("Available Gemini models:")
        for model in models:
            if "gemini" in model.name.lower():
                logger.info(f"  - {model.name}")
    except Exception as e:
        logger.error(f"Error listing models: {e}")
    
    # Create test directory and image
    os.makedirs(TEST_DIR, exist_ok=True)
    test_image_path = os.path.join(TEST_DIR, "gemini_key_test.jpg")
    
    # Create a test image if it doesn't exist
    if not os.path.exists(test_image_path):
        img = Image.new('RGB', (400, 200), color=(73, 109, 137))
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "NEXUS AI Test Image", fill=(255, 255, 255))
        draw.text((10, 50), "Testing Gemini API Key", fill=(255, 255, 255))
        draw.rectangle([(50, 100), (350, 150)], outline=(255, 255, 255), width=2)
        draw.text((100, 115), "Button", fill=(255, 255, 255))
        img.save(test_image_path)
        logger.info(f"Created test image: {test_image_path}")
    
    # Test text generation
    logger.info("\n=== TESTING TEXT GENERATION ===")
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            "Explain how NEXUS AI can learn and adapt to new environments in 3 sentences."
        )
        
        if hasattr(response, 'text') and response.text:
            logger.info("TEXT GENERATION SUCCESS!")
            logger.info(f"Response: {response.text}")
        else:
            logger.error("Empty or invalid response from text generation")
    except Exception as e:
        logger.error(f"Text generation error: {e}")
    
    # Test image analysis
    logger.info("\n=== TESTING IMAGE ANALYSIS ===")
    try:
        img = Image.open(test_image_path)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        response = model.generate_content([
            "Describe what you see in this image, including any text and UI elements.",
            img
        ])
        
        if hasattr(response, 'text') and response.text:
            logger.info("IMAGE ANALYSIS SUCCESS!")
            logger.info(f"Analysis: {response.text}")
        else:
            logger.error("Empty or invalid response from image analysis")
    except Exception as e:
        logger.error(f"Image analysis error: {e}")

if __name__ == "__main__":
    asyncio.run(test_gemini_with_key())
