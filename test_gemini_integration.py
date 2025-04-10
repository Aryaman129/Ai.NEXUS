"""
Test script for Gemini AI integration with NEXUS
"""
import os
import sys
import asyncio
import logging
from pathlib import Path
from PIL import Image

# Add project to Python path
sys.path.append(str(Path(__file__).parent))

# Import our Gemini integration
from src.integrations.gemini_integration import GeminiAI

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test images directory
TEST_IMAGES_DIR = "demos/test_images"

async def test_gemini():
    """Test Gemini AI integration"""
    logger.info("=== TESTING GEMINI AI INTEGRATION ===")
    
    # Get API key from environment if available
    api_key = os.environ.get("GOOGLE_API_KEY")
    
    # Find service account file
    service_account_path = None
    for path in ["autonomus-1743898709312-8589efbc502d.json", 
                "d:/Ai/Nexus/autonomus-1743898709312-8589efbc502d.json"]:
        if os.path.exists(path):
            service_account_path = path
            break
    
    # Initialize Gemini
    gemini = GeminiAI(api_key=api_key, service_account_path=service_account_path)
    
    # Check if Gemini is available
    if not gemini.is_available():
        logger.error("Gemini AI not available. Please check your credentials.")
        
        # Give instructions for getting an API key
        logger.info("\nTo make Gemini work, you can get an API key:")
        logger.info("1. Go to https://makersuite.google.com/app/apikey")
        logger.info("2. Create a new API key")
        logger.info("3. Add it to this script or set as GOOGLE_API_KEY environment variable")
        
        return
    
    # Create test images directory if it doesn't exist
    os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
    
    # Check if we have test images
    test_image_path = os.path.join(TEST_IMAGES_DIR, "example.jpg")
    test_screenshot_path = os.path.join(TEST_IMAGES_DIR, "example_ui.png")
    
    has_test_image = os.path.exists(test_image_path)
    has_test_screenshot = os.path.exists(test_screenshot_path)
    
    if not has_test_image and not has_test_screenshot:
        logger.warning("No test images found. Creating a simple test image.")
        # Create a simple test image
        img = Image.new('RGB', (300, 200), color=(73, 109, 137))
        img.save(test_image_path)
        has_test_image = True
    
    # 1. Test text generation
    logger.info("\nTesting text generation...")
    text_result = await gemini.generate_text(
        prompt="Explain how NEXUS can adapt and learn from new information in 3 sentences."
    )
    
    if text_result.get("success", False):
        logger.info("Text generation SUCCESS!")
        logger.info(f"Response: {text_result['text']}")
    else:
        logger.error(f"Text generation failed: {text_result.get('error')}")
    
    # 2. Test image analysis if we have a test image
    if has_test_image:
        logger.info(f"\nTesting image analysis with {test_image_path}...")
        image_result = await gemini.analyze_image(image_path=test_image_path)
        
        if image_result.get("success", False):
            logger.info("Image analysis SUCCESS!")
            logger.info(f"Analysis: {image_result['analysis'][:200]}...")  # First 200 chars
        else:
            logger.error(f"Image analysis failed: {image_result.get('error')}")
    
    # 3. Test UI detection if we have a screenshot
    if has_test_screenshot:
        logger.info(f"\nTesting UI element detection with {test_screenshot_path}...")
        ui_result = await gemini.detect_ui_elements(image_path=test_screenshot_path)
        
        if ui_result.get("success", False):
            logger.info("UI detection SUCCESS!")
            logger.info(f"Analysis: {ui_result['analysis'][:200]}...")  # First 200 chars
        else:
            logger.error(f"UI detection failed: {ui_result.get('error')}")
    
    # 4. Test chat
    logger.info("\nTesting chat functionality...")
    chat_result = await gemini.chat([
        {"role": "system", "content": "You are NEXUS, an AI that learns and adapts."},
        {"role": "user", "content": "How can you help analyze images?"}
    ])
    
    if chat_result.get("success", False):
        logger.info("Chat SUCCESS!")
        logger.info(f"Response: {chat_result['text'][:200]}...")  # First 200 chars
    else:
        logger.error(f"Chat failed: {chat_result.get('error')}")
    
    logger.info("\n=== TEST SUMMARY ===")
    logger.info(f"Text generation: {'✅ Success' if text_result.get('success', False) else '❌ Failed'}")
    logger.info(f"Image analysis: {'✅ Success' if has_test_image and image_result.get('success', False) else '❌ Failed' if has_test_image else '⚠️ Skipped'}")
    logger.info(f"UI detection: {'✅ Success' if has_test_screenshot and ui_result.get('success', False) else '❌ Failed' if has_test_screenshot else '⚠️ Skipped'}")
    logger.info(f"Chat: {'✅ Success' if chat_result.get('success', False) else '❌ Failed'}")

async def main():
    await test_gemini()

if __name__ == "__main__":
    asyncio.run(main())
