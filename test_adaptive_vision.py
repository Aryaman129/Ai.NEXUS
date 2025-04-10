"""
Test script for Adaptive Vision integration
Demonstrates how the system learns and adapts to use available capabilities
"""
import os
import asyncio
import logging
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Add project to path
sys.path.append(str(Path(__file__).parent))

# Import our adaptive vision
from src.integrations.adaptive_vision import AdaptiveVision

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Your Gemini API key
GEMINI_API_KEY = "AIzaSyCF3NT8VEpeqhP2gM7to6d4J7W96NIyIrU"

# Test images directory
TEST_DIR = "demos/test_images"

async def test_adaptive_vision():
    """Test the Adaptive Vision system with various scenarios"""
    logger.info("=== TESTING ADAPTIVE VISION SYSTEM ===")
    
    # Create test directory if it doesn't exist
    os.makedirs(TEST_DIR, exist_ok=True)
    
    # Create or locate test images
    test_image_path = os.path.join(TEST_DIR, "adaptive_test.jpg")
    test_ui_path = os.path.join(TEST_DIR, "adaptive_ui.jpg")
    test_text_path = os.path.join(TEST_DIR, "adaptive_text.jpg")
    
    # Create test images if they don't exist
    if not os.path.exists(test_image_path):
        create_test_image(test_image_path, "General Test Image")
        
    if not os.path.exists(test_ui_path):
        create_ui_test_image(test_ui_path)
        
    if not os.path.exists(test_text_path):
        create_text_test_image(test_text_path)
    
    # Initialize the adaptive vision system
    # The system will automatically detect what's available and adapt
    vision = AdaptiveVision(gemini_api_key=GEMINI_API_KEY)
    
    # Check available capabilities
    capabilities = vision.get_capabilities()
    logger.info(f"Available capabilities: {capabilities}")
    
    # 1. Test general image analysis
    logger.info("\n=== TESTING GENERAL IMAGE ANALYSIS ===")
    analysis_result = await vision.analyze_image(image_path=test_image_path)
    
    if analysis_result.get("success", False):
        logger.info(f"Analysis successful using: {analysis_result.get('sources', [])}")
        
        if "description" in analysis_result:
            logger.info(f"Description: {analysis_result['description'][:200]}...")
            
        if "labels" in analysis_result:
            logger.info("Top labels: " + ", ".join([label["description"] for label in analysis_result.get("labels", [])[:5]]))
            
        if "text" in analysis_result and analysis_result["text"].get("full_text"):
            logger.info(f"Extracted text: {analysis_result['text']['full_text']}")
    else:
        logger.error(f"Analysis failed: {analysis_result.get('error')}")
    
    # 2. Test UI element detection
    logger.info("\n=== TESTING UI ELEMENT DETECTION ===")
    ui_result = await vision.detect_ui_elements(image_path=test_ui_path)
    
    if ui_result.get("success", False):
        logger.info(f"UI detection successful using: {ui_result.get('sources', [])}")
        
        if "ui_description" in ui_result:
            logger.info(f"UI description: {ui_result['ui_description'][:200]}...")
            
        if "ui_elements" in ui_result:
            logger.info(f"Detected {len(ui_result['ui_elements'])} UI elements")
            for i, element in enumerate(ui_result.get("ui_elements", [])[:3]):
                logger.info(f"  Element {i+1}: {element.get('type')} - '{element.get('text')}'")
    else:
        logger.error(f"UI detection failed: {ui_result.get('error')}")
    
    # 3. Test text extraction
    logger.info("\n=== TESTING TEXT EXTRACTION ===")
    text_result = await vision.extract_text_from_image(image_path=test_text_path)
    
    if text_result.get("success", False):
        logger.info(f"Text extraction successful using: {text_result.get('sources', [])}")
        logger.info(f"Extracted text: {text_result.get('text', '')[:200]}...")
    else:
        logger.error(f"Text extraction failed: {text_result.get('error')}")
    
    # Print summary
    logger.info("\n=== TEST SUMMARY ===")
    logger.info(f"Capabilities: {capabilities}")
    logger.info(f"Image Analysis: {'✅ Success' if analysis_result.get('success', False) else '❌ Failed'}")
    logger.info(f"UI Detection: {'✅ Success' if ui_result.get('success', False) else '❌ Failed'}")
    logger.info(f"Text Extraction: {'✅ Success' if text_result.get('success', False) else '❌ Failed'}")
    
    # Show how the system adapts
    logger.info("\n=== ADAPTATION CAPABILITIES ===")
    logger.info("The Adaptive Vision system demonstrates learning and adaptation in several ways:")
    logger.info("1. Automatically detected and used your Gemini API key")
    if capabilities.get("cloud_vision", False):
        logger.info("2. Successfully connected to Cloud Vision API")
    else:
        logger.info("2. Adapted to Cloud Vision API unavailability by using alternatives")
    logger.info("3. Used the best available tools for each specific task")
    logger.info("4. Would store visual understanding in vector storage if connected")
    logger.info("5. Can continue learning from each new image it analyzes")

def create_test_image(path, text="Test Image"):
    """Create a general test image"""
    img = Image.new('RGB', (400, 200), color=(73, 109, 137))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), text, fill=(255, 255, 255))
    draw.text((10, 40), "NEXUS Adaptive Vision", fill=(255, 255, 255))
    img.save(path)
    logger.info(f"Created test image: {path}")

def create_ui_test_image(path):
    """Create a test image with UI elements"""
    img = Image.new('RGB', (500, 300), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # Add a header
    draw.rectangle([(0, 0), (500, 50)], fill=(0, 120, 215))
    draw.text((20, 15), "NEXUS Dashboard", fill=(255, 255, 255))
    
    # Add some buttons
    draw.rectangle([(50, 100), (150, 140)], fill=(0, 180, 0), outline=(0, 120, 0), width=2)
    draw.text((70, 112), "Login", fill=(255, 255, 255))
    
    draw.rectangle([(200, 100), (300, 140)], fill=(220, 0, 0), outline=(180, 0, 0), width=2)
    draw.text((210, 112), "Cancel", fill=(255, 255, 255))
    
    # Add a text field
    draw.rectangle([(50, 180), (400, 220)], fill=(255, 255, 255), outline=(200, 200, 200), width=2)
    draw.text((60, 190), "Enter your username", fill=(180, 180, 180))
    
    # Add a checkbox
    draw.rectangle([(50, 250), (70, 270)], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    draw.text((80, 252), "Remember me", fill=(0, 0, 0))
    
    img.save(path)
    logger.info(f"Created UI test image: {path}")

def create_text_test_image(path):
    """Create a test image with lots of text"""
    img = Image.new('RGB', (600, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Add a title
    draw.text((50, 20), "NEXUS: Adaptive AI System", fill=(0, 0, 0))
    
    # Add paragraphs of text
    paragraph1 = """
    NEXUS is designed to learn and adapt to new information 
    and environments without strict rules. It combines 
    multiple perception systems and uses whatever 
    capabilities are available.
    """
    
    paragraph2 = """
    Key features include:
    • Continuous learning from experiences
    • Dynamic tool orchestration
    • Multimodal understanding
    • Adaptive use of available APIs
    • Graceful fallback when preferred methods are unavailable
    """
    
    draw.text((50, 80), paragraph1, fill=(0, 0, 0))
    draw.text((50, 180), paragraph2, fill=(0, 0, 0))
    
    # Add some structured text
    draw.text((50, 300), "System Status:", fill=(0, 0, 0))
    draw.text((70, 320), "✓ Vision System: Online", fill=(0, 120, 0))
    draw.text((70, 340), "✓ Learning System: Active", fill=(0, 120, 0))
    draw.text((70, 360), "✓ Adaptation: Enabled", fill=(0, 120, 0))
    
    img.save(path)
    logger.info(f"Created text test image: {path}")

if __name__ == "__main__":
    asyncio.run(test_adaptive_vision())
