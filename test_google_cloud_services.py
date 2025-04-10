"""
Test Google Cloud Services Integration for NEXUS
This script tests all Google Cloud APIs to verify which ones are working
"""
import os
import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from PIL import Image
import json

# Add the project directory to Python path
sys.path.append(str(Path(__file__).parent))

# Import our Google Cloud Services integration
from src.integrations.google_cloud_services import GoogleCloudServices

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test image path - update this to a real image for testing
TEST_IMAGE_PATH = "demos/test_images/example.jpg"
TEST_SCREENSHOT_PATH = "demos/test_images/example_ui.png"

async def test_service_initialization():
    """Test if services initialize correctly"""
    logger.info("===== TESTING SERVICE INITIALIZATION =====")
    
    # Look for the service account file in standard locations
    service_account_paths = [
        "autonomus-1743898709312-8589efbc502d.json",  # Current directory
        os.path.expanduser("~/autonomus-1743898709312-8589efbc502d.json"),  # Home dir
        "d:/Ai/Nexus/autonomus-1743898709312-8589efbc502d.json",  # Project dir
        os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")  # Environment variable
    ]
    
    service_account_path = None
    for path in service_account_paths:
        if path and os.path.exists(path):
            service_account_path = path
            logger.info(f"Found service account at: {path}")
            break
    
    if not service_account_path:
        logger.error("No service account file found. Tests may fail.")
    
    # Initialize the services
    google_services = GoogleCloudServices(service_account_path=service_account_path)
    
    # Check which services are available
    service_status = google_services.service_status()
    
    logger.info("Service Status:")
    for service, available in service_status.items():
        logger.info(f"  {service}: {'Available' if available else 'Not Available'}")
    
    available_services = google_services.available_services
    logger.info(f"Available services: {available_services}")
    
    return google_services

async def test_vision_api(google_services):
    """Test the Cloud Vision API integration"""
    logger.info("\n===== TESTING CLOUD VISION API =====")
    
    if not google_services.vision_client:
        logger.warning("Cloud Vision API not available. Skipping test.")
        return False
    
    # Make sure test image exists
    if not os.path.exists(TEST_IMAGE_PATH):
        logger.warning(f"Test image not found at {TEST_IMAGE_PATH}. Creating a simple test image.")
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color = (73, 109, 137))
        os.makedirs(os.path.dirname(TEST_IMAGE_PATH), exist_ok=True)
        img.save(TEST_IMAGE_PATH)
    
    # Test image analysis
    logger.info(f"Analyzing test image: {TEST_IMAGE_PATH}")
    try:
        results = await google_services.analyze_image(image_path=TEST_IMAGE_PATH)
        
        if results.get("success", False):
            logger.info("Vision API test: SUCCESS")
            
            # Display some results
            if "labels" in results:
                logger.info("Labels detected:")
                for label in results["labels"][:3]:  # Show top 3
                    logger.info(f"  {label['description']} ({label['score']:.2f})")
            
            if "text" in results and results["text"].get("full_text"):
                logger.info(f"Text detected: {results['text']['full_text'][:100]}...")
            
            if "objects" in results:
                logger.info(f"Objects detected: {len(results['objects'])}")
            
            if "landmarks" in results:
                logger.info(f"Landmarks detected: {len(results['landmarks'])}")
            
            # Save full results to file for inspection
            with open("vision_api_results.json", "w") as f:
                json.dump(results, f, indent=2)
                logger.info("Full results saved to vision_api_results.json")
            
            return True
        else:
            logger.error(f"Vision API test failed: {results.get('error', 'Unknown error')}")
            return False
    
    except Exception as e:
        logger.error(f"Error testing Vision API: {e}")
        return False

async def test_ui_detection(google_services):
    """Test UI element detection"""
    logger.info("\n===== TESTING UI ELEMENT DETECTION =====")
    
    if not google_services.vision_client:
        logger.warning("Cloud Vision API not available. Skipping UI detection test.")
        return False
    
    # Make sure test screenshot exists
    if not os.path.exists(TEST_SCREENSHOT_PATH):
        logger.warning(f"Test screenshot not found at {TEST_SCREENSHOT_PATH}. Skipping test.")
        return False
    
    # Test UI detection
    logger.info(f"Analyzing UI elements in: {TEST_SCREENSHOT_PATH}")
    try:
        results = await google_services.detect_ui_elements(image_path=TEST_SCREENSHOT_PATH)
        
        if results.get("success", False):
            logger.info("UI Detection test: SUCCESS")
            
            # Display some results
            if "ui_elements" in results:
                logger.info(f"UI elements detected: {len(results['ui_elements'])}")
                for i, element in enumerate(results["ui_elements"][:3]):  # Show top 3
                    logger.info(f"  {i+1}. {element['type']} with text '{element.get('text', 'N/A')}'")
            
            if "text_blocks" in results:
                logger.info(f"Text blocks detected: {len(results['text_blocks'])}")
            
            # Save full results to file for inspection
            with open("ui_detection_results.json", "w") as f:
                json.dump(results, f, indent=2)
                logger.info("Full results saved to ui_detection_results.json")
            
            return True
        else:
            logger.error(f"UI Detection test failed: {results.get('error', 'Unknown error')}")
            return False
    
    except Exception as e:
        logger.error(f"Error testing UI detection: {e}")
        return False

async def test_generative_ai(google_services):
    """Test the Generative AI (Gemini) integration"""
    logger.info("\n===== TESTING GENERATIVE AI (GEMINI) =====")
    
    if not google_services.genai_configured:
        logger.warning("Generative AI not available. Skipping test.")
        return False
    
    # Test text generation
    test_prompt = "Explain what NEXUS AI is in 3 sentences."
    logger.info(f"Generating text with prompt: '{test_prompt}'")
    
    try:
        results = await google_services.generate_text(prompt=test_prompt, max_tokens=100)
        
        if results.get("success", False):
            logger.info("Generative AI text test: SUCCESS")
            logger.info(f"Generated text:\n{results['generated_text']}")
            return True
        else:
            logger.error(f"Generative AI text test failed: {results.get('error', 'Unknown error')}")
            return False
    
    except Exception as e:
        logger.error(f"Error testing Generative AI text generation: {e}")
        return False

async def test_image_with_genai(google_services):
    """Test the Generative AI (Gemini) image analysis"""
    logger.info("\n===== TESTING GEMINI IMAGE ANALYSIS =====")
    
    if not google_services.genai_configured:
        logger.warning("Generative AI not available. Skipping image analysis test.")
        return False
    
    # Make sure test image exists
    if not os.path.exists(TEST_IMAGE_PATH):
        logger.warning(f"Test image not found at {TEST_IMAGE_PATH}. Skipping test.")
        return False
    
    # Test image analysis with Gemini
    prompt = "What do you see in this image? Describe it in detail."
    logger.info(f"Analyzing image with Gemini and prompt: '{prompt}'")
    
    try:
        results = await google_services.analyze_image_with_genai(
            image_path=TEST_IMAGE_PATH,
            prompt=prompt
        )
        
        if results.get("success", False):
            logger.info("Gemini image analysis test: SUCCESS")
            logger.info(f"Analysis:\n{results['analysis'][:200]}...")  # First 200 chars
            
            # Save full results to file for inspection
            with open("gemini_image_results.txt", "w") as f:
                f.write(results['analysis'])
                logger.info("Full results saved to gemini_image_results.txt")
            
            return True
        else:
            logger.error(f"Gemini image analysis test failed: {results.get('error', 'Unknown error')}")
            return False
    
    except Exception as e:
        logger.error(f"Error testing Gemini image analysis: {e}")
        return False

async def test_custom_search(google_services):
    """Test the Custom Search API integration"""
    logger.info("\n===== TESTING CUSTOM SEARCH API =====")
    
    if not google_services.search_client:
        logger.warning("Custom Search API not available. Skipping test.")
        return False
    
    # Test web search
    test_query = "NEXUS AI autonomous systems"
    logger.info(f"Searching for: '{test_query}'")
    
    try:
        results = await google_services.search_web(query=test_query, num_results=5)
        
        if results.get("success", False):
            logger.info("Custom Search API test: SUCCESS")
            
            # Display some results
            if "results" in results:
                logger.info(f"Found {len(results['results'])} results:")
                for i, result in enumerate(results["results"][:3]):  # Show top 3
                    logger.info(f"  {i+1}. {result.get('title', 'No title')}")
                    logger.info(f"     URL: {result.get('link', 'No URL')}")
            
            # Save full results to file for inspection
            with open("search_api_results.json", "w") as f:
                json.dump(results, f, indent=2)
                logger.info("Full results saved to search_api_results.json")
            
            return True
        else:
            logger.error(f"Custom Search API test failed: {results.get('error', 'Unknown error')}")
            return False
    
    except Exception as e:
        logger.error(f"Error testing Custom Search API: {e}")
        return False

async def main():
    """Run all tests"""
    logger.info(f"Starting Google Cloud Services tests at {datetime.now()}")
    
    # Initialize services
    google_services = await test_service_initialization()
    
    # Track test results
    test_results = {}
    
    # Run individual tests based on what's available
    if "vision" in google_services.available_services:
        test_results["vision_api"] = await test_vision_api(google_services)
        test_results["ui_detection"] = await test_ui_detection(google_services)
    else:
        logger.warning("Vision API not available. Skipping Vision tests.")
    
    if "generative_ai" in google_services.available_services:
        test_results["generative_ai"] = await test_generative_ai(google_services)
        test_results["gemini_image"] = await test_image_with_genai(google_services)
    else:
        logger.warning("Generative AI not available. Skipping Gemini tests.")
    
    if "search" in google_services.available_services:
        test_results["custom_search"] = await test_custom_search(google_services)
    else:
        logger.warning("Custom Search API not available. Skipping Search test.")
    
    # Print summary
    logger.info("\n===== TEST SUMMARY =====")
    passed = sum(1 for result in test_results.values() if result)
    total = len(test_results)
    
    logger.info(f"Passed: {passed}/{total} tests")
    
    for test, result in test_results.items():
        logger.info(f"  {test}: {'PASS' if result else 'FAIL'}")
    
    logger.info(f"Google Cloud Services tests completed at {datetime.now()}")

if __name__ == "__main__":
    asyncio.run(main())
