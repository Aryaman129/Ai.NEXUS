"""
Quick test script for Google Cloud APIs
Tests which APIs are functioning with the current credentials
"""
import os
import sys
import asyncio
import logging
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_vision_api():
    """Test if Cloud Vision API is working"""
    logger.info("Testing Cloud Vision API...")
    
    # Find service account file
    service_account_path = None
    for path in ["autonomus-1743898709312-8589efbc502d.json", 
                "d:/Ai/Nexus/autonomus-1743898709312-8589efbc502d.json"]:
        if os.path.exists(path):
            service_account_path = path
            break
    
    if not service_account_path:
        logger.error("No service account file found!")
        return False
    
    try:
        from google.oauth2 import service_account
        from google.cloud import vision
        
        # Initialize Vision client
        credentials = service_account.Credentials.from_service_account_file(service_account_path)
        client = vision.ImageAnnotatorClient(credentials=credentials)
        
        # Test with a simple feature request to check if it works
        # We'll create a tiny test image
        from PIL import Image
        import io
        
        # Generate a small test image
        img = Image.new('RGB', (100, 100), color=(73, 109, 137))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        content = img_byte_arr.getvalue()
        
        # Create Vision API image
        vision_image = vision.Image(content=content)
        
        # Try label detection
        response = client.label_detection(image=vision_image)
        
        if response.error.message:
            logger.error(f"Vision API error: {response.error.message}")
            return False
        
        # If we got here, it worked
        labels = response.label_annotations
        logger.info(f"Vision API test success! Found {len(labels)} labels")
        for label in labels[:3]:
            logger.info(f"  Label: {label.description} ({label.score:.2f})")
        
        return True
        
    except Exception as e:
        logger.error(f"Vision API test failed: {str(e)}")
        return False

async def test_search_api():
    """Test if Custom Search API is working"""
    logger.info("Testing Custom Search API...")
    
    # Find service account file
    service_account_path = None
    for path in ["autonomus-1743898709312-8589efbc502d.json", 
                "d:/Ai/Nexus/autonomus-1743898709312-8589efbc502d.json"]:
        if os.path.exists(path):
            service_account_path = path
            break
    
    if not service_account_path:
        logger.error("No service account file found!")
        return False
    
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        
        # Initialize credentials
        credentials = service_account.Credentials.from_service_account_file(
            service_account_path,
            scopes=['https://www.googleapis.com/auth/cse']
        )
        
        # Get the service account info to extract project ID
        with open(service_account_path, 'r') as f:
            sa_info = json.load(f)
            project_id = sa_info.get('project_id')
            logger.info(f"Project ID from service account: {project_id}")
        
        # Try to build search client
        search_client = build(
            'customsearch', 'v1',
            credentials=credentials,
            cache_discovery=False
        )
        
        # Attempt a simple search (will need search engine ID)
        # Note: This will likely fail unless you've set up a Custom Search Engine
        try:
            # You would need to replace this with your actual search engine ID
            search_engine_id = f"{project_id}:searchapi"
            logger.info(f"Using search engine ID: {search_engine_id}")
            
            results = search_client.cse().list(
                q="NEXUS AI test",
                cx=search_engine_id
            ).execute()
            
            logger.info("Custom Search API test success!")
            if 'items' in results:
                logger.info(f"Found {len(results['items'])} search results")
            else:
                logger.info("Search returned no items, but API is working")
            
            return True
            
        except Exception as search_e:
            # API client initialized but search failed (likely due to missing search engine ID)
            logger.warning(f"Search failed but API client initialized: {search_e}")
            logger.info("Custom Search API available but needs configuration")
            return "partial"  # Partially working
        
    except Exception as e:
        logger.error(f"Custom Search API test failed: {str(e)}")
        return False

async def test_generative_ai():
    """Test if Generative AI API (Gemini) is working"""
    logger.info("Testing Generative AI (Gemini) API...")
    
    # Find service account file
    service_account_path = None
    for path in ["autonomus-1743898709312-8589efbc502d.json", 
                "d:/Ai/Nexus/autonomus-1743898709312-8589efbc502d.json"]:
        if os.path.exists(path):
            service_account_path = path
            break
    
    if not service_account_path:
        logger.error("No service account file found!")
        return False
    
    try:
        from google.oauth2 import service_account
        import google.generativeai as genai
        
        # Initialize credentials
        credentials = service_account.Credentials.from_service_account_file(
            service_account_path
        )
        
        # Get the service account info to extract project ID
        with open(service_account_path, 'r') as f:
            sa_info = json.load(f)
            project_id = sa_info.get('project_id')
            logger.info(f"Project ID from service account: {project_id}")
        
        # Configure Gemini with credentials
        genai.configure(credentials=credentials)
        
        # Test generating text
        try:
            model = genai.GenerativeModel('gemini-pro')
            
            # Generate a simple response
            response = model.generate_content("What is NEXUS AI in 2 sentences?")
            
            if response.text:
                logger.info("Generative AI (Gemini) test success!")
                logger.info(f"Response: {response.text}")
                return True
            else:
                logger.warning("Generative AI responded but with empty text")
                return "partial"
                
        except Exception as gen_e:
            # Client initialized but generation failed
            logger.warning(f"Generation failed but API initialized: {gen_e}")
            logger.info("Generative AI available but needs configuration")
            return "partial"  # Partially working
        
    except Exception as e:
        logger.error(f"Generative AI test failed: {str(e)}")
        return False

async def test_gemini_vision():
    """Test if Gemini Vision (multimodal) is working"""
    logger.info("Testing Gemini Vision (multimodal) API...")
    
    # Find service account file
    service_account_path = None
    for path in ["autonomus-1743898709312-8589efbc502d.json", 
                "d:/Ai/Nexus/autonomus-1743898709312-8589efbc502d.json"]:
        if os.path.exists(path):
            service_account_path = path
            break
    
    if not service_account_path:
        logger.error("No service account file found!")
        return False
    
    try:
        from google.oauth2 import service_account
        import google.generativeai as genai
        from PIL import Image
        
        # Initialize credentials
        credentials = service_account.Credentials.from_service_account_file(
            service_account_path
        )
        
        # Configure Gemini with credentials
        genai.configure(credentials=credentials)
        
        # Test multimodal with a simple image
        try:
            # Generate a small test image
            img = Image.new('RGB', (100, 100), color=(73, 109, 137))
            
            # Use Gemini Pro Vision
            model = genai.GenerativeModel('gemini-pro-vision')
            
            # Analyze the image
            response = model.generate_content([
                "What do you see in this image?",
                img
            ])
            
            if response.text:
                logger.info("Gemini Vision test success!")
                logger.info(f"Response: {response.text}")
                return True
            else:
                logger.warning("Gemini Vision responded but with empty text")
                return "partial"
                
        except Exception as vis_e:
            # Client initialized but vision failed
            logger.warning(f"Vision analysis failed but API initialized: {vis_e}")
            logger.info("Gemini Vision available but needs configuration")
            return "partial"  # Partially working
        
    except Exception as e:
        logger.error(f"Gemini Vision test failed: {str(e)}")
        return False

async def main():
    """Run all API tests and show summary"""
    logger.info("=== GOOGLE CLOUD API TESTS ===")
    
    # Run all tests
    vision_result = await test_vision_api()
    search_result = await test_search_api()
    genai_result = await test_generative_ai()
    vision_ai_result = await test_gemini_vision()
    
    # Print summary table
    logger.info("\n=== API TEST RESULTS ===")
    logger.info("API                | Status")
    logger.info("------------------|---------")
    logger.info(f"Cloud Vision API   | {'✅ Working' if vision_result else '❌ Not Working'}")
    
    if search_result == "partial":
        logger.info("Custom Search API  | ⚠️ Partially Working (needs configuration)")
    else:
        logger.info(f"Custom Search API  | {'✅ Working' if search_result else '❌ Not Working'}")
        
    if genai_result == "partial":
        logger.info("Generative AI     | ⚠️ Partially Working (needs configuration)")
    else:
        logger.info(f"Generative AI     | {'✅ Working' if genai_result else '❌ Not Working'}")
        
    if vision_ai_result == "partial":
        logger.info("Gemini Vision     | ⚠️ Partially Working (needs configuration)")
    else:
        logger.info(f"Gemini Vision     | {'✅ Working' if vision_ai_result else '❌ Not Working'}")
    
    logger.info("\n=== SUMMARY ===")
    working_count = sum(1 for x in [vision_result, search_result, genai_result, vision_ai_result] 
                        if x is True)
    partial_count = sum(1 for x in [vision_result, search_result, genai_result, vision_ai_result] 
                        if x == "partial")
    logger.info(f"Working APIs: {working_count}/4")
    logger.info(f"Partially Working APIs: {partial_count}/4")
    logger.info(f"Non-Working APIs: {4 - working_count - partial_count}/4")

if __name__ == "__main__":
    asyncio.run(main())
