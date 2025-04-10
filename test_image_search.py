"""
Test script for Google Custom Search API image search functionality
"""
import os
import sys
import asyncio
import logging
import json
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_image_search():
    """Test if Custom Search API can retrieve images"""
    logger.info("Testing Custom Search API for image search...")
    
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
        # Initialize credentials
        credentials = service_account.Credentials.from_service_account_file(
            service_account_path,
            scopes=['https://www.googleapis.com/auth/cse']
        )
        
        # Get project ID from service account
        with open(service_account_path, 'r') as f:
            sa_info = json.load(f)
            project_id = sa_info.get('project_id')
        
        # Try API key method as an alternative
        try:
            # Initialize search client with credentials
            search_client = build(
                'customsearch', 'v1',
                credentials=credentials,
                cache_discovery=False
            )
            
            # Test with a general search engine if we don't have a specific ID
            search_engine_id = "017576662512468239146:omuauf_lfve"  # This is a public example CSE ID
            
            # Try image search
            logger.info("Attempting image search...")
            results = search_client.cse().list(
                q="artificial intelligence robot",
                cx=search_engine_id,
                searchType="image",
                num=3
            ).execute()
            
            if 'items' in results:
                logger.info(f"SUCCESS! Found {len(results['items'])} image results")
                for i, item in enumerate(results['items']):
                    if 'link' in item:
                        logger.info(f"Image {i+1}: {item['link']}")
                return True
            else:
                logger.info("Search executed but no image results returned")
                logger.info(f"Response: {results}")
                return "partial"
                
        except Exception as e:
            logger.error(f"Image search failed: {e}")
            
            # Try again with direct API key approach (from project ID)
            logger.info("Trying alternative method with API key...")
            
            # We'll use a heuristic to create a likely API key from the project ID
            # This won't work in most cases but we'll try
            api_key = f"{project_id}-key"  # Placeholder - not a real key
            
            try:
                # Try with API key
                search_client = build(
                    'customsearch', 'v1',
                    developerKey=api_key,
                    cache_discovery=False
                )
                
                results = search_client.cse().list(
                    q="artificial intelligence robot",
                    cx=search_engine_id,
                    searchType="image",
                    num=3
                ).execute()
                
                if 'items' in results:
                    logger.info(f"SUCCESS with API key! Found {len(results['items'])} image results")
                    return True
                else:
                    logger.info("API key approach executed but no results")
                    return "partial"
                    
            except Exception as api_e:
                logger.error(f"API key approach also failed: {api_e}")
                return False
            
    except Exception as e:
        logger.error(f"Error setting up Custom Search: {e}")
        return False

async def main():
    """Run the image search test"""
    logger.info("=== GOOGLE CUSTOM SEARCH IMAGE TEST ===")
    
    # Run test
    result = await test_image_search()
    
    # Show result
    logger.info("\n=== IMAGE SEARCH TEST RESULT ===")
    
    if result is True:
        logger.info("Image Search: ✅ WORKING")
        logger.info("You can use Google Custom Search API to retrieve images!")
    elif result == "partial":
        logger.info("Image Search: ⚠️ PARTIALLY WORKING")
        logger.info("API connected but search engine needs configuration")
    else:
        logger.info("Image Search: ❌ NOT WORKING")
        logger.info("To make it work, you need to:")
        logger.info("1. Create a Custom Search Engine in Google Cloud Console")
        logger.info("2. Configure it to search the entire web")
        logger.info("3. Enable image search")
        logger.info("4. Get the Search Engine ID (cx)")

if __name__ == "__main__":
    asyncio.run(main())
