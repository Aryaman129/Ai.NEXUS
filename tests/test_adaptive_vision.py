#!/usr/bin/env python
"""
Test script for NEXUS Adaptive Vision System

This script tests the adaptive vision system to ensure it can:
1. Dynamically select different vision capabilities based on what's available
2. Process images using the best available method
3. Learn from past image analysis to improve future performance
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from PIL import Image
import time

# Add the parent directory to the path so we can import NEXUS modules
sys.path.append(str(Path(__file__).parent.parent))

# Import NEXUS modules
from src.integrations.adaptive_vision import AdaptiveVision
from src.integrations.nexus_visual_intelligence import NexusVisualIntelligence
from src.modules.vision_ui_automation import VisionUIAutomation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create directory for test data if it doesn't exist
test_data_dir = Path("test_data")
test_data_dir.mkdir(exist_ok=True)
test_images_dir = test_data_dir / "images"
test_images_dir.mkdir(exist_ok=True)


def create_test_image():
    """Create a simple test image with text and shapes"""
    # Only create if it doesn't exist
    test_image_path = test_images_dir / "test_image.png"
    if test_image_path.exists():
        return str(test_image_path)
    
    # Create a white image
    width, height = 800, 600
    image = Image.new('RGB', (width, height), color='white')
    
    try:
        # Try to draw on the image if PIL.ImageDraw is available
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(image)
        
        # Draw some shapes
        draw.rectangle([(50, 50), (200, 200)], outline='black', width=2)
        draw.ellipse([(300, 100), (500, 300)], outline='blue', width=3)
        
        # Draw some text
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except IOError:
            font = ImageFont.load_default()
        
        draw.text((100, 400), "NEXUS Vision Test", fill='black', font=font)
        draw.text((100, 450), "This is a test image", fill='black', font=font)
        
    except Exception as e:
        logger.warning(f"Couldn't add drawing to test image: {e}")
    
    # Save the image
    image.save(str(test_image_path))
    logger.info(f"Created test image at {test_image_path}")
    
    return str(test_image_path)


async def test_adaptive_vision():
    """Test the AdaptiveVision component"""
    
    # Initialize AdaptiveVision
    logger.info("Initializing AdaptiveVision...")
    vision = AdaptiveVision()
    
    # Get available capabilities
    capabilities = vision.get_capabilities() if hasattr(vision, 'get_capabilities') else {"error": "get_capabilities not available"}
    logger.info(f"Available capabilities: {capabilities}")
    
    # Create test image
    test_image_path = create_test_image()
    
    # Test image analysis
    logger.info("Testing image analysis...")
    analysis_result = await vision.analyze_image(image_path=test_image_path)
    
    # Log the results
    if analysis_result.get("success", False):
        logger.info("✓ Image analysis successful")
        logger.info(f"Method used: {analysis_result.get('method', 'unknown')}")
        logger.info(f"Analysis: {analysis_result.get('analysis', '')[:100]}...")
    else:
        logger.warning(f"✗ Image analysis failed: {analysis_result.get('error', 'Unknown error')}")
    
    # Test text extraction
    logger.info("Testing text extraction...")
    text_result = await vision.extract_text(image_path=test_image_path)
    
    if text_result.get("success", False):
        logger.info("✓ Text extraction successful")
        logger.info(f"Method used: {text_result.get('method', 'unknown')}")
        logger.info(f"Extracted text: {text_result.get('text', '')}")
    else:
        logger.warning(f"✗ Text extraction failed: {text_result.get('error', 'Unknown error')}")
    
    return {
        "capabilities": capabilities,
        "analysis_result": analysis_result,
        "text_result": text_result
    }


async def test_nexus_visual_intelligence():
    """Test the NexusVisualIntelligence component"""
    
    # Initialize NexusVisualIntelligence
    logger.info("Initializing NexusVisualIntelligence...")
    visual_intel = NexusVisualIntelligence()
    
    # Get capabilities
    if hasattr(visual_intel, 'get_capabilities'):
        capabilities = await visual_intel.get_capabilities()
        logger.info(f"Visual Intelligence capabilities: {capabilities}")
    else:
        capabilities = {"error": "get_capabilities not available"}
        logger.info("Visual Intelligence does not have get_capabilities method")
    
    # Create test image
    test_image_path = create_test_image()
    
    # Test UI detection
    logger.info("Testing UI element detection...")
    if hasattr(visual_intel, 'detect_ui_elements'):
        ui_result = await visual_intel.detect_ui_elements(image_path=test_image_path)
    else:
        ui_result = {"error": "detect_ui_elements not available"}
        logger.info("Visual Intelligence does not have detect_ui_elements method")
    
    if ui_result.get("success", False):
        logger.info("✓ UI element detection successful")
        logger.info(f"Method used: {ui_result.get('method', 'unknown')}")
        logger.info(f"Elements detected: {len(ui_result.get('elements', []))}")
        for i, element in enumerate(ui_result.get("elements", [])[:3]):
            logger.info(f"  Element {i+1}: {element.get('type', 'unknown')} at {element.get('coordinates', 'unknown')}")
    else:
        logger.warning(f"✗ UI element detection failed: {ui_result.get('error', 'Unknown error')}")
    
    # Test image analysis
    logger.info("Testing comprehensive image analysis...")
    if hasattr(visual_intel, 'analyze_image'):
        analysis_result = await visual_intel.analyze_image(image_path=test_image_path)
    else:
        analysis_result = {"error": "analyze_image not available"}
        logger.info("Visual Intelligence does not have analyze_image method")
    
    if analysis_result.get("success", False):
        logger.info("✓ Comprehensive image analysis successful")
        logger.info(f"Services used: {', '.join(analysis_result.get('services_used', ['unknown']))}")
    else:
        logger.warning(f"✗ Comprehensive image analysis failed: {analysis_result.get('error', 'Unknown error')}")
    
    return {
        "capabilities": capabilities,
        "ui_result": ui_result,
        "analysis_result": analysis_result
    }


async def test_learning_improvement():
    """Test if the vision system improves with repeated analysis"""
    
    # Initialize AdaptiveVision
    logger.info("Testing learning improvement in vision system...")
    vision = AdaptiveVision()
    
    # Create test image
    test_image_path = create_test_image()
    
    # Run multiple analyses to see if performance improves
    timing_results = []
    
    for i in range(3):
        logger.info(f"Learning test - Run {i+1}")
        
        # Measure performance
        start_time = time.time()
        analysis_result = await vision.analyze_image(image_path=test_image_path)
        end_time = time.time()
        
        execution_time = end_time - start_time
        success = analysis_result.get("success", False)
        
        timing_results.append({
            "run": i+1,
            "execution_time": execution_time,
            "success": success,
            "method": analysis_result.get("method", "unknown")
        })
        
        logger.info(f"Run {i+1}: {execution_time:.2f}s using {analysis_result.get('method', 'unknown')}")
        
        # Small delay between runs
        await asyncio.sleep(1)
    
    # Check if time improved
    if len(timing_results) > 1:
        first_time = timing_results[0]["execution_time"]
        last_time = timing_results[-1]["execution_time"]
        
        if last_time < first_time:
            logger.info(f"✓ Vision system improved performance by {(first_time - last_time):.2f}s")
        else:
            logger.info(f"✗ No performance improvement observed")
    
    return timing_results


async def test_ui_automation():
    """Test basic UI automation capabilities"""
    
    logger.info("Testing UI automation system...")
    
    # Initialize VisionUIAutomation
    ui_automation = VisionUIAutomation()
    
    # Check if initialization method exists and call it if it does
    if hasattr(ui_automation, 'initialize'):
        await ui_automation.initialize()
    
    # Test capability to identify screen elements
    logger.info("Testing ability to take screenshot...")
    if hasattr(ui_automation, 'take_screenshot'):
        screenshot = await ui_automation.take_screenshot()
    else:
        screenshot = None
        logger.info("UI Automation does not have take_screenshot method")
    
    screenshot_success = screenshot is not None
    if screenshot_success:
        logger.info("✓ Successfully captured screenshot")
        
        # Save the screenshot for reference
        screenshot_path = test_images_dir / "test_screenshot.png"
        screenshot.save(str(screenshot_path))
        logger.info(f"Screenshot saved to {screenshot_path}")
    else:
        logger.warning("✗ Failed to capture screenshot")
    
    # Test finding UI elements (this will likely detect nothing useful in a test environment)
    if screenshot_success and hasattr(ui_automation, 'find_ui_element'):
        logger.info("Testing UI element detection...")
        elements = await ui_automation.find_ui_element("button", screenshot)
        
        if elements:
            logger.info(f"✓ Detected {len(elements)} button-like elements")
        else:
            logger.info("No button elements detected in screenshot (expected in test environment)")
    else:
        logger.info("UI Automation does not have find_ui_element method")
    
    # Get action history (should be empty at this point)
    if hasattr(ui_automation, 'get_action_history'):
        action_history = ui_automation.get_action_history()
        logger.info(f"Action history contains {len(action_history)} entries")
    else:
        logger.info("UI Automation does not have get_action_history method")
    
    return {
        "screenshot_success": screenshot_success,
        "action_history_size": len(action_history) if 'action_history' in locals() else 0
    }


async def main():
    """Main test function"""
    logger.info("Starting NEXUS Adaptive Vision System Tests")
    
    try:
        # Test 1: AdaptiveVision
        logger.info("=== Test 1: AdaptiveVision ===")
        vision_results = await test_adaptive_vision()
        
        # Test 2: NexusVisualIntelligence
        logger.info("=== Test 2: NexusVisualIntelligence ===")
        intel_results = await test_nexus_visual_intelligence()
        
        # Test 3: Learning Improvement
        logger.info("=== Test 3: Learning Improvement ===")
        learning_results = await test_learning_improvement()
        
        # Test 4: UI Automation
        logger.info("=== Test 4: UI Automation ===")
        ui_results = await test_ui_automation()
        
        # Print summary
        logger.info("\n=== Test Summary ===")
        
        # AdaptiveVision Summary
        logger.info("AdaptiveVision Capabilities:")
        for capability, available in vision_results["capabilities"].items():
            status = "✓" if available else "✗"
            logger.info(f"  {status} {capability}")
        
        # Vision Analysis Method
        analysis_method = vision_results["analysis_result"].get("method", "unknown")
        text_method = vision_results["text_result"].get("method", "unknown")
        logger.info(f"Adaptive Vision used:")
        logger.info(f"  - {analysis_method} for image analysis")
        logger.info(f"  - {text_method} for text extraction")
        
        # NexusVisualIntelligence Summary
        if "services_used" in intel_results["analysis_result"]:
            services = intel_results["analysis_result"]["services_used"]
            logger.info(f"NexusVisualIntelligence used services: {', '.join(services)}")
        
        # Learning Improvement
        if len(learning_results) > 1:
            first_time = learning_results[0]["execution_time"]
            last_time = learning_results[-1]["execution_time"]
            difference = first_time - last_time
            
            if difference > 0:
                logger.info(f"✓ Learning improved performance by {difference:.2f}s ({(difference/first_time)*100:.1f}%)")
            else:
                logger.info(f"✗ No measurable learning improvement")
        
        # UI Automation
        if ui_results["screenshot_success"]:
            logger.info("✓ UI Automation screenshot capability working")
        else:
            logger.info("✗ UI Automation screenshot capability failed")
            
    except Exception as e:
        logger.error(f"Error during testing: {e}")
    
    logger.info("NEXUS Adaptive Vision System Tests Completed")


if __name__ == "__main__":
    asyncio.run(main())
