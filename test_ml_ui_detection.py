"""Test ML UI Detection Components

This script tests the ML-based UI detection components and outputs the results to a text file
for analysis of any errors or issues that need to be addressed.
"""

import os
import sys
import time
import logging
import traceback
import json
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Add the project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

# Import the ML UI detection components
from src.ai_core.screen_analysis.ml_ui_detection.enhanced_detector_registry import EnhancedDetectorRegistry
from src.ai_core.screen_analysis.ml_ui_detection.adaptive_detector import AdaptiveDetector
from src.ai_core.screen_analysis.ml_ui_detection.huggingface_detector import HuggingFaceDetector
from src.ai_core.screen_analysis.ml_ui_detection.autogluon_detector_adapter import AutoGluonDetectorAdapter
from src.ai_core.screen_analysis.ml_ui_detection.multimodal_detection_coordinator import (
    MultimodalDetectionCoordinator, DetectionRequest, DetectionResult
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set up output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_results")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Path for the output file
TEST_OUTPUT = os.path.join(OUTPUT_DIR, f"ml_ui_detection_test_{time.strftime('%Y%m%d_%H%M%S')}.txt")

# Test image directory - modify this to point to your test images
TEST_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "test_data", "ui_screenshots")

# Create test directory if it doesn't exist
if not os.path.exists(TEST_IMAGES_DIR):
    os.makedirs(TEST_IMAGES_DIR)


def log_to_file(message: str, test_output_path: str = TEST_OUTPUT):
    """Log a message to both the console and the output file
    
    Args:
        message: The message to log
        test_output_path: Path to the output file
    """
    print(message)
    with open(test_output_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def log_separator(test_output_path: str = TEST_OUTPUT):
    """Log a separator line
    
    Args:
        test_output_path: Path to the output file
    """
    separator = "-" * 80
    log_to_file(separator, test_output_path)


def log_section_header(title: str, test_output_path: str = TEST_OUTPUT):
    """Log a section header
    
    Args:
        title: The section title
        test_output_path: Path to the output file
    """
    log_separator(test_output_path)
    log_to_file(f"\n{title.upper()}\n", test_output_path)
    log_separator(test_output_path)


def setup_test_environment():
    """Set up the test environment by initializing the detection system
    
    Returns:
        tuple: (coordinator, registry, test_images)
    """
    # Initialize registry and detectors
    registry = EnhancedDetectorRegistry()
    detectors_initialized = 0
    
    # Create and register adaptive detector
    try:
        adaptive_detector = AdaptiveDetector()
        if adaptive_detector.initialize():
            registry.register_detector("adaptive", adaptive_detector)
            detectors_initialized += 1
            logger.info("Registered adaptive detector")
        else:
            logger.warning("Failed to initialize adaptive detector")
    except Exception as e:
        logger.error(f"Error initializing adaptive detector: {e}")
    
    # Create and register HuggingFace detector
    try:
        huggingface_detector = HuggingFaceDetector()
        if huggingface_detector.is_available and huggingface_detector.initialize():
            registry.register_detector("huggingface", huggingface_detector)
            detectors_initialized += 1
            logger.info("Registered HuggingFace detector")
        else:
            logger.warning("HuggingFace detector not available or failed to initialize")
    except Exception as e:
        logger.error(f"Error with HuggingFace detector: {e}")
    
    # Create and register AutoGluon adapter
    try:
        autogluon_adapter = AutoGluonDetectorAdapter()
        if autogluon_adapter.initialize():
            registry.register_detector("autogluon", autogluon_adapter)
            detectors_initialized += 1
            logger.info("Registered AutoGluon detector adapter")
        else:
            logger.warning("Failed to initialize AutoGluon detector adapter")
    except Exception as e:
        logger.error(f"Error with AutoGluon detector adapter: {e}")
    
    # Create multimodal detection coordinator
    coordinator = MultimodalDetectionCoordinator(registry)
    
    # Try to initialize vision-language model for semantic understanding
    try:
        vlm_success = coordinator.initialize_vlm()
        if vlm_success:
            logger.info("Initialized vision-language model for semantic understanding")
        else:
            logger.warning("Could not initialize vision-language model")
    except Exception as e:
        logger.warning(f"Error initializing vision-language model: {e}")
    
    # Find test images
    test_images = find_test_images()
    
    return coordinator, registry, test_images


def find_test_images():
    """Find test images for UI detection testing
    
    Returns:
        list: List of test image paths
    """
    test_images = []
    
    # Check if the test images directory exists
    if not os.path.exists(TEST_IMAGES_DIR):
        logger.warning(f"Test images directory not found: {TEST_IMAGES_DIR}")
        return test_images
    
    # Look for image files
    for file in os.listdir(TEST_IMAGES_DIR):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
            test_images.append(os.path.join(TEST_IMAGES_DIR, file))
    
    # If no test images found, add some default system screenshots
    if not test_images:
        logger.warning("No test images found. Creating default screenshot for testing.")
        try:
            # Create test directory if needed
            os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
            
            # Try to capture a screenshot using platform-specific methods
            import platform
            system = platform.system()
            screenshot_saved = False
            
            if system == "Windows":
                # Windows screenshot
                try:
                    import pyautogui
                    screenshot = pyautogui.screenshot()
                    screenshot_path = os.path.join(TEST_IMAGES_DIR, "windows_screenshot.png")
                    screenshot.save(screenshot_path)
                    test_images.append(screenshot_path)
                    screenshot_saved = True
                except ImportError:
                    logger.warning("pyautogui not installed for taking screenshots")
            
            if not screenshot_saved:
                # If no screenshot was taken, create a dummy test image
                dummy_img = np.zeros((800, 1200, 3), dtype=np.uint8)
                # Draw some rectangles of different colors to simulate UI elements
                cv2.rectangle(dummy_img, (100, 100), (300, 150), (0, 0, 255), -1)  # Red button
                cv2.rectangle(dummy_img, (400, 100), (600, 150), (0, 255, 0), -1)  # Green button
                cv2.rectangle(dummy_img, (700, 100), (900, 150), (255, 0, 0), -1)  # Blue button
                cv2.rectangle(dummy_img, (100, 200), (900, 600), (200, 200, 200), 2)  # Gray area
                
                # Add some text
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(dummy_img, "Button 1", (150, 135), font, 1, (255, 255, 255), 2)
                cv2.putText(dummy_img, "Button 2", (450, 135), font, 1, (255, 255, 255), 2)
                cv2.putText(dummy_img, "Button 3", (750, 135), font, 1, (255, 255, 255), 2)
                cv2.putText(dummy_img, "Text Area", (400, 350), font, 2, (255, 255, 255), 3)
                
                dummy_path = os.path.join(TEST_IMAGES_DIR, "dummy_ui.png")
                cv2.imwrite(dummy_path, dummy_img)
                test_images.append(dummy_path)
                logger.info(f"Created dummy test image at {dummy_path}")
        except Exception as e:
            logger.error(f"Error creating test images: {e}")
    
    return test_images


def test_detector_registry(registry, test_output_path=TEST_OUTPUT):
    """Test the enhanced detector registry
    
    Args:
        registry: The detector registry to test
        test_output_path: Path to the output file
    """
    log_section_header("Testing Enhanced Detector Registry", test_output_path)
    
    try:
        # Get registered detectors
        detectors = registry.get_all_detectors()
        log_to_file(f"Number of registered detectors: {len(detectors)}", test_output_path)
        
        for name, detector in detectors.items():
            log_to_file(f"\nTesting detector: {name}", test_output_path)
            
            # Test capabilities
            capabilities = detector.get_capabilities()
            log_to_file(f"  Capabilities: {json.dumps(capabilities, indent=2)}", test_output_path)
            
            # Test performance metrics
            metrics = registry.get_detector_metrics(name)
            log_to_file(f"  Performance metrics: {json.dumps(metrics, indent=2)}", test_output_path)
        
        # Test detector selection
        best_detector = registry.select_best_detector()
        log_to_file(f"\nBest detector selected: {best_detector}", test_output_path)
        
        # Test adaptive selection
        context = {"element_type": "button"}
        specialized_detector = registry.select_detector_for_context(context)
        log_to_file(f"Detector selected for 'button' context: {specialized_detector}", test_output_path)
        
        log_to_file("\nEnhanced detector registry tests completed", test_output_path)
        return True
    except Exception as e:
        log_to_file(f"ERROR in detector registry test: {e}", test_output_path)
        log_to_file(traceback.format_exc(), test_output_path)
        return False


def test_individual_detectors(registry, test_image, test_output_path=TEST_OUTPUT):
    """Test individual detectors with a test image
    
    Args:
        registry: The detector registry containing detectors
        test_image: Path to a test image
        test_output_path: Path to the output file
    """
    log_section_header(f"Testing Individual Detectors with image: {os.path.basename(test_image)}", test_output_path)
    
    try:
        # Load test image
        image = cv2.imread(test_image)
        if image is None:
            log_to_file(f"ERROR: Could not load test image {test_image}", test_output_path)
            return False
        
        log_to_file(f"Image loaded with shape: {image.shape}", test_output_path)
        
        # Get all detectors
        detectors = registry.get_all_detectors()
        
        for name, detector in detectors.items():
            log_to_file(f"\nTesting detector: {name}", test_output_path)
            
            try:
                # Time the detection
                start_time = time.time()
                elements = detector.detect_elements(image)
                detection_time = time.time() - start_time
                
                # Log results
                log_to_file(f"  Detection time: {detection_time:.3f} seconds", test_output_path)
                log_to_file(f"  Elements detected: {len(elements)}", test_output_path)
                
                if elements:
                    # Log some details of the first few elements
                    log_to_file("  Sample elements:", test_output_path)
                    for i, element in enumerate(elements[:5]):  # Show up to 5 elements
                        element_type = element.get("type", "unknown")
                        confidence = element.get("confidence", 0.0)
                        bbox = element.get("bbox", (0, 0, 0, 0))
                        log_to_file(f"    Element {i+1}: Type={element_type}, Confidence={confidence:.2f}, BBox={bbox}", test_output_path)
                    
                    if len(elements) > 5:
                        log_to_file(f"    ... and {len(elements) - 5} more elements", test_output_path)
                
                # Save visualization
                output_dir = os.path.join(OUTPUT_DIR, "visualizations")
                os.makedirs(output_dir, exist_ok=True)
                visualization_path = os.path.join(output_dir, f"{name}_{os.path.basename(test_image)}")
                
                visualize_detection(image, elements, visualization_path)
                log_to_file(f"  Visualization saved to: {visualization_path}", test_output_path)
                
            except Exception as e:
                log_to_file(f"  ERROR testing detector {name}: {e}", test_output_path)
                log_to_file(traceback.format_exc(), test_output_path)
        
        log_to_file("\nIndividual detector tests completed", test_output_path)
        return True
    except Exception as e:
        log_to_file(f"ERROR in individual detector test: {e}", test_output_path)
        log_to_file(traceback.format_exc(), test_output_path)
        return False


def test_multimodal_coordinator(coordinator, test_image, test_output_path=TEST_OUTPUT):
    """Test the multimodal detection coordinator
    
    Args:
        coordinator: The multimodal detection coordinator
        test_image: Path to a test image
        test_output_path: Path to the output file
    """
    log_section_header(f"Testing Multimodal Detection Coordinator with image: {os.path.basename(test_image)}", test_output_path)
    
    try:
        # Load test image
        image = cv2.imread(test_image)
        if image is None:
            log_to_file(f"ERROR: Could not load test image {test_image}", test_output_path)
            return False
        
        # Create detection request
        log_to_file("Testing detection with default settings", test_output_path)
        request = DetectionRequest(image=image)
        
        # Time the detection
        start_time = time.time()
        result = coordinator.detect(request)
        detection_time = time.time() - start_time
        
        # Log results
        log_to_file(f"Detection time: {detection_time:.3f} seconds", test_output_path)
        log_to_file(f"Elements detected: {len(result.elements)}", test_output_path)
        log_to_file(f"Detection errors: {result.errors}", test_output_path)
        log_to_file(f"Performance: {json.dumps(result.performance, indent=2)}", test_output_path)
        
        # Test with different confidence thresholds
        confidence_thresholds = [0.3, 0.5, 0.7, 0.9]
        for threshold in confidence_thresholds:
            log_to_file(f"\nTesting with confidence threshold: {threshold}", test_output_path)
            request = DetectionRequest(image=image, min_confidence=threshold)
            result = coordinator.detect(request)
            log_to_file(f"Elements detected: {len(result.elements)}", test_output_path)
        
        # Test with semantic analysis enabled vs disabled
        log_to_file("\nTesting with semantic analysis disabled", test_output_path)
        request = DetectionRequest(image=image, semantic_analysis=False)
        result_no_semantic = coordinator.detect(request)
        
        log_to_file("\nTesting with semantic analysis enabled", test_output_path)
        request = DetectionRequest(image=image, semantic_analysis=True)
        result_with_semantic = coordinator.detect(request)
        
        log_to_file(f"Elements with semantic analysis disabled: {len(result_no_semantic.elements)}", test_output_path)
        log_to_file(f"Elements with semantic analysis enabled: {len(result_with_semantic.elements)}", test_output_path)
        
        # Check for semantic descriptions
        semantic_count = 0
        for element in result_with_semantic.elements:
            if "semantic_description" in element:
                semantic_count += 1
        
        log_to_file(f"Elements with semantic descriptions: {semantic_count}", test_output_path)
        
        # Save visualization
        output_dir = os.path.join(OUTPUT_DIR, "visualizations")
        os.makedirs(output_dir, exist_ok=True)
        visualization_path = os.path.join(output_dir, f"coordinator_{os.path.basename(test_image)}")
        
        visualize_detection(image, result_with_semantic.elements, visualization_path)
        log_to_file(f"Visualization saved to: {visualization_path}", test_output_path)
        
        # Get performance statistics
        stats = coordinator.get_performance_statistics()
        log_to_file(f"\nPerformance statistics: {json.dumps(stats, indent=2)}", test_output_path)
        
        log_to_file("\nMultimodal detection coordinator tests completed", test_output_path)
        return True
    except Exception as e:
        log_to_file(f"ERROR in multimodal coordinator test: {e}", test_output_path)
        log_to_file(traceback.format_exc(), test_output_path)
        return False


def visualize_detection(image, elements, output_path):
    """Visualize detected elements on the image and save to file
    
    Args:
        image: The original image
        elements: List of detected elements
        output_path: Path to save the visualization
    """
    # Create a copy for visualization
    viz_image = image.copy()
    
    # Color mapping for different element types
    color_map = {
        "button": (0, 255, 0),       # Green
        "checkbox": (255, 0, 0),    # Red
        "radio_button": (0, 0, 255), # Blue
        "dropdown": (255, 255, 0),  # Yellow
        "text_field": (255, 0, 255), # Magenta
        "icon": (0, 255, 255),      # Cyan
        "menu": (128, 128, 0),      # Olive
        "link": (128, 0, 128),      # Purple
        "image": (0, 128, 128),     # Teal
        "unknown": (128, 128, 128)  # Gray
    }
    
    # Draw each element
    font = cv2.FONT_HERSHEY_SIMPLEX
    for element in elements:
        # Get bounding box
        bbox = element.get("bbox")
        if not bbox:
            continue
        
        # Get element type and color
        element_type = element.get("type", "unknown")
        color_bgr = color_map.get(element_type, color_map["unknown"])  # OpenCV uses BGR
        
        # Get confidence
        confidence = element.get("confidence", 0.0)
        
        # Draw rectangle
        x1, y1, x2, y2 = bbox
        cv2.rectangle(viz_image, (x1, y1), (x2, y2), color_bgr, 2)
        
        # Add label with confidence
        label = f"{element_type} ({confidence:.2f})"
        cv2.putText(viz_image, label, (x1, y1-5), font, 0.5, color_bgr, 1)
    
    # Save visualization
    cv2.imwrite(output_path, viz_image)


def analyze_test_results(test_output_path=TEST_OUTPUT):
    """Analyze test results and generate a summary
    
    Args:
        test_output_path: Path to the test output file
    """
    if not os.path.exists(test_output_path):
        logger.error(f"Test output file not found: {test_output_path}")
        return
    
    log_section_header("Test Results Analysis", test_output_path)
    
    try:
        # Read the test output file
        with open(test_output_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Count errors
        error_count = content.count("ERROR")
        
        # Count warnings
        warning_count = content.count("WARNING")
        
        # Count successful tests
        success_count = content.count("tests completed")
        
        # Count detectors
        detector_count = content.count("Testing detector:")
        
        # Count detected elements (approximate)
        import re
        element_counts = re.findall(r"Elements detected: (\d+)", content)
        total_elements = sum(int(count) for count in element_counts) if element_counts else 0
        
        # Generate summary
        log_to_file("\nTEST SUMMARY:", test_output_path)
        log_to_file(f"Total errors: {error_count}", test_output_path)
        log_to_file(f"Total warnings: {warning_count}", test_output_path)
        log_to_file(f"Successful test sections: {success_count}", test_output_path)
        log_to_file(f"Detectors tested: {detector_count}", test_output_path)
        log_to_file(f"Total UI elements detected: {total_elements}", test_output_path)
        
        # Check for specific issues
        if "Could not initialize vision-language model" in content:
            log_to_file("ISSUE: Vision-language model initialization failed", test_output_path)
        
        if "AutoGluon detector not available" in content or "Failed to initialize AutoGluon detector" in content:
            log_to_file("ISSUE: AutoGluon detector not available or failed to initialize", test_output_path)
        
        if "HuggingFace detector not available" in content or "Failed to initialize HuggingFace detector" in content:
            log_to_file("ISSUE: HuggingFace detector not available or failed to initialize", test_output_path)
        
        # Overall status
        if error_count > 0:
            log_to_file("\nOVERALL STATUS: FAILED - Critical errors detected", test_output_path)
        elif warning_count > 0:
            log_to_file("\nOVERALL STATUS: PARTIAL SUCCESS - Warnings detected", test_output_path)
        else:
            log_to_file("\nOVERALL STATUS: SUCCESS - All tests passed", test_output_path)
        
    except Exception as e:
        log_to_file(f"ERROR analyzing test results: {e}", test_output_path)
        log_to_file(traceback.format_exc(), test_output_path)


def run_all_tests():
    """Run all tests and generate a report"""
    # Initialize the output file
    with open(TEST_OUTPUT, "w", encoding="utf-8") as f:
        f.write(f"ML UI Detection Test Results - {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Set up test environment
    log_section_header("Setting up test environment")
    coordinator, registry, test_images = setup_test_environment()
    
    if not test_images:
        log_to_file("ERROR: No test images available. Unable to proceed with tests.")
        return
    
    # Log test configuration
    log_to_file(f"Test output: {TEST_OUTPUT}")
    log_to_file(f"Number of test images: {len(test_images)}")
    log_to_file("Test images:")
    for img in test_images:
        log_to_file(f"  - {img}")
    
    # Test enhanced detector registry
    test_detector_registry(registry)
    
    # Test each detector with each test image
    for test_image in test_images:
        test_individual_detectors(registry, test_image)
    
    # Test multimodal coordinator with each test image
    for test_image in test_images:
        test_multimodal_coordinator(coordinator, test_image)
    
    # Analyze test results
    analyze_test_results()
    
    log_section_header("Test Complete")
    log_to_file(f"Test results saved to: {TEST_OUTPUT}")
    log_to_file(f"Visualizations saved to: {os.path.join(OUTPUT_DIR, 'visualizations')}")


if __name__ == "__main__":
    run_all_tests()
