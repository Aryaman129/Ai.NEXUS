"""
Test script for NEXUS ML-based UI element detection system.

This script demonstrates the adaptive UI detection capabilities of NEXUS,
showcasing different detector types and their results.
"""

import cv2
import numpy as np
import os
import logging
import time
from typing import Dict, List, Optional

import sys
import os

# Add src directory to path if not already there
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import from our package
from ai_core.screen_analysis import VisualMemorySystem
from ai_core.screen_analysis.visualization import DetectionVisualizer

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_test_image(image_path: str) -> np.ndarray:
    """Load a test image from the given path"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Test image not found: {image_path}")
        
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
        
    return image

def create_test_images_directory():
    """Create directory for test images if it doesn't exist"""
    os.makedirs("test_images", exist_ok=True)
    
    # Create a simple UI test image if none exist
    test_image_path = os.path.join("test_images", "sample_ui.png")
    if not os.path.exists(test_image_path):
        # Create a blank image
        img = np.ones((600, 800, 3), dtype=np.uint8) * 255
        
        # Draw a button
        cv2.rectangle(img, (100, 100), (250, 150), (200, 200, 200), -1)
        cv2.rectangle(img, (100, 100), (250, 150), (100, 100, 100), 2)
        cv2.putText(img, "Button", (130, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Draw a text field
        cv2.rectangle(img, (100, 200), (400, 240), (240, 240, 240), -1)
        cv2.rectangle(img, (100, 200), (400, 240), (100, 100, 100), 2)
        cv2.putText(img, "Text Field", (130, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Draw a checkbox
        cv2.rectangle(img, (100, 300), (130, 330), (240, 240, 240), -1)
        cv2.rectangle(img, (100, 300), (130, 330), (100, 100, 100), 2)
        cv2.putText(img, "Checkbox", (140, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Save the image
        cv2.imwrite(test_image_path, img)
        logger.info(f"Created sample UI test image: {test_image_path}")
    
    return "test_images"

def test_ui_detection():
    """Test UI element detection with different detectors"""
    # Create test images directory
    test_dir = create_test_images_directory()
    test_image_path = os.path.join(test_dir, "sample_ui.png")
    
    # Load test image
    image = load_test_image(test_image_path)
    
    # Create output directory
    os.makedirs("detection_results", exist_ok=True)
    
    # Initialize the visual memory system with different configurations
    
    # 1. Legacy detection only
    logger.info("Testing legacy detection...")
    legacy_config = {
        "use_ml_detection": False
    }
    legacy_system = VisualMemorySystem(legacy_config)
    
    # 2. ML detection with OpenCV fallback
    logger.info("Testing ML detection with OpenCV fallback...")
    ml_config = {
        "use_ml_detection": True,
        "detector_config": {
            "opencv": {
                "min_button_width": 30,
                "min_button_height": 20
            }
        }
    }
    ml_system = VisualMemorySystem(ml_config)
    
    # 3. ML detection with blended approach
    logger.info("Testing blended detection...")
    blended_config = {
        "use_ml_detection": True,
        "use_blended_detection": True
    }
    blended_system = VisualMemorySystem(blended_config)
    
    # Run detection with each system
    start_time = time.time()
    legacy_elements = legacy_system.detect_ui_elements(image)
    legacy_time = time.time() - start_time
    logger.info(f"Legacy detection found {len(legacy_elements)} elements in {legacy_time:.3f}s")
    
    start_time = time.time()
    ml_elements = ml_system.detect_ui_elements(image)
    ml_time = time.time() - start_time
    logger.info(f"ML detection found {len(ml_elements)} elements in {ml_time:.3f}s")
    
    start_time = time.time()
    blended_elements = blended_system.detect_ui_elements(image)
    blended_time = time.time() - start_time
    logger.info(f"Blended detection found {len(blended_elements)} elements in {blended_time:.3f}s")
    
    # Visualize results
    visualizer = DetectionVisualizer()
    
    # Legacy detection visualization
    legacy_vis = visualizer.visualize_detections(
        image, legacy_elements, 
        color_by="type", 
        show_confidence=True,
        show_detector=True
    )
    cv2.imwrite(os.path.join("detection_results", "legacy_detection.png"), legacy_vis)
    
    # ML detection visualization
    ml_vis = visualizer.visualize_detections(
        image, ml_elements, 
        color_by="type", 
        show_confidence=True,
        show_detector=True
    )
    cv2.imwrite(os.path.join("detection_results", "ml_detection.png"), ml_vis)
    
    # Blended detection visualization
    blended_vis = visualizer.visualize_detections(
        image, blended_elements, 
        color_by="type", 
        show_confidence=True,
        show_detector=True
    )
    cv2.imwrite(os.path.join("detection_results", "blended_detection.png"), blended_vis)
    
    # Create comparison visualization
    detections_by_detector = {
        "Legacy": legacy_elements,
        "ML": ml_elements,
        "Blended": blended_elements
    }
    comparison_vis = visualizer.create_comparison_visualization(image, detections_by_detector)
    cv2.imwrite(os.path.join("detection_results", "comparison.png"), comparison_vis)
    
    logger.info("Detection test complete. Results saved to detection_results directory.")
    
    return {
        "legacy": {
            "elements": legacy_elements,
            "time": legacy_time
        },
        "ml": {
            "elements": ml_elements,
            "time": ml_time
        },
        "blended": {
            "elements": blended_elements,
            "time": blended_time
        }
    }

def show_element_statistics(results):
    """Display statistics about detected elements"""
    for detector_name, result in results.items():
        elements = result["elements"]
        time_taken = result["time"]
        
        # Count element types
        element_types = {}
        for element in elements:
            element_type = element.get("type", "unknown")
            if element_type not in element_types:
                element_types[element_type] = 0
            element_types[element_type] += 1
            
        # Calculate average confidence
        avg_confidence = sum(e.get("confidence", 0) for e in elements) / len(elements) if elements else 0
        
        print(f"\n{detector_name.capitalize()} Detection Statistics:")
        print(f"  Total elements: {len(elements)}")
        print(f"  Average confidence: {avg_confidence:.2f}")
        print(f"  Time taken: {time_taken:.3f}s")
        print("  Element types:")
        for element_type, count in element_types.items():
            print(f"    {element_type}: {count}")

if __name__ == "__main__":
    print("Testing NEXUS ML-based UI Element Detection")
    print("==========================================")
    
    # Run the test
    results = test_ui_detection()
    
    # Show statistics
    show_element_statistics(results)
    
    print("\nTest complete! Detection visualizations saved to 'detection_results' directory.")
