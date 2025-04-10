"""
Simple ML-based UI Detector Test

This script provides a simple test of our ML-based UI detection components,
outputting results to a text file for error analysis.
"""

import os
import sys
import time
import logging
import traceback
from datetime import datetime
import numpy as np
import cv2

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Output file
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"ui_detector_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")


def log_to_file(message):
    """Log message to both console and file"""
    print(message)
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(f"{message}\n")


def log_separator():
    """Log a separator line"""
    separator = "-" * 80
    log_to_file(separator)


def log_section(title):
    """Log a section header"""
    log_separator()
    log_to_file(f"\n{title.upper()}\n")
    log_separator()


def create_test_image():
    """Create a test image with UI elements
    
    Returns:
        tuple: (image, path)
    """
    # Create a test image directory
    test_dir = os.path.join(OUTPUT_DIR, "test_images")
    os.makedirs(test_dir, exist_ok=True)
    
    # Create a test image with simulated UI elements
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    img.fill(240)  # Light gray background
    
    # Add a blue header
    cv2.rectangle(img, (0, 0), (800, 60), (200, 150, 100), -1)
    
    # Add a green button
    cv2.rectangle(img, (50, 100), (200, 150), (100, 200, 100), -1)
    cv2.putText(img, "Button", (90, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add a red button
    cv2.rectangle(img, (250, 100), (400, 150), (100, 100, 200), -1)
    cv2.putText(img, "Cancel", (290, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add a text field
    cv2.rectangle(img, (50, 200), (400, 250), (255, 255, 255), -1)
    cv2.rectangle(img, (50, 200), (400, 250), (100, 100, 100), 1)
    
    # Add a checkbox
    cv2.rectangle(img, (50, 300), (70, 320), (255, 255, 255), -1)
    cv2.rectangle(img, (50, 300), (70, 320), (100, 100, 100), 1)
    cv2.putText(img, "Enable feature", (80, 318), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Add a dropdown
    cv2.rectangle(img, (50, 350), (200, 380), (255, 255, 255), -1)
    cv2.rectangle(img, (50, 350), (200, 380), (100, 100, 100), 1)
    cv2.rectangle(img, (180, 350), (200, 380), (200, 200, 200), -1)
    cv2.putText(img, "Select", (80, 372), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Save the image
    img_path = os.path.join(test_dir, "test_ui.png")
    cv2.imwrite(img_path, img)
    
    return img, img_path


def test_adaptive_detector(test_image, image_path):
    """Test the AdaptiveDetector
    
    Args:
        test_image: The test image
        image_path: Path to the test image
    """
    log_section("Testing AdaptiveDetector")
    
    try:
        # Import the adaptive detector
        sys.path.append(os.path.dirname(__file__))
        from src.ai_core.screen_analysis.ml_ui_detection.adaptive_detector import AdaptiveDetector
        
        log_to_file("Successfully imported AdaptiveDetector")
        
        # Initialize detector
        detector = AdaptiveDetector()
        init_success = detector.initialize()
        log_to_file(f"Initialization success: {init_success}")
        
        # Get capabilities
        capabilities = detector.get_capabilities()
        log_to_file(f"Detector capabilities: {capabilities}")
        
        # Detect elements
        start_time = time.time()
        elements = detector.detect_elements(test_image)
        detection_time = time.time() - start_time
        
        log_to_file(f"Detection time: {detection_time:.3f} seconds")
        log_to_file(f"Detected {len(elements)} elements")
        
        # Show detected elements
        log_to_file("\nDetected elements:")
        for i, element in enumerate(elements):
            log_to_file(f"Element {i+1}:")
            log_to_file(f"  Type: {element.get('type', 'unknown')}")
            log_to_file(f"  Confidence: {element.get('confidence', 0.0):.2f}")
            log_to_file(f"  Bounding box: {element.get('bbox', 'N/A')}")
            log_to_file(f"  Detection method: {element.get('detection_method', 'N/A')}")
        
        # Save visualization
        viz_dir = os.path.join(OUTPUT_DIR, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        viz_path = os.path.join(viz_dir, "adaptive_detector_results.png")
        
        # Create a copy for visualization
        viz_image = test_image.copy()
        
        # Draw each element
        for element in elements:
            bbox = element.get("bbox")
            if not bbox:
                continue
            
            element_type = element.get("type", "unknown")
            confidence = element.get("confidence", 0.0)
            
            # Choose color based on element type
            if element_type == "button":
                color = (0, 255, 0)  # Green
            elif element_type == "checkbox":
                color = (255, 0, 0)  # Red
            elif element_type == "text_field":
                color = (0, 0, 255)  # Blue
            elif element_type == "dropdown":
                color = (255, 255, 0)  # Yellow
            else:
                color = (128, 128, 128)  # Gray
            
            # Draw rectangle
            x1, y1, x2, y2 = bbox
            cv2.rectangle(viz_image, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"{element_type} ({confidence:.2f})"
            cv2.putText(viz_image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Save visualization
        cv2.imwrite(viz_path, viz_image)
        log_to_file(f"Visualization saved to: {viz_path}")
        
        return True
    except Exception as e:
        log_to_file(f"ERROR in adaptive detector test: {e}")
        log_to_file(traceback.format_exc())
        return False


def test_huggingface_detector(test_image, image_path):
    """Test the HuggingFaceDetector
    
    Args:
        test_image: The test image
        image_path: Path to the test image
    """
    log_section("Testing HuggingFaceDetector")
    
    try:
        # Import the HuggingFace detector
        sys.path.append(os.path.dirname(__file__))
        from src.ai_core.screen_analysis.ml_ui_detection.huggingface_detector import HuggingFaceDetector
        
        log_to_file("Successfully imported HuggingFaceDetector")
        
        # Initialize detector
        detector = HuggingFaceDetector()
        log_to_file(f"HuggingFace available: {detector.is_available}")
        
        if not detector.is_available:
            log_to_file("Skipping HuggingFace detector test as it's not available")
            return False
        
        init_success = detector.initialize()
        log_to_file(f"Initialization success: {init_success}")
        
        if not init_success:
            log_to_file("Failed to initialize HuggingFace detector, skipping tests")
            return False
        
        # Get capabilities
        capabilities = detector.get_capabilities()
        log_to_file(f"Detector capabilities: {capabilities}")
        
        # Detect elements
        start_time = time.time()
        elements = detector.detect_elements(test_image)
        detection_time = time.time() - start_time
        
        log_to_file(f"Detection time: {detection_time:.3f} seconds")
        log_to_file(f"Detected {len(elements)} elements")
        
        # Show detected elements
        log_to_file("\nDetected elements:")
        for i, element in enumerate(elements):
            log_to_file(f"Element {i+1}:")
            log_to_file(f"  Type: {element.get('type', 'unknown')}")
            log_to_file(f"  Confidence: {element.get('confidence', 0.0):.2f}")
            log_to_file(f"  Bounding box: {element.get('bbox', 'N/A')}")
            log_to_file(f"  Original label: {element.get('original_label', 'N/A')}")
        
        # Save visualization
        viz_dir = os.path.join(OUTPUT_DIR, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        viz_path = os.path.join(viz_dir, "huggingface_detector_results.png")
        
        # Create a copy for visualization
        viz_image = test_image.copy()
        
        # Draw each element
        for element in elements:
            bbox = element.get("bbox")
            if not bbox:
                continue
            
            element_type = element.get("type", "unknown")
            confidence = element.get("confidence", 0.0)
            
            # Choose color based on element type
            if element_type == "button":
                color = (0, 255, 0)  # Green
            elif element_type == "checkbox":
                color = (255, 0, 0)  # Red
            elif element_type == "text_field":
                color = (0, 0, 255)  # Blue
            elif element_type == "dropdown":
                color = (255, 255, 0)  # Yellow
            else:
                color = (128, 128, 128)  # Gray
            
            # Draw rectangle
            x1, y1, x2, y2 = bbox
            cv2.rectangle(viz_image, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"{element_type} ({confidence:.2f})"
            cv2.putText(viz_image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Save visualization
        cv2.imwrite(viz_path, viz_image)
        log_to_file(f"Visualization saved to: {viz_path}")
        
        return True
    except Exception as e:
        log_to_file(f"ERROR in HuggingFace detector test: {e}")
        log_to_file(traceback.format_exc())
        return False


def test_detector_registry():
    """Test the EnhancedDetectorRegistry"""
    log_section("Testing EnhancedDetectorRegistry")
    
    try:
        # Import the detector registry
        sys.path.append(os.path.dirname(__file__))
        from src.ai_core.screen_analysis.ml_ui_detection.enhanced_detector_registry import EnhancedDetectorRegistry
        
        log_to_file("Successfully imported EnhancedDetectorRegistry")
        
        # Initialize registry
        registry = EnhancedDetectorRegistry()
        log_to_file("Registry initialized")
        
        # Test registry methods
        detectors = registry.get_all_detectors()
        log_to_file(f"Number of registered detectors: {len(detectors)}")
        
        # Register a dummy detector
        from src.ai_core.screen_analysis.ui_detection.detector_interface import UIDetectorInterface
        
        class DummyDetector(UIDetectorInterface):
            def __init__(self, config=None):
                self.config = config or {}
                self.initialized = True
            
            def initialize(self, config=None):
                return True
            
            def get_capabilities(self):
                return {"name": "dummy", "version": "1.0", "element_types": ["button", "text_field"]}
            
            def detect_elements(self, screenshot, context=None):
                return [{"type": "button", "confidence": 0.9, "bbox": (10, 10, 100, 40)}]
            
            def supports_incremental_learning(self):
                """Whether this detector supports incremental learning"""
                return False
        
        dummy_detector = DummyDetector()
        registry.register_detector("dummy", dummy_detector)
        
        # Check registration
        detectors = registry.get_all_detectors()
        log_to_file(f"Number of registered detectors after registration: {len(detectors)}")
        
        # Test detector selection
        best_detector = registry.select_best_detector()
        log_to_file(f"Best detector: {best_detector}")
        
        # Test context-based selection
        context = {"element_type": "button"}
        context_detector = registry.select_detector_for_context(context)
        log_to_file(f"Detector selected for button context: {context_detector}")
        
        # Test performance updates
        registry.update_detector_performance("dummy", {
            "success_rate": 0.95,
            "avg_confidence": 0.9,
            "avg_detection_time": 0.1
        })
        
        metrics = registry.get_detector_metrics("dummy")
        log_to_file(f"Updated metrics for dummy detector: {metrics}")
        
        return True
    except Exception as e:
        log_to_file(f"ERROR in detector registry test: {e}")
        log_to_file(traceback.format_exc())
        return False


def test_multimodal_coordinator(test_image, image_path):
    """Test the MultimodalDetectionCoordinator
    
    Args:
        test_image: The test image
        image_path: Path to the test image
    """
    log_section("Testing MultimodalDetectionCoordinator")
    
    try:
        # Import the coordinator
        sys.path.append(os.path.dirname(__file__))
        from src.ai_core.screen_analysis.ml_ui_detection.multimodal_detection_coordinator import (
            MultimodalDetectionCoordinator, DetectionRequest, DetectionResult
        )
        from src.ai_core.screen_analysis.ml_ui_detection.enhanced_detector_registry import EnhancedDetectorRegistry
        
        log_to_file("Successfully imported MultimodalDetectionCoordinator")
        
        # Create registry and coordinator
        registry = EnhancedDetectorRegistry()
        coordinator = MultimodalDetectionCoordinator(registry)
        log_to_file("Coordinator initialized")
        
        # Initialize VLM if possible
        vlm_success = coordinator.initialize_vlm()
        log_to_file(f"VLM initialization success: {vlm_success}")
        
        # Test with empty registry
        request = DetectionRequest(image=test_image)
        result = coordinator.detect(request)
        
        log_to_file(f"Detection with empty registry - Elements: {len(result.elements)}")
        log_to_file(f"Errors: {result.errors}")
        
        # Try to add a detector
        try:
            from src.ai_core.screen_analysis.ml_ui_detection.adaptive_detector import AdaptiveDetector
            detector = AdaptiveDetector()
            if detector.initialize():
                registry.register_detector("adaptive", detector)
                log_to_file("Registered adaptive detector")
                
                # Test with detector
                request = DetectionRequest(image=test_image)
                result = coordinator.detect(request)
                
                log_to_file(f"Detection with adaptive detector - Elements: {len(result.elements)}")
                
                # Get performance statistics
                stats = coordinator.get_performance_statistics()
                log_to_file(f"Performance statistics: {stats}")
        except Exception as e:
            log_to_file(f"Error registering adaptive detector: {e}")
        
        # Test with different confidence threshold
        try:
            request = DetectionRequest(image=test_image, min_confidence=0.5)
            result = coordinator.detect(request)
            log_to_file(f"Detection with confidence threshold 0.5 - Elements: {len(result.elements)}")
        except Exception as e:
            log_to_file(f"Error testing with confidence threshold: {e}")
        
        # Test with semantic analysis toggle
        try:
            request = DetectionRequest(image=test_image, semantic_analysis=True)
            result = coordinator.detect(request)
            semantic_count = sum(1 for e in result.elements if "semantic_description" in e)
            log_to_file(f"Detection with semantic analysis - Elements: {len(result.elements)}")
            log_to_file(f"Elements with semantic descriptions: {semantic_count}")
        except Exception as e:
            log_to_file(f"Error testing with semantic analysis: {e}")
        
        # Save visualization
        if result.elements:
            viz_dir = os.path.join(OUTPUT_DIR, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            viz_path = os.path.join(viz_dir, "coordinator_results.png")
            
            # Create a copy for visualization
            viz_image = test_image.copy()
            
            # Draw each element
            for element in result.elements:
                bbox = element.get("bbox")
                if not bbox:
                    continue
                
                element_type = element.get("type", "unknown")
                confidence = element.get("confidence", 0.0)
                
                # Choose color based on element type
                if element_type == "button":
                    color = (0, 255, 0)  # Green
                elif element_type == "checkbox":
                    color = (255, 0, 0)  # Red
                elif element_type == "text_field":
                    color = (0, 0, 255)  # Blue
                elif element_type == "dropdown":
                    color = (255, 255, 0)  # Yellow
                else:
                    color = (128, 128, 128)  # Gray
                
                # Draw rectangle
                x1, y1, x2, y2 = bbox
                cv2.rectangle(viz_image, (x1, y1), (x2, y2), color, 2)
                
                # Add label
                label = f"{element_type} ({confidence:.2f})"
                cv2.putText(viz_image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Save visualization
            cv2.imwrite(viz_path, viz_image)
            log_to_file(f"Visualization saved to: {viz_path}")
        
        return True
    except Exception as e:
        log_to_file(f"ERROR in multimodal coordinator test: {e}")
        log_to_file(traceback.format_exc())
        return False


def analyze_test_results():
    """Analyze test results and provide summary"""
    log_section("Test Results Analysis")
    
    try:
        # Read the output file
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Count errors
        error_count = content.count("ERROR")
        
        # Count warnings
        warning_count = content.count("WARNING")
        
        # Count detectors tested
        detector_count = content.count("Testing")
        
        # Count elements detected
        import re
        element_counts = re.findall(r"Detected (\d+) elements", content)
        total_elements = sum(int(count) for count in element_counts) if element_counts else 0
        
        # Generate summary
        log_to_file("\nTEST SUMMARY:")
        log_to_file(f"Total errors: {error_count}")
        log_to_file(f"Total warnings: {warning_count}")
        log_to_file(f"Components tested: {detector_count}")
        log_to_file(f"Total UI elements detected: {total_elements}")
        
        # Check for specific issues
        if "Could not initialize vision-language model" in content or "VLM initialization success: False" in content:
            log_to_file("ISSUE: Vision-language model initialization failed")
        
        if "Failed to initialize HuggingFace detector" in content or "HuggingFace available: False" in content:
            log_to_file("ISSUE: HuggingFace detector not available or failed to initialize")
        
        # Overall status
        if error_count > 0:
            log_to_file("\nOVERALL STATUS: FAILED - Critical errors detected")
        elif warning_count > 0:
            log_to_file("\nOVERALL STATUS: PARTIAL SUCCESS - Warnings detected")
        else:
            log_to_file("\nOVERALL STATUS: SUCCESS - All tests passed")
    except Exception as e:
        log_to_file(f"ERROR analyzing test results: {e}")
        log_to_file(traceback.format_exc())


def main():
    """Main function"""
    # Initialize output file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"Simple ML UI Detector Test - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    log_section("Test Environment Setup")
    
    # Create test image
    test_image, image_path = create_test_image()
    log_to_file(f"Created test image at: {image_path}")
    
    # Run tests
    registry_success = test_detector_registry()
    log_to_file(f"Registry test {'succeeded' if registry_success else 'failed'}")
    
    adaptive_success = test_adaptive_detector(test_image, image_path)
    log_to_file(f"Adaptive detector test {'succeeded' if adaptive_success else 'failed'}")
    
    huggingface_success = test_huggingface_detector(test_image, image_path)
    log_to_file(f"HuggingFace detector test {'succeeded' if huggingface_success else 'failed'}")
    
    coordinator_success = test_multimodal_coordinator(test_image, image_path)
    log_to_file(f"Multimodal coordinator test {'succeeded' if coordinator_success else 'failed'}")
    
    # Analyze results
    analyze_test_results()
    
    log_section("Test Complete")
    log_to_file(f"Test results saved to: {OUTPUT_FILE}")
    log_to_file(f"Visualizations saved to: {os.path.join(OUTPUT_DIR, 'visualizations')}")


if __name__ == "__main__":
    main()
