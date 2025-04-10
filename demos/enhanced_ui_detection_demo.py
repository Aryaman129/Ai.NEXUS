"""Enhanced UI Detection Demonstration

This script demonstrates the enhanced ML-based UI detection capabilities
of the NEXUS system, showcasing the new adaptive detector registry,
multimodal detection coordinator, and various specialized detectors.
"""

import sys
import os
import time
import logging
import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Any, Optional

# Configure paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import NEXUS components
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

# Color mapping for visualization
COLOR_MAP = {
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


def setup_detection_system():
    """Set up the enhanced UI detection system
    
    Returns:
        MultimodalDetectionCoordinator: The detection coordinator
    """
    logger.info("Setting up enhanced UI detection system...")
    
    # Create enhanced detector registry
    registry = EnhancedDetectorRegistry()
    
    # Create adaptive detector
    adaptive_detector = AdaptiveDetector()
    if adaptive_detector.initialize():
        registry.register_detector("adaptive", adaptive_detector)
        logger.info("Registered adaptive detector")
    
    # Create HuggingFace detector
    huggingface_detector = HuggingFaceDetector()
    if huggingface_detector.is_available and huggingface_detector.initialize():
        registry.register_detector("huggingface", huggingface_detector)
        logger.info("Registered HuggingFace detector")
    
    # Create AutoGluon adapter
    try:
        autogluon_adapter = AutoGluonDetectorAdapter()
        if autogluon_adapter.initialize():
            registry.register_detector("autogluon", autogluon_adapter)
            logger.info("Registered AutoGluon detector adapter")
    except Exception as e:
        logger.warning(f"Could not initialize AutoGluon detector: {e}")
    
    # Create multimodal detection coordinator
    coordinator = MultimodalDetectionCoordinator(registry)
    
    # Initialize vision-language model for semantic understanding
    try:
        vlm_success = coordinator.initialize_vlm()
        if vlm_success:
            logger.info("Initialized vision-language model for semantic understanding")
        else:
            logger.warning("Could not initialize vision-language model")
    except Exception as e:
        logger.warning(f"Error initializing vision-language model: {e}")
    
    return coordinator


def load_image(image_path):
    """Load an image from the given path
    
    Args:
        image_path: Path to the image file
        
    Returns:
        np.ndarray: The loaded image
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        return image
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        sys.exit(1)


def visualize_detection(image, elements, output_path=None, show=True):
    """Visualize detected elements on the image
    
    Args:
        image: The original image
        elements: List of detected elements
        output_path: Optional path to save the visualization
        show: Whether to display the visualization
    """
    # Create a copy for visualization
    viz_image = image.copy()
    
    # Convert to RGB for PIL
    viz_image = cv2.cvtColor(viz_image, cv2.COLOR_BGR2RGB)
    
    # Create PIL image for drawing
    pil_image = Image.fromarray(viz_image)
    draw = ImageDraw.Draw(pil_image)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()
    
    # Draw each element
    for i, element in enumerate(elements):
        # Get bounding box
        bbox = element.get("bbox")
        if not bbox:
            continue
        
        # Get element type and color
        element_type = element.get("type", "unknown")
        color_rgb = COLOR_MAP.get(element_type, COLOR_MAP["unknown"])
        
        # Get confidence
        confidence = element.get("confidence", 0.0)
        
        # Draw rectangle
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=2)
        
        # Add label with confidence
        label = f"{element_type} ({confidence:.2f})"
        draw.text((x1, y1-15), label, fill=color_rgb, font=font)
        
        # Draw method source if available
        method = element.get("detection_method", element.get("source_detector", "unknown"))
        draw.text((x1, y2+5), method, fill=(128, 128, 128), font=font)
    
    # Convert back to numpy array
    result_image = np.array(pil_image)
    
    # Save if output path provided
    if output_path:
        output_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, output_image)
        logger.info(f"Saved visualization to {output_path}")
    
    # Show if requested
    if show:
        plt.figure(figsize=(12, 8))
        plt.imshow(result_image)
        plt.axis('off')
        plt.title(f"Detected {len(elements)} UI Elements")
        plt.tight_layout()
        plt.show()
    
    return result_image


def print_detection_details(elements):
    """Print details of detected elements
    
    Args:
        elements: List of detected elements
    """
    print(f"\nDetected {len(elements)} UI elements:\n")
    print("{:<5} {:<15} {:<25} {:<15} {:<10}".format("No.", "Type", "Position", "Confidence", "Detector"))
    print("-" * 80)
    
    for i, element in enumerate(elements):
        element_type = element.get("type", "unknown")
        bbox = element.get("bbox", (0, 0, 0, 0))
        confidence = element.get("confidence", 0.0)
        detector = element.get("source_detector", element.get("detection_method", "unknown"))
        
        print("{:<5} {:<15} {:<25} {:<15.2f} {:<10}".format(
            i+1, element_type, f"({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})", confidence, detector
        ))
    
    print("\n")


def run_analysis_loop(coordinator, image_path):
    """Run an interactive analysis loop for the image
    
    Args:
        coordinator: The multimodal detection coordinator
        image_path: Path to the image file
    """
    image = load_image(image_path)
    
    while True:
        print("\nEnhanced UI Detection Analysis Options:")
        print("1. Run detection with all available detectors")
        print("2. Run detection with adaptive detector only")
        print("3. Run detection with HuggingFace detector only")
        print("4. Run detection with AutoGluon detector only")
        print("5. Try different confidence threshold")
        print("6. Toggle semantic analysis")
        print("7. Visualize detection results")
        print("8. Print performance statistics")
        print("9. Load a different image")
        print("0. Exit")
        
        choice = input("\nEnter your choice: ")
        
        if choice == "0":
            break
        
        elif choice == "1":
            # Run with all detectors
            request = DetectionRequest(
                image=image,
                semantic_analysis=True
            )
            
            start_time = time.time()
            result = coordinator.detect(request)
            end_time = time.time()
            
            print(f"\nDetection completed in {end_time - start_time:.2f} seconds")
            print_detection_details(result.elements)
            visualize_detection(image, result.elements)
        
        elif choice == "2":
            # Run with adaptive detector only
            request = DetectionRequest(
                image=image,
                priority_detectors=["adaptive"],
                semantic_analysis=False
            )
            
            result = coordinator.detect(request)
            print_detection_details(result.elements)
            visualize_detection(image, result.elements)
        
        elif choice == "3":
            # Run with HuggingFace detector only
            request = DetectionRequest(
                image=image,
                priority_detectors=["huggingface"],
                semantic_analysis=True
            )
            
            result = coordinator.detect(request)
            print_detection_details(result.elements)
            visualize_detection(image, result.elements)
        
        elif choice == "4":
            # Run with AutoGluon detector only
            request = DetectionRequest(
                image=image,
                priority_detectors=["autogluon"],
                semantic_analysis=False
            )
            
            result = coordinator.detect(request)
            print_detection_details(result.elements)
            visualize_detection(image, result.elements)
        
        elif choice == "5":
            # Try different confidence threshold
            try:
                threshold = float(input("Enter confidence threshold (0.0-1.0): "))
                threshold = max(0.0, min(1.0, threshold))
                
                request = DetectionRequest(
                    image=image,
                    min_confidence=threshold,
                    semantic_analysis=True
                )
                
                result = coordinator.detect(request)
                print_detection_details(result.elements)
                visualize_detection(image, result.elements)
            except ValueError:
                print("Invalid threshold value, using default")
        
        elif choice == "6":
            # Toggle semantic analysis
            use_semantic = input("Enable semantic analysis? (y/n): ").lower() == 'y'
            
            request = DetectionRequest(
                image=image,
                semantic_analysis=use_semantic
            )
            
            result = coordinator.detect(request)
            print_detection_details(result.elements)
            visualize_detection(image, result.elements)
            
            # Show semantic descriptions if available
            if use_semantic:
                print("\nSemantic Descriptions:")
                for i, element in enumerate(result.elements):
                    if "semantic_description" in element:
                        print(f"Element {i+1} ({element['type']}): {element['semantic_description']}")
        
        elif choice == "7":
            # Run detection if not already done
            try:
                if not result.elements:
                    request = DetectionRequest(image=image)
                    result = coordinator.detect(request)
            except NameError:
                request = DetectionRequest(image=image)
                result = coordinator.detect(request)
            
            # Visualize with option to save
            save_path = input("Enter path to save visualization (or press Enter to skip): ")
            if save_path.strip():
                visualize_detection(image, result.elements, output_path=save_path)
            else:
                visualize_detection(image, result.elements)
        
        elif choice == "8":
            # Print performance statistics
            stats = coordinator.get_performance_statistics()
            
            print("\nPerformance Statistics:")
            print(f"Average total detection time: {stats.get('avg_total_time', 0):.3f} seconds")
            print(f"Number of detection runs: {stats.get('history_count', 0)}")
            
            print("\nDetector-specific statistics:")
            detector_stats = stats.get("detector_stats", {})
            for detector_name, detector_stats in detector_stats.items():
                print(f"\n{detector_name}:")
                print(f"  Average time: {detector_stats.get('avg_time', 0):.3f} seconds")
                print(f"  Average elements detected: {detector_stats.get('avg_elements', 0):.1f}")
                print(f"  Average confidence: {detector_stats.get('avg_confidence', 0):.3f}")
        
        elif choice == "9":
            # Load a different image
            new_path = input("Enter path to new image: ")
            try:
                new_image = load_image(new_path)
                image = new_image
                image_path = new_path
                print(f"Loaded new image from {new_path}")
            except Exception as e:
                print(f"Error loading image: {e}")
        
        else:
            print("Invalid choice, try again")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Enhanced UI Detection Demo")
    parser.add_argument("-i", "--image", type=str, help="Path to the image to analyze")
    args = parser.parse_args()
    
    # Set up detection system
    coordinator = setup_detection_system()
    
    # If no image provided, use a default or ask for one
    image_path = args.image
    if not image_path:
        # Try common screenshot locations
        default_paths = [
            os.path.join(os.path.expanduser("~"), "Desktop", "screenshot.png"),
            os.path.join(os.path.expanduser("~"), "Pictures", "Screenshots", "screenshot.png"),
            os.path.join(os.path.dirname(__file__), "..", "test_data", "ui_screenshots", "sample_ui.png")
        ]
        
        for path in default_paths:
            if os.path.exists(path):
                image_path = path
                logger.info(f"Using default image: {image_path}")
                break
        
        if not image_path:
            image_path = input("Enter path to image file for analysis: ")
    
    # Run analysis loop
    run_analysis_loop(coordinator, image_path)


if __name__ == "__main__":
    main()
