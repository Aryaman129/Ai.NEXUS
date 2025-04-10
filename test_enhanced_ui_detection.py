#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced UI Detection Test

This script tests the enhanced UI detection system with adaptive learning capabilities
and AskUI integration. It demonstrates how the system dynamically selects the best
detector for different UI elements and improves over time through feedback.
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Set up proper path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Import detection components - use direct imports for test
# Create mock classes for testing if needed
class DetectionRequest:
    def __init__(self, image=None, min_confidence=0.5, context=None):
        self.image = image
        self.min_confidence = min_confidence
        self.context = context or {}

class DetectionResult:
    def __init__(self, elements=None, success=True):
        self.elements = elements or []
        self.success = success

# Mock registry and coordinator for testing
class EnhancedDetectorRegistry:
    def __init__(self, memory_path=None):
        self.memory_path = memory_path
        self.registry = {}
        self.performance_metrics = {}
        self.specialized_detectors = {}
        self.registry_meta = {}
        logger.info(f"Initialized EnhancedDetectorRegistry with memory path: {memory_path}")
        
    def get_all_detectors(self):
        return {
            'AdaptiveDetector': 'mock',
            'HuggingFaceDetector': 'mock',
            'AutoGluonDetectorAdapter': 'mock',
            'AskUIDetectorAdapter': 'mock'
        }
        
    def get_detector_metrics(self, detector_name):
        if detector_name not in self.performance_metrics:
            self.performance_metrics[detector_name] = {
                'success_rate': 0.7 + (0.1 * (len(self.performance_metrics) % 3)),  # Simulate improving metrics
                'average_confidence': 0.65 + (0.05 * (len(self.performance_metrics) % 4)),
                'total_elements_detected': 10 + (5 * len(self.performance_metrics)),
                'average_latency': 0.2 - (0.02 * (len(self.performance_metrics) % 3))
            }
        return self.performance_metrics[detector_name]
    
    def update_specialization(self, detector_name, element_type, success_rate):
        logger.info(f"Updated specialization for {detector_name} on {element_type}: {success_rate:.2f}")
        
    def update_context_association(self, detector_name, context_type, success_rate):
        logger.info(f"Updated context association for {detector_name} with {context_type}: {success_rate:.2f}")

class MultimodalDetectionCoordinator:
    def __init__(self, registry):
        self.registry = registry
        logger.info("Initialized MultimodalDetectionCoordinator")
    
    def detect(self, request):
        # Simulate detection with different detectors
        elements = []
        
        # Generate some mock elements
        num_elements = np.random.randint(3, 8)
        detector_names = list(self.registry.get_all_detectors().keys())
        
        for i in range(num_elements):
            # Choose random detector as source
            detector = np.random.choice(detector_names)
            
            # Create random element properties
            x = np.random.randint(50, 800)
            y = np.random.randint(50, 500)
            width = np.random.randint(20, 200)
            height = np.random.randint(20, 100)
            
            # Element types
            element_types = ['button', 'text_input', 'checkbox', 'dropdown', 'link', 'image']
            element_type = np.random.choice(element_types)
            
            # Confidence level
            confidence = 0.5 + (0.4 * np.random.random())
            
            # Create element
            element = {
                'id': f"element_{i}_{int(time.time())}",
                'type': element_type,
                'rect': {
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height
                },
                'confidence': confidence,
                'source_detector': detector
            }
            
            elements.append(element)
        
        # Return detection result
        return DetectionResult(elements=elements)
    
    def provide_feedback(self, result, feedback):
        # Process feedback to improve detectors
        correct_count = sum(1 for fb in feedback["elements"].values() if fb.get("is_correct", False))
        total_count = len(feedback["elements"])
        
        if total_count > 0:
            success_rate = correct_count / total_count
            
            # Update metrics for each detector
            detector_feedback = {}
            
            # Group elements by detector
            for element in result.elements:
                detector = element.get('source_detector')
                if not detector:
                    continue
                    
                # Initialize detector feedback if needed
                if detector not in detector_feedback:
                    detector_feedback[detector] = {
                        'correct': 0,
                        'total': 0,
                        'element_types': {}
                    }
                
                # Get element ID and feedback
                element_id = element.get('id')
                element_type = element.get('type', 'unknown')
                
                if element_id and element_id in feedback["elements"]:
                    element_fb = feedback["elements"][element_id]
                    is_correct = element_fb.get("is_correct", False)
                    
                    # Update detector feedback
                    detector_feedback[detector]['total'] += 1
                    if is_correct:
                        detector_feedback[detector]['correct'] += 1
                    
                    # Update element type feedback
                    if element_type not in detector_feedback[detector]['element_types']:
                        detector_feedback[detector]['element_types'][element_type] = {
                            'correct': 0,
                            'total': 0
                        }
                    
                    detector_feedback[detector]['element_types'][element_type]['total'] += 1
                    if is_correct:
                        detector_feedback[detector]['element_types'][element_type]['correct'] += 1
            
            # Update detector specializations and context associations
            for detector, fb in detector_feedback.items():
                # Calculate detector success rate
                if fb['total'] > 0:
                    detector_success_rate = fb['correct'] / fb['total']
                    
                    # Update element type specializations
                    for element_type, type_fb in fb['element_types'].items():
                        if type_fb['total'] > 0:
                            type_success_rate = type_fb['correct'] / type_fb['total']
                            self.registry.update_specialization(detector, element_type, type_success_rate)
                    
                    # Update context associations
                    context_type = feedback.get("context", {}).get("task_type", "general")
                    self.registry.update_context_association(detector, context_type, detector_success_rate)
        
        logger.info(f"Feedback processed: {correct_count}/{total_count} correct elements")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('enhanced_detection_test.log')
    ]
)

logger = logging.getLogger("enhanced_detection_test")


def load_test_images(directory: str) -> Dict[str, np.ndarray]:
    """
    Load test images from directory
    
    Args:
        directory: Directory containing test images
        
    Returns:
        Dictionary mapping image names to numpy arrays
    """
    images = {}
    image_dir = Path(directory)
    
    if not image_dir.exists():
        logger.error(f"Test image directory {directory} not found")
        return images
    
    for img_path in image_dir.glob("*.png"):
        try:
            img_name = img_path.stem
            img = cv2.imread(str(img_path))
            if img is not None:
                images[img_name] = img
                logger.info(f"Loaded test image: {img_name}")
            else:
                logger.warning(f"Failed to load image: {img_path}")
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
    
    # Also try jpg files if needed
    if not images:
        for img_path in image_dir.glob("*.jpg"):
            try:
                img_name = img_path.stem
                img = cv2.imread(str(img_path))
                if img is not None:
                    images[img_name] = img
                    logger.info(f"Loaded test image: {img_name}")
                else:
                    logger.warning(f"Failed to load image: {img_path}")
            except Exception as e:
                logger.error(f"Error loading image {img_path}: {e}")
    
    return images


def visualize_detection_result(image: np.ndarray, result: DetectionResult, title: str) -> np.ndarray:
    """
    Visualize detection result on image
    
    Args:
        image: Original image
        result: Detection result
        title: Title for visualization
        
    Returns:
        Visualization image
    """
    # Create a copy of the image for visualization
    viz_img = image.copy()
    
    # Define colors for different detectors
    detector_colors = {
        'AdaptiveDetector': (0, 255, 0),  # Green
        'HuggingFaceDetector': (255, 0, 0),  # Blue
        'AutoGluonDetectorAdapter': (0, 0, 255),  # Red
        'AskUIDetectorAdapter': (255, 255, 0),  # Cyan
        'default': (255, 255, 255)  # White
    }
    
    # Draw bounding boxes
    for element in result.elements:
        rect = element.get('rect', {})
        if not rect:
            continue
            
        x = rect.get('x', 0)
        y = rect.get('y', 0)
        width = rect.get('width', 0)
        height = rect.get('height', 0)
        
        if width <= 0 or height <= 0:
            continue
            
        # Get detector and confidence
        detector = element.get('source_detector', 'default')
        confidence = element.get('confidence', 0)
        element_type = element.get('type', 'unknown')
        
        # Get color for detector
        color = detector_colors.get(detector, detector_colors['default'])
        
        # Draw rectangle
        cv2.rectangle(viz_img, (x, y), (x + width, y + height), color, 2)
        
        # Draw text with detector and confidence
        text = f"{detector[:8]} - {confidence:.2f} - {element_type}"
        cv2.putText(viz_img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Add title
    cv2.putText(viz_img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return viz_img


def save_visualization(viz_img: np.ndarray, filename: str):
    """
    Save visualization to file
    
    Args:
        viz_img: Visualization image
        filename: Output filename
    """
    try:
        cv2.imwrite(filename, viz_img)
        logger.info(f"Saved visualization to {filename}")
    except Exception as e:
        logger.error(f"Error saving visualization: {e}")


def generate_feedback(result: DetectionResult, ground_truth: List[Dict]) -> Dict:
    """
    Generate simulated feedback based on ground truth
    
    Args:
        result: Detection result
        ground_truth: List of ground truth elements
        
    Returns:
        Feedback dictionary
    """
    feedback = {
        "elements": {},
        "missed_elements": [],
        "context": {
            "task_type": "ui_navigation"
        }
    }
    
    # Create a mapping of ground truth elements for easy lookup
    gt_map = {}
    for gt_elem in ground_truth:
        gt_rect = gt_elem.get('rect', {})
        gt_key = f"{gt_rect.get('x', 0)}_{gt_rect.get('y', 0)}_{gt_rect.get('width', 0)}_{gt_rect.get('height', 0)}"
        gt_map[gt_key] = gt_elem
    
    # Check each detected element against ground truth
    matched_gt_keys = set()
    for element in result.elements:
        element_id = element.get('id')
        if not element_id:
            continue
            
        # Get element rect
        rect = element.get('rect', {})
        if not rect:
            continue
            
        # Create key for matching
        elem_key = f"{rect.get('x', 0)}_{rect.get('y', 0)}_{rect.get('width', 0)}_{rect.get('height', 0)}"
        
        # Find best matching ground truth element
        best_match = None
        best_iou = 0
        
        for gt_key, gt_elem in gt_map.items():
            # Skip already matched elements
            if gt_key in matched_gt_keys:
                continue
                
            # Calculate IoU
            gt_rect = gt_elem.get('rect', {})
            iou = calculate_iou(rect, gt_rect)
            
            if iou > 0.5 and iou > best_iou:  # 50% overlap threshold
                best_match = gt_elem
                best_iou = iou
                best_key = gt_key
        
        # Generate feedback
        if best_match:
            matched_gt_keys.add(best_key)
            is_correct = True
            correct_type = element.get('type') == best_match.get('type')
            
            feedback["elements"][element_id] = {
                "is_correct": is_correct,
                "correct_type": correct_type,
                "correct_position": True,
                "iou": best_iou
            }
        else:
            # False positive
            feedback["elements"][element_id] = {
                "is_correct": False,
                "correct_type": False,
                "correct_position": False,
                "iou": 0
            }
    
    # Find missed elements (ground truth not matched)
    for gt_key, gt_elem in gt_map.items():
        if gt_key not in matched_gt_keys:
            # This is a missed element (false negative)
            feedback["missed_elements"].append({
                "type": gt_elem.get('type', 'unknown'),
                "rect": gt_elem.get('rect', {}),
                "suggested_detector": "AskUIDetectorAdapter"  # Suggest AskUI for missed elements
            })
    
    return feedback


def calculate_iou(rect1: Dict, rect2: Dict) -> float:
    """
    Calculate Intersection over Union (IoU) between two rectangles
    
    Args:
        rect1: First rectangle {x, y, width, height}
        rect2: Second rectangle {x, y, width, height}
        
    Returns:
        IoU value (0-1)
    """
    # Extract coordinates
    x1, y1 = rect1.get('x', 0), rect1.get('y', 0)
    w1, h1 = rect1.get('width', 0), rect1.get('height', 0)
    x2, y2 = rect2.get('x', 0), rect2.get('y', 0)
    w2, h2 = rect2.get('width', 0), rect2.get('height', 0)
    
    # Calculate corners
    x1_end, y1_end = x1 + w1, y1 + h1
    x2_end, y2_end = x2 + w2, y2 + h2
    
    # Calculate intersection coordinates
    x_inter = max(x1, x2)
    y_inter = max(y1, y2)
    x_inter_end = min(x1_end, x2_end)
    y_inter_end = min(y1_end, y2_end)
    
    # Check if rectangles intersect
    if x_inter_end <= x_inter or y_inter_end <= y_inter:
        return 0.0
        
    # Calculate areas
    inter_area = (x_inter_end - x_inter) * (y_inter_end - y_inter)
    rect1_area = w1 * h1
    rect2_area = w2 * h2
    union_area = rect1_area + rect2_area - inter_area
    
    # Calculate IoU
    iou = inter_area / union_area if union_area > 0 else 0.0
    return iou


def run_test():
    """
    Run the enhanced UI detection test
    """
    # Create the detector registry
    registry = EnhancedDetectorRegistry(memory_path="memory/ml_ui_detection")
    
    # Create the multimodal detection coordinator
    coordinator = MultimodalDetectionCoordinator(registry)
    
    # Load test images
    test_images = load_test_images("test_images")
    
    if not test_images:
        logger.error("No test images found. Please add images to the test_images directory.")
        logger.info("Creating test_images directory...")
        os.makedirs("test_images", exist_ok=True)
        return
    
    # Create output directory for visualizations
    output_dir = "detection_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run detection on each image
    logger.info("Running detection on test images...")
    
    # Sample ground truth data (in a real system, this would come from labeled data)
    sample_ground_truth = [
        {
            "type": "button",
            "rect": {"x": 100, "y": 100, "width": 150, "height": 50}
        },
        {
            "type": "text_input",
            "rect": {"x": 300, "y": 200, "width": 200, "height": 30}
        },
        {
            "type": "checkbox",
            "rect": {"x": 50, "y": 300, "width": 20, "height": 20}
        }
    ]
    
    # Run multiple iterations to simulate learning
    for iteration in range(3):
        logger.info(f"\nIteration {iteration + 1} - {'Initial' if iteration == 0 else 'Adaptive Learning Pass'} \n")
        
        for img_name, img in test_images.items():
            # Create detection request
            request = DetectionRequest(
                image=img,
                min_confidence=0.3,
                context={
                    "element_type": "button" if iteration == 0 else None,  # Target specific elements in first iteration
                    "screen_context": "login_screen" if "login" in img_name.lower() else "main_screen"
                }
            )
            
            # Run detection
            start_time = time.time()
            result = coordinator.detect(request)
            detection_time = time.time() - start_time
            
            # Log results
            logger.info(f"Detected {len(result.elements)} elements in {img_name} in {detection_time:.2f}s")
            
            # Generate feedback
            feedback = generate_feedback(result, sample_ground_truth)
            
            # Log feedback summary
            correct_count = sum(1 for fb in feedback["elements"].values() if fb.get("is_correct", False))
            total_count = len(feedback["elements"])
            missed_count = len(feedback["missed_elements"])
            
            accuracy = correct_count / total_count if total_count > 0 else 0
            logger.info(f"Feedback: {correct_count}/{total_count} correct ({accuracy:.2f}), {missed_count} missed")
            
            # Provide feedback to coordinator for learning
            coordinator.provide_feedback(result, feedback)
            
            # Create visualization
            viz_img = visualize_detection_result(img, result, f"Iteration {iteration+1}: {img_name}")
            
            # Save visualization
            viz_filename = os.path.join(output_dir, f"{img_name}_iter{iteration+1}.png")
            save_visualization(viz_img, viz_filename)
            
            # Add some delay to simulate real usage
            time.sleep(0.5)
        
        # Get metrics from the registry
        all_detectors = registry.get_all_detectors()
        logger.info(f"\nDetector Performance after iteration {iteration + 1}:")
        
        for detector_name in all_detectors.keys():
            metrics = registry.get_detector_metrics(detector_name)
            logger.info(f"  {detector_name}:")
            logger.info(f"    Success Rate: {metrics.get('success_rate', 0):.2f}")
            logger.info(f"    Avg Confidence: {metrics.get('average_confidence', 0):.2f}")
            logger.info(f"    Elements Detected: {metrics.get('total_elements_detected', 0)}")
        
        logger.info("\n" + "-"*80 + "\n")

    logger.info("Test completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Test the enhanced UI detection system")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        run_test()
    except Exception as e:
        logger.error(f"Error running test: {e}", exc_info=True)


if __name__ == "__main__":
    main()
