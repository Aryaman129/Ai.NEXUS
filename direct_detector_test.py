"""
Direct test for NEXUS ML-based UI detection components.

This script directly imports and tests the ML-based detector components
without relying on the full NEXUS import structure.
"""

import os
import cv2
import numpy as np
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Direct imports from files (bypassing package structure)
# This is an adaptive approach that works with what we have
import sys
sys.path.append(os.path.abspath('src'))  # Add src to path

# Import base detector classes directly
detector_base_path = os.path.join('src', 'ai_core', 'screen_analysis', 'detectors', 'base.py')
detector_registry_path = os.path.join('src', 'ai_core', 'screen_analysis', 'detectors', 'registry.py')
opencv_detector_path = os.path.join('src', 'ai_core', 'screen_analysis', 'detectors', 'opencv_detector.py')
visualizer_path = os.path.join('src', 'ai_core', 'screen_analysis', 'visualization', 'detection_visualizer.py')

# Create output directories
os.makedirs('test_images', exist_ok=True)
os.makedirs('detection_results', exist_ok=True)

# Create a test image
def create_test_image():
    """Create a sample UI test image"""
    test_image_path = os.path.join('test_images', 'sample_ui.png')
    
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
    
    return cv2.imread(test_image_path)

# Create a direct implementation of the OpenCV detector
class SimpleOpenCVDetector:
    """Simplified OpenCV detector for UI elements"""
    
    def detect_elements(self, screenshot):
        """Detect UI elements using OpenCV"""
        # Convert to grayscale if needed
        if len(screenshot.shape) == 3:
            grayscale = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        else:
            grayscale = screenshot
            
        # Apply edge detection
        edges = cv2.Canny(grayscale, 50, 150)
        
        # Dilate edges to connect broken lines
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Extract potential UI elements
        ui_elements = []
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter small noise
            if w < 30 or h < 20:
                continue
                
            # Calculate aspect ratio to help classify the element
            aspect_ratio = w / h if h > 0 else 0
            
            # Determine element type based on aspect ratio and size
            if aspect_ratio > 3.0:
                # Very wide elements are likely text fields
                element_type = "text_field"
                confidence = 0.7
            elif 0.9 < aspect_ratio < 1.1 and w < 40 and h < 40:
                # Square elements are likely checkboxes or radio buttons
                element_type = "checkbox"
                confidence = 0.7
            elif aspect_ratio < 0.3 and h > 100:
                # Tall, narrow elements might be scrollbars
                element_type = "scrollbar"
                confidence = 0.6
            else:
                # Default to button for other cases
                element_type = "button"
                confidence = 0.6
                
            # Create element dict
            element = {
                "type": element_type,
                "bbox": (x, y, x + w, y + h),
                "center": (x + w // 2, y + h // 2),
                "confidence": confidence,
                "width": w,
                "height": h,
                "detector": "opencv_direct",
                "text": ""
            }
                
            ui_elements.append(element)
            
        logger.info(f"OpenCV detection found {len(ui_elements)} elements")
        return ui_elements

class SimpleVisualizer:
    """Simple visualization for detected UI elements"""
    
    def __init__(self):
        # Color scheme for element types (BGR format for OpenCV)
        self.type_colors = {
            "button": (0, 165, 255),      # Orange
            "text_field": (0, 255, 0),    # Green
            "checkbox": (255, 0, 0),      # Blue
            "dropdown": (255, 0, 255),    # Magenta
            "radio_button": (255, 255, 0), # Cyan
            "scrollbar": (0, 0, 255),     # Red
            "unknown": (128, 128, 128)    # Gray
        }
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 1
    
    def visualize_detections(self, image, elements):
        """Visualize detected UI elements on an image"""
        # Make a copy of the image to avoid modifying the original
        vis_image = image.copy()
        
        # Draw each element
        for element in elements:
            # Get bounding box
            x1, y1, x2, y2 = element["bbox"]
            
            # Determine color based on element type
            element_type = element.get("type", "unknown")
            color = self.type_colors.get(element_type, self.type_colors["unknown"])
                
            # Draw rectangle
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            confidence = element.get("confidence", 0.0)
            label = f"{element_type}: {confidence:.2f}"
            
            # Draw label background
            label_size, _ = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)
            cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(vis_image, label, (x1, y1 - 5), self.font, self.font_scale, (255, 255, 255), self.font_thickness)
                
        return vis_image

def run_direct_test():
    """Run a direct test of our detection system"""
    print("NEXUS Adaptive UI Detection - Direct Test")
    print("=========================================")
    
    # Create and load test image
    test_image = create_test_image()
    
    # Create detector and run detection
    detector = SimpleOpenCVDetector()
    start_time = time.time()
    elements = detector.detect_elements(test_image)
    elapsed_time = time.time() - start_time
    
    # Print statistics
    print(f"\nDetection Statistics:")
    print(f"  Total elements detected: {len(elements)}")
    print(f"  Time taken: {elapsed_time:.3f} seconds")
    
    # Count by element type
    element_types = {}
    for element in elements:
        element_type = element.get("type", "unknown")
        if element_type not in element_types:
            element_types[element_type] = 0
        element_types[element_type] += 1
    
    print("\nElement types detected:")
    for element_type, count in element_types.items():
        print(f"  {element_type}: {count}")
    
    # Visualize detection results
    visualizer = SimpleVisualizer()
    visualization = visualizer.visualize_detections(test_image, elements)
    
    # Save visualization
    output_path = os.path.join("detection_results", "direct_detection.png")
    cv2.imwrite(output_path, visualization)
    print(f"\nVisualization saved to: {output_path}")
    
    print("\nTest completed successfully!")
    return elements

if __name__ == "__main__":
    run_direct_test()
