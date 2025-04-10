"""
NEXUS Adaptive UI Element Detection with Screenshot Capture

This script demonstrates the adaptive UI detection capabilities 
by analyzing the current screen contents at runtime.
"""

import os
import cv2
import numpy as np
import logging
import time
from datetime import datetime

# Import screenshot capabilities
try:
    from PIL import ImageGrab
    SCREEN_CAPTURE_AVAILABLE = True
except ImportError:
    SCREEN_CAPTURE_AVAILABLE = False
    print("WARNING: PIL.ImageGrab not available. Install pillow package for screen capture.")

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create output directory
os.makedirs('detection_results', exist_ok=True)

class AdaptiveUIDetector:
    """Adaptive UI element detector that learns from interactions"""
    
    def __init__(self):
        self.detection_history = {}
        self.confidence_thresholds = {
            "button": 0.6,
            "text_field": 0.65,
            "icon": 0.7,
            "control": 0.65,
            "list_item": 0.6,
            "cell": 0.6,
            "panel": 0.55,
            "table_cell": 0.55,
            "text": 0.65
        }
    
    def detect_elements(self, screenshot):
        """Detect UI elements using an adaptive approach"""
        # Convert to grayscale
        if len(screenshot.shape) == 3:
            grayscale = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        else:
            grayscale = screenshot
            
        # Apply various detection strategies
        elements = []
        
        # 1. First strategy: Edge-based detection
        elements.extend(self._detect_with_edges(grayscale, screenshot))
        
        # 2. Second strategy: Color-based detection for UI components
        elements.extend(self._detect_with_color(screenshot))
        
        # Remove duplicates and overlapping elements
        elements = self._filter_overlapping_elements(elements)
        
        # Apply confidence thresholds based on learning
        elements = [e for e in elements if e["confidence"] >= 
                  self.confidence_thresholds.get(e["type"], 0.6)]
            
        logger.info(f"Adaptive detection found {len(elements)} elements")
        return elements
    
    def _detect_with_edges(self, grayscale, original):
        """Detect UI elements using edge detection"""
        # Apply Canny edge detection with adaptive thresholds
        median_val = np.median(grayscale)
        lower = int(max(0, (1.0 - 0.33) * median_val))
        upper = int(min(255, (1.0 + 0.33) * median_val))
        
        edges = cv2.Canny(grayscale, lower, upper)
        
        # Dilate edges to connect broken lines
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Extract potential UI elements
        ui_elements = []
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter small noise
            if w < 20 or h < 15 or w * h < 300:
                continue
                
            # Calculate aspect ratio to help classify the element
            aspect_ratio = w / h if h > 0 else 0
            
            # Calculate the fill ratio (contour area / bounding box area)
            contour_area = cv2.contourArea(contour)
            bbox_area = w * h
            fill_ratio = contour_area / bbox_area if bbox_area > 0 else 0
            
            # Get region of image for color analysis
            roi = original[y:y+h, x:x+w]
            
            # Determine element type based on aspect ratio, size, and fill ratio
            if w > 150 and h > 30 and aspect_ratio > 4.0:
                # Very wide elements are likely list items
                element_type = "list_item"
                confidence = 0.7 + (0.2 * min(fill_ratio, 0.5))
            elif aspect_ratio > 3.0:
                # Wide elements are likely text fields
                element_type = "text_field"
                confidence = 0.65 + (0.15 * min(fill_ratio, 0.6))
            elif 0.9 < aspect_ratio < 1.1 and w < 40 and h < 40:
                # Square elements are likely icons or controls
                element_type = "icon"
                confidence = 0.8 if fill_ratio > 0.7 else 0.7
            elif w > 100 and h > 25:
                # Large elements might be cells or panels
                element_type = "cell"
                confidence = 0.7 if fill_ratio > 0.6 else 0.6
            elif w < 40 and h < 40:
                # Small elements are likely icons or controls
                element_type = "control"
                confidence = 0.75 if fill_ratio > 0.6 else 0.65
            else:
                # Default to button for other cases
                element_type = "button"
                confidence = 0.6 + (0.2 * min(fill_ratio, 0.8))
                
            # Create element dict
            element = {
                "type": element_type,
                "bbox": (x, y, x + w, y + h),
                "center": (x + w // 2, y + h // 2),
                "confidence": confidence,
                "width": w,
                "height": h,
                "strategy": "edge"
            }
                
            ui_elements.append(element)
            
        return ui_elements
    
    def _detect_with_color(self, image):
        """Detect UI elements using color analysis"""
        elements = []
        
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for typical UI elements
        # Light backgrounds (white, light gray, etc.)
        lower_light = np.array([0, 0, 180])
        upper_light = np.array([180, 30, 255])
        
        # Dark elements (dark gray, black)
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 30, 60])
        
        # Blue/accent elements
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([140, 255, 255])
        
        # Create masks
        light_mask = cv2.inRange(hsv, lower_light, upper_light)
        dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Process each mask to find elements
        for mask_type, mask in [("light", light_mask), ("dark", dark_mask), ("accent", blue_mask)]:
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Filter by size
                if cv2.contourArea(contour) < 300:
                    continue
                    
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Determine element type based on color and shape
                if mask_type == "light" and w > h * 3:
                    element_type = "text_field"
                    confidence = 0.7
                elif mask_type == "light" and w > 100 and h > 30:
                    element_type = "panel"
                    confidence = 0.65
                elif mask_type == "dark" and w < 40 and h < 40:
                    element_type = "icon"
                    confidence = 0.75
                elif mask_type == "accent":
                    element_type = "button"
                    confidence = 0.8
                else:
                    element_type = "cell"
                    confidence = 0.6
                
                # Create element
                element = {
                    "type": element_type,
                    "bbox": (x, y, x + w, y + h),
                    "center": (x + w // 2, y + h // 2),
                    "confidence": confidence,
                    "width": w,
                    "height": h,
                    "strategy": f"color_{mask_type}"
                }
                elements.append(element)
        
        return elements
    
    def _filter_overlapping_elements(self, elements, overlap_threshold=0.6):
        """Filter out overlapping elements, keeping the higher confidence ones"""
        if not elements:
            return []
            
        # Sort by confidence (descending)
        elements.sort(key=lambda e: e["confidence"], reverse=True)
        
        filtered_elements = []
        for element in elements:
            should_keep = True
            
            for kept_element in filtered_elements:
                iou = self._calculate_iou(element["bbox"], kept_element["bbox"])
                if iou > overlap_threshold:
                    should_keep = False
                    break
                    
            if should_keep:
                filtered_elements.append(element)
                
        return filtered_elements
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate intersection over union for two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - intersection_area
        
        if union_area == 0:
            return 0.0
            
        return intersection_area / union_area
    
    def update_from_feedback(self, element, was_correct):
        """Learn from user feedback to improve detection"""
        element_type = element.get("type", "unknown")
        strategy = element.get("strategy", "unknown")
        confidence = element.get("confidence", 0.5)
        
        # Update detection history
        key = f"{element_type}_{strategy}"
        if key not in self.detection_history:
            self.detection_history[key] = {"correct": 0, "total": 0}
            
        self.detection_history[key]["total"] += 1
        if was_correct:
            self.detection_history[key]["correct"] += 1
            
        # Adjust confidence thresholds based on performance
        if self.detection_history[key]["total"] >= 5:
            accuracy = self.detection_history[key]["correct"] / self.detection_history[key]["total"]
            
            # If accuracy is low, increase the threshold to reduce false positives
            if accuracy < 0.6 and self.confidence_thresholds.get(element_type, 0.6) < 0.8:
                self.confidence_thresholds[element_type] += 0.05
                logger.info(f"Increasing threshold for {element_type} to {self.confidence_thresholds[element_type]:.2f}")
                
            # If accuracy is high, decrease the threshold to reduce false negatives
            elif accuracy > 0.8 and self.confidence_thresholds.get(element_type, 0.6) > 0.5:
                self.confidence_thresholds[element_type] -= 0.05
                logger.info(f"Decreasing threshold for {element_type} to {self.confidence_thresholds[element_type]:.2f}")

class UIVisualizer:
    """Visualization for detected UI elements"""
    
    def __init__(self):
        # Color scheme for element types (BGR format for OpenCV)
        self.type_colors = {
            "button": (0, 165, 255),     # Orange
            "text_field": (0, 255, 0),   # Green
            "icon": (255, 0, 0),         # Blue
            "control": (255, 0, 255),    # Magenta
            "list_item": (255, 255, 0),  # Cyan
            "cell": (0, 255, 255),       # Yellow
            "panel": (128, 0, 128),      # Purple
            "table_cell": (0, 128, 255), # Light Orange
            "text": (255, 255, 255),     # White
            "unknown": (128, 128, 128)   # Gray
        }
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.4
        self.font_thickness = 1
    
    def visualize_detections(self, image, elements, show_labels=True, show_strategy=True):
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
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 1)
            
            if show_labels:
                # Prepare label
                confidence = element.get("confidence", 0.0)
                strategy = element.get("strategy", "")
                
                if show_strategy:
                    label = f"{element_type}: {confidence:.2f} ({strategy})"
                else:
                    label = f"{element_type}: {confidence:.2f}"
                
                # Draw label background (smaller and more transparent for better visibility)
                label_size, _ = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)
                cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1), color, -1)
                
                # Draw label text
                cv2.putText(vis_image, label, (x1, y1 - 2), self.font, self.font_scale, (255, 255, 255), self.font_thickness)
                
        return vis_image

def capture_and_analyze_screen():
    """Capture the current screen and analyze for UI elements"""
    if not SCREEN_CAPTURE_AVAILABLE:
        print("ERROR: Screen capture not available. Please install pillow package.")
        return []
    
    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Capture screen
    print("Capturing screen...")
    screenshot = ImageGrab.grab()
    screenshot_np = np.array(screenshot)
    screenshot_rgb = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
    
    # Save screenshot
    screenshot_path = os.path.join("detection_results", f"screenshot_{timestamp}.png")
    cv2.imwrite(screenshot_path, screenshot_rgb)
    print(f"Screenshot saved to: {screenshot_path}")
    
    # Create detector
    detector = AdaptiveUIDetector()
    
    # Run detection
    print("Analyzing UI elements...")
    start_time = time.time()
    elements = detector.detect_elements(screenshot_rgb)
    elapsed_time = time.time() - start_time
    
    # Print statistics
    print(f"\nUI Detection Statistics:")
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
    for element_type, count in sorted(element_types.items()):
        print(f"  {element_type}: {count}")
    
    # Count by detection strategy
    strategy_counts = {}
    for element in elements:
        strategy = element.get("strategy", "unknown")
        if strategy not in strategy_counts:
            strategy_counts[strategy] = 0
        strategy_counts[strategy] += 1
    
    print("\nDetection strategies used:")
    for strategy, count in sorted(strategy_counts.items()):
        print(f"  {strategy}: {count}")
    
    # Visualize detection results
    visualizer = UIVisualizer()
    
    # Create visualization with labels
    vis_with_labels = visualizer.visualize_detections(screenshot_rgb, elements, show_labels=True, show_strategy=True)
    vis_path = os.path.join("detection_results", f"detection_{timestamp}.png")
    cv2.imwrite(vis_path, vis_with_labels)
    
    print(f"\nDetection visualization saved to: {vis_path}")
    return elements

def main():
    print("NEXUS Adaptive UI Detection - Screen Analysis")
    print("=========================================")
    
    # Create output directory
    try:
        os.makedirs("detection_results", exist_ok=True)
    except Exception as e:
        print(f"Error creating directories: {e}")
    
    capture_and_analyze_screen()
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
