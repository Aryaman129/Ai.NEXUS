"""
NEXUS UI Element Detection on Apple Music Interface

This script demonstrates the adaptive UI detection capabilities
on a real-world example of the Apple Music interface.
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

# Create output directory
os.makedirs('detection_results', exist_ok=True)

class AdvancedUIDetector:
    """Enhanced UI element detector for real-world interfaces"""
    
    def detect_elements(self, screenshot):
        """Detect UI elements using an enhanced approach"""
        # Convert to grayscale
        if len(screenshot.shape) == 3:
            grayscale = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        else:
            grayscale = screenshot
            
        # Apply various detection strategies
        elements = []
        
        # 1. First strategy: Contour-based detection
        elements.extend(self._detect_with_contours(grayscale, screenshot))
        
        # 2. Second strategy: Rectangle detection (for buttons and cells)
        elements.extend(self._detect_rectangles(grayscale, screenshot))
        
        # 3. Third strategy: Text region detection
        elements.extend(self._detect_text_regions(grayscale, screenshot))
        
        # Remove duplicates and overlapping elements
        elements = self._filter_overlapping_elements(elements)
            
        logger.info(f"Advanced detection found {len(elements)} elements")
        return elements
    
    def _detect_with_contours(self, grayscale, original):
        """Detect UI elements using contour detection"""
        # Apply edge detection
        edges = cv2.Canny(grayscale, 50, 150)
        
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
            if w < 20 or h < 15:
                continue
                
            # Calculate aspect ratio to help classify the element
            aspect_ratio = w / h if h > 0 else 0
            
            # Determine element type based on aspect ratio and size
            if aspect_ratio > 5.0:
                # Very wide elements are likely text fields or list items
                element_type = "list_item"
                confidence = 0.7
            elif aspect_ratio > 3.0:
                element_type = "text_field"
                confidence = 0.7
            elif 0.9 < aspect_ratio < 1.1 and w < 25 and h < 25:
                # Small square elements are likely checkboxes or icons
                element_type = "icon"
                confidence = 0.8
            elif w > 150 and h > 25:
                # Large elements might be cells or panels
                element_type = "cell"
                confidence = 0.7
            elif w < 30 and h < 30:
                # Small elements are likely icons or controls
                element_type = "control"
                confidence = 0.75
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
                "strategy": "contour"
            }
                
            ui_elements.append(element)
            
        return ui_elements
    
    def _detect_rectangles(self, grayscale, original):
        """Detect rectangular UI elements"""
        elements = []
        
        # Use HoughLinesP to detect lines
        edges = cv2.Canny(grayscale, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=50, maxLineGap=10)
        
        if lines is not None:
            # Create a blank image for visualization of lines
            line_image = np.zeros_like(grayscale)
            
            # Draw lines
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), 255, 2)
            
            # Dilate lines to connect nearby ones
            kernel = np.ones((5, 5), np.uint8)
            dilated_lines = cv2.dilate(line_image, kernel, iterations=1)
            
            # Find contours of potential rectangles
            contours, _ = cv2.findContours(dilated_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by size
                if w < 40 or h < 25 or w * h < 1000:
                    continue
                
                # Create element
                element = {
                    "type": "panel" if w > 200 and h > 100 else "table_cell",
                    "bbox": (x, y, x + w, y + h),
                    "center": (x + w // 2, y + h // 2),
                    "confidence": 0.65,
                    "width": w,
                    "height": h,
                    "strategy": "rectangle"
                }
                elements.append(element)
        
        return elements
    
    def _detect_text_regions(self, grayscale, original):
        """Detect regions likely to contain text"""
        elements = []
        
        # Apply threshold to find dark text on light background
        _, binary = cv2.threshold(grayscale, 180, 255, cv2.THRESH_BINARY_INV)
        
        # Apply horizontal dilation to connect letters within words
        kernel_h = np.ones((1, 15), np.uint8)
        dilated_h = cv2.dilate(binary, kernel_h, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated_h, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size and shape (text regions are typically wider than tall)
            if w < 30 or h < 10 or h > 40 or w/h < 2:
                continue
            
            # Create element
            element = {
                "type": "text",
                "bbox": (x, y, x + w, y + h),
                "center": (x + w // 2, y + h // 2),
                "confidence": 0.7,
                "width": w,
                "height": h,
                "strategy": "text"
            }
            elements.append(element)
        
        return elements
    
    def _filter_overlapping_elements(self, elements, overlap_threshold=0.7):
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

class EnhancedVisualizer:
    """Enhanced visualization for detected UI elements"""
    
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

def save_image(path, image):
    """Save an image with proper compression for UI screenshots"""
    # Use PNG for lossless quality
    cv2.imwrite(path, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    logger.info(f"Saved image to: {path}")

def process_apple_music_image():
    """Process the Apple Music interface image"""
    # Load the image
    image_path = os.path.join("detection_results", "apple_music_screenshot.png")
    
    # Check if we need to take a screenshot
    if not os.path.exists(image_path):
        logger.info("Apple Music screenshot not found. Please provide an image.")
        return
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return
    
    # Create detector
    detector = AdvancedUIDetector()
    
    # Run detection
    start_time = time.time()
    elements = detector.detect_elements(image)
    elapsed_time = time.time() - start_time
    
    # Print statistics
    print(f"\nApple Music UI Detection Statistics:")
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
    visualizer = EnhancedVisualizer()
    
    # Create visualization with labels
    vis_with_labels = visualizer.visualize_detections(image, elements, show_labels=True, show_strategy=True)
    save_image(os.path.join("detection_results", "apple_music_detection_with_labels.png"), vis_with_labels)
    
    # Create clean visualization without labels
    vis_without_labels = visualizer.visualize_detections(image, elements, show_labels=False)
    save_image(os.path.join("detection_results", "apple_music_detection_clean.png"), vis_without_labels)
    
    print("\nDetection visualizations saved to:")
    print(f"  detection_results/apple_music_detection_with_labels.png")
    print(f"  detection_results/apple_music_detection_clean.png")
    
    print("\nAnalysis complete!")
    return elements

def main():
    print("NEXUS Adaptive UI Detection - Apple Music Interface Analysis")
    print("=========================================================")
    print("Analyzing Apple Music interface for UI elements...")
    
    # Save image
    try:
        os.makedirs("detection_results", exist_ok=True)
    except Exception as e:
        print(f"Error creating directories: {e}")
    
    process_apple_music_image()

if __name__ == "__main__":
    main()
