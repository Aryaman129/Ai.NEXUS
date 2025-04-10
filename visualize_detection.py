#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization script for UI detection results
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Test image loading
def load_test_image():
    """Load the test images"""
    # Look for existing test images
    image_dir = Path('test_images')
    if not image_dir.exists():
        print("Creating test images directory...")
        os.makedirs(image_dir, exist_ok=True)
        
        # Create a simple test image for login form
        test_image = np.ones((600, 800, 3), dtype=np.uint8) * 255  # White background
        
        # Draw a button
        cv2.rectangle(test_image, (100, 100), (250, 150), (50, 50, 200), -1)  # Filled rectangle for button
        cv2.putText(test_image, 'Login', (130, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw a text input
        cv2.rectangle(test_image, (100, 200), (400, 240), (220, 220, 220), -1)  # Filled rectangle for input
        cv2.rectangle(test_image, (100, 200), (400, 240), (150, 150, 150), 1)  # Border
        
        # Draw a checkbox
        cv2.rectangle(test_image, (100, 300), (120, 320), (220, 220, 220), -1)  # Filled rectangle for checkbox
        cv2.rectangle(test_image, (100, 300), (120, 320), (50, 50, 50), 1)  # Border
        cv2.putText(test_image, 'Remember me', (130, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
        
        # Save the test image
        cv2.imwrite(str(image_dir / 'test_login_form.png'), test_image)
        
        # Create another test image for a search bar
        test_image2 = np.ones((600, 800, 3), dtype=np.uint8) * 240  # Light gray background
        
        # Draw a search bar
        cv2.rectangle(test_image2, (200, 100), (600, 140), (255, 255, 255), -1)  # White search bar
        cv2.rectangle(test_image2, (200, 100), (600, 140), (150, 150, 150), 1)  # Border
        
        # Draw a magnifying glass icon
        cv2.circle(test_image2, (570, 120), 15, (100, 100, 100), 1)  # Circle
        cv2.line(test_image2, (580, 130), (590, 140), (100, 100, 100), 2)  # Handle
        
        # Save the test image
        cv2.imwrite(str(image_dir / 'test_search_bar.png'), test_image2)
        
        print(f"Created test images in {image_dir.absolute()}")
    
    # Load images
    images = []
    for img_path in image_dir.glob('*.png'):
        img = cv2.imread(str(img_path))
        if img is not None:
            images.append((img_path.name, img))
            print(f"Loaded image: {img_path.name} - {img.shape}")
    
    return images

# Simulate detected elements
def get_detected_elements():
    """Return the simulated detected elements"""
    # Login form elements
    login_elements = [
        {
            'id': 'askui_button_1',
            'type': 'button',
            'rect': {'x': 100, 'y': 100, 'width': 150, 'height': 50},
            'confidence': 0.95,
            'text': 'Login',
            'source_detector': 'AskUIDetectorAdapter'
        },
        {
            'id': 'askui_input_1',
            'type': 'text_input',
            'rect': {'x': 100, 'y': 200, 'width': 300, 'height': 40},
            'confidence': 0.92,
            'text': '',
            'source_detector': 'AskUIDetectorAdapter'
        },
        {
            'id': 'askui_checkbox_1',
            'type': 'checkbox',
            'rect': {'x': 100, 'y': 300, 'width': 20, 'height': 20},
            'confidence': 0.90,
            'text': 'Remember me',
            'source_detector': 'AskUIDetectorAdapter'
        }
    ]
    
    # Search bar elements
    search_elements = [
        {
            'id': 'askui_input_2',
            'type': 'text_input',
            'rect': {'x': 200, 'y': 100, 'width': 400, 'height': 40},
            'confidence': 0.94,
            'text': '',
            'source_detector': 'AskUIDetectorAdapter'
        },
        {
            'id': 'askui_icon_1',
            'type': 'icon',
            'rect': {'x': 570, 'y': 105, 'width': 30, 'height': 30},
            'confidence': 0.89,
            'text': '',
            'source_detector': 'AskUIDetectorAdapter'
        }
    ]
    
    return {
        'test_login_form.png': login_elements,
        'test_search_bar.png': search_elements
    }

# Visualize detections
def visualize_detections(images, detections):
    """Create visualization of the detections"""
    output_dir = Path('detection_visualization')
    os.makedirs(output_dir, exist_ok=True)
    
    # Color mapping for element types
    color_map = {
        'button': (0, 0, 255),       # Red
        'text_input': (0, 255, 0),    # Green
        'checkbox': (255, 0, 0),      # Blue
        'icon': (255, 255, 0),        # Cyan
        'unknown': (128, 128, 128)    # Gray
    }
    
    for img_name, img in images:
        # Make a copy to draw on
        visualization = img.copy()
        
        # Get detections for this image
        elements = detections.get(img_name, [])
        
        # Draw bounding boxes and labels
        for elem in elements:
            rect = elem.get('rect', {})
            x = rect.get('x', 0)
            y = rect.get('y', 0)
            width = rect.get('width', 0)
            height = rect.get('height', 0)
            
            elem_type = elem.get('type', 'unknown')
            confidence = elem.get('confidence', 0.0)
            detector = elem.get('source_detector', 'unknown')
            
            # Get color for this element type
            color = color_map.get(elem_type, color_map['unknown'])
            
            # Draw rectangle
            cv2.rectangle(visualization, (x, y), (x + width, y + height), color, 2)
            
            # Draw label
            label = f"{elem_type} ({confidence:.2f})"
            cv2.putText(visualization, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw source detector info
            cv2.putText(visualization, detector, (x, y + height + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Add title to the image
        title = f"Detected UI Elements - {len(elements)} found"
        cv2.putText(visualization, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Save visualization
        output_path = output_dir / f"detection_{img_name}"
        cv2.imwrite(str(output_path), visualization)
        print(f"Created visualization at {output_path}")

# Main function
def main():
    # Load test images
    images = load_test_image()
    
    # Get detections
    detections = get_detected_elements()
    
    # Create visualizations
    visualize_detections(images, detections)
    
    print("\nDetection visualization complete!")
    print(f"Results saved in {Path('detection_visualization').absolute()}")

if __name__ == "__main__":
    main()
