"""
NEXUS Enhanced UI Intelligence Test

This script demonstrates the capabilities of the NEXUS Enhanced UI Intelligence System,
which combines computer vision, LLMs, and adaptive learning for advanced UI understanding.
"""

import os
import cv2
import numpy as np
import time
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any

# Setup path to include src
import sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import the enhanced UI intelligence system
from ai_core.screen_analysis.enhanced_ui.nexus_ui_intelligence import (
    NexusUIIntelligence, create_ui_intelligence
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create output directories
os.makedirs("test_data", exist_ok=True)
os.makedirs("test_results", exist_ok=True)
os.makedirs("data/ui_templates", exist_ok=True)
os.makedirs("data/calibration", exist_ok=True)

def create_test_ui_image() -> np.ndarray:
    """Create a test UI image with various controls"""
    # Create base image (white background)
    ui_image = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Draw a navbar at the top
    cv2.rectangle(ui_image, (0, 0), (800, 50), (50, 50, 50), -1)
    
    # Draw a logo in navbar
    cv2.circle(ui_image, (30, 25), 20, (0, 120, 215), -1)
    cv2.putText(ui_image, "N", (25, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Draw menu items in navbar
    menu_items = ["Home", "Dashboard", "Reports", "Settings"]
    for i, item in enumerate(menu_items):
        x = 100 + i * 120
        cv2.putText(ui_image, item, (x, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw a sidebar on the left
    cv2.rectangle(ui_image, (0, 50), (180, 600), (240, 240, 240), -1)
    
    # Draw sidebar items
    sidebar_items = ["Profile", "Messages", "Tasks", "Analytics", "Help"]
    for i, item in enumerate(sidebar_items):
        y = 90 + i * 50
        cv2.putText(ui_image, item, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 2)
    
    # Draw a main content area title
    cv2.putText(ui_image, "Dashboard Overview", (200, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 50, 50), 2)
    
    # Draw buttons
    button_positions = [
        ((220, 150), (350, 190), "View Reports"),
        ((370, 150), (500, 190), "Export Data"),
        ((520, 150), (650, 190), "Refresh")
    ]
    
    for (x1, y1), (x2, y2), text in button_positions:
        # Button background
        cv2.rectangle(ui_image, (x1, y1), (x2, y2), (0, 120, 215), -1)
        # Button text
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x1 + (x2 - x1 - text_size[0]) // 2
        text_y = y1 + (y2 - y1 + text_size[1]) // 2
        cv2.putText(ui_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw a search box
    cv2.rectangle(ui_image, (590, 80), (780, 110), (255, 255, 255), -1)
    cv2.rectangle(ui_image, (590, 80), (780, 110), (200, 200, 200), 1)
    cv2.putText(ui_image, "Search...", (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
    
    # Draw data cards
    card_positions = [
        ((220, 220), (380, 320), "Users", "1,254"),
        ((400, 220), (560, 320), "Revenue", "$45,678"),
        ((580, 220), (740, 320), "Orders", "267")
    ]
    
    for (x1, y1), (x2, y2), title, value in card_positions:
        # Card background
        cv2.rectangle(ui_image, (x1, y1), (x2, y2), (255, 255, 255), -1)
        cv2.rectangle(ui_image, (x1, y1), (x2, y2), (220, 220, 220), 1)
        # Card title
        cv2.putText(ui_image, title, (x1 + 15, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
        # Card value
        cv2.putText(ui_image, value, (x1 + 15, y1 + 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 50, 50), 2)
    
    # Draw a table
    table_y = 350
    # Table header
    cv2.rectangle(ui_image, (220, table_y), (740, table_y + 40), (240, 240, 240), -1)
    cv2.rectangle(ui_image, (220, table_y), (740, table_y + 40), (200, 200, 200), 1)
    headers = ["ID", "Name", "Date", "Status"]
    header_widths = [50, 170, 170, 130]
    header_x = 220
    for header, width in zip(headers, header_widths):
        cv2.putText(ui_image, header, (header_x + 10, table_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 2)
        header_x += width
    
    # Table rows
    row_data = [
        ["#1", "John Smith", "2025-04-05", "Completed"],
        ["#2", "Jane Doe", "2025-04-06", "Pending"],
        ["#3", "Bob Johnson", "2025-04-07", "Processing"]
    ]
    
    for i, row in enumerate(row_data):
        row_y = table_y + 40 + i * 40
        cv2.rectangle(ui_image, (220, row_y), (740, row_y + 40), (255, 255, 255), -1)
        cv2.rectangle(ui_image, (220, row_y), (740, row_y + 40), (220, 220, 220), 1)
        
        cell_x = 220
        for j, (cell, width) in enumerate(zip(row, header_widths)):
            # For the status column, add color coding
            if j == 3:
                status_colors = {
                    "Completed": (0, 180, 0),    # Green
                    "Pending": (0, 120, 215),    # Blue
                    "Processing": (255, 150, 0)  # Orange
                }
                text_color = status_colors.get(cell, (80, 80, 80))
            else:
                text_color = (80, 80, 80)
                
            cv2.putText(ui_image, cell, (cell_x + 10, row_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            cell_x += width
    
    # Save the image
    test_image_path = os.path.join("test_data", "test_ui.png")
    cv2.imwrite(test_image_path, ui_image)
    logger.info(f"Created test UI image: {test_image_path}")
    
    return ui_image

def capture_real_screenshot() -> Optional[np.ndarray]:
    """Capture the current screen"""
    try:
        from PIL import ImageGrab
        screenshot = ImageGrab.grab()
        screenshot_np = np.array(screenshot)
        return cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.error(f"Error capturing screenshot: {e}")
        return None

def test_basic_detection(ui_intelligence: NexusUIIntelligence, 
                        test_image: np.ndarray,
                        description: str) -> None:
    """Test basic UI element detection"""
    print(f"\n=== Testing Basic Detection on {description} ===")
    
    start_time = time.time()
    
    # Analyze the image
    analysis = ui_intelligence.analyze_screenshot(test_image)
    
    elapsed_time = time.time() - start_time
    
    # Print results
    print(f"Detection completed in {elapsed_time:.3f} seconds")
    print(f"Detected {len(analysis.elements)} UI elements")
    print(f"Overall confidence: {analysis.confidence:.2f}")
    
    # Count by element type
    element_types = {}
    for element in analysis.elements:
        element_type = element.element_type
        if element_type not in element_types:
            element_types[element_type] = 0
        element_types[element_type] += 1
    
    print("\nElement Types:")
    for element_type, count in sorted(element_types.items()):
        print(f"  {element_type}: {count}")

def test_interaction_suggestions(ui_intelligence: NexusUIIntelligence) -> None:
    """Test interaction suggestions"""
    print("\n=== Testing Interaction Suggestions ===")
    
    if not ui_intelligence.current_analysis:
        print("No current analysis available. Run detection first.")
        return
    
    # Get general interaction suggestions
    suggestions = ui_intelligence.suggest_interactions()
    
    print(f"Generated {len(suggestions)} interaction suggestions:")
    for i, suggestion in enumerate(suggestions[:5]):  # Show top 5
        element_id = suggestion.get("element_id", "")
        element = ui_intelligence.get_element_by_id(element_id)
        
        element_type = suggestion.get("element_type", "unknown")
        element_text = suggestion.get("element_text", "")
        action = suggestion.get("action", "")
        description = suggestion.get("description", "")
        
        print(f"  {i+1}. {action.upper()} {element_type} '{element_text}': {description}")

def test_semantic_understanding(ui_intelligence: NexusUIIntelligence) -> None:
    """Test semantic understanding of UI"""
    print("\n=== Testing Semantic Understanding ===")
    
    if not ui_intelligence.current_analysis:
        print("No current analysis available. Run detection first.")
        return
    
    # Look for elements with semantic attributes
    semantic_elements = []
    
    for element in ui_intelligence.current_analysis.elements:
        if hasattr(element, "attributes") and element.attributes:
            if "purpose" in element.attributes or "content_description" in element.attributes:
                semantic_elements.append(element)
    
    print(f"Found {len(semantic_elements)} elements with semantic understanding:")
    for i, element in enumerate(semantic_elements[:5]):  # Show top 5
        purpose = element.attributes.get("purpose", "")
        content = element.attributes.get("content_description", "")
        text = element.text if hasattr(element, "text") and element.text else ""
        
        print(f"  {i+1}. {element.element_type.upper()}: '{text}'")
        if purpose:
            print(f"     Purpose: {purpose}")
        if content:
            print(f"     Content: {content}")

def test_interaction_planning(ui_intelligence: NexusUIIntelligence) -> None:
    """Test interaction path planning"""
    print("\n=== Testing Interaction Planning ===")
    
    test_goals = [
        "Search for recent orders",
        "Export the data to CSV",
        "View detailed reports"
    ]
    
    for goal in test_goals:
        print(f"\nGoal: {goal}")
        interaction_path = ui_intelligence.plan_interaction(goal)
        
        if interaction_path:
            print(f"  Planned {len(interaction_path)} steps:")
            for step in interaction_path:
                element_id = step.get("element_id", "")
                element = ui_intelligence.get_element_by_id(element_id)
                element_type = element.element_type if element else "unknown"
                action = step.get("action", "")
                description = step.get("description", "")
                
                print(f"    {step.get('step', 0)}. {action.upper()} {element_type}: {description}")
        else:
            print("  No interaction path found")

def main():
    print("NEXUS Enhanced UI Intelligence Test")
    print("==================================")
    
    # Create test UI image
    print("\nCreating test UI image...")
    test_image = create_test_ui_image()
    
    # Initialize UI Intelligence system
    print("\nInitializing NEXUS UI Intelligence system...")
    ui_intelligence = create_ui_intelligence()
    
    # Test with synthetic UI
    test_basic_detection(ui_intelligence, test_image, "Synthetic UI")
    
    # Test interaction suggestions
    test_interaction_suggestions(ui_intelligence)
    
    # Test semantic understanding (if available)
    test_semantic_understanding(ui_intelligence)
    
    # Test interaction planning
    test_interaction_planning(ui_intelligence)
    
    # Test with real screenshot (if available)
    print("\nCapturing real screenshot...")
    real_screenshot = capture_real_screenshot()
    if real_screenshot is not None:
        # Save the screenshot
        screenshot_path = os.path.join("test_data", "real_screenshot.png")
        cv2.imwrite(screenshot_path, real_screenshot)
        
        # Test with real screenshot
        test_basic_detection(ui_intelligence, real_screenshot, "Real Screenshot")
    
    print("\nTest completed!")

if __name__ == "__main__":
    main()
