#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Template image generator for UI detection.
Creates common UI element templates to be used with OpenCV detector.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Template directories
TEMPLATE_DIR = Path('ui_templates')
TEMPLATE_CATEGORIES = ['buttons', 'inputs', 'checkboxes', 'icons', 'toggles']

def create_template_directories():
    """Create template directories by category"""
    for category in TEMPLATE_CATEGORIES:
        os.makedirs(TEMPLATE_DIR / category, exist_ok=True)
    print(f"Created template directories in {TEMPLATE_DIR.absolute()}")

def create_button_templates():
    """Create templates for common button styles"""
    button_dir = TEMPLATE_DIR / 'buttons'
    
    # Rectangular button with border
    rect_btn = np.ones((60, 150, 3), dtype=np.uint8) * 240
    cv2.rectangle(rect_btn, (0, 0), (149, 59), (180, 180, 180), 1)
    cv2.putText(rect_btn, 'Button', (45, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
    cv2.imwrite(str(button_dir / 'rect_button.png'), rect_btn)
    
    # Rounded rectangular button
    rounded_btn = np.ones((60, 150, 3), dtype=np.uint8) * 240
    # Drawing rounded corners manually
    cv2.rectangle(rounded_btn, (10, 0), (139, 59), (30, 144, 255), -1)  # Filled rectangle
    cv2.rectangle(rounded_btn, (0, 10), (149, 49), (30, 144, 255), -1)  # Complete the rounded effect
    # Add text
    cv2.putText(rounded_btn, 'OK', (65, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(str(button_dir / 'rounded_button.png'), rounded_btn)
    
    # Success button (green)
    success_btn = np.ones((60, 150, 3), dtype=np.uint8) * 240
    cv2.rectangle(success_btn, (0, 0), (149, 59), (46, 204, 113), -1)
    cv2.putText(success_btn, 'Submit', (45, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(str(button_dir / 'success_button.png'), success_btn)
    
    # Danger button (red)
    danger_btn = np.ones((60, 150, 3), dtype=np.uint8) * 240
    cv2.rectangle(danger_btn, (0, 0), (149, 59), (231, 76, 60), -1)
    cv2.putText(danger_btn, 'Cancel', (45, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(str(button_dir / 'danger_button.png'), danger_btn)
    
    # Icon button
    icon_btn = np.ones((60, 60, 3), dtype=np.uint8) * 240
    cv2.circle(icon_btn, (30, 30), 25, (52, 152, 219), -1)
    # Draw a simple plus icon
    cv2.line(icon_btn, (20, 30), (40, 30), (255, 255, 255), 2)
    cv2.line(icon_btn, (30, 20), (30, 40), (255, 255, 255), 2)
    cv2.imwrite(str(button_dir / 'icon_button.png'), icon_btn)
    
    print(f"Created {5} button templates")

def create_input_templates():
    """Create templates for common input fields"""
    input_dir = TEMPLATE_DIR / 'inputs'
    
    # Standard text input
    text_input = np.ones((40, 200, 3), dtype=np.uint8) * 255
    cv2.rectangle(text_input, (0, 0), (199, 39), (220, 220, 220), 1)
    cv2.imwrite(str(input_dir / 'text_input.png'), text_input)
    
    # Filled text input
    filled_input = np.ones((40, 200, 3), dtype=np.uint8) * 255
    cv2.rectangle(filled_input, (0, 0), (199, 39), (220, 220, 220), 1)
    cv2.putText(filled_input, 'Username', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
    cv2.imwrite(str(input_dir / 'filled_input.png'), filled_input)
    
    # Search input with icon
    search_input = np.ones((40, 200, 3), dtype=np.uint8) * 255
    cv2.rectangle(search_input, (0, 0), (199, 39), (220, 220, 220), 1)
    # Add a search icon
    cv2.circle(search_input, (180, 20), 8, (150, 150, 150), 1)
    cv2.line(search_input, (186, 26), (195, 35), (150, 150, 150), 1)
    cv2.imwrite(str(input_dir / 'search_input.png'), search_input)
    
    # Password input with dots
    password_input = np.ones((40, 200, 3), dtype=np.uint8) * 255
    cv2.rectangle(password_input, (0, 0), (199, 39), (220, 220, 220), 1)
    # Add dots for password
    for i in range(5):
        cv2.circle(password_input, (20 + i * 10, 20), 3, (120, 120, 120), -1)
    cv2.imwrite(str(input_dir / 'password_input.png'), password_input)
    
    # Dropdown input
    dropdown_input = np.ones((40, 200, 3), dtype=np.uint8) * 255
    cv2.rectangle(dropdown_input, (0, 0), (199, 39), (220, 220, 220), 1)
    # Add dropdown arrow
    cv2.line(dropdown_input, (180, 15), (190, 25), (150, 150, 150), 1)
    cv2.line(dropdown_input, (190, 25), (200, 15), (150, 150, 150), 1)
    cv2.imwrite(str(input_dir / 'dropdown_input.png'), dropdown_input)
    
    print(f"Created {5} input templates")

def create_checkbox_templates():
    """Create templates for checkboxes and radio buttons"""
    checkbox_dir = TEMPLATE_DIR / 'checkboxes'
    
    # Unchecked checkbox
    unchecked_box = np.ones((30, 100, 3), dtype=np.uint8) * 255
    cv2.rectangle(unchecked_box, (5, 5), (25, 25), (150, 150, 150), 1)
    cv2.putText(unchecked_box, 'Option', (35, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    cv2.imwrite(str(checkbox_dir / 'unchecked_box.png'), unchecked_box)
    
    # Checked checkbox
    checked_box = np.ones((30, 100, 3), dtype=np.uint8) * 255
    cv2.rectangle(checked_box, (5, 5), (25, 25), (150, 150, 150), 1)
    # Draw check mark
    cv2.line(checked_box, (8, 15), (14, 20), (50, 50, 50), 2)
    cv2.line(checked_box, (14, 20), (22, 8), (50, 50, 50), 2)
    cv2.putText(checked_box, 'Option', (35, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    cv2.imwrite(str(checkbox_dir / 'checked_box.png'), checked_box)
    
    # Unchecked radio button
    unchecked_radio = np.ones((30, 100, 3), dtype=np.uint8) * 255
    cv2.circle(unchecked_radio, (15, 15), 10, (150, 150, 150), 1)
    cv2.putText(unchecked_radio, 'Option', (35, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    cv2.imwrite(str(checkbox_dir / 'unchecked_radio.png'), unchecked_radio)
    
    # Checked radio button
    checked_radio = np.ones((30, 100, 3), dtype=np.uint8) * 255
    cv2.circle(checked_radio, (15, 15), 10, (150, 150, 150), 1)
    cv2.circle(checked_radio, (15, 15), 5, (100, 100, 100), -1)
    cv2.putText(checked_radio, 'Option', (35, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    cv2.imwrite(str(checkbox_dir / 'checked_radio.png'), checked_radio)
    
    print(f"Created {4} checkbox templates")

def create_icon_templates():
    """Create templates for common UI icons"""
    icon_dir = TEMPLATE_DIR / 'icons'
    
    # Search icon (magnifying glass)
    search_icon = np.ones((40, 40, 3), dtype=np.uint8) * 255
    cv2.circle(search_icon, (20, 20), 12, (100, 100, 100), 2)
    cv2.line(search_icon, (28, 28), (35, 35), (100, 100, 100), 2)
    cv2.imwrite(str(icon_dir / 'search_icon.png'), search_icon)
    
    # Close/X icon
    close_icon = np.ones((40, 40, 3), dtype=np.uint8) * 255
    cv2.line(close_icon, (10, 10), (30, 30), (231, 76, 60), 2)
    cv2.line(close_icon, (30, 10), (10, 30), (231, 76, 60), 2)
    cv2.imwrite(str(icon_dir / 'close_icon.png'), close_icon)
    
    # Menu/hamburger icon
    menu_icon = np.ones((40, 40, 3), dtype=np.uint8) * 255
    for i in range(3):
        cv2.line(menu_icon, (10, 13 + i*8), (30, 13 + i*8), (100, 100, 100), 2)
    cv2.imwrite(str(icon_dir / 'menu_icon.png'), menu_icon)
    
    # Settings/gear icon
    settings_icon = np.ones((40, 40, 3), dtype=np.uint8) * 255
    # Draw simple gear
    cv2.circle(settings_icon, (20, 20), 10, (100, 100, 100), 1)
    for i in range(6):
        angle = i * 60 * np.pi / 180
        x1 = int(20 + 10 * np.cos(angle))
        y1 = int(20 + 10 * np.sin(angle))
        x2 = int(20 + 16 * np.cos(angle))
        y2 = int(20 + 16 * np.sin(angle))
        cv2.line(settings_icon, (x1, y1), (x2, y2), (100, 100, 100), 2)
    cv2.imwrite(str(icon_dir / 'settings_icon.png'), settings_icon)
    
    print(f"Created {4} icon templates")

def create_toggle_templates():
    """Create templates for toggle switches"""
    toggle_dir = TEMPLATE_DIR / 'toggles'
    
    # Toggle switch OFF
    toggle_off = np.ones((30, 60, 3), dtype=np.uint8) * 255
    cv2.rectangle(toggle_off, (5, 5), (55, 25), (200, 200, 200), -1)
    cv2.circle(toggle_off, (15, 15), 10, (255, 255, 255), -1)
    cv2.imwrite(str(toggle_dir / 'toggle_off.png'), toggle_off)
    
    # Toggle switch ON
    toggle_on = np.ones((30, 60, 3), dtype=np.uint8) * 255
    cv2.rectangle(toggle_on, (5, 5), (55, 25), (46, 204, 113), -1)
    cv2.circle(toggle_on, (45, 15), 10, (255, 255, 255), -1)
    cv2.imwrite(str(toggle_dir / 'toggle_on.png'), toggle_on)
    
    print(f"Created {2} toggle templates")

def main():
    """Create all template images for UI detection"""
    print("Creating UI element templates for OpenCV detector...")
    
    # Create template directories
    create_template_directories()
    
    # Create templates by category
    create_button_templates()
    create_input_templates()
    create_checkbox_templates()
    create_icon_templates()
    create_toggle_templates()
    
    print("\nTemplate creation complete!")
    print(f"Created templates in: {TEMPLATE_DIR.absolute()}")
    print("These templates will enhance OpenCV-based detection of UI elements")

if __name__ == "__main__":
    main()
