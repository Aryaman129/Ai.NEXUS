"""
Vision-Based UI Automation for NEXUS
Provides computer vision-based UI element detection and interaction capabilities.
"""
import logging
import asyncio
import os
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

# Vision and automation libraries
import numpy as np
import cv2
from PIL import Image, ImageGrab
import pyautogui

# Import NEXUS modules
from .enhanced_vision import EnhancedVision

logger = logging.getLogger(__name__)

class VisionUIAutomation:
    """Vision-based UI automation using computer vision and pyautogui"""
    
    def __init__(self):
        """Initialize the vision-based UI automation module"""
        self.screenshots_dir = Path("screenshots")
        self.screenshots_dir.mkdir(exist_ok=True)
        
        # Basic UI element templates
        self.templates_dir = Path("templates")
        self.templates_dir.mkdir(exist_ok=True)
        
        # Configure PyAutoGUI to be safer
        pyautogui.PAUSE = 0.5  # Add a pause between actions
        pyautogui.FAILSAFE = True  # Move mouse to corner to abort
        
        # Initialize the enhanced vision module
        self.vision = EnhancedVision()
        
        # Recent actions history for learning
        self.action_history = []
        self.max_history_size = 100
        
        logger.info("Vision UI Automation module initialized")
    
    async def initialize(self) -> bool:
        """Initialize the vision automation module"""
        try:
            logger.info("Initializing Vision UI Automation module")
            
            # Check for required dependencies
            if not self._check_dependencies():
                logger.error("Missing required dependencies")
                return False
            
            # Initialize enhanced vision
            await self.vision.initialize()
            
            logger.info("Vision UI Automation module initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Vision UI Automation: {e}")
            return False
    
    def _check_dependencies(self) -> bool:
        """Check if required dependencies are available"""
        try:
            # Check OpenCV
            cv2_version = cv2.__version__
            logger.info(f"OpenCV version: {cv2_version}")
            
            # Check PyAutoGUI
            pyautogui_version = pyautogui.__version__
            logger.info(f"PyAutoGUI version: {pyautogui_version}")
            
            # Check NumPy
            numpy_version = np.__version__
            logger.info(f"NumPy version: {numpy_version}")
            
            return True
        except Exception as e:
            logger.error(f"Dependency check failed: {e}")
            return False
    
    async def take_screenshot(self, save_path: Optional[str] = None) -> Optional[Image.Image]:
        """Take a screenshot using the enhanced vision module"""
        return await self.vision.take_screenshot(save_to_file=save_path)
    
    async def find_element_by_template(self, 
                                      screenshot: Optional[Image.Image] = None,
                                      template_path: Optional[str] = None,
                                      threshold: float = 0.7) -> Optional[Tuple[int, int, int, int]]:
        """Find an element on the screen using template matching
        
        Args:
            screenshot: Optional screenshot to search in (will take a new one if None)
            template_path: Path to the template image to find
            threshold: Matching threshold (0.0-1.0), higher is more strict
            
        Returns:
            Tuple of (x, y, width, height) if found, None otherwise
        """
        try:
            # Take screenshot if not provided
            if screenshot is None:
                screenshot = await self.take_screenshot()
                if screenshot is None:
                    return None
            
            # Convert PIL image to numpy array
            screenshot_np = np.array(screenshot)
            screenshot_np = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
            
            # Load template
            if template_path and os.path.exists(template_path):
                template = cv2.imread(template_path)
                if template is None:
                    logger.error(f"Failed to load template: {template_path}")
                    return None
                
                # Perform template matching
                result = cv2.matchTemplate(screenshot_np, template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                if max_val >= threshold:
                    # Get the position of the match
                    w, h = template.shape[1], template.shape[0]
                    x, y = max_loc
                    
                    logger.info(f"Found template match at ({x}, {y}) with confidence {max_val:.2f}")
                    return (x, y, w, h)
                else:
                    logger.info(f"No match found above threshold {threshold} (best: {max_val:.2f})")
                    return None
            else:
                logger.error(f"Template path is invalid or file doesn't exist: {template_path}")
                return None
        
        except Exception as e:
            logger.error(f"Error finding element by template: {e}")
            return None
    
    async def find_text_on_screen(self, 
                                 text: str,
                                 screenshot: Optional[Image.Image] = None) -> Optional[Tuple[int, int, int, int]]:
        """Find text on the screen using OCR
        
        Args:
            text: Text to search for
            screenshot: Optional screenshot to search in (will take a new one if None)
            
        Returns:
            Tuple of (x, y, width, height) if found, None otherwise
        """
        try:
            # Take screenshot if not provided
            if screenshot is None:
                screenshot = await self.take_screenshot()
                if screenshot is None:
                    return None
            
            # Analyze screenshot to extract text
            np_image = np.array(screenshot)
            analysis_result = await self.vision.analyze_screenshot(screenshot, analysis_types=["text"])
            
            if "text" not in analysis_result:
                logger.warning("No text analysis available in the result")
                return None
                
            text_elements = analysis_result["text"]
            
            # Look for the target text
            for element in text_elements:
                if text.lower() in element["text"].lower():
                    position = element["position"]
                    logger.info(f"Found text '{text}' at ({position['x']}, {position['y']})")
                    return (position["x"], position["y"], position["width"], position["height"])
            
            logger.info(f"Text '{text}' not found on screen")
            return None
            
        except Exception as e:
            logger.error(f"Error finding text on screen: {e}")
            return None
    
    async def find_ui_element(self,
                             element_type: str,
                             screenshot: Optional[Image.Image] = None) -> List[Tuple[int, int, int, int]]:
        """Find UI elements of a specific type
        
        Args:
            element_type: Type of UI element to find ("button", "text_field", "icon", etc.)
            screenshot: Optional screenshot to search in
            
        Returns:
            List of tuples (x, y, width, height) of matching elements
        """
        try:
            # Take screenshot if not provided
            if screenshot is None:
                screenshot = await self.take_screenshot()
                if screenshot is None:
                    return []
            
            # Analyze screenshot to detect UI elements
            analysis_result = await self.vision.analyze_screenshot(screenshot, analysis_types=["ui_elements"])
            
            if "ui_elements" not in analysis_result:
                logger.warning("No UI element analysis available in the result")
                return []
                
            ui_elements = analysis_result["ui_elements"]
            
            # Filter by element type
            matching_elements = []
            for element in ui_elements:
                if element["type"] == element_type:
                    position = element["position"]
                    matching_elements.append((
                        position["x"], 
                        position["y"], 
                        position["width"], 
                        position["height"]
                    ))
            
            logger.info(f"Found {len(matching_elements)} '{element_type}' elements")
            return matching_elements
            
        except Exception as e:
            logger.error(f"Error finding UI elements: {e}")
            return []
    
    async def click_element(self, 
                           x: int, 
                           y: int, 
                           width: Optional[int] = None, 
                           height: Optional[int] = None,
                           clicks: int = 1,
                           button: str = "left") -> bool:
        """Click on a UI element at the specified position
        
        Args:
            x: X coordinate of the element
            y: Y coordinate of the element
            width: Optional width of the element (to click in center)
            height: Optional height of the element (to click in center)
            clicks: Number of clicks (1 for single, 2 for double)
            button: Mouse button to use ("left", "right", "middle")
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Calculate center point if width and height are provided
            if width is not None and height is not None:
                center_x = x + width // 2
                center_y = y + height // 2
            else:
                center_x, center_y = x, y
            
            # Move the mouse to the position
            pyautogui.moveTo(center_x, center_y, duration=0.2)
            
            # Click
            pyautogui.click(center_x, center_y, clicks=clicks, button=button, duration=0.1)
            
            # Record the action
            self._record_action({
                "type": "click",
                "position": (center_x, center_y),
                "clicks": clicks,
                "button": button,
                "timestamp": time.time()
            })
            
            logger.info(f"Clicked at ({center_x}, {center_y}) with {button} button, {clicks} times")
            return True
            
        except Exception as e:
            logger.error(f"Error clicking element: {e}")
            return False
    
    async def type_text(self, text: str, click_first: bool = True, x: int = None, y: int = None) -> bool:
        """Type text at the current cursor position or at specified coordinates
        
        Args:
            text: Text to type
            click_first: Whether to click first before typing
            x: Optional x coordinate to click before typing
            y: Optional y coordinate to click before typing
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Click first if requested and coordinates are provided
            if click_first and x is not None and y is not None:
                await self.click_element(x, y)
                await asyncio.sleep(0.5)  # Wait for click to register
            
            # Type the text
            pyautogui.typewrite(text, interval=0.05)
            
            # Record the action
            self._record_action({
                "type": "type",
                "text": text,
                "position": (x, y) if x is not None and y is not None else None,
                "timestamp": time.time()
            })
            
            logger.info(f"Typed text: '{text}'")
            return True
            
        except Exception as e:
            logger.error(f"Error typing text: {e}")
            return False
    
    async def press_key(self, key: str) -> bool:
        """Press a specific key
        
        Args:
            key: Key to press (e.g., "enter", "tab", "esc")
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Press the key
            pyautogui.press(key)
            
            # Record the action
            self._record_action({
                "type": "keypress",
                "key": key,
                "timestamp": time.time()
            })
            
            logger.info(f"Pressed key: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Error pressing key: {e}")
            return False
    
    async def drag_and_drop(self, 
                           start_x: int, 
                           start_y: int, 
                           end_x: int, 
                           end_y: int, 
                           duration: float = 0.5) -> bool:
        """Perform a drag and drop operation
        
        Args:
            start_x: Starting X coordinate
            start_y: Starting Y coordinate
            end_x: Ending X coordinate
            end_y: Ending Y coordinate
            duration: Duration of the drag in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Move to start position
            pyautogui.moveTo(start_x, start_y, duration=0.2)
            
            # Perform drag
            pyautogui.dragTo(end_x, end_y, duration=duration, button="left")
            
            # Record the action
            self._record_action({
                "type": "drag",
                "start": (start_x, start_y),
                "end": (end_x, end_y),
                "duration": duration,
                "timestamp": time.time()
            })
            
            logger.info(f"Dragged from ({start_x}, {start_y}) to ({end_x}, {end_y})")
            return True
            
        except Exception as e:
            logger.error(f"Error performing drag and drop: {e}")
            return False
    
    async def scroll(self, clicks: int, x: Optional[int] = None, y: Optional[int] = None) -> bool:
        """Scroll up or down
        
        Args:
            clicks: Number of clicks to scroll (positive for down, negative for up)
            x: Optional X coordinate to position mouse before scrolling
            y: Optional Y coordinate to position mouse before scrolling
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Move mouse to position if specified
            if x is not None and y is not None:
                pyautogui.moveTo(x, y, duration=0.2)
            
            # Perform scroll
            pyautogui.scroll(clicks)
            
            # Record the action
            self._record_action({
                "type": "scroll",
                "clicks": clicks,
                "position": (x, y) if x is not None and y is not None else None,
                "timestamp": time.time()
            })
            
            direction = "down" if clicks < 0 else "up"
            logger.info(f"Scrolled {direction} {abs(clicks)} clicks")
            return True
            
        except Exception as e:
            logger.error(f"Error scrolling: {e}")
            return False
    
    def _record_action(self, action: Dict[str, Any]):
        """Record an action in the history"""
        self.action_history.append(action)
        
        # Trim history if too long
        if len(self.action_history) > self.max_history_size:
            self.action_history = self.action_history[-self.max_history_size:]
    
    def get_action_history(self) -> List[Dict[str, Any]]:
        """Get the action history"""
        return self.action_history
    
    async def register_with_nexus(self, tool_registry):
        """
        Register this module as a tool with the NEXUS tool registry
        
        Args:
            tool_registry: The NEXUS tool registry to register with
        """
        logger.info("Registering Vision UI Automation with NEXUS")
        
        # Register the core functions
        tool_registry.register_tool(
            "find_by_template",
            self.find_element_by_template,
            categories=["ui_automation", "computer_vision"],
            description="Find UI elements using template matching"
        )
        
        tool_registry.register_tool(
            "find_text",
            self.find_text_on_screen,
            categories=["ui_automation", "text_recognition"],
            description="Find text on the screen using OCR"
        )
        
        tool_registry.register_tool(
            "find_ui_element",
            self.find_ui_element,
            categories=["ui_automation", "element_detection"],
            description="Find UI elements of a specific type (buttons, text fields, etc.)"
        )
        
        tool_registry.register_tool(
            "click",
            self.click_element,
            categories=["ui_automation", "interaction"],
            description="Click on a UI element at the specified position"
        )
        
        tool_registry.register_tool(
            "type_text",
            self.type_text,
            categories=["ui_automation", "interaction"],
            description="Type text at the current cursor position or specified coordinates"
        )
        
        tool_registry.register_tool(
            "press_key",
            self.press_key,
            categories=["ui_automation", "interaction"],
            description="Press a specific keyboard key"
        )
        
        return True
