"""
Mouse Controller for NEXUS

This module provides safe mouse automation capabilities with natural movement
patterns and safety boundaries to prevent unintended actions.

Key features:
- Natural movement with acceleration/deceleration
- Safety boundaries to prevent dangerous actions
- Configurable movement speeds and click patterns
- Learning from user interaction patterns
"""
import os
import time
import random
import logging
import math
from typing import Dict, List, Tuple, Optional, Union, Callable
import pyautogui
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

# Safety first: enable pyautogui failsafe
pyautogui.FAILSAFE = True

class MouseController:
    """
    Mouse controller with safe automation and natural movement
    
    This class provides methods for controlling the mouse with natural-looking
    movements and built-in safety measures to prevent dangerous actions.
    """
    
    def __init__(self, safety_manager=None, config: Optional[Dict] = None):
        """
        Initialize the mouse controller
        
        Args:
            safety_manager: Optional safety manager instance
            config: Optional configuration dictionary with the following keys:
                - movement_speed: Base movement speed (pixels/second)
                - click_delay: Delay between clicks (seconds)
                - double_click_interval: Interval for double clicks (seconds)
                - natural_movement: Whether to use natural movement
                - safe_zones: List of safe zones to allow interaction
                - danger_zones: List of dangerous zones to prevent interaction
        """
        self.config = config or {}
        
        # Set default configuration
        self.movement_speed = self.config.get("movement_speed", 1000)  # pixels/second
        self.click_delay = self.config.get("click_delay", 0.1)  # seconds
        self.double_click_interval = self.config.get("double_click_interval", 0.2)  # seconds
        self.natural_movement = self.config.get("natural_movement", True)
        
        # Safety zones
        self.safe_zones = self.config.get("safe_zones", [])
        self.danger_zones = self.config.get("danger_zones", [])
        
        # Link to safety manager if provided
        self.safety_manager = safety_manager
        
        # Get screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Set default danger zones if none provided
        if not self.danger_zones:
            # Default danger zones: close buttons in corners
            corner_size = 50
            self.danger_zones = [
                # Top-right corner (window close buttons)
                (self.screen_width - corner_size, 0, self.screen_width, corner_size),
                # Browser tab close buttons area
                (self.screen_width - 200, 0, self.screen_width, 40)
            ]
            
        # Store current position
        self.current_x, self.current_y = pyautogui.position()
        
        # Performance tracking
        self.movement_times = []
        
        logger.info(f"MouseController initialized with screen size: {self.screen_width}x{self.screen_height}")
    
    def move_to(self, x: int, y: int, 
               speed_factor: float = 1.0, 
               natural: Optional[bool] = None,
               safety_override: bool = False) -> Dict:
        """
        Move the mouse cursor to the specified coordinates
        
        Args:
            x: X coordinate to move to
            y: Y coordinate to move to
            speed_factor: Multiplier for movement speed (1.0 = normal)
            natural: Whether to use natural movement (None = use default)
            safety_override: Whether to override safety checks
            
        Returns:
            Result dictionary with success status and details
        """
        start_time = time.time()
        
        # Get current position
        start_x, start_y = pyautogui.position()
        
        # Set natural movement flag
        use_natural = natural if natural is not None else self.natural_movement
        
        # Check safety boundaries
        if not safety_override and not self._is_safe_location(x, y):
            logger.warning(f"Attempted to move to unsafe location: ({x}, {y})")
            return {
                "success": False,
                "reason": "unsafe_region",
                "details": "Target coordinates are in a dangerous zone",
                "attempted_coords": (x, y)
            }
            
        # Clamp coordinates to screen boundaries
        x = max(0, min(x, self.screen_width - 1))
        y = max(0, min(y, self.screen_height - 1))
        
        try:
            if use_natural:
                # Use natural movement with bezier curve
                self._natural_move(start_x, start_y, x, y, speed_factor)
            else:
                # Use direct movement
                pyautogui.moveTo(x, y, duration=self._calculate_duration(start_x, start_y, x, y, speed_factor))
                
            # Update current position
            self.current_x, self.current_y = pyautogui.position()
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            self._update_performance_metrics(elapsed_time)
            
            return {
                "success": True,
                "elapsed_time": elapsed_time,
                "distance": self._calculate_distance(start_x, start_y, x, y),
                "coords": (self.current_x, self.current_y)
            }
            
        except Exception as e:
            logger.error(f"Error moving mouse: {e}")
            return {
                "success": False,
                "reason": "error",
                "details": str(e),
                "attempted_coords": (x, y)
            }
    
    def click(self, x: Optional[int] = None, y: Optional[int] = None, 
             button: str = "left", clicks: int = 1, 
             interval: Optional[float] = None,
             safety_override: bool = False) -> Dict:
        """
        Click at the specified coordinates or current position
        
        Args:
            x: Optional X coordinate to click at
            y: Optional Y coordinate to click at
            button: Mouse button to click ("left", "right", "middle")
            clicks: Number of clicks
            interval: Interval between clicks (None = use default)
            safety_override: Whether to override safety checks
            
        Returns:
            Result dictionary with success status and details
        """
        start_time = time.time()
        
        # Move to position if specified
        if x is not None and y is not None:
            move_result = self.move_to(x, y, safety_override=safety_override)
            if not move_result["success"]:
                return move_result
                
        # Get current position for logging
        current_x, current_y = pyautogui.position()
        
        # Check safety again at current position
        if not safety_override and not self._is_safe_location(current_x, current_y):
            logger.warning(f"Attempted to click at unsafe location: ({current_x}, {current_y})")
            return {
                "success": False,
                "reason": "unsafe_region",
                "details": "Click position is in a dangerous zone",
                "attempted_coords": (current_x, current_y)
            }
            
        try:
            # Set interval
            click_interval = interval if interval is not None else self.double_click_interval
            
            # Execute click(s)
            if clicks == 1:
                pyautogui.click(button=button)
            else:
                for i in range(clicks):
                    pyautogui.click(button=button)
                    if i < clicks - 1:
                        time.sleep(click_interval)
                        
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            
            return {
                "success": True,
                "elapsed_time": elapsed_time,
                "clicks": clicks,
                "button": button,
                "coords": (current_x, current_y)
            }
            
        except Exception as e:
            logger.error(f"Error clicking mouse: {e}")
            return {
                "success": False,
                "reason": "error",
                "details": str(e),
                "attempted_coords": (current_x, current_y)
            }
    
    def double_click(self, x: Optional[int] = None, y: Optional[int] = None, 
                    button: str = "left",
                    safety_override: bool = False) -> Dict:
        """
        Double-click at the specified coordinates or current position
        
        Args:
            x: Optional X coordinate to click at
            y: Optional Y coordinate to click at
            button: Mouse button to click ("left", "right", "middle")
            safety_override: Whether to override safety checks
            
        Returns:
            Result dictionary with success status and details
        """
        return self.click(x, y, button, clicks=2, safety_override=safety_override)
    
    def right_click(self, x: Optional[int] = None, y: Optional[int] = None,
                   safety_override: bool = False) -> Dict:
        """
        Right-click at the specified coordinates or current position
        
        Args:
            x: Optional X coordinate to click at
            y: Optional Y coordinate to click at
            safety_override: Whether to override safety checks
            
        Returns:
            Result dictionary with success status and details
        """
        return self.click(x, y, button="right", safety_override=safety_override)
    
    def drag_to(self, x: int, y: int, 
               button: str = "left",
               speed_factor: float = 0.7,
               safety_override: bool = False) -> Dict:
        """
        Drag from current position to the specified coordinates
        
        Args:
            x: X coordinate to drag to
            y: Y coordinate to drag to
            button: Mouse button to use for dragging
            speed_factor: Multiplier for movement speed (1.0 = normal)
            safety_override: Whether to override safety checks
            
        Returns:
            Result dictionary with success status and details
        """
        start_time = time.time()
        
        # Get current position
        start_x, start_y = pyautogui.position()
        
        # Check safety boundaries
        if not safety_override and not self._is_safe_location(x, y):
            logger.warning(f"Attempted to drag to unsafe location: ({x}, {y})")
            return {
                "success": False,
                "reason": "unsafe_region",
                "details": "Target coordinates are in a dangerous zone",
                "attempted_coords": (x, y)
            }
            
        # Clamp coordinates to screen boundaries
        x = max(0, min(x, self.screen_width - 1))
        y = max(0, min(y, self.screen_height - 1))
        
        try:
            # Execute drag operation
            duration = self._calculate_duration(start_x, start_y, x, y, speed_factor)
            pyautogui.dragTo(x, y, duration=duration, button=button)
            
            # Update current position
            self.current_x, self.current_y = pyautogui.position()
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            
            return {
                "success": True,
                "elapsed_time": elapsed_time,
                "distance": self._calculate_distance(start_x, start_y, x, y),
                "coords": (self.current_x, self.current_y)
            }
            
        except Exception as e:
            logger.error(f"Error dragging mouse: {e}")
            return {
                "success": False,
                "reason": "error",
                "details": str(e),
                "attempted_coords": (x, y)
            }
    
    def scroll(self, clicks: int, x: Optional[int] = None, y: Optional[int] = None) -> Dict:
        """
        Scroll the mouse wheel
        
        Args:
            clicks: Number of clicks to scroll (positive = up, negative = down)
            x: Optional X coordinate to scroll at
            y: Optional Y coordinate to scroll at
            
        Returns:
            Result dictionary with success status and details
        """
        start_time = time.time()
        
        # Move to position if specified
        if x is not None and y is not None:
            move_result = self.move_to(x, y)
            if not move_result["success"]:
                return move_result
                
        try:
            # Execute scroll
            pyautogui.scroll(clicks)
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            
            return {
                "success": True,
                "elapsed_time": elapsed_time,
                "scroll_amount": clicks
            }
            
        except Exception as e:
            logger.error(f"Error scrolling: {e}")
            return {
                "success": False,
                "reason": "error",
                "details": str(e)
            }
    
    def get_current_position(self) -> Tuple[int, int]:
        """
        Get the current mouse position
        
        Returns:
            Tuple of (x, y) coordinates
        """
        self.current_x, self.current_y = pyautogui.position()
        return (self.current_x, self.current_y)
    
    def add_safe_zone(self, left: int, top: int, right: int, bottom: int):
        """
        Add a safe zone where interaction is always allowed
        
        Args:
            left: Left coordinate of zone
            top: Top coordinate of zone
            right: Right coordinate of zone
            bottom: Bottom coordinate of zone
        """
        self.safe_zones.append((left, top, right, bottom))
        
    def add_danger_zone(self, left: int, top: int, right: int, bottom: int):
        """
        Add a danger zone where interaction is prevented
        
        Args:
            left: Left coordinate of zone
            top: Top coordinate of zone
            right: Right coordinate of zone
            bottom: Bottom coordinate of zone
        """
        self.danger_zones.append((left, top, right, bottom))
        
    def clear_zones(self):
        """Clear all safe and danger zones"""
        self.safe_zones = []
        self.danger_zones = []
        
    def _is_safe_location(self, x: int, y: int) -> bool:
        """
        Check if the specified coordinates are safe to interact with
        
        Args:
            x: X coordinate to check
            y: Y coordinate to check
            
        Returns:
            True if location is safe, False otherwise
        """
        # Check with safety manager if available
        if self.safety_manager and hasattr(self.safety_manager, 'is_safe_location'):
            return self.safety_manager.is_safe_location(x, y)
            
        # Check if in any safe zones (these override danger zones)
        for left, top, right, bottom in self.safe_zones:
            if left <= x <= right and top <= y <= bottom:
                return True
                
        # Check if in any danger zones
        for left, top, right, bottom in self.danger_zones:
            if left <= x <= right and top <= y <= bottom:
                return False
                
        # If not in any explicit zone, consider it safe
        return True
    
    def _calculate_distance(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """
        Calculate the distance between two points
        
        Args:
            x1: Starting X coordinate
            y1: Starting Y coordinate
            x2: Ending X coordinate
            y2: Ending Y coordinate
            
        Returns:
            Distance in pixels
        """
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    def _calculate_duration(self, x1: int, y1: int, x2: int, y2: int, speed_factor: float = 1.0) -> float:
        """
        Calculate the duration for a mouse movement
        
        Args:
            x1: Starting X coordinate
            y1: Starting Y coordinate
            x2: Ending X coordinate
            y2: Ending Y coordinate
            speed_factor: Multiplier for movement speed (1.0 = normal)
            
        Returns:
            Duration in seconds
        """
        distance = self._calculate_distance(x1, y1, x2, y2)
        base_duration = distance / self.movement_speed
        adjusted_duration = base_duration / speed_factor
        
        # Ensure minimum duration for very short movements
        return max(0.05, min(2.0, adjusted_duration))
    
    def _natural_move(self, x1: int, y1: int, x2: int, y2: int, speed_factor: float = 1.0):
        """
        Move the mouse cursor with natural-looking movement
        
        Args:
            x1: Starting X coordinate
            y1: Starting Y coordinate
            x2: Ending X coordinate
            y2: Ending Y coordinate
            speed_factor: Multiplier for movement speed (1.0 = normal)
        """
        # Calculate base duration
        duration = self._calculate_duration(x1, y1, x2, y2, speed_factor)
        
        # Calculate number of steps based on distance and speed
        distance = self._calculate_distance(x1, y1, x2, y2)
        steps = int(min(50, max(10, distance / 20)))
        
        # Generate a Bezier curve for natural movement
        points = self._generate_bezier_curve(x1, y1, x2, y2, steps)
        
        # Calculate time per step
        step_time = duration / len(points)
        
        # Execute movement along the curve
        last_time = time.time()
        for i, (x, y) in enumerate(points):
            # Move to point
            pyautogui.moveTo(x, y)
            
            # Calculate sleep time to maintain consistent speed
            elapsed = time.time() - last_time
            sleep_time = max(0, step_time - elapsed)
            
            # Sleep if not the last point
            if i < len(points) - 1:
                time.sleep(sleep_time)
                
            last_time = time.time()
    
    def _generate_bezier_curve(self, x1: int, y1: int, x2: int, y2: int, steps: int) -> List[Tuple[int, int]]:
        """
        Generate a Bezier curve for natural mouse movement
        
        Args:
            x1: Starting X coordinate
            y1: Starting Y coordinate
            x2: Ending X coordinate
            y2: Ending Y coordinate
            steps: Number of points to generate
            
        Returns:
            List of (x, y) coordinates along the curve
        """
        # Calculate distance and midpoint
        distance = self._calculate_distance(x1, y1, x2, y2)
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        # Add randomness to control points based on distance
        control_range = min(100, distance * 0.4)
        
        # Generate control points with random offsets
        c1_x = mid_x + random.uniform(-control_range, control_range)
        c1_y = mid_y + random.uniform(-control_range, control_range)
        
        # Ensure control points are within screen bounds
        c1_x = max(0, min(self.screen_width, c1_x))
        c1_y = max(0, min(self.screen_height, c1_y))
        
        # Generate points along the curve
        points = []
        for i in range(steps + 1):
            t = i / steps
            # Quadratic Bezier formula: B(t) = (1-t)²P₀ + 2(1-t)tP₁ + t²P₂
            x = (1 - t) ** 2 * x1 + 2 * (1 - t) * t * c1_x + t ** 2 * x2
            y = (1 - t) ** 2 * y1 + 2 * (1 - t) * t * c1_y + t ** 2 * y2
            points.append((int(x), int(y)))
            
        return points
    
    def _update_performance_metrics(self, elapsed_time: float):
        """Update performance metrics for monitoring"""
        # Keep last 30 movement times for rolling average
        self.movement_times.append(elapsed_time)
        if len(self.movement_times) > 30:
            self.movement_times.pop(0)
            
    def get_performance_stats(self) -> Dict:
        """
        Get current performance statistics
        
        Returns:
            Dictionary with performance metrics
        """
        avg_time = sum(self.movement_times) / len(self.movement_times) if self.movement_times else 0
        return {
            "avg_movement_time": avg_time,
            "movement_speed": self.movement_speed,
            "natural_movement": self.natural_movement
        }
