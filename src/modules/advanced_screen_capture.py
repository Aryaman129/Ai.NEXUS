"""
Advanced Screen Capture Module for NEXUS
Provides high-performance screen capture using python-mss
"""
import os
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from PIL import Image
import time

logger = logging.getLogger(__name__)

class AdvancedScreenCapture:
    """
    High-performance screen capture using python-mss
    
    This module provides faster and more reliable screen capture compared
    to standard PyAutoGUI/PIL methods, especially for larger displays or
    when frequent captures are needed.
    """
    
    def __init__(self):
        """Initialize the advanced screen capture module"""
        self.mss_available = False
        self.sct = None
        self._initialize_mss()
        
        # Performance tracking
        self.total_capture_time = 0
        self.capture_count = 0
        
        logger.info(f"Advanced Screen Capture initialized (MSS available: {self.mss_available})")
    
    def _initialize_mss(self):
        """Initialize MSS (Multi-platform Screen Shot) module if available"""
        try:
            import mss
            self.mss_available = True
            self.sct = mss.mss()
            logger.info("MSS screen capture initialized successfully")
        except ImportError:
            logger.info("MSS not available, will use fallback screen capture methods")
            self.mss_available = False
    
    def capture_screen(self, region: Optional[Tuple[int, int, int, int]] = None) -> Image.Image:
        """
        Capture the screen or a region of the screen
        
        Args:
            region: Optional tuple (left, top, width, height) to capture specific region
                  If None, captures the entire screen
        
        Returns:
            PIL Image of the captured screen/region
        """
        start_time = time.time()
        
        try:
            if self.mss_available and self.sct:
                # MSS-based capture (faster)
                if region:
                    left, top, width, height = region
                    monitor = {"left": left, "top": top, "width": width, "height": height}
                else:
                    # Capture primary monitor
                    monitor = self.sct.monitors[1]  # 0 is all monitors combined, 1 is primary
                
                # Capture and convert to PIL
                sct_img = self.sct.grab(monitor)
                img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            else:
                # Fallback to PyAutoGUI
                import pyautogui
                
                if region:
                    left, top, width, height = region
                    img = pyautogui.screenshot(region=(left, top, width, height))
                else:
                    img = pyautogui.screenshot()
        
        except Exception as e:
            logger.error(f"Error capturing screen: {e}")
            # Last resort fallback
            import pyautogui
            img = pyautogui.screenshot(region=region)
        
        # Update performance tracking
        capture_time = time.time() - start_time
        self.total_capture_time += capture_time
        self.capture_count += 1
        avg_time = self.total_capture_time / self.capture_count
        
        if self.capture_count % 10 == 0:
            logger.debug(f"Average screen capture time: {avg_time:.4f}s (MSS: {self.mss_available})")
        
        return img
    
    def capture_to_array(self, region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        Capture the screen directly to a numpy array
        
        Args:
            region: Optional tuple (left, top, width, height) to capture specific region
        
        Returns:
            Numpy array of the captured screen
        """
        if self.mss_available and self.sct:
            try:
                # MSS direct to numpy (faster than going through PIL)
                if region:
                    left, top, width, height = region
                    monitor = {"left": left, "top": top, "width": width, "height": height}
                else:
                    # Capture primary monitor
                    monitor = self.sct.monitors[1]
                
                # Grab and convert to numpy
                sct_img = self.sct.grab(monitor)
                img_array = np.array(sct_img)
                return img_array[:, :, :3]  # Drop alpha channel
                
            except Exception as e:
                logger.error(f"Error capturing screen to array: {e}")
        
        # Fallback: capture to PIL and convert
        pil_img = self.capture_screen(region)
        return np.array(pil_img)
    
    def capture_multi_monitor(self) -> Dict[int, Image.Image]:
        """
        Capture all monitors
        
        Returns:
            Dictionary of monitor index to PIL Image
        """
        if not self.mss_available or not self.sct:
            logger.warning("Multi-monitor capture requires MSS")
            return {0: self.capture_screen()}
        
        try:
            # Get monitor information
            monitors = self.sct.monitors[1:]  # Skip the combined one
            
            # Capture each monitor
            captures = {}
            for i, monitor in enumerate(monitors, 1):
                sct_img = self.sct.grab(monitor)
                img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                captures[i] = img
            
            return captures
            
        except Exception as e:
            logger.error(f"Error in multi-monitor capture: {e}")
            return {0: self.capture_screen()}
    
    def get_monitor_info(self) -> List[Dict[str, Any]]:
        """
        Get information about available monitors
        
        Returns:
            List of monitor information dictionaries
        """
        if not self.mss_available or not self.sct:
            # Fallback using PyAutoGUI
            import pyautogui
            screen_size = pyautogui.size()
            return [{
                "index": 1,
                "left": 0,
                "top": 0,
                "width": screen_size[0],
                "height": screen_size[1],
                "is_primary": True
            }]
        
        try:
            # Get monitor information from MSS
            monitors_info = []
            
            for i, monitor in enumerate(self.sct.monitors[1:], 1):  # Skip the combined one
                monitor_info = {
                    "index": i,
                    "left": monitor["left"],
                    "top": monitor["top"],
                    "width": monitor["width"],
                    "height": monitor["height"],
                    "is_primary": (i == 1)  # Assume the first monitor is primary
                }
                monitors_info.append(monitor_info)
            
            return monitors_info
            
        except Exception as e:
            logger.error(f"Error getting monitor info: {e}")
            # Fallback
            import pyautogui
            screen_size = pyautogui.size()
            return [{
                "index": 1,
                "left": 0,
                "top": 0,
                "width": screen_size[0],
                "height": screen_size[1],
                "is_primary": True
            }]
