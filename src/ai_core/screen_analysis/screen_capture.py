"""
Screen Capture Module for NEXUS

This module provides efficient screen capture capabilities optimized for Windows
and NVIDIA GPUs. It uses DirectX-based capture for high performance and low latency.

Key features:
- DirectX (DXGI) screen capture for high performance
- GPU-accelerated image processing
- Adjustable capture rate based on system load
- Region-of-interest tracking to minimize processing
"""
import os
import time
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

# Set up logging
logger = logging.getLogger(__name__)

# Constants for capture methods
CAPTURE_METHOD_DXGI = "dxgi"  # DirectX (fastest on Windows)
CAPTURE_METHOD_GDI = "gdi"    # GDI (fallback)
CAPTURE_METHOD_PIL = "pil"    # PIL (most compatible)

class ScreenCapture:
    """
    High-performance screen capture optimized for NVIDIA GPUs on Windows
    
    This class provides methods for capturing the screen contents efficiently,
    with optimizations for real-time analysis and minimal resource usage.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the screen capture module
        
        Args:
            config: Optional configuration dictionary with the following keys:
                - capture_method: Method to use for capture (dxgi, gdi, pil)
                - target_fps: Target frames per second for capture
                - gpu_acceleration: Whether to use GPU acceleration
                - monitor_index: Index of monitor to capture (0 = primary)
        """
        self.config = config or {}
        
        # Set default configuration
        self.capture_method = self.config.get("capture_method", CAPTURE_METHOD_DXGI)
        self.target_fps = self.config.get("target_fps", 12)  # Default 12 FPS
        self.gpu_acceleration = self.config.get("gpu_acceleration", True)
        self.monitor_index = self.config.get("monitor_index", 0)  # Primary monitor
        
        # Initialize capture objects
        self._init_capture_method()
        
        # Performance tracking
        self.last_capture_time = 0
        self.frame_times = []
        self.current_fps = 0
        
        logger.info(f"ScreenCapture initialized with method: {self.capture_method}, "
                  f"target FPS: {self.target_fps}, GPU acceleration: {self.gpu_acceleration}")
    
    def _init_capture_method(self):
        """Initialize the appropriate screen capture method"""
        if self.capture_method == CAPTURE_METHOD_DXGI:
            self._init_dxgi_capture()
        elif self.capture_method == CAPTURE_METHOD_GDI:
            self._init_gdi_capture()
        else:
            self._init_pil_capture()
            
    def _init_dxgi_capture(self):
        """Initialize DirectX (DXGI) screen capture - fastest on Windows"""
        try:
            import d3dshot
            self.capture_device = d3dshot.create(capture_output="numpy")
            logger.info("Successfully initialized DXGI capture")
        except ImportError:
            logger.warning("d3dshot not available, falling back to GDI capture")
            self.capture_method = CAPTURE_METHOD_GDI
            self._init_gdi_capture()
    
    def _init_gdi_capture(self):
        """Initialize GDI screen capture - good fallback on Windows"""
        try:
            import mss
            self.capture_device = mss.mss()
            # Get monitor info
            monitors = self.capture_device.monitors
            if self.monitor_index + 1 < len(monitors):
                self.monitor = monitors[self.monitor_index + 1]  # mss uses 1-based indexing
            else:
                self.monitor = monitors[1]  # Primary monitor
            logger.info("Successfully initialized GDI capture")
        except ImportError:
            logger.warning("mss not available, falling back to PIL capture")
            self.capture_method = CAPTURE_METHOD_PIL
            self._init_pil_capture()
    
    def _init_pil_capture(self):
        """Initialize PIL screen capture - most compatible but slower"""
        try:
            from PIL import ImageGrab
            self.capture_device = ImageGrab
            logger.info("Successfully initialized PIL capture")
        except ImportError:
            logger.error("No screen capture method available")
            raise ImportError("No screen capture methods available. "
                             "Please install either 'd3dshot', 'mss', or 'Pillow'.")
                             
    def capture_screen(self, region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        Capture the screen contents
        
        Args:
            region: Optional region to capture (left, top, right, bottom)
                   If None, captures the entire screen
                   
        Returns:
            NumPy array containing the screen image (RGB format)
        """
        # Enforce FPS limit for resource management
        self._enforce_fps_limit()
        
        # Capture based on selected method
        if self.capture_method == CAPTURE_METHOD_DXGI:
            return self._capture_dxgi(region)
        elif self.capture_method == CAPTURE_METHOD_GDI:
            return self._capture_gdi(region)
        else:
            return self._capture_pil(region)
    
    def _capture_dxgi(self, region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """Capture using DirectX/DXGI"""
        start_time = time.time()
        
        if region:
            # Convert region format for d3dshot
            left, top, right, bottom = region
            width, height = right - left, bottom - top
            image = self.capture_device.screenshot(region=(left, top, right, bottom))
        else:
            image = self.capture_device.screenshot()
            
        self._update_performance_metrics(start_time)
        return image
    
    def _capture_gdi(self, region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """Capture using GDI (mss)"""
        start_time = time.time()
        
        if region:
            # Convert region format for mss
            left, top, right, bottom = region
            width, height = right - left, bottom - top
            monitor = {"left": left, "top": top, "width": width, "height": height}
            sct_img = self.capture_device.grab(monitor)
        else:
            sct_img = self.capture_device.grab(self.monitor)
            
        # Convert to numpy array in RGB format
        image = np.array(sct_img)
        if image.shape[2] == 4:  # BGRA to RGB
            image = image[:, :, [2, 1, 0]]
            
        self._update_performance_metrics(start_time)
        return image
    
    def _capture_pil(self, region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """Capture using PIL"""
        start_time = time.time()
        
        if region:
            image = np.array(self.capture_device.grab(region))
        else:
            image = np.array(self.capture_device.grab())
            
        # Convert RGBA to RGB if needed
        if image.shape[2] == 4:
            image = image[:, :, :3]
            
        self._update_performance_metrics(start_time)
        return image
    
    def _enforce_fps_limit(self):
        """Enforce FPS limit to avoid excessive resource usage"""
        current_time = time.time()
        elapsed = current_time - self.last_capture_time
        target_frame_time = 1.0 / self.target_fps
        
        if elapsed < target_frame_time:
            # Sleep to maintain target FPS
            sleep_time = target_frame_time - elapsed
            time.sleep(sleep_time)
            
        self.last_capture_time = time.time()
    
    def _update_performance_metrics(self, start_time: float):
        """Update performance metrics for monitoring"""
        elapsed = time.time() - start_time
        
        # Keep last 30 frame times for rolling average
        self.frame_times.append(elapsed)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
            
        # Calculate current FPS
        if self.frame_times:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            self.current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
    
    def get_performance_stats(self) -> Dict:
        """
        Get current performance statistics
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            "fps": self.current_fps,
            "avg_frame_time": sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0,
            "capture_method": self.capture_method,
            "target_fps": self.target_fps
        }
        
    def set_target_fps(self, fps: int):
        """
        Set the target FPS for capture
        
        Args:
            fps: Target frames per second
        """
        self.target_fps = max(1, min(60, fps))  # Clamp between 1-60 FPS
        logger.info(f"Target FPS set to {self.target_fps}")
        
    def set_monitor(self, monitor_index: int):
        """
        Set the monitor to capture
        
        Args:
            monitor_index: Index of monitor (0 = primary)
        """
        self.monitor_index = monitor_index
        # Reinitialize capture to apply the change
        self._init_capture_method()
        
    def set_capture_method(self, method: str):
        """
        Set the capture method
        
        Args:
            method: One of 'dxgi', 'gdi', or 'pil'
        """
        if method in (CAPTURE_METHOD_DXGI, CAPTURE_METHOD_GDI, CAPTURE_METHOD_PIL):
            self.capture_method = method
            self._init_capture_method()
        else:
            logger.warning(f"Unknown capture method: {method}")
            
    def get_available_monitors(self) -> List[Dict]:
        """
        Get information about available monitors
        
        Returns:
            List of monitor information dictionaries
        """
        monitors = []
        
        if self.capture_method == CAPTURE_METHOD_DXGI:
            # For d3dshot
            for i, device in enumerate(self.capture_device.displays):
                monitors.append({
                    "index": i,
                    "name": f"Display {i}",
                    "resolution": (device.resolution[0], device.resolution[1]),
                    "is_primary": device.is_primary
                })
        elif self.capture_method == CAPTURE_METHOD_GDI:
            # For mss
            for i, monitor in enumerate(self.capture_device.monitors[1:], 0):
                monitors.append({
                    "index": i,
                    "name": f"Display {i}",
                    "resolution": (monitor["width"], monitor["height"]),
                    "is_primary": i == 0
                })
        else:
            # For PIL (limited information)
            try:
                from PIL import ImageGrab
                img = ImageGrab.grab()
                monitors.append({
                    "index": 0,
                    "name": "Primary Display",
                    "resolution": img.size,
                    "is_primary": True
                })
            except:
                pass
                
        return monitors
