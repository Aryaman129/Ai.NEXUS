"""
Unit tests for screen capture module

These tests verify that the screen capture functionality works correctly
with appropriate performance characteristics for real-time monitoring.
"""
import os
import time
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from src.ai_core.screen_analysis.screen_capture import ScreenCapture, CAPTURE_METHOD_PIL

# Mock for pyautogui.size()
def mock_screen_size():
    return (1920, 1080)

# Mock for PIL ImageGrab
class MockImageGrab:
    @staticmethod
    def grab(bbox=None):
        """Return a mock PIL image"""
        from PIL import Image
        import numpy as np
        
        if bbox:
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
        else:
            width, height = mock_screen_size()
            
        # Create a gradient image for testing
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                arr[y, x, 0] = x % 256  # R
                arr[y, x, 1] = y % 256  # G
                arr[y, x, 2] = (x + y) % 256  # B
                
        return Image.fromarray(arr)

# Mock for d3dshot
class MockD3DShot:
    @staticmethod
    def create(capture_output=None):
        return MockD3DShotInstance()
        
class MockD3DShotInstance:
    def screenshot(self, region=None):
        """Return a mock numpy array"""
        if region:
            left, top, right, bottom = region
            width = right - left
            height = bottom - top
        else:
            width, height = mock_screen_size()
            
        # Create a gradient image for testing
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                arr[y, x, 0] = x % 256  # R
                arr[y, x, 1] = y % 256  # G
                arr[y, x, 2] = (x + y) % 256  # B
                
        return arr
        
    @property
    def displays(self):
        class Display:
            def __init__(self, id, is_primary=False):
                self.id = id
                self.is_primary = is_primary
                self.resolution = (1920, 1080)
                
        return [Display(0, True), Display(1, False)]

# Mock for mss
class MockMSS:
    def __init__(self):
        self.monitors = [
            {"left": 0, "top": 0, "width": 1920, "height": 1080, "name": "Monitor 1"},
            {"left": 0, "top": 0, "width": 1920, "height": 1080, "name": "Monitor 1"},
        ]
    
    def grab(self, monitor):
        """Return a mock mss screenshot"""
        width = monitor["width"]
        height = monitor["height"]
        
        # Create a mock mss screenshot object
        class MockShot:
            def __init__(self, width, height):
                self.rgb = np.zeros((height, width, 3), dtype=np.uint8)
                for y in range(height):
                    for x in range(width):
                        self.rgb[y, x, 0] = x % 256  # R
                        self.rgb[y, x, 1] = y % 256  # G
                        self.rgb[y, x, 2] = (x + y) % 256  # B
                        
            def __array__(self):
                return self.rgb
                
        return MockShot(width, height)


@pytest.fixture
def capture():
    """Create a ScreenCapture instance for testing"""
    # Force PIL method for testing since it requires fewer dependencies
    with patch('src.ai_core.screen_analysis.screen_capture.pyautogui') as mock_pyautogui:
        mock_pyautogui.size.return_value = mock_screen_size()
        
        with patch('PIL.ImageGrab', MockImageGrab):
            # Create capture with PIL method
            capture = ScreenCapture(config={"capture_method": CAPTURE_METHOD_PIL})
            
            yield capture


def test_screen_capture_initialization():
    """Test screen capture initialization with different methods"""
    # Test with DXGI method
    with patch('src.ai_core.screen_analysis.screen_capture.pyautogui') as mock_pyautogui:
        mock_pyautogui.size.return_value = mock_screen_size()
        
        with patch('d3dshot.create', MockD3DShot.create):
            capture = ScreenCapture(config={"capture_method": "dxgi"})
            assert capture.capture_method == "dxgi"
            
    # Test with GDI method
    with patch('src.ai_core.screen_analysis.screen_capture.pyautogui') as mock_pyautogui:
        mock_pyautogui.size.return_value = mock_screen_size()
        
        with patch('mss.mss', MockMSS):
            capture = ScreenCapture(config={"capture_method": "gdi"})
            assert capture.capture_method == "gdi"
            
    # Test with PIL method
    with patch('src.ai_core.screen_analysis.screen_capture.pyautogui') as mock_pyautogui:
        mock_pyautogui.size.return_value = mock_screen_size()
        
        with patch('PIL.ImageGrab', MockImageGrab):
            capture = ScreenCapture(config={"capture_method": "pil"})
            assert capture.capture_method == "pil"


def test_capture_screen_full(capture):
    """Test capturing the full screen"""
    # Capture the screen
    image = capture.capture_screen()
    
    # Check image dimensions
    assert image.shape[0] == 1080  # Height
    assert image.shape[1] == 1920  # Width
    assert image.shape[2] == 3     # RGB channels
    
    # Check that image contains data (not all zeros)
    assert np.mean(image) > 0


def test_capture_screen_region(capture):
    """Test capturing a specific region of the screen"""
    # Define region (left, top, right, bottom)
    region = (100, 100, 300, 200)
    
    # Capture the region
    image = capture.capture_screen(region)
    
    # Check image dimensions
    assert image.shape[0] == 100  # Height (200 - 100)
    assert image.shape[1] == 200  # Width (300 - 100)
    assert image.shape[2] == 3    # RGB channels
    
    # Check that image contains data (not all zeros)
    assert np.mean(image) > 0


def test_fps_limiting(capture):
    """Test that FPS limiting works correctly"""
    # Set a low target FPS
    capture.set_target_fps(5)  # 5 FPS = 200ms between frames
    
    # First capture
    start_time = time.time()
    capture.capture_screen()
    
    # Second capture (should be rate-limited)
    capture.capture_screen()
    end_time = time.time()
    
    # Check that at least 200ms elapsed
    elapsed = end_time - start_time
    assert elapsed >= 0.19  # Allow a small margin of error
    
    # Restore normal FPS and check that it's faster
    capture.set_target_fps(30)  # 30 FPS
    
    start_time = time.time()
    capture.capture_screen()
    capture.capture_screen()  # Should be much faster now
    end_time = time.time()
    
    elapsed = end_time - start_time
    assert elapsed < 0.1  # Should be well under 100ms


def test_performance_metrics(capture):
    """Test performance metrics tracking"""
    # Capture several frames
    for _ in range(5):
        capture.capture_screen()
        
    # Get performance stats
    stats = capture.get_performance_stats()
    
    # Check that stats include expected fields
    assert "fps" in stats
    assert "avg_frame_time" in stats
    assert "capture_method" in stats
    assert stats["capture_method"] == "pil"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
