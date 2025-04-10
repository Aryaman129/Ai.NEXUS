"""
Unit tests for UI element detector module

These tests verify that the UI element detection works correctly, identifying
buttons, text fields, and other UI components in screen captures.
"""
import os
import time
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import torch

from src.ai_core.screen_analysis.ui_detector import UIDetector

# Create mock images for UI elements
def create_mock_button(width=100, height=40):
    """Create a mock button image"""
    img = np.ones((height, width, 3), dtype=np.uint8) * 200  # Light gray
    # Add border
    img[0:2, :] = img[-2:, :] = img[:, 0:2] = img[:, -2:] = [100, 100, 100]  # Darker border
    return img

def create_mock_text_field(width=200, height=30):
    """Create a mock text field image"""
    img = np.ones((height, width, 3), dtype=np.uint8) * 255  # White
    # Add border
    img[0:1, :] = img[-1:, :] = img[:, 0:1] = img[:, -1:] = [180, 180, 180]  # Gray border
    return img

def create_mock_checkbox(size=24):
    """Create a mock checkbox image"""
    img = np.ones((size, size, 3), dtype=np.uint8) * 240  # Light gray
    # Add border
    img[0:1, :] = img[-1:, :] = img[:, 0:1] = img[:, -1:] = [100, 100, 100]  # Darker border
    # Add check mark
    for i in range(size//4, size*3//4):
        img[i, i] = [0, 120, 0]  # Green checkmark
        img[i, size-i-1] = [0, 120, 0]
    return img

def create_mock_screen():
    """Create a mock screen with various UI elements"""
    screen = np.ones((600, 800, 3), dtype=np.uint8) * 240  # Background
    
    # Add a button
    button = create_mock_button()
    screen[100:140, 50:150] = button
    
    # Add a text field
    text_field = create_mock_text_field()
    screen[200:230, 50:250] = text_field
    
    # Add a checkbox
    checkbox = create_mock_checkbox()
    screen[300:324, 50:74] = checkbox
    
    return screen

# Mock for YOLO model
class MockYOLO:
    def __init__(self, model_path=None):
        self.conf = 0.45
        self.iou = 0.45
        self.max_det = 100
        
    def to(self, device):
        return self
        
    def __call__(self, image, verbose=False):
        """Mock detection on an image"""
        # Create mock results based on the image
        height, width = image.shape[:2]
        
        # Create a Results object with mock detections
        results = [MockResults(image)]
        return results
        
class MockResults:
    def __init__(self, image):
        self.image = image
        height, width = image.shape[:2]
        
        # Analyze image to find potential UI elements
        self.boxes = self._detect_elements(image, height, width)
        self.names = {
            0: "button",
            1: "text_field",
            2: "checkbox",
            3: "icon",
            4: "menu",
            5: "window_control"
        }
        
    def _detect_elements(self, image, height, width):
        """Simple detection based on color patterns"""
        # This is a very simplified detection for testing
        detections = []
        
        # Look for button-like patterns (light gray rectangles with darker borders)
        # For testing, we'll detect elements at fixed positions based on create_mock_screen
        
        # Button at position (50, 100, 150, 140)
        detections.append(self._create_mock_detection(50, 100, 150, 140, 0.92, 0))  # Button
        
        # Text field at position (50, 200, 250, 230)
        detections.append(self._create_mock_detection(50, 200, 250, 230, 0.88, 1))  # Text field
        
        # Checkbox at position (50, 300, 74, 324)
        detections.append(self._create_mock_detection(50, 300, 74, 324, 0.75, 2))  # Checkbox
        
        return MockBoxes(detections)
        
    def _create_mock_detection(self, x1, y1, x2, y2, conf, class_id):
        """Create a mock detection tensor"""
        # Format: [x1, y1, x2, y2, confidence, class_id]
        return np.array([x1, y1, x2, y2, conf, class_id])

class MockBoxes:
    def __init__(self, detections):
        self.detections = detections
        
    @property
    def data(self):
        """Return detection data in tensor format"""
        # Create a mock tensor
        class MockTensor:
            def __init__(self, detections):
                self.detections = detections
                
            def cpu(self):
                return MockTensor(self.detections)
                
            def numpy(self):
                return np.array([self.detections])
                
        return MockTensor(self.detections)
        
    def __iter__(self):
        for det in self.detections:
            yield MockBox(det)
            
    def __len__(self):
        return len(self.detections)
            
class MockBox:
    def __init__(self, detection):
        self.data = MockTensor(detection)
        
    class MockTensor:
        def __init__(self, detection):
            self.detection = detection
            
        def cpu(self):
            return self
            
        def numpy(self):
            return np.array([self.detection])


@pytest.fixture
def ui_detector():
    """Create a UIDetector instance for testing"""
    with patch('ultralytics.YOLO', MockYOLO):
        with patch('torch.cuda.is_available', return_value=True):
            detector = UIDetector(config={"confidence_threshold": 0.5})
            yield detector


def test_detector_initialization():
    """Test UI detector initialization"""
    with patch('ultralytics.YOLO', MockYOLO):
        with patch('torch.cuda.is_available', return_value=True):
            detector = UIDetector(config={"confidence_threshold": 0.5})
            assert detector.confidence_threshold == 0.5
            assert detector.device.type == "cuda"
            
        with patch('torch.cuda.is_available', return_value=False):
            detector = UIDetector(config={"use_gpu": True})
            assert detector.device.type == "cpu"


def test_element_detection(ui_detector):
    """Test detection of UI elements in a mock screen"""
    # Create a mock screen with UI elements
    screen = create_mock_screen()
    
    # Detect elements
    elements = ui_detector.detect_elements(screen)
    
    # Check that elements were detected
    assert len(elements) == 3
    
    # Check that each element has the expected properties
    for element in elements:
        assert "type" in element
        assert "confidence" in element
        assert "bbox" in element
        assert "center" in element
        
    # Check types of detected elements
    element_types = [element["type"] for element in elements]
    assert "button" in element_types
    assert "text_field" in element_types
    assert "checkbox" in element_types


def test_confidence_filtering(ui_detector):
    """Test that confidence threshold filters low-confidence detections"""
    # Create a mock screen
    screen = create_mock_screen()
    
    # Set a high confidence threshold
    ui_detector.set_confidence_threshold(0.9)
    
    # Detect elements
    elements = ui_detector.detect_elements(screen)
    
    # Only button should be above 0.9 confidence
    assert len(elements) == 1
    assert elements[0]["type"] == "button"
    
    # Set a lower threshold
    ui_detector.set_confidence_threshold(0.7)
    
    # Detect elements
    elements = ui_detector.detect_elements(screen)
    
    # Button and text field should be above 0.7
    assert len(elements) == 2


def test_region_of_interest(ui_detector):
    """Test detection within a region of interest"""
    # Create a mock screen
    screen = create_mock_screen()
    
    # Define a region that includes only the button
    roi = (40, 90, 160, 150)
    
    # Detect elements in ROI
    elements = ui_detector.detect_elements(screen, roi)
    
    # Should only detect the button
    assert len(elements) == 1
    assert elements[0]["type"] == "button"
    
    # Check that coordinates are adjusted to screen coordinates
    bbox = elements[0]["bbox"]
    assert bbox[0] >= roi[0]
    assert bbox[1] >= roi[1]


def test_performance_metrics(ui_detector):
    """Test performance metrics tracking"""
    # Create a mock screen
    screen = create_mock_screen()
    
    # Detect elements multiple times
    for _ in range(5):
        ui_detector.detect_elements(screen)
        
    # Get performance stats
    stats = ui_detector.get_performance_stats()
    
    # Check that stats include expected fields
    assert "fps" in stats
    assert "avg_inference_time" in stats
    assert "device" in stats
    assert "confidence_threshold" in stats


def test_element_type_mapping(ui_detector):
    """Test mapping of class names to UI element types"""
    # Test with common COCO classes
    assert ui_detector._map_class_to_ui_element("person") == "icon"
    assert ui_detector._map_class_to_ui_element("keyboard") == "text_field"
    assert ui_detector._map_class_to_ui_element("tv") == "window"
    
    # Test with actual UI element classes
    assert ui_detector._map_class_to_ui_element("button") == "button"
    assert ui_detector._map_class_to_ui_element("checkbox") == "checkbox"
    assert ui_detector._map_class_to_ui_element("unknown_class") == "unknown"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
