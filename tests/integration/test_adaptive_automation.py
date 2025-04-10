"""
Integration tests for adaptive screen automation system.

These tests verify that the complete adaptive automation system functions correctly,
including screen capture, UI detection, automation, and clarification capabilities.

The system should demonstrate learning and adaptation from interactions over time.
"""
import os
import time
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path

# Component imports
from src.ai_core.screen_analysis.screen_capture import ScreenCapture, CAPTURE_METHOD_PIL
from src.ai_core.screen_analysis.ui_detector import UIDetector
from src.ai_core.screen_analysis.visual_memory import VisualMemorySystem
from src.ai_core.automation.mouse_controller import MouseController
from src.ai_core.automation.keyboard_controller import KeyboardController
from src.ai_core.automation.safety_manager import SafetyManager
from src.ai_core.automation.clarification_engine import ClarificationEngine


class MockScreenCapture(ScreenCapture):
    """Mock screen capture for testing"""
    def __init__(self, mock_images=None):
        super().__init__(config={"capture_method": CAPTURE_METHOD_PIL})
        self.mock_images = mock_images or {}
        self.current_image = None
        
    def capture_screen(self, region=None):
        """Return a mock screen image"""
        if self.current_image is not None:
            return self.current_image
        return np.zeros((800, 1200, 3), dtype=np.uint8)
    
    def set_mock_image(self, name_or_array):
        """Set the current mock image"""
        if isinstance(name_or_array, str):
            self.current_image = self.mock_images.get(name_or_array)
        else:
            self.current_image = name_or_array


class MockUIDetector(UIDetector):
    """Mock UI detector for testing"""
    def __init__(self):
        # Don't load actual model
        self.model = None
        self.device = "cpu"
        self.class_names = ["button", "text_field", "checkbox", "dropdown", "link", "icon"]
        self.mock_detections = {}
        
    def detect_elements(self, image, region_of_interest=None):
        """Return mock detections"""
        # Use image hash as key, or return empty list
        img_hash = hash(str(image.tobytes()[:100]))
        return self.mock_detections.get(img_hash, [])
    
    def set_mock_detections(self, image, detections):
        """Set mock detections for an image"""
        img_hash = hash(str(image.tobytes()[:100]))
        self.mock_detections[img_hash] = detections


@pytest.fixture
def integration_system():
    """Set up a full integration system with mocked components"""
    # Create temp directory for memory
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    # Create mocked components
    screen_capture = MockScreenCapture()
    ui_detector = MockUIDetector()
    visual_memory = VisualMemorySystem(config={"memory_path": temp_dir})
    safety_manager = SafetyManager()
    mouse = MouseController(safety_manager=safety_manager)
    keyboard = KeyboardController(safety_manager=safety_manager)
    
    # Create clarification engine with always-yes response
    clarification_engine = ClarificationEngine(config={"memory_path": temp_dir})
    clarification_engine.set_response_callback(lambda q: "Yes")
    
    # Mock the actual automation to avoid real actions
    mouse.move_to = MagicMock(return_value={"success": True})
    mouse.click = MagicMock(return_value={"success": True})
    keyboard.type_text = MagicMock(return_value={"success": True})
    
    # Create a test image with UI elements
    ui_test_image = np.zeros((800, 1200, 3), dtype=np.uint8)
    # Add some colored regions to simulate UI elements
    ui_test_image[100:150, 200:300] = [0, 0, 255]  # Blue button
    ui_test_image[200:250, 200:500] = [0, 255, 0]  # Green text field
    
    # Set up mock detections for test image
    ui_detector.set_mock_detections(ui_test_image, [
        {
            "id": 1,
            "class": "button",
            "confidence": 0.92,
            "bbox": [200, 100, 300, 150],
            "text": "Submit",
            "center": [250, 125]
        },
        {
            "id": 2,
            "class": "text_field",
            "confidence": 0.88,
            "bbox": [200, 200, 500, 250],
            "text": "Name",
            "center": [350, 225]
        }
    ])
    
    # Add the test image to the screen capture
    screen_capture.mock_images["ui_test"] = ui_test_image
    screen_capture.set_mock_image("ui_test")
    
    # Return all components as a dict
    system = {
        "screen_capture": screen_capture,
        "ui_detector": ui_detector,
        "visual_memory": visual_memory,
        "safety_manager": safety_manager,
        "mouse": mouse,
        "keyboard": keyboard,
        "clarification_engine": clarification_engine,
        "temp_dir": temp_dir
    }
    
    yield system
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


def test_detection_to_action(integration_system):
    """Test complete flow from UI detection to action with memory and adaptation"""
    # Extract components
    screen_capture = integration_system["screen_capture"]
    ui_detector = integration_system["ui_detector"]
    visual_memory = integration_system["visual_memory"]
    mouse = integration_system["mouse"]
    keyboard = integration_system["keyboard"]
    
    # Capture screen
    screen_image = screen_capture.capture_screen()
    
    # Detect UI elements
    ui_elements = ui_detector.detect_elements(screen_image)
    
    # Should have found elements
    assert len(ui_elements) == 2
    assert ui_elements[0]["class"] == "button"
    assert ui_elements[1]["class"] == "text_field"
    
    # Create visual signatures and store in memory
    for element in ui_elements:
        # Extract element image from screen
        x1, y1, x2, y2 = element["bbox"]
        element_img = screen_image[y1:y2, x1:x2]
        
        # Create a visual signature (simplified for test)
        signature = np.resize(element_img, (64, 64))
        
        # Store in memory
        pattern = {
            "visual_signature": signature,
            "type": element["class"],
            "text": element["text"],
            "bbox_size": (x2-x1, y2-y1)
        }
        pattern_id = visual_memory.store_pattern(pattern)
        
        # Associate with the UI element
        element["memory_pattern_id"] = pattern_id
    
    # Interact with button
    button = ui_elements[0]
    x, y = button["center"]
    
    # Perform action
    mouse.move_to(x, y)
    mouse.click()
    
    # Record successful interaction
    visual_memory.record_interaction(
        pattern_id=button["memory_pattern_id"],
        action="click",
        success=True,
        context={"window_title": "Test Window"}
    )
    
    # Interact with text field
    text_field = ui_elements[1]
    x, y = text_field["center"]
    
    # Perform action
    mouse.move_to(x, y)
    mouse.click()
    keyboard.type_text("Test User")
    
    # Record successful interaction
    visual_memory.record_interaction(
        pattern_id=text_field["memory_pattern_id"],
        action="type",
        success=True,
        context={"window_title": "Test Window"}
    )
    
    # Check that interactions were recorded
    assert len(visual_memory.interaction_history) == 2
    
    # Verify memory has learned from interactions
    button_patterns = visual_memory.find_similar_patterns(
        signature=np.resize(screen_image[100:150, 200:300], (64, 64)),
        element_type="button"
    )
    assert len(button_patterns) > 0
    assert button_patterns[0]["successful_interactions"] == 1
    
    # Simulate second screen with similar elements but slightly different
    ui_test_image2 = np.zeros((800, 1200, 3), dtype=np.uint8)
    # Add similar colored regions but slightly shifted
    ui_test_image2[110:160, 210:310] = [0, 0, 255]  # Blue button (shifted)
    ui_test_image2[200:250, 200:500] = [0, 255, 0]  # Green text field (same)
    
    # Set up mock detections for second test image
    ui_detector.set_mock_detections(ui_test_image2, [
        {
            "id": 3,
            "class": "button",
            "confidence": 0.85,  # Lower confidence
            "bbox": [210, 110, 310, 160],
            "text": "Submit",
            "center": [260, 135]
        },
        {
            "id": 4,
            "class": "text_field",
            "confidence": 0.88,
            "bbox": [200, 200, 500, 250],
            "text": "Name",
            "center": [350, 225]
        }
    ])
    
    # Update screen capture
    screen_capture.set_mock_image(ui_test_image2)
    
    # Capture new screen
    screen_image2 = screen_capture.capture_screen()
    
    # Detect UI elements
    ui_elements2 = ui_detector.detect_elements(screen_image2)
    
    # Enhance detection with visual memory
    enhanced_elements = visual_memory.enhance_detection(
        layout_results={"window_title": "Test Window"},
        ui_elements=ui_elements2,
        screen_image=screen_image2
    )
    
    # Button should have enhanced confidence due to previous success
    button2 = enhanced_elements[0]
    assert "memory_match" in button2
    assert button2["memory_match"]["similarity"] > 0.5
    assert button2["memory_match"]["successful_interactions"] == 1
    
    # Original confidence was 0.85, should be higher now
    assert button2["enhanced_confidence"] > 0.85


def test_safety_and_clarification(integration_system):
    """Test safety checks and clarification process"""
    # Extract components
    safety_manager = integration_system["safety_manager"]
    clarification_engine = integration_system["clarification_engine"]
    mouse = integration_system["mouse"]
    
    # Create a dangerous action
    action = {
        "type": "mouse_click",
        "x": 10,
        "y": 10,  # Typically top-left corner is dangerous
        "context": {
            "active_application": "notepad.exe",
            "window_title": "Untitled - Notepad"
        }
    }
    
    # Add a danger zone
    safety_manager.add_danger_zone(0, 0, 50, 50)
    
    # Validate action (should be unsafe)
    validation = safety_manager.validate_action(action)
    assert validation["is_safe"] is False
    
    # Create a low confidence action
    context = {
        "element_type": "button",
        "element_text": "Delete All",
        "action": "click"
    }
    
    # Mock clarification response
    mock_callback = MagicMock(return_value="Yes, go ahead")
    clarification_engine.set_response_callback(mock_callback)
    
    # Ask for clarification
    result = clarification_engine.ask_for_clarification(
        scenario="ui_element_action", 
        context=context,
        confidence=0.5
    )
    
    # Should have asked and received approval
    assert result["proceed"] is True
    assert result["confidence"] > 0.5
    
    # Check that clarification was recorded
    assert len(clarification_engine.clarification_history) == 1
    
    # Try again with the same context
    result = clarification_engine.ask_for_clarification(
        scenario="ui_element_action", 
        context=context,
        confidence=0.6
    )
    
    # Should have higher confidence now
    assert result["confidence"] > 0.6
    
    # Change mock to reject
    mock_callback.return_value = "No, don't do that"
    
    # Ask for clarification
    result = clarification_engine.ask_for_clarification(
        scenario="ui_element_action", 
        context=context,
        confidence=0.6
    )
    
    # Should not proceed
    assert result["proceed"] is False
    assert result["confidence"] < 0.6
    
    # Get statistics
    stats = clarification_engine.get_clarification_statistics()
    assert stats["total_clarifications"] == 3
    assert "ui_element_action" in stats["most_common_scenarios"]


def test_adaptive_automation_flow():
    """Test the complete adaptive automation flow with learning over time"""
    # Create a complete system with mocked components
    with patch('src.ai_core.screen_analysis.screen_capture.ScreenCapture') as MockCapture, \
         patch('src.ai_core.screen_analysis.ui_detector.UIDetector') as MockDetector, \
         patch('src.ai_core.automation.mouse_controller.MouseController') as MockMouse, \
         patch('src.ai_core.automation.keyboard_controller.KeyboardController') as MockKeyboard:
        
        # Set up the mocks
        mock_capture = MockCapture.return_value
        mock_detector = MockDetector.return_value
        mock_mouse = MockMouse.return_value
        mock_keyboard = MockKeyboard.return_value
        
        # Mock detection results for different screens
        mock_detector.detect_elements.side_effect = [
            # First detection
            [
                {"id": 1, "class": "button", "confidence": 0.8, "text": "Login", 
                 "bbox": [100, 100, 200, 150], "center": [150, 125]}
            ],
            # Second detection (same screen)
            [
                {"id": 1, "class": "button", "confidence": 0.8, "text": "Login", 
                 "bbox": [100, 100, 200, 150], "center": [150, 125]}
            ],
            # Third detection (different screen after login)
            [
                {"id": 2, "class": "button", "confidence": 0.75, "text": "Settings", 
                 "bbox": [300, 200, 400, 250], "center": [350, 225]}
            ]
        ]
        
        # Create temp directory for memory
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create real components with mocked dependencies
            visual_memory = VisualMemorySystem(config={"memory_path": temp_dir})
            safety_manager = SafetyManager()
            clarification_engine = ClarificationEngine(config={"memory_path": temp_dir})
            
            # Mock successful actions
            mock_mouse.move_to.return_value = {"success": True}
            mock_mouse.click.return_value = {"success": True}
            mock_keyboard.type_text.return_value = {"success": True}
            
            # Set a response callback that approves actions but asks questions
            questions = []
            def mock_response(question):
                questions.append(question)
                return "Yes, go ahead"
            
            clarification_engine.set_response_callback(mock_response)
            
            # Simulate first interaction - login screen
            # 1. Capture screen
            mock_capture.capture_screen.return_value = np.zeros((600, 800, 3))
            screen = mock_capture.capture_screen()
            
            # 2. Detect UI elements
            ui_elements = mock_detector.detect_elements(screen)
            assert len(ui_elements) == 1
            assert ui_elements[0]["text"] == "Login"
            
            # 3. Low confidence, so ask for clarification
            login_context = {
                "element_type": "button",
                "element_text": "Login",
                "action": "click"
            }
            result = clarification_engine.ask_for_clarification(
                scenario="ui_element_action",
                context=login_context,
                confidence=0.6
            )
            assert result["proceed"] is True
            assert len(questions) == 1
            assert "Login" in questions[0]
            
            # 4. Store UI element in visual memory
            pattern = {
                "visual_signature": np.zeros((64, 64)),
                "type": "button",
                "text": "Login",
                "bbox_size": (100, 50)
            }
            pattern_id = visual_memory.store_pattern(pattern)
            
            # 5. Perform action
            mock_mouse.move_to(150, 125)
            mock_mouse.click()
            
            # 6. Record successful interaction
            visual_memory.record_interaction(
                pattern_id=pattern_id,
                action="click",
                success=True,
                context={"window_title": "Login Screen"}
            )
            
            # Simulate second interaction - same login button, but now with memory
            # 1. Capture screen again
            screen = mock_capture.capture_screen()
            
            # 2. Detect UI elements
            ui_elements = mock_detector.detect_elements(screen)
            
            # 3. Enhance with visual memory
            enhanced_elements = visual_memory.enhance_detection(
                layout_results={"window_title": "Login Screen"},
                ui_elements=ui_elements,
                screen_image=screen
            )
            
            # Should have memory match with previous successful interaction
            assert "memory_match" in enhanced_elements[0]
            assert enhanced_elements[0]["memory_match"]["successful_interactions"] == 1
            
            # 4. Due to previous success, confidence should be higher
            # and no clarification needed
            high_confidence = enhanced_elements[0]["enhanced_confidence"]
            assert high_confidence > 0.8
            assert not clarification_engine.needs_clarification(high_confidence)
            
            # 5. Perform action directly this time
            mock_mouse.move_to(150, 125)
            mock_mouse.click()
            
            # 6. Record another successful interaction
            visual_memory.record_interaction(
                pattern_id=pattern_id,
                action="click",
                success=True,
                context={"window_title": "Login Screen"}
            )
            
            # Simulate third interaction - new settings button after login
            # 1. Capture new screen
            screen = mock_capture.capture_screen()
            
            # 2. Detect UI elements on new screen
            ui_elements = mock_detector.detect_elements(screen)
            assert len(ui_elements) == 1
            assert ui_elements[0]["text"] == "Settings"
            
            # 3. New element, low confidence, need clarification
            settings_context = {
                "element_type": "button",
                "element_text": "Settings",
                "action": "click"
            }
            result = clarification_engine.ask_for_clarification(
                scenario="ui_element_action",
                context=settings_context,
                confidence=0.6
            )
            assert result["proceed"] is True
            assert len(questions) == 2
            assert "Settings" in questions[1]
            
            # 4. Store new UI element in visual memory
            pattern = {
                "visual_signature": np.ones((64, 64)),  # Different signature
                "type": "button",
                "text": "Settings",
                "bbox_size": (100, 50)
            }
            settings_pattern_id = visual_memory.store_pattern(pattern)
            
            # 5. Perform action
            mock_mouse.move_to(350, 225)
            mock_mouse.click()
            
            # 6. Record successful interaction
            visual_memory.record_interaction(
                pattern_id=settings_pattern_id,
                action="click",
                success=True,
                context={"window_title": "Main Dashboard"}
            )
            
            # Verify memory statistics
            stats = visual_memory.get_statistics()
            assert stats["total_patterns"] == 2
            assert stats["successful_interactions"] == 3
            assert stats["element_types"]["button"] == 2
            
            # Verify clarification statistics
            c_stats = clarification_engine.get_clarification_statistics()
            assert c_stats["total_clarifications"] == 2
            assert c_stats["proceed_rate"] == 1.0  # All were approved
            
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
