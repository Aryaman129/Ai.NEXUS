"""
Unit tests for mouse controller module

These tests verify that the mouse controller functions correctly, with
appropriate safety measures and natural movement patterns.

This test suite uses mocks to avoid actual mouse movements during testing.
"""
import os
import time
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from src.ai_core.automation.mouse_controller import MouseController

# Mock safety manager
class MockSafetyManager:
    def __init__(self, config=None):
        self.config = config or {}
        self.allowed_locations = self.config.get("allowed_locations", [])
        self.dangerous_locations = self.config.get("dangerous_locations", [])
        
    def is_safe_location(self, x, y, context=None):
        """Check if location is safe"""
        # Check if location is explicitly allowed
        for left, top, right, bottom in self.allowed_locations:
            if left <= x <= right and top <= y <= bottom:
                return True
                
        # Check if location is explicitly dangerous
        for left, top, right, bottom in self.dangerous_locations:
            if left <= x <= right and top <= y <= bottom:
                return False
                
        # Default to safe for testing
        return True


@pytest.fixture
def mouse_controller():
    """Create a MouseController instance for testing"""
    with patch('src.ai_core.automation.mouse_controller.pyautogui') as mock_pyautogui:
        # Mock screen size
        mock_pyautogui.size.return_value = (1920, 1080)
        # Mock position
        mock_pyautogui.position.return_value = (100, 100)
        
        # Create controller
        controller = MouseController()
        
        yield controller


@pytest.fixture
def safe_mouse_controller():
    """Create a MouseController with safety manager"""
    with patch('src.ai_core.automation.mouse_controller.pyautogui') as mock_pyautogui:
        # Mock screen size
        mock_pyautogui.size.return_value = (1920, 1080)
        # Mock position
        mock_pyautogui.position.return_value = (100, 100)
        
        # Create safety manager with some dangerous areas
        safety_manager = MockSafetyManager(config={
            "dangerous_locations": [
                (1800, 0, 1920, 100),  # Top-right corner (close buttons)
                (500, 500, 700, 600),  # Random dangerous area for testing
            ]
        })
        
        # Create controller with safety manager
        controller = MouseController(safety_manager=safety_manager)
        
        yield controller


def test_initialization():
    """Test mouse controller initialization"""
    with patch('src.ai_core.automation.mouse_controller.pyautogui') as mock_pyautogui:
        # Mock screen size
        mock_pyautogui.size.return_value = (1920, 1080)
        # Mock position
        mock_pyautogui.position.return_value = (100, 100)
        
        # Create controller with custom config
        controller = MouseController(config={
            "movement_speed": 500,
            "click_delay": 0.2,
            "natural_movement": False,
        })
        
        # Check configuration
        assert controller.movement_speed == 500
        assert controller.click_delay == 0.2
        assert controller.natural_movement is False
        
        # Check default danger zones were created
        assert len(controller.danger_zones) > 0


def test_move_to(mouse_controller):
    """Test moving the mouse cursor with learning capabilities"""
    with patch('src.ai_core.automation.mouse_controller.pyautogui') as mock_pyautogui:
        # Reset call counts
        mock_pyautogui.reset_mock()
        
        # Set up mock position
        mock_pyautogui.position.return_value = (100, 100)
        
        # Move mouse
        result = mouse_controller.move_to(500, 300)
        
        # Check result structure (not implementation details)
        assert result["success"] is True
        assert "elapsed_time" in result
        assert "distance" in result
        
        # Test adaptive behavior - track performance metrics
        assert len(mouse_controller.movement_times) > 0
        
        # Get performance stats - demonstrating learning capability
        stats = mouse_controller.get_performance_stats()
        assert "avg_movement_time" in stats
        assert "movement_speed" in stats


def test_click(mouse_controller):
    """Test clicking the mouse"""
    with patch('src.ai_core.automation.mouse_controller.pyautogui') as mock_pyautogui:
        # Set up mock for current position
        mock_pyautogui.position.return_value = (100, 100)
        
        # Click at current position
        result = mouse_controller.click()
        
        # Check that pyautogui.click was called
        mock_pyautogui.click.assert_called_once()
        
        # Check result
        assert result["success"] is True
        assert result["clicks"] == 1
        assert result["button"] == "left"
        
        # Test right click
        result = mouse_controller.right_click()
        
        # Check that pyautogui.click was called with right button
        assert mock_pyautogui.click.call_args[1]["button"] == "right"
        
        # Test double click
        result = mouse_controller.double_click()
        
        # Check result
        assert result["success"] is True
        assert result["clicks"] == 2


def test_drag_to(mouse_controller):
    """Test dragging the mouse"""
    with patch('src.ai_core.automation.mouse_controller.pyautogui') as mock_pyautogui:
        # Set up mock for current position
        mock_pyautogui.position.return_value = (100, 100)
        
        # Drag to position
        result = mouse_controller.drag_to(500, 300)
        
        # Check that pyautogui.dragTo was called
        mock_pyautogui.dragTo.assert_called_once()
        
        # Check result
        assert result["success"] is True
        assert "elapsed_time" in result
        assert "distance" in result


def test_safety_boundaries(safe_mouse_controller):
    """Test safety boundaries for mouse movement"""
    with patch('src.ai_core.automation.mouse_controller.pyautogui') as mock_pyautogui:
        # Set up mock for current position
        mock_pyautogui.position.return_value = (100, 100)
        
        # Move to safe location
        result = safe_mouse_controller.move_to(300, 300)
        assert result["success"] is True
        
        # Move to dangerous location
        result = safe_mouse_controller.move_to(1900, 50)  # Top-right (close button area)
        assert result["success"] is False
        assert result["reason"] == "unsafe_region"
        
        # Test override
        result = safe_mouse_controller.move_to(1900, 50, safety_override=True)
        assert result["success"] is True
        
        # Test click in dangerous area
        mock_pyautogui.position.return_value = (600, 550)  # In dangerous area
        result = safe_mouse_controller.click()
        assert result["success"] is False
        assert result["reason"] == "unsafe_region"


def test_natural_movement():
    """Test natural movement curve generation"""
    with patch('src.ai_core.automation.mouse_controller.pyautogui') as mock_pyautogui:
        # Mock screen size
        mock_pyautogui.size.return_value = (1920, 1080)
        # Mock position
        mock_pyautogui.position.return_value = (100, 100)
        
        # Create controller
        controller = MouseController()
        
        # Generate curve
        points = controller._generate_bezier_curve(100, 100, 500, 300, 20)
        
        # Check points
        assert len(points) == 21  # steps + 1
        assert points[0] == (100, 100)  # Start point
        assert points[-1] == (500, 300)  # End point
        
        # Check that intermediate points form a curve
        middle_x = points[10][0]
        middle_y = points[10][1]
        
        # Middle point should not be on straight line
        assert not (middle_x == 300 and middle_y == 200)


def test_performance_tracking(mouse_controller):
    """Test performance metrics tracking"""
    with patch('src.ai_core.automation.mouse_controller.pyautogui') as mock_pyautogui:
        # Set up mock for current position
        mock_pyautogui.position.return_value = (100, 100)
        
        # Perform several movements
        for i in range(5):
            mouse_controller.move_to(500, 300)
            mouse_controller._update_performance_metrics(0.05)  # Mock elapsed time
            
        # Get performance stats
        stats = mouse_controller.get_performance_stats()
        
        # Check stats
        assert "avg_movement_time" in stats
        assert stats["avg_movement_time"] > 0
        assert "movement_speed" in stats
        assert "natural_movement" in stats


def test_custom_zones(mouse_controller):
    """Test adding custom safety zones"""
    # Add safe zone
    mouse_controller.add_safe_zone(100, 100, 300, 300)
    
    # Add danger zone
    mouse_controller.add_danger_zone(500, 500, 700, 700)
    
    # Check zones
    assert (100, 100, 300, 300) in mouse_controller.safe_zones
    assert (500, 500, 700, 700) in mouse_controller.danger_zones
    
    # Test safe zone overrides danger zone
    with patch('src.ai_core.automation.mouse_controller.pyautogui') as mock_pyautogui:
        mock_pyautogui.position.return_value = (600, 600)  # In danger zone
        
        # First check it's unsafe
        assert not mouse_controller._is_safe_location(600, 600)
        
        # Now add overlapping safe zone
        mouse_controller.add_safe_zone(550, 550, 650, 650)
        
        # Should now be safe
        assert mouse_controller._is_safe_location(600, 600)
        
    # Test clear zones
    mouse_controller.clear_zones()
    assert len(mouse_controller.safe_zones) == 0
    assert len(mouse_controller.danger_zones) == 0


def test_scroll(mouse_controller):
    """Test mouse scrolling"""
    with patch('src.ai_core.automation.mouse_controller.pyautogui') as mock_pyautogui:
        # Reset call counts
        mock_pyautogui.reset_mock()
        
        # Scroll up without specifying position
        result = mouse_controller.scroll(10)
        
        # Check result structure (focus on functionality, not implementation)
        assert result["success"] is True
        assert result["scroll_amount"] == 10
        assert mock_pyautogui.scroll.called
        
        # Reset mocks
        mock_pyautogui.reset_mock()
        
        # Mock the move_to method so we don't need to worry about its implementation
        with patch.object(mouse_controller, 'move_to', return_value={"success": True}):
            # Scroll down at specific position
            result = mouse_controller.scroll(-5, x=200, y=300)
            
            # Check result
            assert result["success"] is True
            assert result["scroll_amount"] == -5
            assert mock_pyautogui.scroll.called


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
