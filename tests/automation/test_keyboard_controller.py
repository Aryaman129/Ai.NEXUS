"""
Unit tests for keyboard controller module

These tests verify that the keyboard controller functions correctly, with
appropriate safety measures and natural typing patterns.

This test suite uses mocks to avoid actual keyboard input during testing.
"""
import os
import time
import pytest
from unittest.mock import MagicMock, patch

from src.ai_core.automation.keyboard_controller import KeyboardController, DANGEROUS_COMBINATIONS

# Mock safety manager
class MockSafetyManager:
    def __init__(self, config=None):
        self.config = config or {}
        self.allowed_combinations = self.config.get("allowed_combinations", [])
        self.dangerous_combinations = self.config.get("dangerous_combinations", [])
        
    def is_safe_key_combination(self, keys, context=None):
        """Check if key combination is safe"""
        keys_set = set(k.lower() for k in keys)
        
        # Check if combination is explicitly allowed
        for combo in self.allowed_combinations:
            if set(combo).issubset(keys_set):
                return True
                
        # Check if combination is explicitly dangerous
        for combo in self.dangerous_combinations:
            if set(combo).issubset(keys_set):
                return False
                
        # Default to safe for testing
        return True


@pytest.fixture
def keyboard_controller():
    """Create a KeyboardController instance for testing"""
    with patch('src.ai_core.automation.keyboard_controller.pyautogui') as mock_pyautogui:
        # Create controller
        controller = KeyboardController()
        
        yield controller


@pytest.fixture
def safe_keyboard_controller():
    """Create a KeyboardController with safety manager"""
    with patch('src.ai_core.automation.keyboard_controller.pyautogui') as mock_pyautogui:
        # Create safety manager with some dangerous combinations
        safety_manager = MockSafetyManager(config={
            "dangerous_combinations": [
                ["ctrl", "w"],
                ["alt", "f4"],
                ["win", "r"],
            ],
            "allowed_combinations": [
                ["ctrl", "c"],
                ["ctrl", "v"],
            ]
        })
        
        # Create controller with safety manager
        controller = KeyboardController(safety_manager=safety_manager)
        
        yield controller


def test_initialization():
    """Test keyboard controller initialization"""
    with patch('src.ai_core.automation.keyboard_controller.pyautogui') as mock_pyautogui:
        # Create controller with custom config
        controller = KeyboardController(config={
            "typing_speed": 12.0,
            "key_press_time": 0.02,
            "natural_typing": False,
            "variable_speed": False,
        })
        
        # Check configuration
        assert controller.typing_speed == 12.0
        assert controller.key_press_time == 0.02
        assert controller.natural_typing is False
        assert controller.variable_speed is False
        
        # Check dangerous combinations
        assert len(DANGEROUS_COMBINATIONS) > 0


def test_type_text(keyboard_controller):
    """Test typing text"""
    with patch('src.ai_core.automation.keyboard_controller.pyautogui') as mock_pyautogui:
        # Type text
        result = keyboard_controller.type_text("Hello, world!")
        
        # Check that pyautogui.write was called
        mock_pyautogui.write.assert_called_once()
        
        # Check result
        assert result["success"] is True
        assert result["char_count"] == 13
        assert "chars_per_second" in result
        
        # Test with natural typing
        keyboard_controller.natural_typing = True
        
        # Use a minimal test since _natural_type calls pyautogui.write directly
        # for each character, which is difficult to test
        with patch.object(keyboard_controller, '_natural_type') as mock_natural_type:
            result = keyboard_controller.type_text("Test")
            mock_natural_type.assert_called_once()
            assert result["success"] is True


def test_press_keys(keyboard_controller):
    """Test pressing key combinations"""
    with patch('src.ai_core.automation.keyboard_controller.pyautogui') as mock_pyautogui:
        # Press single key
        result = keyboard_controller.press_key("enter")
        
        # Check that pyautogui.keyDown and keyUp were called
        mock_pyautogui.keyDown.assert_called_once_with("enter")
        mock_pyautogui.keyUp.assert_called_once_with("enter")
        
        # Check result
        assert result["success"] is True
        
        # Reset mocks
        mock_pyautogui.reset_mock()
        
        # Press key combination
        result = keyboard_controller.press_keys(["ctrl", "c"])
        
        # Check key presses
        assert mock_pyautogui.keyDown.call_count == 2
        assert mock_pyautogui.keyUp.call_count == 2
        
        # Check result
        assert result["success"] is True
        assert result["keys"] == ["ctrl", "c"]


def test_hold_key(keyboard_controller):
    """Test holding a key"""
    with patch('src.ai_core.automation.keyboard_controller.pyautogui') as mock_pyautogui:
        with patch('src.ai_core.automation.keyboard_controller.time.sleep') as mock_sleep:
            # Hold key for 1 second
            result = keyboard_controller.hold_key("shift", 1.0)
            
            # Check that key was pressed, held, and released
            mock_pyautogui.keyDown.assert_called_once_with("shift")
            mock_sleep.assert_called_once_with(1.0)
            mock_pyautogui.keyUp.assert_called_once_with("shift")
            
            # Check result
            assert result["success"] is True
            assert result["key"] == "shift"
            assert result["duration"] == 1.0


def test_modifier_key(keyboard_controller):
    """Test pressing modifier key combinations"""
    with patch('src.ai_core.automation.keyboard_controller.pyautogui') as mock_pyautogui:
        # Press Ctrl+C
        result = keyboard_controller.press_modifier_key("ctrl", "c")
        
        # This should call press_keys
        assert result["success"] is True
        assert result["keys"] == ["ctrl", "c"]


def test_special_keys(keyboard_controller):
    """Test special key functions"""
    with patch('src.ai_core.automation.keyboard_controller.pyautogui') as mock_pyautogui:
        # Test backspace
        result = keyboard_controller.backspace()
        mock_pyautogui.press.assert_called_once_with('backspace')
        assert result["success"] is True
        
        # Reset mock
        mock_pyautogui.reset_mock()
        
        # Test multiple backspaces
        result = keyboard_controller.backspace(3)
        assert mock_pyautogui.press.call_count == 3
        assert result["count"] == 3
        
        # Reset mock
        mock_pyautogui.reset_mock()
        
        # Test enter
        result = keyboard_controller.enter()
        assert result["success"] is True
        
        # Reset mock
        mock_pyautogui.reset_mock()
        
        # Test tab
        result = keyboard_controller.tab(2)
        assert mock_pyautogui.press.call_count == 2
        assert result["count"] == 2


def test_safety_check_dangerous_combinations(safe_keyboard_controller):
    """Test safety checks for dangerous key combinations"""
    with patch('src.ai_core.automation.keyboard_controller.pyautogui') as mock_pyautogui:
        # Try safe combination
        result = safe_keyboard_controller.press_keys(["ctrl", "c"])
        assert result["success"] is True
        
        # Try unsafe combination
        result = safe_keyboard_controller.press_keys(["alt", "f4"])
        assert result["success"] is False
        assert result["reason"] == "dangerous_combination"
        
        # Try another unsafe combination
        result = safe_keyboard_controller.press_keys(["ctrl", "w"])
        assert result["success"] is False
        
        # Test override
        result = safe_keyboard_controller.press_keys(["alt", "f4"], safety_override=True)
        assert result["success"] is True


def test_unsafe_text_detection(keyboard_controller):
    """Test detection of unsafe text"""
    # Safe text
    assert not keyboard_controller._contains_unsafe_text("This is safe text")
    
    # Unsafe text with command indicators
    assert keyboard_controller._contains_unsafe_text("cmd /c del *.*")
    assert keyboard_controller._contains_unsafe_text("powershell -Command")
    assert keyboard_controller._contains_unsafe_text("rm -rf /")
    assert keyboard_controller._contains_unsafe_text("system('rm -rf /')")
    
    # Test text safety in type_text
    with patch('src.ai_core.automation.keyboard_controller.pyautogui') as mock_pyautogui:
        # Safe text
        result = keyboard_controller.type_text("Hello world")
        assert result["success"] is True
        
        # Unsafe text
        result = keyboard_controller.type_text("cmd.exe /c del *")
        assert result["success"] is False
        assert result["reason"] == "unsafe_text"
        
        # Override
        result = keyboard_controller.type_text("cmd.exe /c dir", safety_override=True)
        assert result["success"] is True


def test_dangerous_key_detection(keyboard_controller):
    """Test detection of dangerous keys"""
    # Safe keys
    assert not keyboard_controller._is_dangerous_key("a")
    assert not keyboard_controller._is_dangerous_key("enter")
    
    # Dangerous keys
    assert keyboard_controller._is_dangerous_key("win")
    assert keyboard_controller._is_dangerous_key("alt")
    assert keyboard_controller._is_dangerous_key("delete")
    assert keyboard_controller._is_dangerous_key("f4")
    
    # Dangerous combinations
    assert keyboard_controller._is_dangerous_combination(["alt", "f4"])
    assert keyboard_controller._is_dangerous_combination(["ctrl", "alt", "delete"])
    assert keyboard_controller._is_dangerous_combination(["win", "r"])
    
    # Safe combinations
    assert not keyboard_controller._is_dangerous_combination(["ctrl", "s"])
    assert not keyboard_controller._is_dangerous_combination(["shift", "tab"])


def test_allow_dangerous_combination(keyboard_controller):
    """Test allowing specific dangerous combinations"""
    # First check combination is dangerous
    assert keyboard_controller._is_dangerous_combination(["alt", "f4"])
    
    # Allow it
    keyboard_controller.allow_dangerous_combination(["alt", "f4"])
    
    # Now should be allowed
    assert not keyboard_controller._is_dangerous_combination(["alt", "f4"])
    
    # Test disallowing
    keyboard_controller.disallow_dangerous_combination(["alt", "f4"])
    
    # Should be dangerous again
    assert keyboard_controller._is_dangerous_combination(["alt", "f4"])


def test_natural_typing_speed_variance():
    """Test variance in natural typing speed"""
    with patch('src.ai_core.automation.keyboard_controller.pyautogui') as mock_pyautogui:
        controller = KeyboardController(config={
            "natural_typing": True,
            "variable_speed": True
        })
        
        # Test different character sequences
        special_chars = "/!@#$%^&*()_+"
        repeated_chars = "aaaaaa"
        normal_text = "hello world"
        
        # Track delays for different types of text
        special_delays = []
        repeated_delays = []
        normal_delays = []
        
        # Mock the actual write method to capture delays
        def mock_write(char, interval):
            if char in special_chars:
                special_delays.append(interval)
            elif char in repeated_chars:
                repeated_delays.append(interval)
            else:
                normal_delays.append(interval)
        
        mock_pyautogui.write = mock_write
        
        # Test by calling _natural_type directly
        controller._natural_type(special_chars, 0.1)
        controller._natural_type(repeated_chars, 0.1)
        controller._natural_type(normal_text, 0.1)
        
        # Special characters should have higher delays on average
        assert sum(special_delays) / len(special_delays) > 0.1
        
        # Repeated characters should have lower delays on average
        if repeated_delays:
            # Skip for repeated chars at beginning
            repeated_avg = sum(repeated_delays[1:]) / max(1, len(repeated_delays) - 1)
            assert repeated_avg < 0.1 + 0.05  # Allow small margin


def test_performance_tracking(keyboard_controller):
    """Test performance metrics tracking"""
    with patch('src.ai_core.automation.keyboard_controller.pyautogui') as mock_pyautogui:
        # Simulate typing
        keyboard_controller._update_performance_metrics(0.5, 10)  # 0.5s for 10 chars
        keyboard_controller._update_performance_metrics(0.6, 12)  # 0.6s for 12 chars
        
        # Get performance stats
        stats = keyboard_controller.get_performance_stats()
        
        # Check stats
        assert "avg_typing_speed" in stats
        assert "actual_typing_speed" in stats
        assert stats["actual_typing_speed"] > 0
        assert "natural_typing" in stats


def test_set_typing_speed(keyboard_controller):
    """Test setting typing speed"""
    # Default speed
    assert keyboard_controller.typing_speed == 8.0
    
    # Set new speed
    keyboard_controller.set_typing_speed(15.0)
    assert keyboard_controller.typing_speed == 15.0
    
    # Set speed outside bounds
    keyboard_controller.set_typing_speed(30.0)
    assert keyboard_controller.typing_speed == 20.0  # Should be clamped to max
    
    keyboard_controller.set_typing_speed(0.5)
    assert keyboard_controller.typing_speed == 1.0  # Should be clamped to min


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
