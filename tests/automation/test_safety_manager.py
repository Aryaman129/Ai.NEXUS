"""
Unit tests for safety manager module

These tests verify that the safety manager correctly validates automation actions,
enforces safety boundaries, and adapts to user preferences over time.
"""
import os
import time
import json
import pytest
from unittest.mock import MagicMock, patch

from src.ai_core.automation.safety_manager import SafetyManager

@pytest.fixture
def safety_manager():
    """Create a SafetyManager instance for testing"""
    # Create with default settings
    manager = SafetyManager()
    yield manager


@pytest.fixture
def custom_safety_manager():
    """Create a SafetyManager with custom configuration"""
    # Create with custom settings
    config = {
        "safety_level": "medium",
        "allowed_applications": ["notepad.exe", "chrome.exe"],
        "dangerous_applications": ["cmd.exe", "powershell.exe"],
        "allowed_locations": [
            (100, 100, 500, 500),  # Safe area
        ],
        "dangerous_locations": [
            (0, 0, 50, 50),  # Top-left corner
            (1870, 0, 1920, 50),  # Top-right corner
        ],
        "allowed_key_combinations": [
            ["ctrl", "c"],
            ["ctrl", "v"],
        ],
        "dangerous_key_combinations": [
            ["alt", "f4"],
            ["ctrl", "w"],
        ],
        "enable_learning": True,
        "user_confirmation_required": True,
    }
    
    manager = SafetyManager(config=config)
    yield manager


def test_initialization():
    """Test safety manager initialization"""
    # Default configuration
    manager = SafetyManager()
    assert manager.safety_level == "high"
    assert len(manager.dangerous_applications) > 0
    assert len(manager.dangerous_key_combinations) > 0
    
    # Custom configuration
    config = {
        "safety_level": "low",
        "allowed_applications": ["notepad.exe"],
    }
    manager = SafetyManager(config=config)
    assert manager.safety_level == "low"
    assert "notepad.exe" in manager.allowed_applications


def test_safety_level_settings():
    """Test safety level settings"""
    manager = SafetyManager()
    
    # Test high safety
    manager.set_safety_level("high")
    assert manager.safety_level == "high"
    assert manager.user_confirmation_required is True
    
    # Test medium safety
    manager.set_safety_level("medium")
    assert manager.safety_level == "medium"
    
    # Test low safety
    manager.set_safety_level("low")
    assert manager.safety_level == "low"
    assert manager.user_confirmation_required is False
    
    # Test invalid safety level
    with pytest.raises(ValueError):
        manager.set_safety_level("invalid")


def test_is_safe_location(custom_safety_manager):
    """Test location safety checking"""
    # Test safe location
    assert custom_safety_manager.is_safe_location(300, 300) is True
    
    # Test dangerous location
    assert custom_safety_manager.is_safe_location(25, 25) is False
    
    # Test unsafe application context
    context = {"active_application": "cmd.exe"}
    assert custom_safety_manager.is_safe_location(300, 300, context=context) is False
    
    # Test with safety override
    assert custom_safety_manager.is_safe_location(25, 25, safety_override=True) is True


def test_is_safe_key_combination(custom_safety_manager):
    """Test key combination safety checking"""
    # Test safe combination
    assert custom_safety_manager.is_safe_key_combination(["ctrl", "c"]) is True
    
    # Test dangerous combination
    assert custom_safety_manager.is_safe_key_combination(["alt", "f4"]) is False
    
    # Test with dangerous application context
    context = {"active_application": "cmd.exe"}
    assert custom_safety_manager.is_safe_key_combination(["ctrl", "c"], context=context) is False
    
    # Test with safety override
    assert custom_safety_manager.is_safe_key_combination(["alt", "f4"], safety_override=True) is True


def test_is_safe_application(custom_safety_manager):
    """Test application safety checking"""
    # Test safe application
    assert custom_safety_manager.is_safe_application("notepad.exe") is True
    
    # Test dangerous application
    assert custom_safety_manager.is_safe_application("cmd.exe") is False
    
    # Test unknown application - default to safe
    assert custom_safety_manager.is_safe_application("unknown.exe") is True
    
    # Test with high safety level (unknown apps are unsafe)
    custom_safety_manager.set_safety_level("high")
    assert custom_safety_manager.is_safe_application("unknown.exe") is False


def test_is_safe_text(custom_safety_manager):
    """Test text content safety checking"""
    # Test safe text
    assert custom_safety_manager.is_safe_text("Hello world") is True
    
    # Test potentially dangerous command text
    assert custom_safety_manager.is_safe_text("cmd.exe /c del *") is False
    assert custom_safety_manager.is_safe_text("powershell -Command") is False
    assert custom_safety_manager.is_safe_text("rm -rf /") is False
    
    # Test with safety override
    assert custom_safety_manager.is_safe_text("cmd.exe /c del *", safety_override=True) is True


def test_validate_action(custom_safety_manager):
    """Test validating composite actions"""
    # Test safe mouse action
    action = {
        "type": "mouse_click",
        "x": 300,
        "y": 300,
        "context": {
            "active_application": "notepad.exe",
            "window_title": "Untitled - Notepad"
        }
    }
    validation = custom_safety_manager.validate_action(action)
    assert validation["is_safe"] is True
    
    # Test unsafe mouse action
    action["x"] = 25
    action["y"] = 25
    validation = custom_safety_manager.validate_action(action)
    assert validation["is_safe"] is False
    
    # Test unsafe keyboard action
    action = {
        "type": "keyboard_shortcut",
        "keys": ["alt", "f4"],
        "context": {
            "active_application": "notepad.exe",
            "window_title": "Untitled - Notepad"
        }
    }
    validation = custom_safety_manager.validate_action(action)
    assert validation["is_safe"] is False
    
    # Test unknown action type
    action = {
        "type": "unknown_action"
    }
    validation = custom_safety_manager.validate_action(action)
    assert validation["is_safe"] is False
    assert validation["reason"] == "unknown_action_type"


def test_add_safe_zone(safety_manager):
    """Test adding safe zones"""
    # Add safe zone
    safety_manager.add_safe_zone(100, 100, 300, 300)
    
    # Check that location is now safe
    assert safety_manager.is_safe_location(200, 200) is True
    
    # Make sure we can't add invalid zones
    with pytest.raises(ValueError):
        safety_manager.add_safe_zone(300, 300, 100, 100)  # Right < Left


def test_add_danger_zone(safety_manager):
    """Test adding danger zones"""
    # Add danger zone
    safety_manager.add_danger_zone(100, 100, 300, 300)
    
    # Check that location is now unsafe
    assert safety_manager.is_safe_location(200, 200) is False
    
    # Make sure we can't add invalid zones
    with pytest.raises(ValueError):
        safety_manager.add_danger_zone(300, 300, 100, 100)  # Right < Left


def test_allow_application(safety_manager):
    """Test allowing applications"""
    # Make sure dangerous app is marked unsafe
    safety_manager.dangerous_applications.append("test_app.exe")
    assert safety_manager.is_safe_application("test_app.exe") is False
    
    # Allow the application
    safety_manager.allow_application("test_app.exe")
    
    # Check that it's now allowed
    assert safety_manager.is_safe_application("test_app.exe") is True
    assert "test_app.exe" in safety_manager.allowed_applications
    
    # Disallow again
    safety_manager.disallow_application("test_app.exe")
    assert safety_manager.is_safe_application("test_app.exe") is False


def test_allow_key_combination(safety_manager):
    """Test allowing key combinations"""
    # Make sure dangerous combo is marked unsafe
    safety_manager.dangerous_key_combinations.append(["ctrl", "alt", "x"])
    assert safety_manager.is_safe_key_combination(["ctrl", "alt", "x"]) is False
    
    # Allow the combination
    safety_manager.allow_key_combination(["ctrl", "alt", "x"])
    
    # Check that it's now allowed
    assert safety_manager.is_safe_key_combination(["ctrl", "alt", "x"]) is True
    
    # Disallow again
    safety_manager.disallow_key_combination(["ctrl", "alt", "x"])
    assert safety_manager.is_safe_key_combination(["ctrl", "alt", "x"]) is False


def test_safety_policy_reset(custom_safety_manager):
    """Test resetting safety policies"""
    # Add some custom settings
    custom_safety_manager.add_safe_zone(200, 200, 400, 400)
    custom_safety_manager.allow_application("custom_app.exe")
    custom_safety_manager.allow_key_combination(["ctrl", "alt", "t"])
    
    # Reset safety policy
    custom_safety_manager.reset_safety_policy()
    
    # Check that settings were reset
    assert len(custom_safety_manager.allowed_locations) == 1  # Only the original from fixture
    assert "custom_app.exe" not in custom_safety_manager.allowed_applications
    assert ["ctrl", "alt", "t"] not in custom_safety_manager.allowed_key_combinations


def test_record_action_validation(custom_safety_manager):
    """Test recording validation results"""
    with patch('src.ai_core.automation.safety_manager.time.time', return_value=100):
        # Create an action
        action = {
            "type": "mouse_click",
            "x": 300,
            "y": 300,
            "context": {
                "active_application": "notepad.exe",
                "window_title": "Untitled - Notepad"
            }
        }
        
        # Validate and record
        custom_safety_manager.validate_and_record(action)
        
        # Check history
        assert len(custom_safety_manager.validation_history) == 1
        entry = custom_safety_manager.validation_history[0]
        assert entry["action"]["type"] == "mouse_click"
        assert entry["validation"]["is_safe"] is True
        assert entry["timestamp"] == 100


def test_get_safety_statistics(custom_safety_manager):
    """Test getting safety statistics"""
    # Add some validation history
    custom_safety_manager.validation_history = [
        {
            "action": {"type": "mouse_click"},
            "validation": {"is_safe": True},
            "timestamp": 100
        },
        {
            "action": {"type": "keyboard_shortcut"},
            "validation": {"is_safe": False, "reason": "dangerous_combination"},
            "timestamp": 101
        },
        {
            "action": {"type": "mouse_click"},
            "validation": {"is_safe": True},
            "timestamp": 102
        },
    ]
    
    # Get statistics
    stats = custom_safety_manager.get_safety_statistics()
    
    # Check statistics
    assert stats["total_actions"] == 3
    assert stats["safe_actions"] == 2
    assert stats["unsafe_actions"] == 1
    assert stats["safety_rate"] == 2/3
    assert "action_types" in stats
    assert stats["action_types"]["mouse_click"] == 2
    assert stats["action_types"]["keyboard_shortcut"] == 1


def test_user_confirmation_handling(custom_safety_manager):
    """Test user confirmation handling"""
    # Enable user confirmation
    custom_safety_manager.user_confirmation_required = True
    
    # Test with auto-confirmation callback
    confirm_callback = MagicMock(return_value=True)
    custom_safety_manager.set_confirmation_callback(confirm_callback)
    
    # Create unsafe action
    action = {
        "type": "keyboard_shortcut",
        "keys": ["alt", "f4"]
    }
    
    # Validate with confirmation
    result = custom_safety_manager.validate_with_confirmation(action)
    
    # Should call callback
    confirm_callback.assert_called_once()
    
    # Should be allowed due to confirmation
    assert result["is_safe"] is True
    assert result["required_confirmation"] is True
    
    # Test without confirmation
    confirm_callback.reset_mock()
    confirm_callback.return_value = False
    
    # Validate again
    result = custom_safety_manager.validate_with_confirmation(action)
    
    # Should still call callback
    confirm_callback.assert_called_once()
    
    # Should not be allowed
    assert result["is_safe"] is False
    assert result["required_confirmation"] is True


def test_learning_from_confirmations(custom_safety_manager):
    """Test learning from user confirmations"""
    # Enable learning
    custom_safety_manager.enable_learning = True
    
    # Create confirmation callback that always confirms
    confirm_callback = MagicMock(return_value=True)
    custom_safety_manager.set_confirmation_callback(confirm_callback)
    
    # Create unsafe action
    action = {
        "type": "keyboard_shortcut",
        "keys": ["alt", "f4"],
        "context": {
            "active_application": "notepad.exe",
            "window_title": "Untitled - Notepad"
        }
    }
    
    # Set threshold for learning
    custom_safety_manager.learning_confirmation_threshold = 2
    
    # Validate multiple times
    for i in range(3):
        result = custom_safety_manager.validate_with_confirmation(action)
        assert result["is_safe"] is True
    
    # After enough confirmations, should learn the pattern
    # Check that the combination is now allowed in this context
    context = {"active_application": "notepad.exe"}
    assert custom_safety_manager.is_safe_key_combination(["alt", "f4"], context=context) is True
    
    # But should still be unsafe in general
    assert custom_safety_manager.is_safe_key_combination(["alt", "f4"]) is False


def test_persistent_safety_settings(safety_manager):
    """Test saving and loading safety settings"""
    # Setup test directory
    import tempfile
    test_dir = tempfile.mkdtemp()
    settings_path = os.path.join(test_dir, "safety_settings.json")
    
    # Add some settings
    safety_manager.add_safe_zone(100, 100, 200, 200)
    safety_manager.allow_application("test_app.exe")
    safety_manager.allow_key_combination(["ctrl", "shift", "t"])
    
    # Save settings
    safety_manager.save_settings(settings_path)
    
    # Create new manager and load settings
    new_manager = SafetyManager()
    new_manager.load_settings(settings_path)
    
    # Verify settings were loaded
    assert (100, 100, 200, 200) in new_manager.allowed_locations
    assert "test_app.exe" in new_manager.allowed_applications
    assert ["ctrl", "shift", "t"] in new_manager.allowed_key_combinations
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir)


def test_context_aware_safety(custom_safety_manager):
    """Test context-aware safety rules"""
    # Add context-specific rules
    custom_safety_manager.add_context_rule({
        "active_application": "chrome.exe",
        "allowed_key_combinations": [["ctrl", "w"]],  # Allow closing tabs in Chrome
        "dangerous_locations": [(10, 10, 50, 50)]  # Some specific area in Chrome is dangerous
    })
    
    # Test key combo in Chrome
    chrome_context = {"active_application": "chrome.exe"}
    assert custom_safety_manager.is_safe_key_combination(["ctrl", "w"], context=chrome_context) is True
    
    # Still dangerous in other apps
    notepad_context = {"active_application": "notepad.exe"}
    assert custom_safety_manager.is_safe_key_combination(["ctrl", "w"], context=notepad_context) is False
    
    # Test location in Chrome
    assert custom_safety_manager.is_safe_location(30, 30, context=chrome_context) is False
    
    # Same location is safe in other apps
    assert custom_safety_manager.is_safe_location(30, 30, context=notepad_context) is True


def test_dynamic_safety_level_adjustment(safety_manager):
    """Test dynamically adjusting safety level based on application risk"""
    # Start with medium safety
    safety_manager.set_safety_level("medium")
    
    # Add a high-risk application
    safety_manager.add_high_risk_application("risky_app.exe")
    
    # Create context with the risky app
    context = {"active_application": "risky_app.exe"}
    
    # Test that safety level is temporarily increased
    temp_level = safety_manager._get_effective_safety_level(context)
    assert temp_level == "high"
    
    # Regular context should still use medium
    normal_context = {"active_application": "notepad.exe"}
    assert safety_manager._get_effective_safety_level(normal_context) == "medium"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
