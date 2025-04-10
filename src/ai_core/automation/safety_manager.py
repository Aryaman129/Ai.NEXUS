"""
Safety Manager for NEXUS Automation

This module provides safety controls for automation actions, ensuring that
potentially dangerous operations are properly validated and controlled.

Key features:
- Safety boundaries for mouse and keyboard actions
- Contextual safety checks based on screen content
- Interaction confirmation workflow
- Learning from user safety preferences
"""
import os
import time
import logging
import json
from typing import Dict, List, Tuple, Optional, Union, Set, Callable
from pathlib import Path
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

class SafetyManager:
    """
    Safety manager for automation actions
    
    This class provides methods for ensuring that automation actions are safe,
    with configurable safety levels and user-defined boundaries.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the safety manager
        
        Args:
            config: Optional configuration dictionary with the following keys:
                - safety_level: Safety level (0-3, higher is more restrictive)
                - safe_zones: List of safe screen zones
                - danger_zones: List of dangerous screen zones
                - safe_applications: List of safe application names
                - dangerous_applications: List of dangerous application names
                - require_confirmation: Whether to require confirmation for dangerous actions
                - memory_path: Path to store safety preferences
        """
        self.config = config or {}
        
        # Set default configuration
        self.safety_level = self.config.get("safety_level", 2)  # Default: medium-high
        self.require_confirmation = self.config.get("require_confirmation", True)
        self.memory_path = self.config.get("memory_path", "memory/safety")
        
        # Safety zones
        self.safe_zones = self.config.get("safe_zones", [])
        self.danger_zones = self.config.get("danger_zones", [])
        
        # Application safety
        self.safe_applications = set(self.config.get("safe_applications", []))
        self.dangerous_applications = set(self.config.get("dangerous_applications", []))
        
        # User preferences
        self.user_preferences = {}
        
        # Create memory directory if it doesn't exist
        os.makedirs(self.memory_path, exist_ok=True)
        
        # Load existing preferences if available
        self._load_preferences()
        
        # Callback for confirmation requests
        self.confirmation_callback = None
        
        logger.info(f"SafetyManager initialized with safety level: {self.safety_level}")
    
    def _load_preferences(self):
        """Load user safety preferences from disk"""
        prefs_path = os.path.join(self.memory_path, "safety_preferences.json")
        
        if os.path.exists(prefs_path):
            try:
                with open(prefs_path, "r") as f:
                    self.user_preferences = json.load(f)
                logger.info(f"Loaded safety preferences with {len(self.user_preferences)} entries")
            except Exception as e:
                logger.error(f"Error loading safety preferences: {e}")
                self.user_preferences = {}
    
    def _save_preferences(self):
        """Save user safety preferences to disk"""
        prefs_path = os.path.join(self.memory_path, "safety_preferences.json")
        
        try:
            with open(prefs_path, "w") as f:
                json.dump(self.user_preferences, f, indent=2)
            logger.info(f"Saved safety preferences with {len(self.user_preferences)} entries")
        except Exception as e:
            logger.error(f"Error saving safety preferences: {e}")
    
    def set_safety_level(self, level: int):
        """
        Set the safety level
        
        Args:
            level: Safety level (0-3)
                0: Low - minimal restrictions
                1: Medium - basic restrictions
                2: High - significant restrictions
                3: Maximum - most restrictive
        """
        self.safety_level = max(0, min(3, level))
        logger.info(f"Safety level set to {self.safety_level}")
    
    def set_confirmation_callback(self, callback: Callable):
        """
        Set the callback function for confirmation requests
        
        Args:
            callback: Function to call for confirmation, should accept a
                     description and return a boolean
        """
        self.confirmation_callback = callback
    
    def is_safe_location(self, x: int, y: int, context: Optional[Dict] = None) -> bool:
        """
        Check if a screen location is safe to interact with
        
        Args:
            x: X coordinate
            y: Y coordinate
            context: Optional context information (window title, application, etc.)
            
        Returns:
            True if location is safe, False otherwise
        """
        # Always allow if safety level is 0
        if self.safety_level == 0:
            return True
            
        # Check if in any safe zones (these override danger zones)
        for left, top, right, bottom in self.safe_zones:
            if left <= x <= right and top <= y <= bottom:
                return True
                
        # Check if in any danger zones
        for left, top, right, bottom in self.danger_zones:
            if left <= x <= right and top <= y <= bottom:
                # Safety level 1 still allows danger zones with confirmation
                if self.safety_level == 1 and self.confirmation_callback:
                    description = f"Click in potential danger zone at ({x}, {y})"
                    return self.confirmation_callback(description)
                return False
                
        # Check application safety if context provided
        if context and "application" in context:
            app_name = context["application"]
            
            # Check if application is explicitly marked as dangerous
            if app_name in self.dangerous_applications:
                # Safety level 1 still allows with confirmation
                if self.safety_level <= 1 and self.confirmation_callback:
                    description = f"Interact with potentially dangerous application: {app_name}"
                    return self.confirmation_callback(description)
                return False
                
            # Check if application is explicitly marked as safe
            if app_name in self.safe_applications:
                return True
                
        # Check user preferences for this location
        location_key = f"location_{x//10}_{y//10}"  # Group by 10px squares
        if location_key in self.user_preferences:
            return self.user_preferences[location_key].get("is_safe", True)
            
        # Default behavior based on safety level
        if self.safety_level >= 3:
            # Maximum safety: require confirmation for any location not in safe zones
            if self.confirmation_callback:
                description = f"Interact with location at ({x}, {y})"
                return self.confirmation_callback(description)
            return False
            
        # For other safety levels, default to safe
        return True
    
    def is_safe_key_combination(self, keys: List[str], context: Optional[Dict] = None) -> bool:
        """
        Check if a key combination is safe to execute
        
        Args:
            keys: List of keys in the combination
            context: Optional context information (window title, application, etc.)
            
        Returns:
            True if key combination is safe, False otherwise
        """
        # Always allow if safety level is 0
        if self.safety_level == 0:
            return True
            
        # Convert keys to lowercase set for comparison
        keys_set = set(k.lower() for k in keys)
        
        # Define dangerous combinations based on safety level
        dangerous_combinations = []
        
        # Level 1+ dangerous combinations
        if self.safety_level >= 1:
            dangerous_combinations.extend([
                {"win", "l"},              # Lock screen
                {"alt", "f4"},             # Close app
                {"ctrl", "alt", "delete"},  # System menu
                {"win", "r"},              # Run dialog
            ])
            
        # Level 2+ dangerous combinations
        if self.safety_level >= 2:
            dangerous_combinations.extend([
                {"ctrl", "s"},             # Save
                {"ctrl", "w"},             # Close tab
                {"ctrl", "q"},             # Quit app
                {"alt", "tab"},            # Switch app
                {"win", "d"},              # Show desktop
                {"win", "e"},              # File explorer
            ])
            
        # Level 3+ dangerous combinations
        if self.safety_level >= 3:
            dangerous_combinations.extend([
                {"ctrl", "c"},             # Copy
                {"ctrl", "v"},             # Paste
                {"ctrl", "a"},             # Select all
                {"ctrl", "z"},             # Undo
                {"ctrl", "y"},             # Redo
            ])
            
        # Check if keys match any dangerous combination
        for combo in dangerous_combinations:
            if combo.issubset(keys_set):
                # Check user preferences for this combination
                combo_key = "keys_" + "_".join(sorted(combo))
                if combo_key in self.user_preferences:
                    return self.user_preferences[combo_key].get("is_safe", False)
                    
                # Allow with confirmation for lower safety levels
                if self.confirmation_callback:
                    description = f"Execute potentially dangerous key combination: {' + '.join(keys)}"
                    return self.confirmation_callback(description)
                    
                return False
                
        # No dangerous combinations matched
        return True
    
    def is_safe_text(self, text: str, context: Optional[Dict] = None) -> bool:
        """
        Check if text is safe to type
        
        Args:
            text: Text to type
            context: Optional context information (window title, application, etc.)
            
        Returns:
            True if text is safe, False otherwise
        """
        # Always allow if safety level is 0
        if self.safety_level == 0:
            return True
            
        # Check for command-line indicators
        suspicious_patterns = [
            "cmd /", "cmd.exe", "powershell", "powershell.exe",
            "bash", "sh -c", "python -c", "exec(", "eval(",
            "system(", "subprocess", "os.system", "rundll32",
            "regedit", "format c:", "del ", "rm -rf", "DROP TABLE",
        ]
        
        text_lower = text.lower()
        
        # For higher safety levels, add more suspicious patterns
        if self.safety_level >= 2:
            suspicious_patterns.extend([
                "http://", "https://", "ftp://", "file://",
                "javascript:", "data:", "www.", ".com", ".net", ".org",
            ])
            
        # Check for suspicious patterns
        for pattern in suspicious_patterns:
            if pattern.lower() in text_lower:
                # Safety level 1 allows with confirmation
                if self.safety_level == 1 and self.confirmation_callback:
                    description = f"Type potentially unsafe text containing '{pattern}'"
                    return self.confirmation_callback(description)
                    
                # Check user preferences for this pattern
                pattern_key = f"text_pattern_{pattern.replace(' ', '_')}"
                if pattern_key in self.user_preferences:
                    return self.user_preferences[pattern_key].get("is_safe", False)
                    
                return False
                
        # Length restrictions for higher safety levels
        if self.safety_level >= 3 and len(text) > 100:
            if self.confirmation_callback:
                description = f"Type a long text ({len(text)} characters)"
                return self.confirmation_callback(description)
            return False
            
        return True
    
    def record_user_preference(self, category: str, identifier: str, is_safe: bool):
        """
        Record a user safety preference
        
        Args:
            category: Category of preference (location, keys, text)
            identifier: Specific identifier within category
            is_safe: Whether the action is considered safe
        """
        key = f"{category}_{identifier}"
        
        self.user_preferences[key] = {
            "is_safe": is_safe,
            "last_updated": time.time()
        }
        
        # Save preferences after update
        self._save_preferences()
        
    def validate_action(self, action: Dict, context: Optional[Dict] = None) -> Dict:
        """
        Validate if an action is safe to execute
        
        Args:
            action: Action dictionary containing type and parameters
            context: Optional context information
            
        Returns:
            Dictionary with validation result and details
        """
        action_type = action.get("type", "unknown")
        
        if action_type == "click" or action_type == "move":
            # Validate mouse location
            x = action.get("position", [0, 0])[0]
            y = action.get("position", [0, 0])[1]
            
            if not self.is_safe_location(x, y, context):
                return {
                    "valid": False,
                    "reason": "unsafe_location",
                    "details": f"Location ({x}, {y}) is not safe to interact with"
                }
                
        elif action_type == "type":
            # Validate text
            text = action.get("text", "")
            
            if not self.is_safe_text(text, context):
                return {
                    "valid": False,
                    "reason": "unsafe_text",
                    "details": "Text contains potentially unsafe content"
                }
                
        elif action_type == "keypress":
            # Validate key combination
            keys = action.get("keys", [])
            
            if not self.is_safe_key_combination(keys, context):
                return {
                    "valid": False,
                    "reason": "unsafe_key_combination",
                    "details": f"Key combination {keys} is potentially dangerous"
                }
                
        # All validation passed
        return {
            "valid": True
        }
        
    def get_safety_stats(self) -> Dict:
        """
        Get statistics about safety management
        
        Returns:
            Dictionary with safety statistics
        """
        # Count preferences by category
        location_prefs = 0
        key_prefs = 0
        text_prefs = 0
        
        for key in self.user_preferences:
            if key.startswith("location_"):
                location_prefs += 1
            elif key.startswith("keys_"):
                key_prefs += 1
            elif key.startswith("text_"):
                text_prefs += 1
                
        return {
            "safety_level": self.safety_level,
            "require_confirmation": self.require_confirmation,
            "safe_zones": len(self.safe_zones),
            "danger_zones": len(self.danger_zones),
            "user_preferences": {
                "total": len(self.user_preferences),
                "location": location_prefs,
                "keys": key_prefs,
                "text": text_prefs
            }
        }
