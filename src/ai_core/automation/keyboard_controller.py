"""
Keyboard Controller for NEXUS

This module provides safe keyboard automation capabilities with natural typing
patterns and safety boundaries to prevent unintended actions.

Key features:
- Natural typing with variable timing
- Safety measures to prevent dangerous key combinations
- Support for special keys and modifiers
- Learning from user typing patterns
"""
import os
import time
import random
import logging
from typing import Dict, List, Tuple, Optional, Union, Set
import pyautogui

# Set up logging
logger = logging.getLogger(__name__)

# Safety first: enable pyautogui failsafe
pyautogui.FAILSAFE = True

# Dangerous key combinations that require explicit approval
DANGEROUS_COMBINATIONS = [
    {"ctrl", "alt", "delete"},  # System interrupt
    {"alt", "f4"},              # Close application
    {"win", "r"},               # Run dialog
    {"win", "x"},               # Admin menu
    {"ctrl", "shift", "esc"},   # Task manager
    {"ctrl", "alt", "f4"},      # Forced close application
    {"win", "e"},               # Open File Explorer
    {"win", "i"},               # Open Settings
    {"win", "l"},               # Lock computer
    {"ctrl", "s"},              # Save (can be safe, but might overwrite)
    {"ctrl", "w"},              # Close tab/window
    {"alt", "tab"},             # Switch application
]

class KeyboardController:
    """
    Keyboard controller with safe automation and natural typing
    
    This class provides methods for controlling the keyboard with natural-looking
    typing patterns and built-in safety measures to prevent dangerous actions.
    """
    
    def __init__(self, safety_manager=None, config: Optional[Dict] = None):
        """
        Initialize the keyboard controller
        
        Args:
            safety_manager: Optional safety manager instance
            config: Optional configuration dictionary with the following keys:
                - typing_speed: Base typing speed (chars/second)
                - key_press_time: Key press duration (seconds)
                - natural_typing: Whether to use natural typing
                - allowed_dangerous_combinations: List of allowed dangerous key combinations
                - variable_speed: Whether to vary typing speed
        """
        self.config = config or {}
        
        # Set default configuration
        self.typing_speed = self.config.get("typing_speed", 8.0)  # chars/second
        self.key_press_time = self.config.get("key_press_time", 0.01)  # seconds
        self.natural_typing = self.config.get("natural_typing", True)
        self.variable_speed = self.config.get("variable_speed", True)
        
        # Safety settings
        self.allowed_dangerous_combinations = set(
            [frozenset(combo) for combo in self.config.get("allowed_dangerous_combinations", [])]
        )
        
        # Link to safety manager if provided
        self.safety_manager = safety_manager
        
        # Active modifiers
        self.active_modifiers = set()
        
        # Performance tracking
        self.typing_times = []
        
        logger.info(f"KeyboardController initialized with typing speed: {self.typing_speed} chars/sec")
    
    def type_text(self, text: str, delay: Optional[float] = None, 
                 natural: Optional[bool] = None,
                 safety_override: bool = False) -> Dict:
        """
        Type text with natural timing
        
        Args:
            text: Text to type
            delay: Delay between keystrokes (None = use default)
            natural: Whether to use natural typing (None = use default)
            safety_override: Whether to override safety checks
            
        Returns:
            Result dictionary with success status and details
        """
        start_time = time.time()
        
        # Check for unsafe text
        if not safety_override and self._contains_unsafe_text(text):
            logger.warning(f"Attempted to type potentially unsafe text: {text}")
            return {
                "success": False,
                "reason": "unsafe_text",
                "details": "Text contains potentially unsafe commands or scripts"
            }
            
        # Set natural typing flag
        use_natural = natural if natural is not None else self.natural_typing
        
        # Set delay
        char_delay = delay if delay is not None else 1.0 / self.typing_speed
        
        try:
            if use_natural:
                # Use natural typing with variable delays
                self._natural_type(text, base_delay=char_delay)
            else:
                # Use uniform typing
                pyautogui.write(text, interval=char_delay)
                
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            self._update_performance_metrics(elapsed_time, len(text))
            
            return {
                "success": True,
                "elapsed_time": elapsed_time,
                "char_count": len(text),
                "chars_per_second": len(text) / elapsed_time if elapsed_time > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error typing text: {e}")
            return {
                "success": False,
                "reason": "error",
                "details": str(e)
            }
    
    def press_key(self, key: str, safety_override: bool = False) -> Dict:
        """
        Press and release a single key
        
        Args:
            key: Key to press (e.g., 'a', 'enter', 'space')
            safety_override: Whether to override safety checks
            
        Returns:
            Result dictionary with success status and details
        """
        return self.press_keys([key], safety_override=safety_override)
    
    def press_keys(self, keys: List[str], safety_override: bool = False) -> Dict:
        """
        Press and release a combination of keys
        
        Args:
            keys: List of keys to press (e.g., ['ctrl', 'c'])
            safety_override: Whether to override safety checks
            
        Returns:
            Result dictionary with success status and details
        """
        start_time = time.time()
        
        # Check for dangerous key combinations
        if not safety_override and self._is_dangerous_combination(keys):
            logger.warning(f"Attempted to press dangerous key combination: {keys}")
            return {
                "success": False,
                "reason": "dangerous_combination",
                "details": "Key combination could have dangerous effects",
                "keys": keys
            }
            
        try:
            # Press keys in sequence
            for key in keys:
                pyautogui.keyDown(key)
                
            # Release keys in reverse order
            for key in reversed(keys):
                pyautogui.keyUp(key)
                
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            
            return {
                "success": True,
                "elapsed_time": elapsed_time,
                "keys": keys
            }
            
        except Exception as e:
            logger.error(f"Error pressing keys: {e}")
            # Ensure all keys are released if an error occurs
            for key in keys:
                try:
                    pyautogui.keyUp(key)
                except:
                    pass
                    
            return {
                "success": False,
                "reason": "error",
                "details": str(e),
                "keys": keys
            }
    
    def hold_key(self, key: str, duration: float, safety_override: bool = False) -> Dict:
        """
        Hold a key for a specified duration
        
        Args:
            key: Key to hold
            duration: Duration to hold the key (seconds)
            safety_override: Whether to override safety checks
            
        Returns:
            Result dictionary with success status and details
        """
        start_time = time.time()
        
        # Check for dangerous keys
        if not safety_override and self._is_dangerous_key(key):
            logger.warning(f"Attempted to hold dangerous key: {key}")
            return {
                "success": False,
                "reason": "dangerous_key",
                "details": f"Holding {key} could have dangerous effects",
                "key": key
            }
            
        try:
            # Press key
            pyautogui.keyDown(key)
            
            # Hold for duration
            time.sleep(duration)
            
            # Release key
            pyautogui.keyUp(key)
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            
            return {
                "success": True,
                "elapsed_time": elapsed_time,
                "key": key,
                "duration": duration
            }
            
        except Exception as e:
            logger.error(f"Error holding key: {e}")
            # Ensure key is released if an error occurs
            try:
                pyautogui.keyUp(key)
            except:
                pass
                
            return {
                "success": False,
                "reason": "error",
                "details": str(e),
                "key": key
            }
    
    def press_modifier_key(self, modifier: str, key: str, safety_override: bool = False) -> Dict:
        """
        Press a modifier key combination (e.g., Ctrl+C)
        
        Args:
            modifier: Modifier key ('ctrl', 'alt', 'shift', 'win')
            key: Key to press with modifier
            safety_override: Whether to override safety checks
            
        Returns:
            Result dictionary with success status and details
        """
        return self.press_keys([modifier, key], safety_override=safety_override)
    
    def backspace(self, count: int = 1) -> Dict:
        """
        Press backspace key the specified number of times
        
        Args:
            count: Number of times to press backspace
            
        Returns:
            Result dictionary with success status and details
        """
        start_time = time.time()
        
        try:
            for _ in range(count):
                pyautogui.press('backspace')
                if count > 1:
                    time.sleep(0.05)  # Small delay between presses
                    
            elapsed_time = time.time() - start_time
            
            return {
                "success": True,
                "elapsed_time": elapsed_time,
                "count": count
            }
            
        except Exception as e:
            logger.error(f"Error pressing backspace: {e}")
            return {
                "success": False,
                "reason": "error",
                "details": str(e)
            }
    
    def enter(self) -> Dict:
        """
        Press enter key
        
        Returns:
            Result dictionary with success status and details
        """
        return self.press_key('enter')
    
    def tab(self, count: int = 1) -> Dict:
        """
        Press tab key the specified number of times
        
        Args:
            count: Number of times to press tab
            
        Returns:
            Result dictionary with success status and details
        """
        start_time = time.time()
        
        try:
            for _ in range(count):
                pyautogui.press('tab')
                if count > 1:
                    time.sleep(0.05)  # Small delay between presses
                    
            elapsed_time = time.time() - start_time
            
            return {
                "success": True,
                "elapsed_time": elapsed_time,
                "count": count
            }
            
        except Exception as e:
            logger.error(f"Error pressing tab: {e}")
            return {
                "success": False,
                "reason": "error",
                "details": str(e)
            }
    
    def allow_dangerous_combination(self, keys: List[str]):
        """
        Add a dangerous key combination to the allowed list
        
        Args:
            keys: List of keys in the combination
        """
        self.allowed_dangerous_combinations.add(frozenset(keys))
        
    def disallow_dangerous_combination(self, keys: List[str]):
        """
        Remove a dangerous key combination from the allowed list
        
        Args:
            keys: List of keys in the combination
        """
        keyset = frozenset(keys)
        if keyset in self.allowed_dangerous_combinations:
            self.allowed_dangerous_combinations.remove(keyset)
    
    def set_typing_speed(self, chars_per_second: float):
        """
        Set the typing speed
        
        Args:
            chars_per_second: Characters per second
        """
        self.typing_speed = max(1.0, min(20.0, chars_per_second))
        logger.info(f"Typing speed set to {self.typing_speed} chars/sec")
        
    def _natural_type(self, text: str, base_delay: float):
        """
        Type text with natural-looking timing
        
        Args:
            text: Text to type
            base_delay: Base delay between keystrokes
        """
        # Define variance for different key categories
        adjacent_keys = {
            'q': ['w', 'a'],
            'w': ['q', 'e', 's', 'a'],
            'e': ['w', 'r', 'd', 's'],
            'r': ['e', 't', 'f', 'd'],
            't': ['r', 'y', 'g', 'f'],
            'y': ['t', 'u', 'h', 'g'],
            'u': ['y', 'i', 'j', 'h'],
            'i': ['u', 'o', 'k', 'j'],
            'o': ['i', 'p', 'l', 'k'],
            'p': ['o', '[', ';', 'l'],
            'a': ['q', 'w', 's', 'z'],
            's': ['w', 'e', 'd', 'x', 'z', 'a'],
            'd': ['e', 'r', 'f', 'c', 'x', 's'],
            'f': ['r', 't', 'g', 'v', 'c', 'd'],
            'g': ['t', 'y', 'h', 'b', 'v', 'f'],
            'h': ['y', 'u', 'j', 'n', 'b', 'g'],
            'j': ['u', 'i', 'k', 'm', 'n', 'h'],
            'k': ['i', 'o', 'l', ',', 'm', 'j'],
            'l': ['o', 'p', ';', '.', ',', 'k'],
            'z': ['a', 's', 'x'],
            'x': ['z', 's', 'd', 'c'],
            'c': ['x', 'd', 'f', 'v'],
            'v': ['c', 'f', 'g', 'b'],
            'b': ['v', 'g', 'h', 'n'],
            'n': ['b', 'h', 'j', 'm'],
            'm': ['n', 'j', 'k', ','],
        }
        
        prev_char = None
        
        for i, char in enumerate(text):
            # Calculate delay variance based on typing patterns
            variance = 0.0
            
            if self.variable_speed:
                # Slower for special characters
                if not char.isalnum() and char != ' ':
                    variance += random.uniform(0.05, 0.15)
                    
                # Faster for repeated characters
                if prev_char == char:
                    variance -= random.uniform(0.01, 0.05)
                    
                # Faster for adjacent keys on keyboard
                if prev_char and prev_char.lower() in adjacent_keys and char.lower() in adjacent_keys[prev_char.lower()]:
                    variance -= random.uniform(0.01, 0.03)
                    
                # Pause slightly at punctuation
                if prev_char and prev_char in ['.', '!', '?', ',', ':', ';']:
                    variance += random.uniform(0.1, 0.3)
                    
                # Pause at end of words
                if prev_char and prev_char != ' ' and char == ' ':
                    variance += random.uniform(0.01, 0.05)
                    
                # Random variance to simulate human timing
                variance += random.normalvariate(0, 0.02)
            
            # Calculate final delay
            delay = max(0.01, base_delay + variance)
            
            # Type character
            pyautogui.write(char, interval=delay)
            
            prev_char = char
    
    def _is_dangerous_key(self, key: str) -> bool:
        """
        Check if a key is potentially dangerous
        
        Args:
            key: Key to check
            
        Returns:
            True if key is dangerous, False otherwise
        """
        dangerous_keys = ['win', 'cmd', 'command', 'alt', 'delete', 'f4']
        return key.lower() in dangerous_keys
    
    def _is_dangerous_combination(self, keys: List[str]) -> bool:
        """
        Check if a key combination is potentially dangerous
        
        Args:
            keys: List of keys to check
            
        Returns:
            True if combination is dangerous, False otherwise
        """
        # Check with safety manager if available
        if self.safety_manager and hasattr(self.safety_manager, 'is_safe_key_combination'):
            return not self.safety_manager.is_safe_key_combination(keys)
            
        # Convert keys to lowercase
        keys_lower = [k.lower() for k in keys]
        keys_set = set(keys_lower)
        
        # Check if the combination is in allowed list
        if frozenset(keys_set) in self.allowed_dangerous_combinations:
            return False
            
        # Check if the combination is in dangerous list
        for dangerous_combo in DANGEROUS_COMBINATIONS:
            if dangerous_combo.issubset(keys_set):
                return True
                
        return False
    
    def _contains_unsafe_text(self, text: str) -> bool:
        """
        Check if text contains potentially unsafe commands or scripts
        
        Args:
            text: Text to check
            
        Returns:
            True if text contains unsafe elements, False otherwise
        """
        # This is a basic check that could be expanded
        # Check for command line indicators
        suspicious_patterns = [
            "cmd /", "cmd.exe", "powershell", "exec(", "system(", "eval(",
            "rundll32", "regedit", "format c:", "del ", "rm -rf", "DROP TABLE",
            "shutdown", "taskkill", "msconfig", "netsh", "iptables"
        ]
        
        text_lower = text.lower()
        
        return any(pattern.lower() in text_lower for pattern in suspicious_patterns)
    
    def _update_performance_metrics(self, elapsed_time: float, char_count: int):
        """Update performance metrics for monitoring"""
        # Keep last 30 typing times for rolling average
        self.typing_times.append((elapsed_time, char_count))
        if len(self.typing_times) > 30:
            self.typing_times.pop(0)
            
    def get_performance_stats(self) -> Dict:
        """
        Get current performance statistics
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.typing_times:
            return {
                "avg_typing_speed": self.typing_speed,
                "actual_typing_speed": 0,
                "natural_typing": self.natural_typing
            }
            
        total_time = sum(item[0] for item in self.typing_times)
        total_chars = sum(item[1] for item in self.typing_times)
        
        actual_speed = total_chars / total_time if total_time > 0 else 0
        
        return {
            "avg_typing_speed": self.typing_speed,
            "actual_typing_speed": actual_speed,
            "natural_typing": self.natural_typing
        }
