"""
NEXUS Multi-Application Controller

This module enables NEXUS to control multiple applications simultaneously,
with its own UI remaining accessible. It orchestrates the adaptive automation
capabilities across different windows and applications.
"""
import os
import sys
import time
import logging
import threading
import queue
from typing import Dict, List, Tuple, Any, Optional, Callable
import json
import random
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Import NEXUS components
try:
    from src.ai_core.automation.mouse_controller import MouseController
    from src.ai_core.automation.keyboard_controller import KeyboardController
    from src.ai_core.automation.safety_manager import SafetyManager
    from src.ai_core.automation.clarification_engine import ClarificationEngine
    from src.ai_core.screen_analysis.visual_memory import VisualMemorySystem
    from src.nexus_ui.floating_assistant import FloatingAssistantUI
except ImportError as e:
    logger.error(f"Failed to import NEXUS components: {e}")
    logger.error("Make sure you're running from the project root and all components are installed")
    raise

# Import platform-specific modules
try:
    import pyautogui
    import pygetwindow as gw
    import numpy as np
    import cv2
    from PIL import Image, ImageGrab
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Make sure you've installed all required dependencies")
    raise

class ApplicationInfo:
    """Stores information about a monitored application"""
    
    def __init__(self, title: str, app_type: str = None):
        """Initialize application info"""
        self.title = title
        self.app_type = app_type or "unknown"
        self.window_handle = None
        self.process_id = None
        self.geometry = (0, 0, 0, 0)  # x, y, width, height
        self.is_active = False
        self.is_minimized = False
        self.is_fullscreen = False
        self.last_active = 0  # Timestamp
        self.ui_elements = []  # List of detected UI elements
        self.interaction_history = []  # History of interactions
        self.confidence_level = 0.5  # Initial confidence
        
    def update_from_window(self, window):
        """Update info from a window object"""
        if window:
            self.window_handle = window
            self.geometry = (window.left, window.top, window.width, window.height)
            self.is_active = window.isActive
            self.is_minimized = window.isMinimized
            self.last_active = time.time()
            
            # Check if fullscreen
            screen_width, screen_height = pyautogui.size()
            self.is_fullscreen = (window.width >= screen_width and window.height >= screen_height)
            
            return True
        return False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "title": self.title,
            "app_type": self.app_type,
            "geometry": self.geometry,
            "is_active": self.is_active,
            "is_minimized": self.is_minimized,
            "is_fullscreen": self.is_fullscreen,
            "last_active": self.last_active,
            "ui_elements_count": len(self.ui_elements),
            "interactions_count": len(self.interaction_history),
            "confidence_level": self.confidence_level
        }

class MultiAppController:
    """
    Controls multiple applications with adaptive automation
    
    This controller:
    1. Monitors multiple application windows
    2. Coordinates screen capture and UI analysis across apps
    3. Provides app-specific automation with learning
    4. Handles cross-application workflows
    5. Maintains its own UI while controlling other apps
    """
    
    def __init__(self, config: Dict = None):
        """Initialize the multi-application controller"""
        self.config = config or {}
        
        # Set up directory paths
        memory_base = self.config.get("memory_base_path", "memory/applications")
        os.makedirs(memory_base, exist_ok=True)
        
        # Create core components
        self.safety_manager = SafetyManager(
            config=self.config.get("safety_config", {})
        )
        
        self.mouse = MouseController(
            safety_manager=self.safety_manager,
            config=self.config.get("mouse_config", {})
        )
        
        self.keyboard = KeyboardController(
            safety_manager=self.safety_manager,
            config=self.config.get("keyboard_config", {})
        )
        
        # Create visual memory system
        visual_memory_path = os.path.join(memory_base, "visual_memory")
        os.makedirs(visual_memory_path, exist_ok=True)
        
        self.visual_memory = VisualMemorySystem(config={
            "memory_path": visual_memory_path,
            "max_patterns": self.config.get("max_visual_patterns", 10000),
            "similarity_threshold": self.config.get("visual_similarity_threshold", 0.7),
            "enable_learning": True
        })
        
        # Create clarification engine with UI-based callbacks
        clarification_path = os.path.join(memory_base, "clarification_memory")
        os.makedirs(clarification_path, exist_ok=True)
        
        self.clarification_engine = ClarificationEngine(config={
            "memory_path": clarification_path,
            "confidence_threshold": self.config.get("clarification_threshold", 0.7),
            "enable_learning": True
        })
        
        # Create floating UI
        ui_config = self.config.get("ui_config", {})
        self.ui = FloatingAssistantUI(config=ui_config)
        
        # Connect UI to clarification engine
        self.clarification_engine.set_response_callback(self._ui_clarification_callback)
        
        # Track monitored applications
        self.applications: Dict[str, ApplicationInfo] = {}
        self.active_app = None
        
        # Control flags
        self.monitoring = False
        self.capture_thread = None
        self.analysis_thread = None
        self.command_queue = queue.Queue()
        
        # Task system 
        self.current_task = None
        self.task_queue = queue.Queue()
        self.task_results = {}
        
        # Register UI callbacks
        self.ui.register_callback("on_input", self._handle_ui_input)
        self.ui.register_callback("on_close", self._handle_ui_close)
        
        logger.info("Multi-Application Controller initialized")
    
    def start(self, position: Tuple[int, int] = None):
        """Start the controller with UI and monitoring"""
        # Start UI
        self.ui.start(position)
        
        # Wait for UI to initialize
        time.sleep(0.5)
        
        # Start monitoring in background thread
        self._start_monitoring()
        
        # Show welcome message
        self.ui.show_message("Multi-Application Controller started", "system")
        self.ui.show_message("I'm ready to help you control multiple applications", "nexus")
        self.ui.show_message("Try saying 'list apps' to see monitored applications", "nexus")
        
        logger.info("Multi-Application Controller started")
    
    def _start_monitoring(self):
        """Start monitoring applications in background threads"""
        if self.monitoring:
            logger.warning("Already monitoring applications")
            return
            
        # Start monitor thread
        self.monitoring = True
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._monitor_applications)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        # Start analysis thread
        self.analysis_thread = threading.Thread(target=self._analyze_applications)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
        
        logger.info("Application monitoring started")
    
    def _monitor_applications(self):
        """Monitor applications and capture screenshots"""
        update_interval = self.config.get("monitor_interval", 1.0)  # seconds
        
        try:
            while self.monitoring:
                # Update list of windows
                try:
                    all_windows = gw.getAllWindows()
                    
                    # Filter out the NEXUS UI window
                    non_nexus_windows = [w for w in all_windows if "NEXUS Assistant" not in w.title]
                    
                    # Find active window
                    active_window = gw.getActiveWindow()
                    active_title = active_window.title if active_window else None
                    
                    # Update active app
                    if active_title and active_title != "NEXUS Assistant":
                        if active_title not in self.applications:
                            # New active application
                            app_info = ApplicationInfo(active_title)
                            app_info.update_from_window(active_window)
                            self.applications[active_title] = app_info
                            
                            # Log new application
                            logger.info(f"New application detected: {active_title}")
                            
                        self.active_app = active_title
                        self.applications[active_title].is_active = True
                        self.applications[active_title].last_active = time.time()
                    
                    # Update all applications
                    for window in non_nexus_windows:
                        title = window.title
                        if not title:
                            continue
                            
                        if title not in self.applications:
                            # New application
                            app_info = ApplicationInfo(title)
                            app_info.update_from_window(window)
                            self.applications[title] = app_info
                        else:
                            # Update existing application
                            self.applications[title].update_from_window(window)
                    
                    # Update UI status
                    if self.active_app:
                        self.ui.update_status(f"Active: {self.active_app}")
                    
                except Exception as e:
                    logger.error(f"Error updating window list: {e}")
                
                # Sleep for update interval
                time.sleep(update_interval)
                
        except Exception as e:
            logger.error(f"Error in application monitor thread: {e}")
            self.monitoring = False
    
    def _analyze_applications(self):
        """Analyze application screenshots for UI elements"""
        analysis_interval = self.config.get("analysis_interval", 2.0)  # seconds
        
        try:
            while self.monitoring:
                # Only analyze active app
                if self.active_app and self.active_app in self.applications:
                    app_info = self.applications[self.active_app]
                    
                    if not app_info.is_minimized:
                        try:
                            # Capture screenshot of application window
                            screenshot = self._capture_app_screenshot(app_info)
                            
                            if screenshot is not None:
                                # Perform UI analysis (simplified for this example)
                                # In a real implementation, this would use computer vision
                                elements = self._detect_ui_elements(screenshot, app_info)
                                
                                # Update application UI elements
                                app_info.ui_elements = elements
                                
                                # Log summary
                                logger.debug(f"Analyzed {app_info.title}: {len(elements)} UI elements")
                        
                        except Exception as e:
                            logger.error(f"Error analyzing {app_info.title}: {e}")
                
                # Sleep for analysis interval
                time.sleep(analysis_interval)
                
        except Exception as e:
            logger.error(f"Error in application analysis thread: {e}")
            self.monitoring = False
    
    def _capture_app_screenshot(self, app_info: ApplicationInfo):
        """Capture screenshot of application window"""
        if app_info.is_minimized:
            return None
            
        try:
            window = app_info.window_handle
            if not window:
                return None
                
            # Get window geometry
            x, y, width, height = app_info.geometry
            
            # Capture screenshot
            screenshot = ImageGrab.grab(bbox=(x, y, x + width, y + height))
            
            # Convert to numpy array
            screenshot_np = np.array(screenshot)
            
            return screenshot_np
            
        except Exception as e:
            logger.error(f"Error capturing screenshot: {e}")
            return None
    
    def _detect_ui_elements(self, screenshot, app_info: ApplicationInfo):
        """
        Detect UI elements in screenshot
        
        This is a simplified implementation. In a real system, you would use
        a more sophisticated computer vision approach with ML models.
        """
        # Placeholder for UI element detection
        elements = []
        
        try:
            # Convert to grayscale for processing
            gray = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
            
            # For demo purposes, we'll simulate finding some UI elements
            # In a real implementation, you would use ML-based detection
            
            # Simulate finding buttons
            for i in range(random.randint(1, 5)):
                # Generate random position within the window
                x = random.randint(10, app_info.geometry[2] - 50)
                y = random.randint(10, app_info.geometry[3] - 30)
                width = random.randint(50, 150)
                height = random.randint(20, 40)
                
                elements.append({
                    "type": "button",
                    "bbox": (x, y, x + width, y + height),
                    "center": (x + width // 2, y + height // 2),
                    "confidence": random.uniform(0.6, 0.95),
                    "text": f"Button {i+1}"
                })
            
            # Simulate finding text fields
            for i in range(random.randint(0, 3)):
                x = random.randint(10, app_info.geometry[2] - 100)
                y = random.randint(50, app_info.geometry[3] - 30)
                width = random.randint(100, 250)
                height = random.randint(20, 30)
                
                elements.append({
                    "type": "text_field",
                    "bbox": (x, y, x + width, y + height),
                    "center": (x + width // 2, y + height // 2),
                    "confidence": random.uniform(0.7, 0.9),
                    "text": ""
                })
            
            # Use visual memory to enhance detection
            elements = self.visual_memory.enhance_detection(
                layout_results={"window_title": app_info.title},
                ui_elements=elements,
                screen_image=screenshot
            )
            
        except Exception as e:
            logger.error(f"Error detecting UI elements: {e}")
        
        return elements
    
    def _ui_clarification_callback(self, question: str) -> str:
        """Handle clarification questions through the UI"""
        if not self.ui or not self.ui.is_running:
            logger.warning("UI not available for clarification")
            return "abort"
            
        # Ask question through UI
        response = self.ui.ask_question(question)
        
        return response
    
    def _handle_ui_input(self, text: str):
        """Handle input from UI"""
        # Process commands
        text = text.strip()
        
        if text.lower() in ["exit", "quit"]:
            self.stop()
            return
            
        if text.lower() == "help":
            self._show_help()
            return
            
        if text.lower() == "list apps":
            self._list_applications()
            return
            
        if text.lower().startswith("focus "):
            app_name = text[6:].strip()
            self._focus_application(app_name)
            return
            
        if text.lower().startswith("analyze "):
            app_name = text[8:].strip()
            self._analyze_application(app_name)
            return
            
        if text.lower().startswith("click "):
            target = text[6:].strip()
            self._handle_click_command(target)
            return
            
        if text.lower().startswith("type "):
            content = text[5:].strip()
            self._handle_type_command(content)
            return
            
        # If no command matched, treat as natural language
        self._process_natural_language(text)
    
    def _handle_ui_close(self):
        """Handle UI close event"""
        self.stop()
    
    def _show_help(self):
        """Show help information"""
        help_text = """
Available commands:
- list apps: Show monitored applications
- focus [app]: Switch focus to an application
- analyze [app]: Analyze UI elements in an application
- click [element]: Click on a UI element
- type [text]: Type text in active application
- help: Show this help message
- exit/quit: Exit NEXUS

You can also use natural language to describe what you want to do.
"""
        self.ui.show_message(help_text, "nexus")
    
    def _list_applications(self):
        """List monitored applications"""
        if not self.applications:
            self.ui.show_message("No applications are currently being monitored.", "nexus")
            return
            
        self.ui.show_message("Currently monitored applications:", "nexus")
        
        for title, app in self.applications.items():
            status = "ACTIVE" if app.is_active else ("MINIMIZED" if app.is_minimized else "INACTIVE")
            elements_count = len(app.ui_elements)
            
            self.ui.show_message(f"- {title} ({status}): {elements_count} UI elements detected", "nexus")
    
    def _focus_application(self, app_name: str):
        """Focus on an application"""
        # Find matching application
        matching_apps = [title for title in self.applications if app_name.lower() in title.lower()]
        
        if not matching_apps:
            self.ui.show_message(f"No application matching '{app_name}' found.", "nexus")
            return
            
        # Use exact match if available, otherwise use first match
        if app_name in matching_apps:
            target_app = app_name
        else:
            target_app = matching_apps[0]
            
        # Focus on application
        try:
            app_info = self.applications[target_app]
            window = app_info.window_handle
            
            if window:
                window.activate()
                self.ui.show_message(f"Focused on {target_app}", "nexus")
            else:
                self.ui.show_message(f"Could not focus on {target_app}", "error")
        
        except Exception as e:
            logger.error(f"Error focusing on application: {e}")
            self.ui.show_message(f"Error focusing on application: {e}", "error")
    
    def _analyze_application(self, app_name: str):
        """Analyze UI elements in an application"""
        # Find matching application
        matching_apps = [title for title in self.applications if app_name.lower() in title.lower()]
        
        if not matching_apps:
            self.ui.show_message(f"No application matching '{app_name}' found.", "nexus")
            return
            
        # Use exact match if available, otherwise use first match
        if app_name in matching_apps:
            target_app = app_name
        else:
            target_app = matching_apps[0]
            
        # Analyze application
        try:
            app_info = self.applications[target_app]
            
            # Force an immediate analysis
            screenshot = self._capture_app_screenshot(app_info)
            
            if screenshot is not None:
                # Start analysis
                self.ui.show_message(f"Analyzing {target_app}...", "nexus")
                
                # Detect UI elements
                elements = self._detect_ui_elements(screenshot, app_info)
                
                # Update application
                app_info.ui_elements = elements
                
                # Show results
                self.ui.show_message(f"Analysis complete. Found {len(elements)} UI elements.", "nexus")
                
                # Show details of some elements
                if elements:
                    self.ui.show_message("Some detected elements:", "nexus")
                    for i, element in enumerate(elements[:3]):
                        self.ui.show_message(f"- {element['type']}: {element.get('text', 'No text')} (Confidence: {element['confidence']:.2f})", "nexus")
                    
                    if len(elements) > 3:
                        self.ui.show_message(f"... and {len(elements) - 3} more elements", "nexus")
            else:
                self.ui.show_message(f"Could not capture screenshot of {target_app}", "error")
        
        except Exception as e:
            logger.error(f"Error analyzing application: {e}")
            self.ui.show_message(f"Error analyzing application: {e}", "error")
    
    def _handle_click_command(self, target: str):
        """Handle click command"""
        if not self.active_app or self.active_app not in self.applications:
            self.ui.show_message("No active application to click on", "error")
            return
            
        app_info = self.applications[self.active_app]
        
        # Find matching UI element
        matching_elements = [
            e for e in app_info.ui_elements 
            if target.lower() in e.get("text", "").lower() or 
               target.lower() in e.get("type", "").lower()
        ]
        
        if not matching_elements:
            self.ui.show_message(f"No UI element matching '{target}' found", "error")
            return
            
        # Sort by confidence
        matching_elements.sort(key=lambda e: e.get("confidence", 0), reverse=True)
        best_match = matching_elements[0]
        
        # Get click position
        click_x, click_y = best_match["center"]
        
        # Check confidence for clarification
        confidence = best_match.get("confidence", 0.5)
        element_text = best_match.get("text", "Unknown")
        element_type = best_match.get("type", "element")
        
        if confidence < 0.8:
            # Ask for clarification
            context = {
                "element_type": element_type,
                "element_text": element_text,
                "action": "click"
            }
            
            result = self.clarification_engine.ask_for_clarification(
                scenario="ui_element_action",
                context=context,
                confidence=confidence
            )
            
            if not result["proceed"]:
                self.ui.show_message("Click operation canceled", "nexus")
                return
                
            confidence = result["confidence"]
        
        # Adjust for window position
        window_x, window_y, _, _ = app_info.geometry
        abs_x = window_x + click_x
        abs_y = window_y + click_y
        
        # Perform click
        self.ui.show_message(f"Clicking on {element_type}: '{element_text}'", "nexus")
        
        result = self.mouse.click(abs_x, abs_y)
        
        if result["success"]:
            self.ui.show_message("Click successful", "nexus")
            
            # Record successful interaction
            self.visual_memory.record_interaction(
                pattern_id=best_match.get("memory_match", {}).get("pattern_id"),
                action="click",
                success=True,
                context={"window_title": app_info.title}
            )
        else:
            self.ui.show_message(f"Click failed: {result.get('reason', 'unknown error')}", "error")
    
    def _handle_type_command(self, content: str):
        """Handle type command"""
        if not self.active_app:
            self.ui.show_message("No active application to type in", "error")
            return
            
        # Check if content should be treated as potentially dangerous
        is_dangerous = self.safety_manager.is_dangerous_text(content)
        
        if is_dangerous:
            # Always clarify for potentially dangerous text
            context = {
                "text": content,
                "warning": "This text may include commands or dangerous patterns"
            }
            
            result = self.clarification_engine.ask_for_clarification(
                scenario="dangerous_text",
                context=context,
                confidence=0.3  # Low confidence for dangerous text
            )
            
            if not result["proceed"]:
                self.ui.show_message("Typing operation canceled", "nexus")
                return
        
        # Type the text
        self.ui.show_message(f"Typing: {content[:20]}..." if len(content) > 20 else f"Typing: {content}", "nexus")
        
        result = self.keyboard.type_text(content)
        
        if result["success"]:
            self.ui.show_message("Typing successful", "nexus")
        else:
            self.ui.show_message(f"Typing failed: {result.get('reason', 'unknown error')}", "error")
    
    def _process_natural_language(self, text: str):
        """Process natural language command"""
        # Simplified NLP processing - in a real system this would use an LLM
        text_lower = text.lower()
        
        # Basic intent recognition
        if any(word in text_lower for word in ["click", "press", "select", "choose"]):
            # Click intent
            # Extract target from text
            targets = [word for word in text_lower.split() if word not in ["click", "press", "select", "choose", "on", "the", "a", "an"]]
            if targets:
                target = " ".join(targets)
                self._handle_click_command(target)
            else:
                self.ui.show_message("What would you like me to click on?", "nexus")
                
        elif any(word in text_lower for word in ["type", "enter", "input", "write"]):
            # Type intent
            parts = text_lower.split(' ', 1)
            
            if len(parts) > 1:
                content = parts[1]
                self._handle_type_command(content)
            else:
                self.ui.show_message("What would you like me to type?", "nexus")
                
        elif any(word in text_lower for word in ["focus", "switch", "go to"]):
            # Focus intent
            parts = text_lower.split(' ', 1)
            
            if len(parts) > 1:
                app_name = parts[1]
                self._focus_application(app_name)
            else:
                self.ui.show_message("Which application would you like me to focus on?", "nexus")
                
        else:
            # Unknown intent
            self.ui.show_message("I'm not sure what you want me to do. Try using specific commands like 'click', 'type', or 'focus'.", "nexus")
            self.ui.show_message("Type 'help' to see available commands.", "nexus")
    
    def stop(self):
        """Stop the controller and cleanup"""
        # Stop monitoring
        self.monitoring = False
        
        # Wait for threads to finish
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
            
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=1.0)
        
        # Stop UI
        if self.ui and self.ui.is_running:
            self.ui.stop()
        
        logger.info("Multi-Application Controller stopped")

def run_multi_app_controller():
    """Run the multi-application controller"""
    controller = MultiAppController(config={
        "ui_config": {
            "theme": "dark",
            "opacity": 0.9,
            "width": 400,
            "height": 600
        },
        "monitor_interval": 1.0,
        "analysis_interval": 2.0
    })
    
    # Calculate position (right side of screen)
    screen_width, screen_height = pyautogui.size()
    position = (screen_width - 420, 50)
    
    # Start controller
    controller.start(position)
    
    try:
        # Keep main thread alive
        while controller.ui.is_running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        # Stop on Ctrl+C
        controller.stop()

if __name__ == "__main__":
    # Make sure required directories exist
    os.makedirs("memory/applications/visual_memory", exist_ok=True)
    os.makedirs("memory/applications/clarification_memory", exist_ok=True)
    
    # Run controller
    run_multi_app_controller()
