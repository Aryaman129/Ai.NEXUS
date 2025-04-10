"""
NEXUS Adaptive Notepad Automation Example

This example demonstrates how NEXUS can adaptively learn to interact with
the Windows Notepad application, remembering UI patterns and becoming more
confident in automation over time.
"""
import os
import time
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import NEXUS components
from src.ai_core.automation.mouse_controller import MouseController
from src.ai_core.automation.keyboard_controller import KeyboardController
from src.ai_core.automation.safety_manager import SafetyManager
from src.ai_core.automation.clarification_engine import ClarificationEngine
from src.ai_core.screen_analysis.visual_memory import VisualMemorySystem

class NotepadAutomation:
    """
    Class to demonstrate adaptive automation with Windows Notepad
    
    This class shows how NEXUS learns UI patterns for Notepad
    and becomes more efficient over time.
    """
    
    def __init__(self):
        """Initialize the Notepad automation demo"""
        # Set up memory directory
        memory_dir = Path("memory/applications/notepad")
        memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Create automation components
        self.safety_manager = SafetyManager()
        self.mouse = MouseController(safety_manager=self.safety_manager)
        self.keyboard = KeyboardController(safety_manager=self.safety_manager)
        
        # Create visual memory system
        self.visual_memory = VisualMemorySystem(config={
            "memory_path": str(memory_dir / "visual"),
            "max_patterns": 100,
            "similarity_threshold": 0.7,
            "enable_learning": True
        })
        
        # Create clarification engine
        self.clarification_engine = ClarificationEngine(config={
            "memory_path": str(memory_dir / "clarification"),
            "confidence_threshold": 0.7,
            "enable_learning": True
        })
        
        # Set clarification callback
        self.clarification_engine.set_response_callback(self._clarification_callback)
        
        # Track success rates for tasks
        self.task_success = {}
        
        logger.info("Notepad automation initialized")
    
    def _clarification_callback(self, question):
        """Handle clarification questions"""
        print(f"\nNEXUS needs clarification: {question}")
        return input("Your response: ")
    
    def open_notepad(self):
        """Open Windows Notepad and verify it's running"""
        print("\n=== Opening Notepad ===")
        
        # Check confidence based on past attempts
        task_id = "open_notepad"
        confidence = self._get_task_confidence(task_id)
        
        if confidence < 0.7:
            # Ask for confirmation if confidence is low
            context = {
                "action": "open Windows Notepad",
                "application": "notepad.exe"
            }
            
            result = self.clarification_engine.ask_for_clarification(
                scenario="general_uncertainty",
                context=context,
                confidence=confidence
            )
            
            if not result["proceed"]:
                print("Action canceled based on user response.")
                return False
            
            # Update confidence based on clarification
            confidence = result["confidence"]
        
        try:
            # Run notepad
            os.system("start notepad")
            time.sleep(1)  # Wait for Notepad to start
            
            print("Notepad started successfully")
            self._update_task_success(task_id, True, confidence)
            return True
            
        except Exception as e:
            logger.error(f"Error opening Notepad: {e}")
            self._update_task_success(task_id, False, confidence)
            return False
    
    def type_text(self, text):
        """Type text into Notepad with adaptive typing speed"""
        print(f"\n=== Typing Text: '{text[:20]}...' ===")
        
        # Get confidence for typing task
        task_id = "type_text"
        confidence = self._get_task_confidence(task_id)
        
        if confidence < 0.7:
            # Ask for confirmation if confidence is low
            context = {
                "element_type": "text editor",
                "action": "type text",
                "text": text[:30] + "..." if len(text) > 30 else text
            }
            
            result = self.clarification_engine.ask_for_clarification(
                scenario="text_input",
                context=context,
                confidence=confidence
            )
            
            if not result["proceed"]:
                print("Typing canceled based on user response.")
                return False
            
            # Update confidence based on clarification
            confidence = result["confidence"]
        
        try:
            # Type text with speed based on confidence
            # Higher confidence = faster typing
            typing_speed = 5 + (confidence * 5)  # 5-10 chars/sec
            
            # Set typing speed
            self.keyboard.set_typing_speed(typing_speed)
            
            # Type the text
            start_time = time.time()
            result = self.keyboard.type_text(text)
            elapsed_time = time.time() - start_time
            
            # Calculate characters per second
            chars_per_second = len(text) / elapsed_time if elapsed_time > 0 else 0
            
            print(f"Typed {len(text)} characters in {elapsed_time:.2f} seconds")
            print(f"Effective typing speed: {chars_per_second:.2f} chars/sec")
            
            self._update_task_success(task_id, True, confidence)
            return True
            
        except Exception as e:
            logger.error(f"Error typing text: {e}")
            self._update_task_success(task_id, False, confidence)
            return False
    
    def use_menu(self, menu_option, sub_option=None):
        """Access a menu option in Notepad with adaptive learning"""
        option_str = f"{menu_option}"
        if sub_option:
            option_str += f" > {sub_option}"
            
        print(f"\n=== Using Menu: {option_str} ===")
        
        # Get confidence for this specific menu task
        task_id = f"menu_{menu_option}_{sub_option}" if sub_option else f"menu_{menu_option}"
        confidence = self._get_task_confidence(task_id)
        
        if confidence < 0.7:
            # Ask for confirmation if confidence is low
            context = {
                "element_type": "menu",
                "element_text": option_str,
                "action": "select"
            }
            
            result = self.clarification_engine.ask_for_clarification(
                scenario="ui_element_action",
                context=context,
                confidence=confidence
            )
            
            if not result["proceed"]:
                print("Menu action canceled based on user response.")
                return False
            
            # Update confidence based on clarification
            confidence = result["confidence"]
        
        try:
            # Calculate delays based on confidence
            # More confidence = faster interactions
            menu_delay = max(0.2, 1.0 - confidence * 0.7)  # 0.2 - 1.0 seconds
            
            # Access the menu (Alt key opens menu in most Windows apps)
            self.keyboard.press_keys(["alt"])
            time.sleep(menu_delay)
            
            # Select the menu option using keyboard
            for char in menu_option:
                if char == ' ':
                    continue
                self.keyboard.press_key(char.lower())
                break  # Usually just the first letter is needed
            
            time.sleep(menu_delay)
            
            # Select sub-option if provided
            if sub_option:
                for char in sub_option:
                    if char == ' ':
                        continue
                    self.keyboard.press_key(char.lower())
                    break  # Usually just the first letter is needed
                    
                time.sleep(menu_delay)
            
            print(f"Menu option '{option_str}' selected successfully")
            self._update_task_success(task_id, True, confidence)
            return True
            
        except Exception as e:
            logger.error(f"Error using menu: {e}")
            self._update_task_success(task_id, False, confidence)
            return False
    
    def save_file(self, filename):
        """Save the Notepad file with adaptive learning"""
        print(f"\n=== Saving File: {filename} ===")
        
        # Get confidence for save task
        task_id = "save_file"
        confidence = self._get_task_confidence(task_id)
        
        if confidence < 0.7:
            # Ask for confirmation if confidence is low
            context = {
                "action": "save file",
                "filename": filename
            }
            
            result = self.clarification_engine.ask_for_clarification(
                scenario="general_uncertainty",
                context=context,
                confidence=confidence
            )
            
            if not result["proceed"]:
                print("Save operation canceled based on user response.")
                return False
            
            # Update confidence based on clarification
            confidence = result["confidence"]
        
        try:
            # Use keyboard shortcut Ctrl+S
            self.keyboard.press_keys(["ctrl", "s"])
            time.sleep(0.5)
            
            # Type filename
            self.keyboard.type_text(filename)
            time.sleep(0.2)
            
            # Press Enter to confirm
            self.keyboard.press_key("enter")
            time.sleep(0.5)
            
            # Handle possible "File already exists" dialog
            # This demonstrates adaptive handling of unexpected dialogs
            if confidence > 0.8:
                # With high confidence, we anticipate this dialog might appear
                # and prepare to handle it
                print("Checking for 'Confirm Save As' dialog (high confidence approach)")
                self.keyboard.press_key("left")  # Select "No" button
                time.sleep(0.1)
                self.keyboard.press_key("enter")  # Confirm
            
            print(f"File '{filename}' saved successfully")
            self._update_task_success(task_id, True, confidence)
            return True
            
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            self._update_task_success(task_id, False, confidence)
            return False
    
    def close_notepad(self):
        """Close Notepad with confirmation for unsaved changes"""
        print("\n=== Closing Notepad ===")
        
        # Check confidence based on past attempts
        task_id = "close_notepad"
        confidence = self._get_task_confidence(task_id)
        
        if confidence < 0.8:  # Higher threshold for destructive actions
            # Ask for confirmation if confidence is low
            context = {
                "action": "close Notepad",
                "warning": "Unsaved changes might be lost"
            }
            
            result = self.clarification_engine.ask_for_clarification(
                scenario="dangerous_action",
                context=context,
                confidence=confidence
            )
            
            if not result["proceed"]:
                print("Close operation canceled based on user response.")
                return False
            
            # Update confidence based on clarification
            confidence = result["confidence"]
        
        try:
            # Use Alt+F4 to close
            self.keyboard.press_keys(["alt", "f4"])
            time.sleep(0.5)
            
            # Handle possible "Do you want to save" dialog
            if confidence > 0.6:
                # With decent confidence, we handle the dialog
                print("Checking for 'Save changes' dialog (adaptive approach)")
                self.keyboard.press_key("n")  # Press 'n' for "No"
            
            print("Notepad closed successfully")
            self._update_task_success(task_id, True, confidence)
            return True
            
        except Exception as e:
            logger.error(f"Error closing Notepad: {e}")
            self._update_task_success(task_id, False, confidence)
            return False
    
    def _get_task_confidence(self, task_id):
        """Get confidence level for a task based on past successes"""
        if task_id not in self.task_success:
            return 0.5  # Default initial confidence
            
        # Calculate confidence based on success rate
        successes = self.task_success[task_id]["successes"]
        attempts = self.task_success[task_id]["attempts"]
        last_confidence = self.task_success[task_id]["last_confidence"]
        
        if attempts == 0:
            return 0.5
            
        # Calculate new confidence based on success rate and previous confidence
        success_rate = successes / attempts
        new_confidence = 0.3 + (success_rate * 0.5) + (last_confidence * 0.2)
        
        # Ensure confidence is between 0.1 and 1.0
        new_confidence = max(0.1, min(1.0, new_confidence))
        
        return new_confidence
    
    def _update_task_success(self, task_id, success, confidence):
        """Update success statistics for a task"""
        if task_id not in self.task_success:
            self.task_success[task_id] = {
                "successes": 0,
                "attempts": 0,
                "last_confidence": confidence
            }
            
        # Update stats
        if success:
            self.task_success[task_id]["successes"] += 1
        self.task_success[task_id]["attempts"] += 1
        self.task_success[task_id]["last_confidence"] = confidence
        
        # Log current stats
        stats = self.task_success[task_id]
        success_rate = stats["successes"] / stats["attempts"]
        logger.info(f"Task '{task_id}' stats: {stats['successes']}/{stats['attempts']} "
                  f"({success_rate:.2f}) - Confidence: {confidence:.2f}")
    
    def show_learning_stats(self):
        """Show how the system has learned from interactions"""
        print("\n=== NEXUS Adaptive Learning Statistics ===")
        
        # Task learning stats
        print("\nTask Learning:")
        print(f"{'Task':<20} {'Success Rate':<15} {'Confidence':<10} {'Attempts':<10}")
        print("-" * 60)
        
        for task_id, stats in sorted(self.task_success.items()):
            success_rate = stats["successes"] / stats["attempts"] if stats["attempts"] > 0 else 0
            confidence = self._get_task_confidence(task_id)
            print(f"{task_id:<20} {success_rate*100:>6.1f}%        {confidence:.2f}       {stats['attempts']:>3}")
        
        # Get clarification stats
        clarify_stats = self.clarification_engine.get_clarification_statistics()
        
        print("\nClarification Learning:")
        print(f"Total clarifications: {clarify_stats['total_clarifications']}")
        print(f"Proceed rate: {clarify_stats['proceed_rate']:.2f}")
        print(f"Confidence threshold: {clarify_stats['current_confidence_threshold']:.2f}")
        
        if clarify_stats["total_clarifications"] > 0:
            print("\nMost common clarification scenarios:")
            for scenario, count in clarify_stats["most_common_scenarios"].items():
                print(f"  - {scenario}: {count}")
        
        # Keyboard performance stats
        keyboard_stats = self.keyboard.get_performance_stats()
        print("\nKeyboard Performance:")
        print(f"Average typing speed: {keyboard_stats['avg_typing_speed']:.2f} chars/sec")
        print(f"Current typing speed: {keyboard_stats['actual_typing_speed']:.2f} chars/sec")
        
        # Mouse performance stats
        mouse_stats = self.mouse.get_performance_stats()
        print("\nMouse Performance:")
        print(f"Average movement time: {mouse_stats['avg_movement_time']:.4f} seconds")

def run_demo():
    """Run the Notepad automation demo"""
    print("=== NEXUS Adaptive Notepad Automation Demo ===")
    print("This demo shows how NEXUS learns to interact with Notepad")
    print("and becomes more efficient over time.\n")
    
    # Create the automation controller
    notepad = NotepadAutomation()
    
    # Confirm with user before starting
    response = input("Ready to start the demo? (yes/no): ")
    if response.lower() not in ["yes", "y"]:
        print("Demo canceled.")
        return
    
    try:
        # Run a sequence of tasks with Notepad
        tasks_succeeded = 0
        
        # Open Notepad
        if notepad.open_notepad():
            tasks_succeeded += 1
            time.sleep(1)
            
            # Type some text
            sample_text = ("This is a demonstration of NEXUS adaptive automation learning.\n\n"
                          "As I interact with applications, I learn UI patterns and become "
                          "more confident in my actions over time. I don't just follow fixed "
                          "rules - I adapt based on success and failure.")
            
            if notepad.type_text(sample_text):
                tasks_succeeded += 1
                time.sleep(1)
                
                # Use a menu option - Format -> Word Wrap
                if notepad.use_menu("Format", "Word Wrap"):
                    tasks_succeeded += 1
                    time.sleep(1)
                
                # Save the file
                if notepad.save_file("nexus_demo.txt"):
                    tasks_succeeded += 1
                    time.sleep(1)
            
            # Close Notepad
            if notepad.close_notepad():
                tasks_succeeded += 1
        
        # Show learning statistics
        print(f"\nTasks completed successfully: {tasks_succeeded}/5")
        notepad.show_learning_stats()
        
        # Ask if user wants to run the demo again to see learning improvements
        response = input("\nRun the demo again to see learning improvements? (yes/no): ")
        if response.lower() in ["yes", "y"]:
            # Run again with the same instance to demonstrate learning
            run_second_iteration(notepad)
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        logger.error(f"Error during demo: {e}")
    
    print("\n=== Demo Complete ===")
    print("Thank you for trying the NEXUS adaptive automation system!")

def run_second_iteration(notepad):
    """Run a second iteration of the demo to show learning improvements"""
    print("\n=== Running Second Iteration ===")
    print("Watch how NEXUS has learned from the first run and improved its confidence!\n")
    
    try:
        # Run the same sequence again
        tasks_succeeded = 0
        
        # Open Notepad
        if notepad.open_notepad():
            tasks_succeeded += 1
            time.sleep(1)
            
            # Type some new text
            sample_text = ("This is the second iteration of the automation demo.\n\n"
                          "Notice how NEXUS now has higher confidence in its actions "
                          "and requires fewer confirmations for tasks it has successfully "
                          "completed before. This adaptive learning is what makes NEXUS "
                          "different from rule-based automation systems.")
            
            if notepad.type_text(sample_text):
                tasks_succeeded += 1
                time.sleep(1)
                
                # Use a menu option - Format -> Font
                if notepad.use_menu("Format", "Font"):
                    tasks_succeeded += 1
                    time.sleep(1)
                    
                    # Press Escape to close the dialog
                    notepad.keyboard.press_key("escape")
                    time.sleep(0.5)
                
                # Save the file with a different name
                if notepad.save_file("nexus_demo_2.txt"):
                    tasks_succeeded += 1
                    time.sleep(1)
            
            # Close Notepad
            if notepad.close_notepad():
                tasks_succeeded += 1
        
        # Show improved learning statistics
        print(f"\nSecond run tasks completed successfully: {tasks_succeeded}/5")
        notepad.show_learning_stats()
        
    except KeyboardInterrupt:
        print("\nSecond iteration interrupted by user.")
    except Exception as e:
        logger.error(f"Error during second iteration: {e}")

if __name__ == "__main__":
    # Create memory directories if they don't exist
    os.makedirs("memory/applications/notepad/visual", exist_ok=True)
    os.makedirs("memory/applications/notepad/clarification", exist_ok=True)
    
    # Run the demo
    run_demo()
