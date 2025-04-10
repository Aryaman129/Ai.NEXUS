"""
NEXUS Adaptive Automation Demo

This script demonstrates the adaptive screen automation capabilities
of NEXUS, showing real mouse and keyboard control with learning.
"""
import os
import time
import logging
import argparse
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the Python path
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import NEXUS components
from src.ai_core.automation.mouse_controller import MouseController
from src.ai_core.automation.keyboard_controller import KeyboardController
from src.ai_core.automation.safety_manager import SafetyManager
from src.ai_core.automation.clarification_engine import ClarificationEngine
from src.ai_core.screen_analysis.visual_memory import VisualMemorySystem

def get_user_confirmation(message):
    """Get user confirmation for an action"""
    response = input(f"{message} (yes/no): ").lower()
    return response in ['yes', 'y']

def clarification_callback(question):
    """Callback function for clarification engine"""
    print("\nNEXUS needs clarification:")
    print(f"  {question}")
    response = input("Your response: ")
    return response

def setup_memory_directory():
    """Set up directory for storing visual memories"""
    memory_dir = Path("memory")
    memory_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for different memory types
    (memory_dir / "visual").mkdir(exist_ok=True)
    (memory_dir / "clarification").mkdir(exist_ok=True)
    
    return memory_dir

def demo_mouse_movement(mouse_controller, iterations=5):
    """Demonstrate adaptive mouse movement"""
    print("\n=== Demonstrating Adaptive Mouse Movement ===")
    print("The mouse will move in a pattern and learn from the movements")
    print("Observe how the movement becomes smoother over time")
    
    if not get_user_confirmation("Ready to start mouse movement demo?"):
        return
    
    # Get screen dimensions
    screen_width, screen_height = mouse_controller.screen_width, mouse_controller.screen_height
    
    # Define a pattern - rectangular movement
    points = [
        (screen_width // 4, screen_height // 4),
        (screen_width * 3 // 4, screen_height // 4),
        (screen_width * 3 // 4, screen_height * 3 // 4),
        (screen_width // 4, screen_height * 3 // 4),
        (screen_width // 4, screen_height // 4)
    ]
    
    # Repeat the pattern, demonstrating learning over time
    print("Moving mouse in rectangular pattern...")
    
    for iteration in range(iterations):
        # Show iteration number and average movement time
        stats = mouse_controller.get_performance_stats()
        avg_time = stats.get("avg_movement_time", 0)
        print(f"\nIteration {iteration+1}/{iterations} (Avg. movement time: {avg_time:.4f}s)")
        
        # Move through the pattern
        for i, (x, y) in enumerate(points):
            # Move with varying speed based on iteration (learning effect)
            result = mouse_controller.move_to(
                x, y, 
                speed_factor=1.0 + (iteration * 0.2)  # Speed increases with iterations
            )
            
            # Print coordinates and elapsed time
            print(f"  Point {i+1}: ({x}, {y}) - Time: {result['elapsed_time']:.4f}s")
            
            # Short pause between points
            time.sleep(0.2)
    
    # Return to center and show performance stats
    mouse_controller.move_to(screen_width // 2, screen_height // 2)
    
    # Show final performance stats
    stats = mouse_controller.get_performance_stats()
    print("\nFinal performance statistics after learning:")
    print(f"  Average movement time: {stats['avg_movement_time']:.4f}s")
    print(f"  Movement speed: {stats['movement_speed']} pixels/sec")
    print(f"  Natural movement: {'Enabled' if stats['natural_movement'] else 'Disabled'}")

def demo_adaptive_clarification(clarification_engine):
    """Demonstrate adaptive clarification with learning"""
    print("\n=== Demonstrating Adaptive Clarification ===")
    print("The system will ask questions and learn from your responses")
    
    if not get_user_confirmation("Ready to start clarification demo?"):
        return
    
    # Run several clarification scenarios to demonstrate learning
    scenarios = [
        ("ui_element_action", {
            "element_type": "button",
            "element_text": "Submit",
            "action": "click"
        }),
        ("ui_element_action", {
            "element_type": "button",
            "element_text": "Submit",  # Same as before to demonstrate learning
            "action": "click"
        }),
        ("ui_element_action", {
            "element_type": "link",
            "element_text": "Privacy Policy",
            "action": "click"
        }),
        ("dangerous_action", {
            "action": "delete file",
        }),
    ]
    
    # Initial confidence starts low
    confidence = 0.5
    
    # Run through scenarios
    for i, (scenario, context) in enumerate(scenarios):
        print(f"\nScenario {i+1}/{len(scenarios)} - {scenario}")
        print(f"Initial confidence: {confidence:.2f}")
        
        # Ask for clarification
        result = clarification_engine.ask_for_clarification(
            scenario=scenario,
            context=context,
            confidence=confidence
        )
        
        # Update confidence based on result
        confidence = result["confidence"]
        
        print(f"Decision: {'Proceed' if result['proceed'] else 'Do not proceed'}")
        print(f"Updated confidence: {confidence:.2f}")
        print(f"Learning: System is adapting confidence thresholds")
        
        # Short pause between scenarios
        time.sleep(1)
    
    # Show clarification statistics to demonstrate learning
    stats = clarification_engine.get_clarification_statistics()
    print("\nClarification statistics after learning:")
    print(f"  Total clarifications: {stats['total_clarifications']}")
    print(f"  Proceed rate: {stats['proceed_rate']:.2f}")
    print(f"  Current confidence threshold: {stats['current_confidence_threshold']:.2f}")
    print("  Most common scenarios:")
    for scenario, count in stats["most_common_scenarios"].items():
        print(f"    - {scenario}: {count}")

def demo_visual_memory(visual_memory, mouse_controller):
    """Demonstrate visual memory learning"""
    print("\n=== Demonstrating Visual Memory Learning ===")
    print("The system will simulate interacting with UI elements and learning from experiences")
    
    if not get_user_confirmation("Ready to start visual memory demo?"):
        return
    
    # Create mock screen regions as basic numpy arrays (simplified for demo)
    import numpy as np
    
    # Let's simulate finding and clicking some UI elements
    print("Simulating UI interactions with learning...")
    
    # Screen regions for demo
    regions = {
        "button1": np.random.rand(64, 64),  # Submit button
        "button2": np.random.rand(64, 64),  # Cancel button
        "textbox": np.random.rand(64, 64),  # Username field
    }
    
    # Simulate interactions with different success rates
    elements = [
        ("button1", "Submit", 3, 3),  # 3 successes, 0 failures
        ("button2", "Cancel", 1, 2),  # 1 success, 2 failures
        ("textbox", "Username", 2, 0),  # 2 successes, 0 failures
    ]
    
    # Simulate some UI interactions
    for element_id, text, successes, failures in elements:
        print(f"\nInteracting with: {text} ({element_id})")
        
        # Store the element in visual memory
        pattern = {
            "visual_signature": regions[element_id],
            "type": "button" if "button" in element_id else "text_field",
            "text": text,
            "bbox_size": (100, 40)
        }
        pattern_id = visual_memory.store_pattern(pattern)
        
        # Perform some successful interactions
        for i in range(successes):
            # Simulate clicking by moving mouse to random position (not actually clicking)
            x, y = mouse_controller.screen_width // 2, mouse_controller.screen_height // 2
            mouse_controller.move_to(
                x + np.random.randint(-100, 100),
                y + np.random.randint(-100, 100)
            )
            
            # Record successful interaction
            visual_memory.record_interaction(
                pattern_id=pattern_id,
                action="click",
                success=True,
                context={"window_title": "Demo Window"}
            )
            print(f"  Successful interaction {i+1}/{successes}")
            time.sleep(0.5)
        
        # Perform some failed interactions
        for i in range(failures):
            # Record failed interaction
            visual_memory.record_interaction(
                pattern_id=pattern_id,
                action="click",
                success=False,
                context={"window_title": "Demo Window"}
            )
            print(f"  Failed interaction {i+1}/{failures}")
            time.sleep(0.5)
    
    # Show memory statistics
    stats = visual_memory.get_statistics()
    print("\nVisual memory statistics after learning:")
    print(f"  Total patterns: {stats['total_patterns']}")
    print(f"  Successful interactions: {stats['successful_interactions']}")
    print(f"  Failed interactions: {stats['failed_interactions']}")
    print(f"  Overall success rate: {stats['overall_success_rate']:.2f}")
    print("  Element types:")
    for element_type, count in stats["element_types"].items():
        print(f"    - {element_type}: {count}")
    
    # Demonstrate how the system would enhance confidence for previously seen elements
    print("\nSimulating enhanced detection for previously seen elements...")
    # Create dummy UI elements that would come from a detector
    ui_elements = [
        {
            "type": "button",
            "confidence": 0.7,
            "text": "Submit",
            "bbox": (100, 100, 200, 140),
            "center": (150, 120)
        }
    ]
    
    # Create a dummy screen image
    screen_image = np.zeros((600, 800, 3), dtype=np.uint8)
    
    # Create a layout result with window context
    layout_results = {
        "window_title": "Demo Window",
        "application": "TestApp"
    }
    
    # This is the key part: enhance detection with visual memory
    enhanced = visual_memory.enhance_detection(layout_results, ui_elements, screen_image)
    
    # Show how confidence was enhanced
    for element in enhanced:
        original_confidence = element.get("confidence", 0)
        enhanced_confidence = element.get("enhanced_confidence", 0)
        element_type = element.get("type", "unknown")
        element_text = element.get("text", "")
        
        print(f"\nElement: {element_text} ({element_type})")
        print(f"  Original confidence: {original_confidence:.2f}")
        print(f"  Enhanced confidence: {enhanced_confidence:.2f}")
        print(f"  Confidence boost: {enhanced_confidence - original_confidence:.2f}")
        
        if "memory_match" in element:
            match = element["memory_match"]
            print(f"  Matched pattern: {match['pattern_id']}")
            print(f"  Similarity: {match['similarity']:.2f}")
            print(f"  Successful interactions: {match['successful_interactions']}")
            print(f"  Failed interactions: {match['failed_interactions']}")

def main():
    """Main function to run the NEXUS adaptive automation demo"""
    parser = argparse.ArgumentParser(description="NEXUS Adaptive Automation Demo")
    parser.add_argument("--mouse", action="store_true", help="Run mouse movement demo")
    parser.add_argument("--clarify", action="store_true", help="Run clarification demo")
    parser.add_argument("--visual", action="store_true", help="Run visual memory demo")
    parser.add_argument("--all", action="store_true", help="Run all demos")
    args = parser.parse_args()
    
    # Run all if no specific demo is selected
    if not (args.mouse or args.clarify or args.visual):
        args.all = True
    
    print("=== NEXUS Adaptive Automation Demo ===")
    print("This demo will showcase the adaptive learning capabilities")
    print("of the NEXUS system for screen automation tasks.")
    
    # Set up memory directory
    memory_dir = setup_memory_directory()
    
    # Create components for the demo
    safety_manager = SafetyManager()
    mouse_controller = MouseController(safety_manager=safety_manager)
    keyboard_controller = KeyboardController(safety_manager=safety_manager)
    
    visual_memory = VisualMemorySystem(config={
        "memory_path": str(memory_dir / "visual"),
        "max_patterns": 1000,
        "similarity_threshold": 0.7,
        "enable_learning": True
    })
    
    clarification_engine = ClarificationEngine(config={
        "memory_path": str(memory_dir / "clarification"),
        "confidence_threshold": 0.7,
        "enable_learning": True
    })
    clarification_engine.set_response_callback(clarification_callback)
    
    # Run demos based on arguments
    if args.all or args.mouse:
        demo_mouse_movement(mouse_controller)
    
    if args.all or args.clarify:
        demo_adaptive_clarification(clarification_engine)
    
    if args.all or args.visual:
        demo_visual_memory(visual_memory, mouse_controller)
    
    print("\n=== Demo Complete ===")
    print("Thank you for trying the NEXUS adaptive automation system!")

if __name__ == "__main__":
    main()
