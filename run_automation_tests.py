#!/usr/bin/env python
"""
Test runner for NEXUS Adaptive Screen Automation System

This script provides a convenient way to run tests for the adaptive screen automation
system, either individually or as a complete test suite. It includes performance monitoring
and visualizes learning progress over time.

Usage:
  python run_automation_tests.py --all                   Run all tests
  python run_automation_tests.py --component visual      Run visual memory tests only
  python run_automation_tests.py --component mouse       Run mouse controller tests only
  python run_automation_tests.py --component keyboard    Run keyboard controller tests only
  python run_automation_tests.py --component safety      Run safety manager tests only
  python run_automation_tests.py --component clarify     Run clarification engine tests only
  python run_automation_tests.py --integration           Run integration tests only
  python run_automation_tests.py --learning-curve        Run adaptive learning curve analysis
"""

import os
import sys
import time
import argparse
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the test modules
from src.ai_core.screen_analysis.visual_memory import VisualMemorySystem
from src.ai_core.automation.clarification_engine import ClarificationEngine


def run_component_tests(component):
    """Run tests for a specific component"""
    component_map = {
        "visual": "tests/screen_analysis/test_visual_memory.py",
        "mouse": "tests/automation/test_mouse_controller.py",
        "keyboard": "tests/automation/test_keyboard_controller.py",
        "safety": "tests/automation/test_safety_manager.py",
        "clarify": "tests/automation/test_clarification_engine.py"
    }
    
    if component not in component_map:
        print(f"Error: Unknown component '{component}'. Choose from: {', '.join(component_map.keys())}")
        return False
    
    test_path = component_map[component]
    print(f"Running {component} tests from {test_path}...")
    
    result = subprocess.run(["pytest", "-xvs", test_path], capture_output=True, text=True)
    print(result.stdout)
    
    if result.returncode == 0:
        print(f"✅ {component.capitalize()} tests passed successfully!")
        return True
    else:
        print(f"❌ {component.capitalize()} tests failed!")
        print(result.stderr)
        return False


def run_integration_tests():
    """Run the integration tests"""
    test_path = "tests/integration/test_adaptive_automation.py"
    print(f"Running integration tests from {test_path}...")
    
    result = subprocess.run(["pytest", "-xvs", test_path], capture_output=True, text=True)
    print(result.stdout)
    
    if result.returncode == 0:
        print("✅ Integration tests passed successfully!")
        return True
    else:
        print("❌ Integration tests failed!")
        print(result.stderr)
        return False


def run_all_tests():
    """Run all tests for the adaptive automation system"""
    components = ["visual", "mouse", "keyboard", "safety", "clarify"]
    all_passed = True
    
    # Run component tests
    for component in components:
        if not run_component_tests(component):
            all_passed = False
    
    # Run integration tests
    if not run_integration_tests():
        all_passed = False
    
    return all_passed


def analyze_learning_curve():
    """
    Analyze and visualize how the system learns over time
    by simulating increasingly complex UI interactions
    """
    print("Analyzing learning curve of the adaptive automation system...")
    
    # Create temporary directory for testing
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create memory system for testing
        visual_memory = VisualMemorySystem(config={"memory_path": temp_dir})
        
        # Create clarification engine for testing
        clarification_engine = ClarificationEngine(config={"memory_path": temp_dir})
        clarification_engine.set_response_callback(lambda q: "yes")
        
        # Metrics to track
        iterations = 20
        confidence_scores = []
        clarification_needs = []
        success_rates = []
        pattern_counts = []
        
        # Initial confidence level
        confidence = 0.6
        
        # Simulate learning over iterations
        for i in range(iterations):
            # Store confidence
            confidence_scores.append(confidence)
            
            # Check if clarification needed
            needs_clarification = clarification_engine.needs_clarification(confidence)
            clarification_needs.append(1 if needs_clarification else 0)
            
            # Simulate a UI pattern
            pattern = {
                "visual_signature": np.random.rand(64, 64),
                "type": "button",
                "text": f"Button {i}",
                "bbox_size": (100, 40)
            }
            
            # Store pattern
            pattern_id = visual_memory.store_pattern(pattern)
            
            # Record a successful interaction
            visual_memory.record_interaction(
                pattern_id=pattern_id,
                action="click",
                success=True,
                context={"window_title": "Test Window"}
            )
            
            # Add some failed interactions
            if i % 5 == 0:
                visual_memory.record_interaction(
                    pattern_id=pattern_id,
                    action="click",
                    success=False,
                    context={"window_title": "Test Window"}
                )
            
            # Get current stats
            stats = visual_memory.get_statistics()
            success_rate = stats.get("overall_success_rate", 0.0)
            success_rates.append(success_rate)
            pattern_counts.append(stats.get("total_patterns", 0))
            
            # Simulate confidence increase with experience
            confidence = min(0.95, confidence + 0.02)
            
            # Add clarification record if needed
            if needs_clarification:
                context = {
                    "element_type": "button",
                    "element_text": f"Button {i}",
                    "action": "click"
                }
                clarification_engine.ask_for_clarification(
                    scenario="ui_element_action",
                    context=context,
                    confidence=confidence
                )
        
        # Plot the results
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # First plot: Confidence and Clarification Needs
        iterations_range = range(1, iterations + 1)
        ax1.plot(iterations_range, confidence_scores, 'b-', label='Confidence')
        ax1.set_ylabel('Confidence Score', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_ylim(0, 1.0)
        
        ax1b = ax1.twinx()
        ax1b.bar(iterations_range, clarification_needs, color='r', alpha=0.3, label='Clarification Needed')
        ax1b.set_ylabel('Clarification Needed', color='r')
        ax1b.tick_params(axis='y', labelcolor='r')
        ax1b.set_ylim(0, 1.5)
        
        ax1.set_title('Adaptive Learning: Confidence & Clarification Needs Over Time')
        ax1.set_xlabel('Interaction Iteration')
        ax1.grid(True, alpha=0.3)
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1b.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
        
        # Second plot: Pattern Count and Success Rate
        ax2.plot(iterations_range, pattern_counts, 'g-', label='Pattern Count')
        ax2.set_ylabel('Pattern Count', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        
        ax2b = ax2.twinx()
        ax2b.plot(iterations_range, success_rates, 'm-', label='Success Rate')
        ax2b.set_ylabel('Success Rate', color='m')
        ax2b.tick_params(axis='y', labelcolor='m')
        ax2b.set_ylim(0, 1.1)
        
        ax2.set_title('Adaptive Learning: Pattern Growth & Success Rate')
        ax2.set_xlabel('Interaction Iteration')
        ax2.grid(True, alpha=0.3)
        
        # Add legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2b.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        
        # Create output directory if it doesn't exist
        output_dir = Path("test_results")
        output_dir.mkdir(exist_ok=True)
        
        # Save the figure
        learning_curve_path = output_dir / "adaptive_learning_curve.png"
        plt.savefig(learning_curve_path)
        plt.close()
        
        print(f"Learning curve analysis complete! Results saved to {learning_curve_path}")
        
        # Print some insights
        c_stats = clarification_engine.get_clarification_statistics()
        print("\nLearning Insights:")
        print(f"- Initial confidence: 0.60")
        print(f"- Final confidence: {confidence_scores[-1]:.2f}")
        print(f"- Total clarifications needed: {sum(clarification_needs)}")
        print(f"- Final success rate: {success_rates[-1]:.2f}")
        print(f"- Patterns stored: {pattern_counts[-1]}")
        
        # Calculate learning rate
        if iterations > 1:
            learning_rate = (confidence_scores[-1] - confidence_scores[0]) / iterations
            print(f"- Average confidence increase per iteration: {learning_rate:.4f}")
        
        # Calculate clarification reduction
        if sum(clarification_needs[:5]) > 0:
            initial_clarifications = sum(clarification_needs[:5])
            final_clarifications = sum(clarification_needs[-5:])
            reduction = (initial_clarifications - final_clarifications) / initial_clarifications
            print(f"- Clarification reduction rate: {reduction:.2%}")
        
        return True
        
    except Exception as e:
        print(f"Error in learning curve analysis: {e}")
        return False
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)


def print_system_summary():
    """Print a summary of the adaptive screen automation system"""
    print("\n" + "="*80)
    print("                  NEXUS ADAPTIVE SCREEN AUTOMATION SYSTEM")
    print("="*80)
    print("""
The adaptive screen automation system enables NEXUS to:

1. Capture and analyze screen contents in real-time using GPU acceleration
2. Detect and interact with UI elements using computer vision
3. Learn from successful interactions to improve future performance
4. Ask clarifying questions when uncertainty arises
5. Implement safety measures to prevent unintended consequences
6. Adapt to user behavior and preferences over time

This system represents a significant advancement in AI-human collaboration,
allowing NEXUS to become a true adaptive assistant that learns from experience
rather than following rigid rules.
""")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests for adaptive screen automation system")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Run all tests")
    group.add_argument("--component", choices=["visual", "mouse", "keyboard", "safety", "clarify"], 
                       help="Run tests for specific component")
    group.add_argument("--integration", action="store_true", help="Run integration tests")
    group.add_argument("--learning-curve", action="store_true", help="Run adaptive learning curve analysis")
    
    args = parser.parse_args()
    
    # Print system summary
    print_system_summary()
    
    start_time = time.time()
    
    # Run the specified tests
    if args.all:
        success = run_all_tests()
    elif args.component:
        success = run_component_tests(args.component)
    elif args.integration:
        success = run_integration_tests()
    elif args.learning-curve:
        success = analyze_learning_curve()
    
    elapsed_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*80)
    print(f"Test execution complete in {elapsed_time:.2f} seconds")
    print(f"Overall result: {'SUCCESS' if success else 'FAILURE'}")
    print("="*80 + "\n")
    
    sys.exit(0 if success else 1)
