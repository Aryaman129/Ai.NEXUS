"""
Simple test runner script for NEXUS

This script properly sets up the Python path to ensure imports from 'src' work correctly,
regardless of how the tests are executed.

Usage:
  python run_test.py path/to/test_file.py
"""
import os
import sys
import subprocess
from pathlib import Path

# Add the project root directory to Python's module search path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Check if a test file was specified
if len(sys.argv) > 1:
    test_file = sys.argv[1]
    print(f"Running test: {test_file}")
    
    # Run pytest on the specified file
    result = subprocess.run([sys.executable, "-m", "pytest", test_file, "-v"])
    sys.exit(result.returncode)
else:
    print("Please specify a test file to run.")
    print("Example: python run_test.py tests/automation/test_mouse_controller.py")
    sys.exit(1)
