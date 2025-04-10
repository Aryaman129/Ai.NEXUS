"""
Configuration file for pytest

This file modifies the Python path to ensure imports from src work correctly
in test files.
"""
import os
import sys
from pathlib import Path

# Add the project root directory to Python's module search path
# This allows imports from 'src' to work properly in test files
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
